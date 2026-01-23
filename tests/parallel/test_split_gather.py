"""
Test split and gather operations to verify data preservation.

This test verifies that:
1. split_pack_by_sp splits data correctly
2. _SequenceParallelAllGather reconstructs data exactly
3. The round-trip preserves all information
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.models.qwen2_5_vl.modeling_qwen2_5_vl import _SequenceParallelAllGatherRagged  # noqa: E402
from python.models.qwen2_5_vl.sequence_parallel_utils import split_pack_by_sp  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for sequence-parallel collective test", allow_module_level=True)


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def _test_split_and_gather_equal_lengths():
    """Test split → gather with equal lengths."""
    rank, world_size = dist.get_rank(), dist.get_world_size()

    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Test: Split → Gather (Equal Lengths)")
    print(f"[Rank {rank}] " + "=" * 60)

    # Create test data on rank 0
    if rank == 0:
        seq_len_per_sample = 1024
        hidden_dim = 1280

        # Create unique values so we can track if they're preserved
        global_pack = torch.arange(seq_len_per_sample * hidden_dim, dtype=torch.float32, device="cuda").reshape(
            seq_len_per_sample, hidden_dim
        )

        lengths = [seq_len_per_sample]

        print(f"[Rank {rank}] Global pack shape: {global_pack.shape}")
        print(f"[Rank {rank}] Global pack sum: {global_pack.sum().item():.2f}")
        print(f"[Rank {rank}] Global pack[0, 0:5]: {global_pack[0, :5]}")
        print(f"[Rank {rank}] Global pack[511, 0:5]: {global_pack[511, :5]}")
        print(f"[Rank {rank}] Global pack[512, 0:5]: {global_pack[512, :5]}")
    else:
        global_pack = torch.zeros(1024, 1280, device="cuda")
        lengths = [1024]

    # Broadcast
    dist.broadcast(global_pack, src=0)

    # Split
    pack_local, meta = split_pack_by_sp(global_pack, lengths, rank, world_size, debug=True)

    print(f"[Rank {rank}] Local pack shape: {pack_local.shape}")
    print(f"[Rank {rank}] Local pack sum: {pack_local.sum().item():.2f}")
    print(f"[Rank {rank}] Local pack[0, 0:5]: {pack_local[0, :5]}")

    # Verify split is correct
    expected_len = sum(meta.local_lengths)
    assert pack_local.shape[0] == expected_len, f"Expected {expected_len}, got {pack_local.shape[0]}"

    # Verify the values are correct slices
    if rank == 0:
        expected_slice = global_pack[:expected_len]
        if torch.allclose(pack_local, expected_slice):
            print(f"[Rank {rank}] ✅ Rank 0 slice is correct")
        else:
            print(f"[Rank {rank}] ❌ Rank 0 slice is WRONG!")

    # All-gather back
    group = dist.new_group(ranks=list(range(world_size)))

    local_lengths_tensor = torch.tensor(meta.local_lengths, device=pack_local.device, dtype=torch.int32)
    gathered_lengths = [torch.zeros_like(local_lengths_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, local_lengths_tensor, group=group)
    lengths_matrix = torch.stack(gathered_lengths, dim=0)

    gathered = _SequenceParallelAllGatherRagged.apply(pack_local, meta.total_local_len, group)

    hidden_dim = pack_local.shape[-1]
    rank_chunks = []
    start = 0
    for r in range(world_size):
        total = lengths_matrix[r].sum().item()
        if total > 0:
            rank_chunks.append(gathered[start : start + total])
        else:
            rank_chunks.append(gathered.new_empty((0, hidden_dim)))
        start += total

    rank_offsets = [0] * world_size
    sample_chunks = []
    for sample_idx in range(meta.batch_size):
        parts = []
        for rank_idx in range(world_size):
            part_len = lengths_matrix[rank_idx, sample_idx].item()
            if part_len == 0:
                continue
            offset = rank_offsets[rank_idx]
            parts.append(rank_chunks[rank_idx][offset : offset + part_len])
            rank_offsets[rank_idx] += part_len
        if parts:
            sample_chunks.append(torch.cat(parts, dim=0))

    reconstructed = torch.cat(sample_chunks, dim=0) if sample_chunks else gathered.new_empty((0, hidden_dim))

    print(f"[Rank {rank}] Gathered shape: {gathered.shape}")
    print(f"[Rank {rank}] Gathered sum: {gathered.sum().item():.2f}")

    # Compare with original
    if torch.allclose(reconstructed, global_pack, atol=1e-6):
        print(f"[Rank {rank}] ✅ All-gather EXACTLY reconstructs original!")
    else:
        diff = (reconstructed - global_pack).abs()
        max_diff = diff.max().item()
        print(f"[Rank {rank}] ❌ All-gather differs! Max diff: {max_diff:.6e}")

        # Find where differences are
        large_diff = diff > 1e-6
        if large_diff.any():
            indices = torch.where(large_diff)
            print(f"[Rank {rank}] Differences at positions (first 10): {indices[0][:10].tolist()}")


def _test_split_and_gather_uneven_lengths():
    """Test split → gather with uneven lengths."""
    rank, world_size = dist.get_rank(), dist.get_world_size()

    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Test: Split → Gather (Uneven Lengths)")
    print(f"[Rank {rank}] " + "=" * 60)

    # Create test data with slightly uneven length
    if rank == 0:
        seq_len = 1020  # Not divisible by 2
        hidden_dim = 1280

        global_pack = torch.arange(seq_len * hidden_dim, dtype=torch.float32, device="cuda").reshape(
            seq_len, hidden_dim
        )

        lengths = [seq_len]

        print(f"[Rank {rank}] Global pack shape: {global_pack.shape}")
        print(f"[Rank {rank}] Global pack sum: {global_pack.sum().item():.2f}")
    else:
        global_pack = torch.zeros(1020, 1280, device="cuda")
        lengths = [1020]

    # Broadcast
    dist.broadcast(global_pack, src=0)

    # Split (ceiling division: rank 0 gets 510, rank 1 gets 510)
    pack_local, meta = split_pack_by_sp(global_pack, lengths, rank, world_size, debug=True)

    print(f"[Rank {rank}] Local pack shape: {pack_local.shape}")
    print(f"[Rank {rank}] Local pack sum: {pack_local.sum().item():.2f}")

    # All-gather back
    group = dist.new_group(ranks=list(range(world_size)))

    local_lengths_tensor = torch.tensor(meta.local_lengths, device=pack_local.device, dtype=torch.int32)
    gathered_lengths = [torch.zeros_like(local_lengths_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, local_lengths_tensor, group=group)
    lengths_matrix = torch.stack(gathered_lengths, dim=0)

    gathered = _SequenceParallelAllGatherRagged.apply(pack_local, meta.total_local_len, group)

    hidden_dim = pack_local.shape[-1]
    rank_chunks = []
    start = 0
    for r in range(world_size):
        total = lengths_matrix[r].sum().item()
        if total > 0:
            rank_chunks.append(gathered[start : start + total])
        else:
            rank_chunks.append(gathered.new_empty((0, hidden_dim)))
        start += total

    rank_offsets = [0] * world_size
    sample_chunks = []
    for sample_idx in range(meta.batch_size):
        parts = []
        for rank_idx in range(world_size):
            part_len = lengths_matrix[rank_idx, sample_idx].item()
            if part_len == 0:
                continue
            offset = rank_offsets[rank_idx]
            parts.append(rank_chunks[rank_idx][offset : offset + part_len])
            rank_offsets[rank_idx] += part_len
        if parts:
            sample_chunks.append(torch.cat(parts, dim=0))

    reconstructed = torch.cat(sample_chunks, dim=0) if sample_chunks else gathered.new_empty((0, hidden_dim))

    print(f"[Rank {rank}] Gathered shape: {gathered.shape}")
    print(f"[Rank {rank}] Gathered sum: {gathered.sum().item():.2f}")

    # Compare with original
    if torch.allclose(reconstructed, global_pack, atol=1e-6):
        print(f"[Rank {rank}] ✅ All-gather EXACTLY reconstructs original (uneven case)!")
    else:
        diff = (reconstructed - global_pack).abs()
        max_diff = diff.max().item()
        print(f"[Rank {rank}] ❌ All-gather differs! Max diff: {max_diff:.6e}")


def test_split_and_gather():
    """Main test - entry point for pytest."""
    rank, world_size = init_dist()
    print(f"[Rank {rank}] Initialized (world_size={world_size})")

    # Test 1: Equal lengths
    _test_split_and_gather_equal_lengths()

    if dist.is_initialized():
        dist.barrier()

    # Test 2: Uneven lengths
    _test_split_and_gather_uneven_lengths()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    test_split_and_gather()
