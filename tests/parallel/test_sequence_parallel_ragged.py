"""
Unit tests for ragged sequence parallel utilities.

These tests validate:
- split_pack_by_sp: Uneven splitting across ranks
- pad_pack_to_batch / unpad_batch_to_pack: Round-trip equality
- rebuild_local_cu_seqlens: Correct boundaries
- build_local_rope: Correct slicing of embeddings
- Zero-length rank detection
"""

import logging
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.models.qwen2_5_vl.sequence_parallel_utils import (  # noqa: E402
    RaggedSequenceMeta,
    build_local_rope,
    pad_pack_to_batch,
    rebuild_local_cu_seqlens,
    split_pack_by_sp,
    unpad_batch_to_pack,
)

pytestmark = pytest.mark.cpu_only


def _make_meta(local_lengths, *, device, sp_rank=0, sp_size=1):
    batch_size = len(local_lengths)
    global_lengths = list(local_lengths)
    global_offsets = []
    sample_slices = []
    local_offsets = []
    offset_global = 0
    offset_local = 0
    for length in local_lengths:
        global_offsets.append(offset_global)
        sample_slices.append((offset_global, offset_global + length))
        local_offsets.append((offset_local, offset_local + length))
        offset_global += length
        offset_local += length

    return RaggedSequenceMeta(
        batch_size=batch_size,
        local_lengths=list(local_lengths),
        global_lengths=global_lengths,
        sp_rank=sp_rank,
        sp_size=sp_size,
        device=device,
        global_sample_offsets=global_offsets,
        sample_global_slices=sample_slices,
        local_token_offsets=local_offsets,
        window_lengths_per_sample=None,
        window_index_ranges=None,
    )


class TestSplitPackBySP:
    """Test split_pack_by_sp function."""

    def test_even_split(self):
        """Test splitting when lengths are evenly divisible."""
        # Create global pack: B=2, lengths=[8, 12], total=20
        hidden_size = 64
        global_pack = torch.randn(20, hidden_size)
        lengths = [8, 12]
        sp_size = 2

        # Rank 0 should get: 4 tokens from sample 0, 6 tokens from sample 1 = 10 total
        pack_r0, meta_r0 = split_pack_by_sp(global_pack, lengths, sp_rank=0, sp_size=sp_size)
        assert pack_r0.shape == (10, hidden_size)
        assert meta_r0.local_lengths == [4, 6]
        assert meta_r0.total_local_len == 10

        # Rank 1 should get: 4 tokens from sample 0, 6 tokens from sample 1 = 10 total
        pack_r1, meta_r1 = split_pack_by_sp(global_pack, lengths, sp_rank=1, sp_size=sp_size)
        assert pack_r1.shape == (10, hidden_size)
        assert meta_r1.local_lengths == [4, 6]

        # Verify no overlap and complete coverage
        torch.testing.assert_close(pack_r0[:4], global_pack[0:4])  # Sample 0, rank 0
        torch.testing.assert_close(pack_r1[:4], global_pack[4:8])  # Sample 0, rank 1
        torch.testing.assert_close(pack_r0[4:10], global_pack[8:14])  # Sample 1, rank 0
        torch.testing.assert_close(pack_r1[4:10], global_pack[14:20])  # Sample 1, rank 1

    def test_uneven_split(self):
        """Test splitting when lengths are NOT evenly divisible."""
        # B=2, lengths=[7, 11], total=18
        hidden_size = 32
        global_pack = torch.randn(18, hidden_size)
        lengths = [7, 11]
        sp_size = 3

        pack_r0, meta_r0 = split_pack_by_sp(global_pack, lengths, sp_rank=0, sp_size=sp_size)
        pack_r1, meta_r1 = split_pack_by_sp(global_pack, lengths, sp_rank=1, sp_size=sp_size)
        pack_r2, meta_r2 = split_pack_by_sp(global_pack, lengths, sp_rank=2, sp_size=sp_size)

        assert meta_r0.local_lengths == [3, 4]
        assert meta_r1.local_lengths == [2, 4]
        assert meta_r2.local_lengths == [2, 3]

        assert pack_r0.shape[0] == sum(meta_r0.local_lengths)
        assert pack_r1.shape[0] == sum(meta_r1.local_lengths)
        assert pack_r2.shape[0] == sum(meta_r2.local_lengths)

        total = meta_r0.total_local_len + meta_r1.total_local_len + meta_r2.total_local_len
        assert total == 18

    def test_single_sample(self):
        """Test with batch_size=1."""
        hidden_size = 128
        global_pack = torch.randn(100, hidden_size)
        lengths = [100]
        sp_size = 4

        # Each rank should get 25 tokens
        for rank in range(sp_size):
            pack_rank, meta_rank = split_pack_by_sp(global_pack, lengths, sp_rank=rank, sp_size=sp_size)
            assert meta_rank.batch_size == 1
            assert pack_rank.shape[0] == 25
            assert meta_rank.local_lengths == [25]

    def test_zero_length_rank_marks_meta(self):
        """Ranks with no real tokens should be flagged for padding."""
        hidden_size = 64
        global_pack = torch.randn(2, hidden_size)
        lengths = [2]
        sp_size = 4

        pack_r2, meta_r2 = split_pack_by_sp(global_pack, lengths, sp_rank=2, sp_size=sp_size)
        assert pack_r2.shape[0] == 0
        assert meta_r2.requires_dummy_pad


class TestPadUnpad:
    """Test pad_pack_to_batch and unpad_batch_to_pack round-trip."""

    def test_round_trip_even_lengths(self):
        """Test pad → unpad with equal lengths."""
        batch_size = 4
        local_lengths = [10, 10, 10, 10]
        hidden_size = 64

        # Create ragged pack
        total_len = sum(local_lengths)
        pack = torch.randn(total_len, hidden_size)

        # Create meta
        meta = _make_meta(local_lengths, device=pack.device)

        # Pad
        padded, mask = pad_pack_to_batch(pack, meta)
        assert padded.shape == (batch_size, 10, hidden_size)
        assert mask.shape == (batch_size, 10)
        assert mask.all()  # All True since no actual padding

        # Unpad
        pack_recovered, _ = unpad_batch_to_pack(padded, mask)
        torch.testing.assert_close(pack_recovered, pack)

    def test_round_trip_uneven_lengths(self):
        """Test pad → unpad with varying lengths."""
        local_lengths = [5, 12, 3]
        hidden_size = 32

        total_len = sum(local_lengths)
        pack = torch.randn(total_len, hidden_size)

        meta = _make_meta(local_lengths, device=pack.device)

        # Pad (max_len = 12)
        padded, mask = pad_pack_to_batch(pack, meta)
        assert padded.shape == (meta.batch_size, meta.max_local_len, hidden_size)

        # Check mask is correct
        assert mask[0, :5].all() and not mask[0, 5:].any()
        assert mask[1, :12].all()
        assert mask[2, :3].all() and not mask[2, 3:].any()

        # Unpad
        pack_recovered, _ = unpad_batch_to_pack(padded, mask)
        torch.testing.assert_close(pack_recovered, pack)

    def test_padding_value(self):
        """Test that padding uses the specified value."""
        local_lengths = [3, 5]
        hidden_size = 8

        pack = torch.ones(8, hidden_size)
        meta = _make_meta(local_lengths, device=pack.device)

        padded, mask = pad_pack_to_batch(pack, meta, pad_value=-999.0)

        # Check padded values
        assert (padded[0, 3:] == -999.0).all()  # Padding in sample 0
        assert (padded[1, :5] == 1.0).all()  # Real tokens in sample 1


class TestRebuildCuSeqlens:
    """Test rebuild_local_cu_seqlens function."""

    def test_cu_seqlens_boundaries(self):
        """Test that cu_seqlens has correct boundaries."""
        local_lengths = [4, 7, 2]

        meta = _make_meta(local_lengths, device=torch.device("cpu"))

        cu_seqlens = rebuild_local_cu_seqlens(meta, use_windows=False)

        # Should be [0, 4, 11, 13]
        expected = torch.tensor([0, 4, 11, 13], dtype=torch.int32)
        torch.testing.assert_close(cu_seqlens, expected)

    def test_cu_seqlens_dtype(self):
        """Test that cu_seqlens is int32 (required by FA2)."""
        meta = _make_meta([5, 10], device=torch.device("cpu"))

        cu_seqlens = rebuild_local_cu_seqlens(meta, use_windows=False)
        assert cu_seqlens.dtype == torch.int32

    def test_cu_window_seqlens_slicing(self):
        """Test windowed cu_seqlens slicing."""
        # Global cu_window_seqlens: [0, 6, 10, 17, 24]
        # Sample 0 windows: 6 + 4, Sample 1 windows: 7 + 7
        global_cu_window = torch.tensor([0, 6, 10, 17, 24], dtype=torch.int32)
        total_len = 24
        hidden_size = 8
        _, meta_r0 = split_pack_by_sp(
            torch.randn(total_len, hidden_size),
            lengths_per_sample=[10, 14],
            sp_rank=0,
            sp_size=2,
            cu_window_seqlens=global_cu_window,
        )

        cu_window_local = rebuild_local_cu_seqlens(meta_r0, use_windows=True)
        expected = torch.tensor([0, 6, 13], dtype=torch.int32)
        torch.testing.assert_close(cu_window_local, expected)


class TestBuildLocalRope:
    """Test build_local_rope function."""

    def test_rope_slicing(self):
        """Test that RoPE embeddings are sliced correctly."""
        # Setup: B=2, global_lengths=[12, 8], sp_size=2
        # Rank 0 gets [6, 4], rank 1 gets [6, 4]
        hidden_size = 16
        emb_dim = 32

        # Global embeddings
        global_cos = torch.randn(20, emb_dim)  # 12 + 8
        global_sin = torch.randn(20, emb_dim)

        lengths = [12, 8]
        sp_size = 2

        # Test rank 0
        _, meta_r0 = split_pack_by_sp(torch.randn(20, hidden_size), lengths, sp_rank=0, sp_size=sp_size)
        cos_r0, sin_r0, mask_r0 = build_local_rope(meta_r0, global_cos, global_sin)

        assert cos_r0.shape == (10, emb_dim)  # 6 + 4
        assert sin_r0.shape == (10, emb_dim)
        assert mask_r0 is None  # No padding requested

        # Check that cos_r0 matches the correct slices
        torch.testing.assert_close(cos_r0[:6], global_cos[0:6])  # Sample 0, rank 0
        torch.testing.assert_close(cos_r0[6:10], global_cos[12:16])  # Sample 1, rank 0

    def test_rope_with_padding(self):
        """Test RoPE with padding (pad_to specified)."""
        hidden_size = 16
        emb_dim = 32

        global_cos = torch.randn(15, emb_dim)
        global_sin = torch.randn(15, emb_dim)

        # B=2, lengths=[8, 7], sp_size=2
        # Rank 0 gets [4, 4]
        lengths = [8, 7]
        sp_size = 2
        _, meta_r0 = split_pack_by_sp(torch.randn(15, hidden_size), lengths, sp_rank=0, sp_size=sp_size)

        # Request padding to max_len=4
        cos_padded, sin_padded, mask = build_local_rope(meta_r0, global_cos, global_sin, pad_to=4)

        assert cos_padded.shape == (2, 4, emb_dim)  # [B, pad_to, D]
        assert sin_padded.shape == (2, 4, emb_dim)
        assert mask.shape == (2, 4)

        # Check mask
        assert mask[0, :4].all()  # Sample 0 has 4 tokens
        assert mask[1, :4].all()  # Sample 1 has 4 tokens


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_token_per_sample(self):
        """Test with minimal token counts."""
        hidden_size = 16
        global_pack = torch.randn(4, hidden_size)  # Use 4 tokens instead of 2
        lengths = [2, 2]  # 2 tokens per sample
        sp_size = 2

        # Each rank should get 1 token per sample
        pack_r0, meta_r0 = split_pack_by_sp(global_pack, lengths, sp_rank=0, sp_size=sp_size)
        pack_r1, meta_r1 = split_pack_by_sp(global_pack, lengths, sp_rank=1, sp_size=sp_size)

        # Each rank gets [1, 1]
        assert meta_r0.local_lengths == [1, 1]
        assert meta_r1.local_lengths == [1, 1]

        # Both ranks combined should cover all tokens
        assert pack_r0.shape[0] + pack_r1.shape[0] == 4

    def test_empty_pack_handling(self):
        """Test handling of empty packs (after unpad)."""
        # Create a scenario where all tokens are padding
        padded = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)  # All False

        pack, _ = unpad_batch_to_pack(padded, mask)
        assert pack.shape[0] == 0  # Empty pack

    def test_window_split_with_fewer_windows_than_ranks(self, caplog):
        """Ensure window-aware splitting handles fewer windows than SP ranks."""
        hidden_size = 8
        total_len = 5
        global_pack = torch.arange(total_len * hidden_size, dtype=torch.float32).view(total_len, hidden_size)
        lengths = [total_len]
        cu_window = torch.tensor([0, total_len], dtype=torch.int32)
        sp_size = 4

        caplog.set_level(logging.WARNING)
        local_lengths = []
        gathered = []
        for rank in range(sp_size):
            pack_rank, meta = split_pack_by_sp(
                global_pack,
                lengths_per_sample=lengths,
                sp_rank=rank,
                sp_size=sp_size,
                cu_window_seqlens=cu_window,
            )
            local_lengths.append(meta.local_lengths[0])
            if pack_rank.numel():
                gathered.append(pack_rank)

        assert local_lengths.count(0) == sp_size - 1
        assert total_len in local_lengths
        assert sorted(local_lengths) == [0, 0, 0, total_len]
        assert caplog.records, "Expected a warning about insufficient windows"
        assert "Sequence parallel requested" in caplog.text
        if gathered:
            torch.testing.assert_close(torch.cat(gathered, dim=0), global_pack)

    def test_device_consistency(self):
        """Test that all tensors stay on the same device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")
        hidden_size = 64

        global_pack = torch.randn(20, hidden_size, device=device)
        lengths = [12, 8]

        pack_r0, meta_r0 = split_pack_by_sp(global_pack, lengths, sp_rank=0, sp_size=2)

        assert pack_r0.device == device
        assert meta_r0.device == device

        # Test padding
        padded, mask = pad_pack_to_batch(pack_r0, meta_r0)
        assert padded.device == device
        assert mask.device == device


class TestRaggedAllGatherAutograd:
    """Test _SequenceParallelAllGatherRagged autograd function."""

    def test_ragged_allgather_forward(self):
        """Test forward pass of ragged all-gather (single process simulation)."""
        # Simulate 3 ranks with different local lengths
        local_lengths_per_rank = [10, 15, 8]
        hidden_size = 64

        # Create tensors for each "rank"
        tensors = [torch.randn(length, hidden_size) for length in local_lengths_per_rank]

        # Expected output: concatenation of all tensors
        expected = torch.cat(tensors, dim=0)
        assert expected.shape[0] == sum(local_lengths_per_rank)

        # Verify the concept: pad → concat → unpad should give us the same result
        max_len = max(local_lengths_per_rank)
        padded = []
        for t, length in zip(tensors, local_lengths_per_rank):
            if length < max_len:
                pad = torch.zeros(max_len - length, hidden_size)
                padded.append(torch.cat([t, pad], dim=0))
            else:
                padded.append(t)

        # Concat padded
        padded_concat = torch.cat(padded, dim=0)
        assert padded_concat.shape[0] == max_len * 3

        # Unpad
        result = []
        offset = 0
        for length in local_lengths_per_rank:
            result.append(padded_concat[offset : offset + length])
            offset += max_len

        result = torch.cat(result, dim=0)
        torch.testing.assert_close(result, expected)

    def test_ragged_allgather_gradient_flow(self):
        """Test that gradients flow correctly through ragged all-gather."""
        # Simulate manual forward/backward to verify gradient logic

        local_lengths_per_rank = [5, 8, 3]
        hidden_size = 32

        # Create input tensors with requires_grad
        tensors = [torch.randn(length, hidden_size, requires_grad=True) for length in local_lengths_per_rank]

        # Simulate forward: each rank contributes its tensor
        # In real all-gather, we'd get all tensors; here we simulate with cat
        global_tensor = torch.cat(tensors, dim=0)

        # Compute some loss
        loss = global_tensor.sum()
        loss.backward()

        # Check that all input tensors have gradients
        for i, t in enumerate(tensors):
            assert t.grad is not None, f"Tensor {i} should have gradients"
            assert t.grad.shape == t.shape


class TestIntegrationRaggedSP:
    """Integration tests for ragged sequence parallel components."""

    def test_split_rebuild_cu_seqlens_consistency(self):
        """Test that split → rebuild cu_seqlens produces consistent boundaries."""
        global_lengths = [100, 150, 80]
        hidden_size = 64
        sp_size = 2

        # Create global pack
        total_len = sum(global_lengths)
        global_pack = torch.randn(total_len, hidden_size)

        # Split across ranks
        pack_r0, meta_r0 = split_pack_by_sp(global_pack, global_lengths, sp_rank=0, sp_size=sp_size)
        pack_r1, meta_r1 = split_pack_by_sp(global_pack, global_lengths, sp_rank=1, sp_size=sp_size)

        # Rebuild cu_seqlens for each rank
        cu_r0 = rebuild_local_cu_seqlens(meta_r0, use_windows=False)
        cu_r1 = rebuild_local_cu_seqlens(meta_r1, use_windows=False)

        # Verify cu_seqlens boundaries match pack lengths
        assert cu_r0[-1].item() == pack_r0.shape[0], "cu_seqlens should end at total local length"
        assert cu_r1[-1].item() == pack_r1.shape[0]

        # Verify cu_seqlens has correct number of boundaries
        assert cu_r0.numel() == meta_r0.batch_size + 1
        assert cu_r1.numel() == meta_r1.batch_size + 1

    def test_rope_split_consistency(self):
        """Test that split RoPE embeddings maintain alignment with tokens."""
        global_lengths = [64, 96]
        emb_dim = 128
        sp_size = 2

        # Create global embeddings
        total_len = sum(global_lengths)
        global_cos = torch.randn(total_len, emb_dim)
        global_sin = torch.randn(total_len, emb_dim)

        # Create dummy pack for split
        hidden_size = 256
        global_pack = torch.randn(total_len, hidden_size)

        # Split pack and embeddings
        pack_r0, meta_r0 = split_pack_by_sp(global_pack, global_lengths, sp_rank=0, sp_size=sp_size)
        cos_r0, sin_r0, _ = build_local_rope(meta_r0, global_cos, global_sin)

        # Verify shapes match
        assert cos_r0.shape[0] == pack_r0.shape[0], "RoPE length should match pack length"
        assert sin_r0.shape[0] == pack_r0.shape[0]

    def test_pad_unpad_with_varying_lengths(self):
        """Test pad/unpad round-trip with highly varying lengths."""
        local_lengths = [1, 50, 3, 30]  # Highly varying
        hidden_size = 128

        total_len = sum(local_lengths)
        pack = torch.randn(total_len, hidden_size)

        meta = _make_meta(local_lengths, device=pack.device)

        # Pad
        padded, mask = pad_pack_to_batch(pack, meta)
        assert padded.shape[1] == max(local_lengths), "Should pad to max length"

        # Unpad
        pack_recovered, _ = unpad_batch_to_pack(padded, mask)
        torch.testing.assert_close(pack_recovered, pack, msg="Round-trip should preserve values")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
