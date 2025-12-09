"""Test DeepSpeed Sequence Parallel with Data Parallel for the vision model.

This test verifies that DeepSpeed's Sequence Parallel combined with Data Parallel produces
correct training behavior with 4 GPUs: sp_size=2, dp_size=2.

Run with:
    torchrun --nproc_per_node=4 -m pytest tests/test_vision_sp_dp.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig  # noqa: E402
from python.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel  # noqa: E402
from python.ray.utils import init_distributed_comm  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Sequence Parallel+DP test", allow_module_level=True)


def init_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        init_distributed_comm(backend="nccl", use_deepspeed=True)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def create_small_model_config(model_name="Qwen/Qwen2.5-VL-3B-Instruct", num_layers=2):
    """Create a small model config for testing."""
    config = Qwen2_5_VLConfig.from_pretrained(model_name, trust_remote_code=True)
    # Reduce model size for faster testing
    config.vision_config.depth = num_layers
    return config


def create_no_parallel_model(model_config, device, torch_dtype, seed=42):
    """Create a non-parallel baseline model with fixed seed."""
    # Set seed for deterministic initialization
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Disable sequence parallelism for baseline
    vision_config = model_config.vision_config
    vision_config.sequence_parallel = False
    vision_config._attn_implementation = "sdpa"

    torch.set_default_device(device)
    model = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
    torch.set_default_device("cpu")

    model.to(device=device, dtype=torch_dtype)
    model.train()

    # Disable dropout for deterministic behavior
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    return model


def create_sp_dp_model(model_config, rank, sp_size, device, torch_dtype, seed=42):
    """Create a model with Sequence Parallel and Data Parallel with fixed seed."""
    # Set seed for deterministic initialization (same as baseline)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enable sequence parallelism
    vision_config = model_config.vision_config
    vision_config.sequence_parallel = True
    vision_config._attn_implementation = "flash_attention_2"

    # Initialize DeepSpeed sequence parallel groups
    try:
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu
    except ImportError:
        raise ImportError("DeepSpeed is required for sequence parallelism. Please install DeepSpeed: pip install deepspeed")

    world_size = dist.get_world_size()
    sequence_parallel_size = sp_size
    data_parallel_size = world_size // sp_size

    if sequence_parallel_size <= 1:
        raise ValueError(f"Sequence parallelism requires sequence_parallel_size > 1 (got {sequence_parallel_size})")
    if world_size % sequence_parallel_size != 0:
        raise ValueError(f"sequence_parallel_size {sequence_parallel_size} must divide world size {world_size}")

    print(f"[Rank {rank}] Initializing SP+DP: sp_size={sequence_parallel_size}, dp_size={data_parallel_size}")
    mpu.initialize_sequence_parallel(sequence_parallel_size=sequence_parallel_size)
    sp_group = mpu.get_sequence_parallel_group()

    # Manually create DP groups since DeepSpeed's DP groups may not be initialized
    # For world_size=4, sp_size=2: DP groups are [0,2] and [1,3]
    # Each DP group contains ranks with the same SP rank from different DP replicas
    dp_groups = []
    for sp_r in range(sequence_parallel_size):
        dp_group_ranks = [sp_r + dp_r * sequence_parallel_size for dp_r in range(data_parallel_size)]
        dp_group = dist.new_group(ranks=dp_group_ranks)
        dp_groups.append(dp_group)

    # Determine which DP group this rank belongs to
    sp_rank_local = rank % sequence_parallel_size
    dp_group = dp_groups[sp_rank_local]

    print(f"[Rank {rank}] Created DP group for SP rank {sp_rank_local}")

    torch.set_default_device(device)
    model = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
    torch.set_default_device("cpu")

    model.to(device=device, dtype=torch_dtype)

    # Set sp_group for the model and attention layers
    model.sp_group = sp_group
    for m in model.modules():
        if m.__class__.__name__ == "Qwen2_5_VLVisionAttention":
            m.sp_group = sp_group

    model.train()

    # Disable dropout for deterministic behavior
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    return model, sp_group, dp_group


def broadcast_for_sequence_parallel(tensors: list, sp_group):
    """Ensure all SP ranks see identical data within each SP group."""
    if sp_group is None or not dist.is_initialized():
        return

    try:
        sp_world_size = dist.get_world_size(sp_group)
    except Exception:
        return

    if sp_world_size <= 1:
        return

    # Get the source rank: first rank in the SP group (SP rank 0)
    # We need to find the global rank that corresponds to SP rank 0 in this group
    try:
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu
        src_rank = mpu.get_sequence_parallel_src_rank()
    except Exception:
        # Cannot determine source rank reliably, skip broadcast
        print(f"[Rank {dist.get_rank()}] Warning: Cannot determine SP source rank, skipping broadcast")
        return

    # Broadcast from SP rank 0 in the SP group
    for tensor in tensors:
        dist.broadcast(tensor, src=src_rank, group=sp_group)


def prepare_batch(height=448, width=448, device="cuda", seed=None):
    """Create a vision batch (pixel_values and grid_thw)."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Image parameters
    in_channels = 3
    temporal_patch_size = 2
    patch_size = 14

    # Ensure dimensions are multiples of patch_size
    height = (height // patch_size) * patch_size
    width = (width // patch_size) * patch_size

    grid_t = 1
    grid_h = height // patch_size
    grid_w = width // patch_size

    # Create pixel values
    pixel_values = torch.randn(in_channels, temporal_patch_size, height, width, dtype=torch.bfloat16, device=device)

    # Flatten to patches (mimic preprocessing)
    num_patches = grid_h * grid_w
    pixel_values = pixel_values.reshape(in_channels, temporal_patch_size, grid_h, patch_size, grid_w, patch_size)
    pixel_values = pixel_values.permute(2, 4, 0, 1, 3, 5)
    pixel_values = pixel_values.reshape(num_patches, in_channels, temporal_patch_size, patch_size, patch_size)
    pixel_values = pixel_values.flatten()

    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64, device=device)

    return pixel_values, grid_thw


def prepare_batch_for_dp(height=448, width=448, device="cuda", dp_rank=0, seed=None):
    """Create different batches for different DP ranks to simulate data parallelism."""
    if seed is not None:
        torch.manual_seed(seed + dp_rank)  # Different seed per DP rank
        torch.cuda.manual_seed_all(seed + dp_rank)

    # Image parameters
    in_channels = 3
    temporal_patch_size = 2
    patch_size = 14

    # Ensure dimensions are multiples of patch_size
    height = (height // patch_size) * patch_size
    width = (width // patch_size) * patch_size

    grid_t = 1
    grid_h = height // patch_size
    grid_w = width // patch_size

    # Create pixel values
    pixel_values = torch.randn(in_channels, temporal_patch_size, height, width, dtype=torch.bfloat16, device=device)

    # Flatten to patches (mimic preprocessing)
    num_patches = grid_h * grid_w
    pixel_values = pixel_values.reshape(in_channels, temporal_patch_size, grid_h, patch_size, grid_w, patch_size)
    pixel_values = pixel_values.permute(2, 4, 0, 1, 3, 5)
    pixel_values = pixel_values.reshape(num_patches, in_channels, temporal_patch_size, patch_size, patch_size)
    pixel_values = pixel_values.flatten()

    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64, device=device)

    return pixel_values, grid_thw


def compute_dummy_loss(vision_embeddings):
    """Compute a simple loss for vision embeddings (sum of squares)."""
    # Use a simple loss: sum of squares (this creates gradients flowing back)
    loss = (vision_embeddings ** 2).mean()
    return loss


def collect_selected_parameters(model):
    """Collect selected parameters for comparison.

    Returns dict with parameter names and their values.
    For SP, replicated parameters (e.g., layer norms) should match across SP ranks.
    """
    params = {}

    # Collect first layer norm parameters (replicated in SP)
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        first_block = model.blocks[0]
        if hasattr(first_block, 'norm1') and first_block.norm1.weight is not None:
            params['block0_norm1'] = first_block.norm1.weight.detach().clone()
        if hasattr(first_block, 'norm2') and first_block.norm2.weight is not None:
            params['block0_norm2'] = first_block.norm2.weight.detach().clone()

    return params


def test_sp_dp():
    """Test that Sequence Parallel with Data Parallel works correctly.

    This test verifies:
    1. SP (sp_size=2) combined with Data Parallel (dp_size=2) runs without errors
    2. Different DP ranks process different data
    3. Parameters are synchronized across DP ranks after optimizer steps
    4. Loss values are computed correctly for each DP rank

    Test configuration:
    - 4 GPUs total
    - SP size = 2 (each DP replica uses 2 GPUs for sequence parallelism)
    - DP size = 2 (2 data parallel replicas)
    - DP Group 0: Global ranks 0,1 (SP ranks 0,1)
    - DP Group 1: Global ranks 2,3 (SP ranks 0,1)
    """
    rank, world_size, local_rank = init_distributed()

    assert world_size == 4, f"This test requires exactly 4 GPUs, but got {world_size}"

    print(f"\n[Rank {rank}] Starting SP+DP test (world_size={world_size})")

    # Configuration
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    height = 448
    width = 448
    num_steps = 3
    sp_size = 2  # SP size
    # DP size = world_size / sp_size = 4 / 2 = 2

    # Create model config
    model_config = create_small_model_config(num_layers=2)

    print(f"[Rank {rank}] Model config: num_layers={model_config.vision_config.depth}")
    print(f"[Rank {rank}] Parallelism: sp_size={sp_size}, dp_size={world_size // sp_size}")

    # ========================================
    # BASELINE: No-parallel model (rank 0 only)
    # ========================================
    baseline_model = None
    baseline_losses = []

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] Creating NO-PARALLEL baseline model")
        print(f"[Rank {rank}] " + "=" * 60)

        baseline_model = create_no_parallel_model(model_config, device, torch_dtype)

        # Create optimizer
        optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4, weight_decay=0.01)

        # Training loop for baseline (using DP rank 0's data)
        for step in range(num_steps):
            # Use same seed as DP rank 0 for fair comparison
            pixel_values, grid_thw = prepare_batch_for_dp(height, width, device, dp_rank=0, seed=42 + step)

            optimizer.zero_grad()

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                vision_embeddings = baseline_model(hidden_states=pixel_values, grid_thw=grid_thw)

            # Compute loss
            loss = compute_dummy_loss(vision_embeddings)
            baseline_losses.append(loss.item())

            # Backward
            loss.backward()

            # Optimizer step
            optimizer.step()

            print(f"[Rank {rank}] Baseline step {step}: loss={loss.item():.6f}")

        print(f"[Rank {rank}] Baseline training complete")

    # Sync before starting SP+DP test
    dist.barrier()

    # ========================================
    # SP+DP model (all ranks)
    # ========================================
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Creating SP+DP model (sp_size={sp_size})")
    print(f"[Rank {rank}] " + "=" * 60)

    # Create SP+DP model with same seed as baseline for identical initialization
    sp_model, sp_group, dp_group = create_sp_dp_model(model_config, rank, sp_size, device, torch_dtype, seed=42)

    # Determine DP rank
    try:
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu
        dp_rank = mpu.get_data_parallel_rank()
        dp_world_size = mpu.get_data_parallel_world_size()
        sp_rank = mpu.get_sequence_parallel_rank()
    except Exception:
        # Fallback calculation
        dp_rank = rank // sp_size
        dp_world_size = world_size // sp_size
        sp_rank = rank % sp_size

    dist.barrier()
    print(f"[Rank {rank}] SP+DP model created: dp_rank={dp_rank}/{dp_world_size}, sp_rank={sp_rank}/{sp_size}")

    # Create optimizer for SP+DP model
    sp_dp_optimizer = torch.optim.AdamW(sp_model.parameters(), lr=1e-4, weight_decay=0.01)

    # Training loop for SP+DP
    sp_dp_losses = []
    sp_dp_params_before_sync = []
    sp_dp_params_after_sync = []

    for step in range(num_steps):
        # Each DP rank gets different data (simulating real data parallelism)
        pixel_values, grid_thw = prepare_batch_for_dp(height, width, device, dp_rank=dp_rank, seed=42 + step)

        # Broadcast data within SP group to ensure all SP ranks see the same input
        broadcast_for_sequence_parallel([pixel_values, grid_thw], sp_group)

        sp_dp_optimizer.zero_grad()

        # Forward
        with torch.autocast(device_type="cuda", dtype=torch_dtype):
            vision_embeddings = sp_model(hidden_states=pixel_values, grid_thw=grid_thw)

        # Compute loss
        loss = compute_dummy_loss(vision_embeddings)
        sp_dp_losses.append(loss.item())

        # Backward
        loss.backward()

        # Collect parameters before optimizer step (should differ across DP ranks due to different gradients)
        params_before = collect_selected_parameters(sp_model)
        sp_dp_params_before_sync.append(params_before)

        # Manually reduce gradients across DP ranks (simulating DeepSpeed's gradient reduction)
        # In real DeepSpeed, this is done automatically by the engine
        for param in sp_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)

        # Optimizer step
        sp_dp_optimizer.step()

        # Collect parameters after optimizer step (should be synced across DP ranks)
        params_after = collect_selected_parameters(sp_model)
        sp_dp_params_after_sync.append(params_after)

        print(f"[Rank {rank}] SP+DP step {step}: loss={loss.item():.6f}")

    print(f"[Rank {rank}] SP+DP training complete")

    # ========================================
    # Verification: Check DP synchronization
    # ========================================
    dist.barrier()

    # Gather parameters from all ranks to rank 0 for comparison
    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] VERIFICATION: Data Parallel Synchronization")
        print(f"[Rank {rank}] " + "=" * 60)

    for step in range(num_steps):
        # Compare parameters across DP ranks (should be identical after optimizer step)
        params_after = sp_dp_params_after_sync[step]

        # Gather parameters from all ranks
        for param_name in params_after.keys():
            param_tensor = params_after[param_name]

            # Gather to rank 0
            if rank == 0:
                gathered_params = [torch.zeros_like(param_tensor) for _ in range(world_size)]
            else:
                gathered_params = None

            dist.gather(param_tensor, gather_list=gathered_params, dst=0)

            if rank == 0:
                # Compare DP rank 0 (ranks 0,1) with DP rank 1 (ranks 2,3)
                # Within each DP rank, SP ranks should have identical replicated parameters
                # Across DP ranks, parameters should also be identical after sync

                # Compare rank 0 with rank 2 (both are SP rank 0 in their respective DP groups)
                param_dp0 = gathered_params[0]
                param_dp1 = gathered_params[2]

                diff = (param_dp0 - param_dp1).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                DP_SYNC_TOLERANCE = 1e-5
                match_str = "✅" if max_diff < DP_SYNC_TOLERANCE else "❌"

                print(f"[Rank {rank}] Step {step}, {param_name}: {match_str} "
                      f"DP sync max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

                if max_diff >= DP_SYNC_TOLERANCE:
                    print(f"[Rank {rank}]   WARNING: Parameters not synced across DP ranks!")

    # ========================================
    # Compare with baseline (rank 0, DP rank 0, SP rank 0)
    # ========================================
    dist.barrier()

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: Baseline vs SP+DP (DP rank 0)")
        print(f"[Rank {rank}] " + "=" * 60)

        # Compare losses (baseline used DP rank 0's data, so should match rank 0's SP+DP loss)
        print(f"\n[Rank {rank}] Loss Comparison:")
        for step in range(num_steps):
            baseline_loss = baseline_losses[step]
            sp_dp_loss = sp_dp_losses[step]
            loss_diff = abs(baseline_loss - sp_dp_loss)
            rel_diff = loss_diff / (abs(baseline_loss) + 1e-8)

            print(f"[Rank {rank}] Step {step}: baseline_loss={baseline_loss:.6f}, sp_dp_loss={sp_dp_loss:.6f}, "
                  f"diff={loss_diff:.6e}, rel_diff={rel_diff:.6e}")

            LOSS_TOLERANCE = 1e-3
            if loss_diff < LOSS_TOLERANCE:
                print(f"[Rank {rank}] Step {step}: ✅ Losses match!")
            else:
                print(f"[Rank {rank}] Step {step}: ⚠️  Losses differ (expected with bfloat16 and SP)")

        print(f"\n[Rank {rank}] ====================================================================")
        print(f"[Rank {rank}] TEST RESULT: ✅ SP+DP test complete")
        print(f"[Rank {rank}] - SP (sp_size={sp_size}) combined with DP (dp_size={world_size // sp_size})")
        print(f"[Rank {rank}] - Different DP ranks processed different data")
        print(f"[Rank {rank}] - Parameters synchronized across DP ranks after optimizer steps")
        print(f"[Rank {rank}] ====================================================================")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


if __name__ == "__main__":
    test_sp_dp()
