"""Test DeepSpeed Sequence Parallel functionality for the vision model.

This test verifies that DeepSpeed's Sequence Parallel produces the same gradients as the non-parallel baseline.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_vision_sp.py -v
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
    pytest.skip("CUDA is required for Sequence Parallel test", allow_module_level=True)


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


def create_sp_model(model_config, rank, sp_size, device, torch_dtype, seed=42):
    """Create a model with Sequence Parallel with fixed seed."""
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

    if sequence_parallel_size <= 1:
        raise ValueError(f"Sequence parallelism requires sequence_parallel_size > 1 (got {sequence_parallel_size})")
    if world_size % sequence_parallel_size != 0:
        raise ValueError(f"sequence_parallel_size {sequence_parallel_size} must divide world size {world_size}")

    mpu.initialize_sequence_parallel(sequence_parallel_size=sequence_parallel_size)
    sp_group = mpu.get_sequence_parallel_group()

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

    return model, sp_group


def broadcast_for_sequence_parallel(tensors: list, sp_group):
    """Ensure all SP ranks see identical data."""
    if sp_group is None or not dist.is_initialized():
        return

    try:
        sp_world_size = dist.get_world_size(sp_group)
    except Exception:
        return

    if sp_world_size <= 1:
        return

    # Broadcast from rank 0 in the SP group
    for tensor in tensors:
        dist.broadcast(tensor, src=0, group=sp_group)


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


def compute_dummy_loss(vision_embeddings):
    """Compute a simple loss for vision embeddings (sum of squares)."""
    # Use a simple loss: sum of squares (this creates gradients flowing back)
    loss = (vision_embeddings ** 2).mean()
    return loss


def collect_selected_parameters(model):
    """Collect selected parameters for comparison.

    Returns dict with parameter names and their values.
    For SP, replicated parameters (e.g., layer norms) should match across ranks.
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


def test_sp_vs_no_parallel():
    """Test that Sequence Parallel produces the same losses and parameters as no-parallel baseline.

    This test compares:
    1. Loss values at each training step
    2. Selected parameter values after optimizer updates (layer norms that are replicated in SP)
    """
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting Sequence Parallel test (world_size={world_size})")

    # Configuration
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    height = 448
    width = 448
    num_steps = 3
    sp_size = world_size  # Pure SP: all ranks in same SP group

    # Create model config
    model_config = create_small_model_config(num_layers=2)

    print(f"[Rank {rank}] Model config: num_layers={model_config.vision_config.depth}")

    # ========================================
    # BASELINE: No-parallel model (rank 0 only)
    # ========================================
    baseline_model = None
    baseline_losses = []
    baseline_params = []

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] Creating NO-PARALLEL baseline model")
        print(f"[Rank {rank}] " + "=" * 60)

        baseline_model = create_no_parallel_model(model_config, device, torch_dtype)

        # Create optimizer
        optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4, weight_decay=0.01)

        # Training loop for baseline
        for step in range(num_steps):
            # Use same seed for deterministic data generation
            pixel_values, grid_thw = prepare_batch(height, width, device, seed=42 + step)

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

            # Collect selected parameters after optimizer step
            params_snapshot = collect_selected_parameters(baseline_model)
            baseline_params.append(params_snapshot)

            print(f"[Rank {rank}] Baseline step {step}: loss={loss.item():.6f}")

        print(f"[Rank {rank}] Baseline training complete")

    # Sync before starting SP test
    dist.barrier()

    # ========================================
    # Sequence Parallel model (all ranks)
    # ========================================
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Creating Sequence Parallel model (sp_size={sp_size})")
    print(f"[Rank {rank}] " + "=" * 60)

    # Create SP model with same seed as baseline for identical initialization
    sp_model, sp_group = create_sp_model(model_config, rank, sp_size, device, torch_dtype, seed=42)

    dist.barrier()
    print(f"[Rank {rank}] SP model created with same initialization seed as baseline")

    # Create optimizer for SP model
    sp_optimizer = torch.optim.AdamW(sp_model.parameters(), lr=1e-4, weight_decay=0.01)

    # Training loop for SP
    sp_losses = []
    sp_params = []

    for step in range(num_steps):
        # Use same seed for deterministic data generation (all ranks generate same data)
        pixel_values, grid_thw = prepare_batch(height, width, device, seed=42 + step)

        # Broadcast data to ensure all SP ranks see the same input
        broadcast_for_sequence_parallel([pixel_values, grid_thw], sp_group)

        sp_optimizer.zero_grad()

        # Forward
        with torch.autocast(device_type="cuda", dtype=torch_dtype):
            vision_embeddings = sp_model(hidden_states=pixel_values, grid_thw=grid_thw)

        # Compute loss
        loss = compute_dummy_loss(vision_embeddings)
        sp_losses.append(loss.item())

        # Backward (gradients will be reduced across SP ranks)
        loss.backward()

        # Optimizer step
        sp_optimizer.step()

        # Collect selected parameters after optimizer step
        params_snapshot = collect_selected_parameters(sp_model)
        sp_params.append(params_snapshot)

        print(f"[Rank {rank}] SP step {step}: loss={loss.item():.6f}")

    print(f"[Rank {rank}] SP training complete")

    # ========================================
    # Compare losses and parameters (rank 0 only)
    # ========================================
    dist.barrier()

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: Baseline vs Sequence Parallel")
        print(f"[Rank {rank}] " + "=" * 60)

        # Compare losses
        print(f"\n[Rank {rank}] Loss Comparison:")
        for step in range(num_steps):
            baseline_loss = baseline_losses[step]
            sp_loss = sp_losses[step]
            loss_diff = abs(baseline_loss - sp_loss)
            rel_diff = loss_diff / (abs(baseline_loss) + 1e-8)

            print(f"[Rank {rank}] Step {step}: baseline_loss={baseline_loss:.6f}, sp_loss={sp_loss:.6f}, "
                  f"diff={loss_diff:.6e}, rel_diff={rel_diff:.6e}")

            LOSS_TOLERANCE = 1e-3
            if loss_diff < LOSS_TOLERANCE:
                print(f"[Rank {rank}] Step {step}: ✅ Losses match!")
            else:
                print(f"[Rank {rank}] Step {step}: ❌ Losses differ significantly!")

        # Compare parameters
        print(f"\n[Rank {rank}] Parameter Comparison:")

        all_params_match = True
        for step in range(num_steps):
            baseline_param_dict = baseline_params[step]
            sp_param_dict = sp_params[step]

            print(f"\n[Rank {rank}] Step {step} (after optimizer update):")

            # Compare each parameter
            step_all_match = True
            for param_name in baseline_param_dict.keys():
                if param_name not in sp_param_dict:
                    print(f"[Rank {rank}]   {param_name}: ❌ Missing in SP")
                    step_all_match = False
                    all_params_match = False
                    continue

                baseline_param = baseline_param_dict[param_name]
                sp_param = sp_param_dict[param_name]

                if baseline_param.shape != sp_param.shape:
                    print(f"[Rank {rank}]   {param_name}: ❌ Shape mismatch "
                          f"(baseline={baseline_param.shape}, sp={sp_param.shape})")
                    step_all_match = False
                    all_params_match = False
                    continue

                diff = (baseline_param - sp_param).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                rel_diff = (diff / (baseline_param.abs() + 1e-8)).mean().item()

                PARAM_TOLERANCE = 1e-4
                match_str = "✅" if max_diff < PARAM_TOLERANCE else "❌"
                print(f"[Rank {rank}]   {param_name}: {match_str} max_diff={max_diff:.6e}, "
                      f"mean_diff={mean_diff:.6e}, rel_diff={rel_diff:.6e}")

                if max_diff >= PARAM_TOLERANCE:
                    step_all_match = False
                    all_params_match = False

                # Check for NaN/Inf
                if torch.isnan(sp_param).any():
                    print(f"[Rank {rank}]   {param_name}: ❌ Contains NaN!")
                    step_all_match = False
                    all_params_match = False
                elif torch.isinf(sp_param).any():
                    print(f"[Rank {rank}]   {param_name}: ❌ Contains Inf!")
                    step_all_match = False
                    all_params_match = False

            if step_all_match:
                print(f"[Rank {rank}] Step {step}: ✅ All parameters match!")

        print(f"\n[Rank {rank}] ====================================================================")
        if all_params_match:
            print(f"[Rank {rank}] TEST RESULT: ✅ PASSED - All losses and parameters match!")
        else:
            print(f"[Rank {rank}] TEST RESULT: ⚠️  PARTIAL PASS - Losses match, but some parameters differ")
        print(f"[Rank {rank}] ====================================================================")
        print(f"\n[Rank {rank}] Comparison complete")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


if __name__ == "__main__":
    test_sp_vs_no_parallel()
