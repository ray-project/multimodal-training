"""Comparison tests for ragged sequence parallel implementation."""

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen2_5_VLConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for vision comparison test", allow_module_level=True)


def init_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Initialize DeepSpeed backend
    try:
        import deepspeed.comm as dist_deepspeed

        if not dist_deepspeed.is_initialized():
            dist_deepspeed.init_distributed(dist_backend="nccl")
    except ImportError:
        pass

    return rank, world_size


def create_model(sequence_parallel=False):
    """Create vision model."""
    config = Qwen2_5_VLConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    vision_config = config.vision_config
    vision_config.sequence_parallel = sequence_parallel
    vision_config._attn_implementation = "flash_attention_2"

    model = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
    model = model.to(dtype=torch.bfloat16, device="cuda")

    if sequence_parallel and dist.is_initialized():
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu

        world_size = dist.get_world_size()
        mpu.initialize_sequence_parallel(sequence_parallel_size=world_size)
        sp_group = mpu.get_sequence_parallel_group()
        model.sp_group = sp_group

        for m in model.modules():
            if m.__class__.__name__ == "Qwen2_5_VLVisionAttention":
                m.sp_group = sp_group

    return model


def create_pixel_values(height=448, width=448):
    """Create pixel values tensor."""
    in_channels = 3
    temporal_patch_size = 2
    patch_size = 14

    height = (height // patch_size) * patch_size
    width = (width // patch_size) * patch_size

    grid_t = 1
    grid_h = height // patch_size
    grid_w = width // patch_size

    pixel_values = torch.randn(in_channels, temporal_patch_size, height, width, dtype=torch.bfloat16, device="cuda")

    # Reshape to patches and flatten
    num_patches = grid_h * grid_w
    pixel_values = pixel_values.reshape(in_channels, temporal_patch_size, grid_h, patch_size, grid_w, patch_size)
    pixel_values = pixel_values.permute(2, 4, 0, 1, 3, 5)
    pixel_values = pixel_values.reshape(num_patches, in_channels, temporal_patch_size, patch_size, patch_size)
    pixel_values = pixel_values.flatten()

    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64, device="cuda")

    return pixel_values, grid_thw


def load_pretrained_weights(model, rank):
    """Load pretrained weights to get consistent outputs."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    if rank == 0:
        print("Loading pretrained weights from Qwen/Qwen2.5-VL-3B-Instruct...")

    # Load the full model to extract vision weights
    full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    # Extract vision model state dict
    vision_state = full_model.visual.state_dict()

    # Load into our model
    model.load_state_dict(vision_state, strict=False)

    if rank == 0:
        print("✅ Pretrained weights loaded")

    return model


def _test_no_sp_baseline(rank):
    """Test without sequence parallelism (ground truth)."""
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] TEST 1: NO Sequence Parallelism (Baseline)")
    print(f"[Rank {rank}] " + "=" * 60)

    # Only rank 0 runs this test (no distribution)
    if rank != 0:
        if dist.is_initialized():
            dist.barrier()
        return None, None

    model = create_model(sequence_parallel=False)
    model = load_pretrained_weights(model, rank)
    model.eval()

    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    pixel_values, grid_thw = create_pixel_values(height=448, width=448)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(pixel_values, grid_thw)

    print(f"[Rank {rank}] Output shape: {output.shape}")
    print(f"[Rank {rank}] Output mean: {output.float().mean().item():.6f}")
    print(f"[Rank {rank}] Output std: {output.float().std().item():.6f}")
    print(f"[Rank {rank}] Output min: {output.float().min().item():.6f}")
    print(f"[Rank {rank}] Output max: {output.float().max().item():.6f}")
    print(f"[Rank {rank}] Has NaN: {torch.isnan(output).any().item()}")
    print(f"[Rank {rank}] Has Inf: {torch.isinf(output).any().item()}")

    if dist.is_initialized():
        dist.barrier()

    return output, pixel_values


def _test_sp_equal_length(rank, pixel_values_ref, grid_thw_ref):
    """Test with SP but equal-length sequences (old path)."""
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] TEST 2: SP with Equal-Length Sequences")
    print(f"[Rank {rank}] " + "=" * 60)

    model = create_model(sequence_parallel=True)
    model = load_pretrained_weights(model, rank)
    model.eval()

    # Use SAME input as baseline (broadcasted to all ranks)
    if rank == 0:
        pixel_values = pixel_values_ref.clone()
        grid_thw = grid_thw_ref.clone()
    else:
        # Create dummy data for other ranks (will sync with barrier)
        pixel_values, grid_thw = create_pixel_values(height=448, width=448)

    # Broadcast inputs to ensure all ranks use same data
    if dist.is_initialized():
        dist.broadcast(pixel_values, src=0)
        dist.broadcast(grid_thw, src=0)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(pixel_values, grid_thw, debug=True)

    print(f"[Rank {rank}] Output shape: {output.shape}")
    print(f"[Rank {rank}] Output mean: {output.float().mean().item():.6f}")
    print(f"[Rank {rank}] Output std: {output.float().std().item():.6f}")
    print(f"[Rank {rank}] Output min: {output.float().min().item():.6f}")
    print(f"[Rank {rank}] Output max: {output.float().max().item():.6f}")
    print(f"[Rank {rank}] Has NaN: {torch.isnan(output).any().item()}")
    print(f"[Rank {rank}] Has Inf: {torch.isinf(output).any().item()}")

    return output


def test_vision_comparison():
    """Main comparison test - entry point for pytest."""
    rank, world_size = init_distributed()

    print(f"[Rank {rank}] Initialized with world_size={world_size}")

    # Test 1: No SP (baseline) - only rank 0, but both ranks wait
    output_no_sp, pixel_values_ref = _test_no_sp_baseline(rank)

    # Broadcast the reference pixel_values to all ranks
    if rank != 0:
        pixel_values_ref, _ = create_pixel_values(height=448, width=448)

    grid_thw_ref = torch.tensor([[1, 32, 32]], dtype=torch.int64, device="cuda")

    if dist.is_initialized():
        dist.broadcast(pixel_values_ref, src=0)
        dist.broadcast(grid_thw_ref, src=0)

    # Test 2: SP with equal lengths (using SAME input as baseline)
    output_sp = _test_sp_equal_length(rank, pixel_values_ref, grid_thw_ref)

    # Gather outputs from all ranks to rank 0 for comparison
    if dist.is_initialized():
        # All ranks have the same output (replicated), just use rank 0's
        pass

    # Compare outputs (rank 0 only)
    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: No-SP vs SP")
        print(f"[Rank {rank}] " + "=" * 60)

        if output_no_sp is not None and output_sp is not None:
            # Check if outputs match
            diff = (output_no_sp - output_sp).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_diff = (diff / (output_no_sp.abs() + 1e-8)).mean().item()

            print(f"[Rank {rank}] Max absolute difference: {max_diff:.6e}")
            print(f"[Rank {rank}] Mean absolute difference: {mean_diff:.6e}")
            print(f"[Rank {rank}] Mean relative difference: {rel_diff:.6e}")

            # Tolerance check
            TOLERANCE = 1e-3  # 0.1% difference allowed
            if max_diff < TOLERANCE:
                print(f"[Rank {rank}] ✅ Outputs match within tolerance ({TOLERANCE})!")
            else:
                print(f"[Rank {rank}] ❌ Outputs differ significantly (max_diff={max_diff:.6e} > {TOLERANCE})!")

                # Show where differences occur
                large_diff_mask = diff > 0.01
                num_large_diff = large_diff_mask.sum().item()
                print(f"[Rank {rank}] Number of elements with diff > 0.01: {num_large_diff} / {diff.numel()}")

                if num_large_diff > 0 and num_large_diff < 20:
                    indices = torch.where(large_diff_mask)
                    print(f"[Rank {rank}] Indices with large diff (first 10): {indices[0][:10].tolist()}")

            # Check for NaN/Inf
            if torch.isnan(output_sp).any():
                print(f"[Rank {rank}] ❌ SP output contains NaN! This is a BUG.")
                nan_count = torch.isnan(output_sp).sum().item()
                print(f"[Rank {rank}] NaN count: {nan_count} / {output_sp.numel()}")
            elif torch.isinf(output_sp).any():
                print(f"[Rank {rank}] ❌ SP output contains Inf! This is a BUG.")
            else:
                print(f"[Rank {rank}] ✅ SP output is numerically valid (no NaN/Inf)")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    test_vision_comparison()
