"""Detailed diagnostic test for vision sequence parallelism."""

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig  # noqa: E402
from python.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for detailed vision diagnostics", allow_module_level=True)


def init_dist():
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


def create_simple_inputs():
    """Create simple test inputs."""
    # Single 448x448 image
    in_channels, temp_patch, patch_size = 3, 2, 14
    height, width = 448, 448

    grid_h, grid_w = height // patch_size, width // patch_size
    num_patches = grid_h * grid_w

    # Create pixel values
    pixel_values = torch.randn(in_channels, temp_patch, height, width, dtype=torch.bfloat16, device="cuda")

    # Reshape to patches
    pixel_values = pixel_values.reshape(in_channels, temp_patch, grid_h, patch_size, grid_w, patch_size)
    pixel_values = pixel_values.permute(2, 4, 0, 1, 3, 5)
    pixel_values = pixel_values.reshape(num_patches, in_channels, temp_patch, patch_size, patch_size)
    pixel_values = pixel_values.flatten()

    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64, device="cuda")

    return pixel_values, grid_thw


def test_intermediate_outputs():
    """Test and compare intermediate outputs at each stage."""
    rank, world_size = init_dist()

    print(f"[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Detailed Diagnostic Test")
    print(f"[Rank {rank}] " + "=" * 60)

    # Create config
    config = Qwen2_5_VLVisionConfig()
    config.sequence_parallel = True  # Enable SP for both ranks
    config._attn_implementation = "flash_attention_2"

    # Create model
    model = Qwen2_5_VisionTransformerPretrainedModel(config)
    model = model.to(dtype=torch.bfloat16, device="cuda")

    # Initialize sequence parallelism AFTER model creation
    if config.sequence_parallel:
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu

        mpu.initialize_sequence_parallel(sequence_parallel_size=world_size)
        sp_group = mpu.get_sequence_parallel_group()
        model.sp_group = sp_group
        for m in model.modules():
            if m.__class__.__name__ == "Qwen2_5_VLVisionAttention":
                m.sp_group = sp_group

    # Sync weights across ranks (important!)
    print(f"[Rank {rank}] Synchronizing weights across ranks...")
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    model.eval()

    # Create same inputs on all ranks
    torch.manual_seed(42 + rank)  # Different seed per rank initially
    pixel_values, grid_thw = create_simple_inputs()

    # Broadcast inputs from rank 0
    if rank == 0:
        print(f"[Rank {rank}] Broadcasting inputs to all ranks...")

    dist.broadcast(pixel_values, src=0)
    dist.broadcast(grid_thw, src=0)

    # Run forward with debug
    print(f"[Rank {rank}] Running forward pass...")

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(pixel_values, grid_thw, debug=True)

    print(f"[Rank {rank}] Output shape: {output.shape}")
    print(f"[Rank {rank}] Output mean: {output.float().mean().item():.6f}")
    print(f"[Rank {rank}] Output std: {output.float().std().item():.6f}")
    print(f"[Rank {rank}] Has NaN: {torch.isnan(output).any().item()}")

    # Compare outputs across ranks (should be identical after all-gather)
    # All ranks must participate in collective operations!
    outputs = [torch.zeros_like(output) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, output)

    # Only rank 0 does the comparison
    if rank == 0:
        # Check if all ranks have same output
        for i in range(1, len(outputs)):
            if torch.allclose(outputs[0], outputs[i], atol=1e-5):
                print(f"[Rank 0] ✅ Output matches between rank 0 and rank {i}")
            else:
                diff = (outputs[0] - outputs[i]).abs().max().item()
                print(f"[Rank 0] ❌ Output DIFFERS between rank 0 and rank {i}: max_diff={diff:.6e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_intermediate_outputs()
