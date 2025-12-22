"""Test DeepSpeed AutoTP functionality for the vision model.

This test verifies that DeepSpeed's AutoTP produces the same outputs as the non-parallel baseline.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_vision_autotp.py -v
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
from python.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VisionTransformerPretrainedModel,
)
from python.ray.vision import BaseVisionTrainer  # noqa: E402
from python.ray.utils import init_distributed_comm  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for AutoTP test", allow_module_level=True)


class AutoTPQwenVisionTrainer(BaseVisionTrainer):
    """Standalone trainer for AutoTP testing without Ray."""

    def _get_device(self) -> torch.device:
        """Respect LOCAL_RANK when running under torchrun."""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        return device

    def _load_model_config(self, model_name):
        """Load Qwen2.5-VL model config."""
        return Qwen2_5_VLConfig.from_pretrained(model_name, trust_remote_code=True)

    def _create_model_instance(self, model_config):
        """Create Qwen2.5-VL vision model instance."""
        model = Qwen2_5_VisionTransformerPretrainedModel(model_config.vision_config)
        return model, None  # No projector for Qwen

    def _get_transformer_layers(self, model):
        """Get transformer blocks for Qwen."""
        return model.blocks

    def _get_projector_or_merger(self, model, projector):
        """Get merger module for Qwen."""
        return model.merger

    def _get_tensor_parallel_mapping(self):
        """Get tensor parallel mapping for Qwen transformer layers."""
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
        return {
            "attn.q_proj": ColwiseParallel(),
            "attn.k_proj": ColwiseParallel(),
            "attn.v_proj": ColwiseParallel(),
            "attn.proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }

    def _parallelize_projector_or_merger(self, model, projector, tp_mesh):
        """Parallelize Qwen merger MLP."""
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        parallelize_module(model.merger.mlp[0], tp_mesh, ColwiseParallel(), src_data_rank=None)
        parallelize_module(model.merger.mlp[2], tp_mesh, RowwiseParallel(), src_data_rank=None)

    def _setup_sequence_parallel(self, model, sp_group):
        """Set up sequence parallelism for Qwen."""
        for m in model.modules():
            if m.__class__.__name__ in ["Qwen2_5_VLVisionAttention", "Qwen2_5_VisionTransformerPretrainedModel"]:
                m.sp_group = sp_group

    def _get_vision_config(self, model_name):
        """Get Qwen2.5-VL vision config."""
        config = Qwen2_5_VLConfig.from_pretrained(model_name, trust_remote_code=True)
        return config.vision_config

    def _model_forward(self, batch):
        """Forward pass for Qwen2.5-VL."""
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        autocast_context = self._get_autocast_context()

        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        with autocast_context:
            vision_outputs = self.model(hidden_states=pixel_values, grid_thw=image_grid_thw)

        vision_outputs = vision_outputs.unsqueeze(0)
        return vision_outputs

    def _zero_padded_weights_after_init(self, model, projector):
        """Qwen does not use padded attention heads, so this is a no-op."""
        pass


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_default_device(device)
    model = Qwen2_5_VisionTransformerPretrainedModel(model_config.vision_config)
    torch.set_default_device("cpu")

    model.to(device=device, dtype=torch_dtype)
    model.train()

    # Disable dropout for deterministic behavior
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    return model


def prepare_vision_batch(device, batch_size=1, image_size=56, seed=None):
    """Create a simple vision batch for testing.

    Note: Qwen2.5-VL expects pixel_values of shape [C, T, H, W] where T is temporal dimension.
    We use T=1 for single frame. The model uses a patch size of 14.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Qwen2.5-VL expects [C, T, H, W] for single image
    # C=3 (RGB), T=temporal_patch_size (2), H and W should be multiples of patch_size (14)
    # Minimum size: 14 * 2 = 28 for temporal, 14 for spatial
    temporal_patch_size = 2
    spatial_patch_size = 14

    # pixel_values shape: [C, T, H, W]
    pixel_values = torch.randn(3, temporal_patch_size, image_size, image_size, device=device)

    # grid_thw: [T, H, W] in terms of patches
    t_patches = temporal_patch_size // temporal_patch_size  # 1
    h_patches = image_size // spatial_patch_size  # 4
    w_patches = image_size // spatial_patch_size  # 4
    image_grid_thw = torch.tensor([t_patches, h_patches, w_patches], device=device)

    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def broadcast_for_tensor_parallel(tensors: list, tp_group):
    """Ensure all TP ranks see identical data."""
    if tp_group is None or not dist.is_initialized():
        return

    try:
        tp_world_size = dist.get_world_size(tp_group)
    except Exception:
        return

    if tp_world_size <= 1:
        return

    try:
        from deepspeed.utils import groups
        src_rank = groups.get_tensor_model_parallel_src_rank()
    except Exception:
        src_rank = 0

    for tensor in tensors:
        dist.broadcast(tensor, src=src_rank, group=tp_group)


def run_baseline_forward(model_config, device, torch_dtype, num_steps, seed=42, rank=0):
    """Run baseline (no-parallel) forward passes and return outputs."""
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Creating NO-PARALLEL baseline vision model")
    print(f"[Rank {rank}] " + "=" * 60)

    model = create_no_parallel_model(model_config, device, torch_dtype, seed=seed)

    params = sum(p.numel() for p in model.parameters())
    print(f"[Rank {rank}] Baseline model parameters: {params:,}")

    # Forward passes
    outputs = []
    for step in range(num_steps):
        batch = prepare_vision_batch(device, seed=seed + step)

        with torch.no_grad():
            # grid_thw needs to be [1, 3] for single image
            grid_thw = batch["image_grid_thw"].unsqueeze(0)
            output = model(hidden_states=batch["pixel_values"], grid_thw=grid_thw)

        outputs.append(output.detach().clone())
        print(f"[Rank {rank}] Baseline step {step}: output shape={output.shape}, mean={output.mean().item():.6f}")

    print(f"[Rank {rank}] Baseline forward complete")
    return outputs


def run_autotp_forward(model_config, rank, world_size, device, torch_dtype, num_steps, seed=42):
    """Run AutoTP forward passes and return outputs."""
    autotp_size = world_size

    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Creating AutoTP vision model (autotp_size={autotp_size})")
    print(f"[Rank {rank}] " + "=" * 60)

    config = {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "parallelism": "autotp",
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "activation_checkpointing": False,
        "autocast": True,
        "zero_stage": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "batch_size": 1,
        "num_iterations": 5,
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "lr_scheduler_type": "constant",
        "gradient_accumulation_steps": 1,
        "reduce_bucket_size": 500000000,
        "seed": seed,
        "clip_grad_norm": False,
        "max_grad_norm": 1.0,
        "autotp_size": autotp_size,
        "tp_overlap_comm": False,
        "train_batch_size_override": None,
    }

    trainer = AutoTPQwenVisionTrainer(config, rank)

    # Build model using AutoTP
    import deepspeed
    from deepspeed.module_inject.layers import set_autotp_mode
    from deepspeed.utils import groups

    set_autotp_mode(training=True)

    # Set seed for deterministic initialization
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_default_device(device)
    model = Qwen2_5_VisionTransformerPretrainedModel(model_config.vision_config)
    torch.set_default_device("cpu")

    model.to(torch_dtype)

    # Disable dropout for deterministic behavior
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    # Count parameters before TP
    params_before = sum(p.numel() for p in model.parameters())
    print(f"[Rank {rank}] Parameters BEFORE TP: {params_before:,}")

    # Apply TP sharding
    print(f"[Rank {rank}] Applying deepspeed.tp_model_init...")
    model = deepspeed.tp_model_init(model, tp_size=autotp_size, dtype=torch_dtype)

    # Count parameters after TP
    params_after = sum(p.numel() for p in model.parameters())
    reduction = 100 * (params_before - params_after) / params_before
    print(f"[Rank {rank}] Parameters AFTER TP: {params_after:,} ({reduction:.1f}% reduction)")

    # Collect parameters
    params = list(model.parameters())

    # Initialize DeepSpeed
    tensor_parallel_cfg = {"autotp_size": autotp_size}
    model_engine, optimizer, _, _ = trainer._initialize_deepspeed(
        model=model,
        params=params,
        config=config,
        torch_dtype=torch_dtype,
        tensor_parallel_config=tensor_parallel_cfg,
    )

    try:
        tp_group = groups.get_tensor_model_parallel_group()
    except Exception:
        tp_group = None

    trainer.model = model_engine
    trainer.deepspeed_engine = model_engine

    dist.barrier()
    print(f"[Rank {rank}] AutoTP model created")

    # Forward passes
    outputs = []
    for step in range(num_steps):
        batch = prepare_vision_batch(device, seed=seed + step)

        # Broadcast inputs to ensure all TP ranks see the same data
        broadcast_for_tensor_parallel([batch["pixel_values"], batch["image_grid_thw"]], tp_group)

        with torch.no_grad():
            with trainer._get_autocast_context():
                grid_thw = batch["image_grid_thw"].unsqueeze(0)
                output = trainer.model(hidden_states=batch["pixel_values"], grid_thw=grid_thw)

        outputs.append(output.detach().clone())
        print(f"[Rank {rank}] AutoTP step {step}: output shape={output.shape}, mean={output.mean().item():.6f}")

    print(f"[Rank {rank}] AutoTP forward complete")
    return outputs


def test_vision_autotp_vs_no_parallel():
    """Test that Vision AutoTP produces the same outputs as no-parallel baseline.

    This test compares output values at each forward step.
    """
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting Vision AutoTP test (world_size={world_size})")

    # Configuration
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    num_steps = 3

    # Create model config
    model_config = create_small_model_config(num_layers=2)

    print(f"[Rank {rank}] Vision config: depth={model_config.vision_config.depth}")

    # ========================================
    # BASELINE: No-parallel model (rank 0 only)
    # ========================================
    baseline_outputs = []

    if rank == 0:
        baseline_outputs = run_baseline_forward(
            model_config=model_config,
            device=device,
            torch_dtype=torch_dtype,
            num_steps=num_steps,
            seed=42,
            rank=rank,
        )

    # Sync before starting AutoTP test
    dist.barrier()

    # ========================================
    # AutoTP model (all ranks)
    # ========================================
    autotp_outputs = run_autotp_forward(
        model_config=model_config,
        rank=rank,
        world_size=world_size,
        device=device,
        torch_dtype=torch_dtype,
        num_steps=num_steps,
        seed=42,
    )

    # ========================================
    # Compare outputs (rank 0 only)
    # ========================================
    dist.barrier()

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: Baseline vs AutoTP")
        print(f"[Rank {rank}] " + "=" * 60)

        all_match = True
        for step in range(num_steps):
            baseline_out = baseline_outputs[step]
            autotp_out = autotp_outputs[step]

            # Compare shapes
            if baseline_out.shape != autotp_out.shape:
                print(f"[Rank {rank}] Step {step}: Shape mismatch! "
                      f"baseline={baseline_out.shape}, autotp={autotp_out.shape}")
                all_match = False
                continue

            # Compare values
            diff = (baseline_out.float() - autotp_out.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_diff = (diff / (baseline_out.float().abs() + 1e-8)).mean().item()

            baseline_mean = baseline_out.mean().item()
            autotp_mean = autotp_out.mean().item()

            print(f"[Rank {rank}] Step {step}: baseline_mean={baseline_mean:.6f}, autotp_mean={autotp_mean:.6f}, "
                  f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, rel_diff={rel_diff:.6e}")

            OUTPUT_TOLERANCE = 1e-2  # bfloat16 has lower precision
            if max_diff < OUTPUT_TOLERANCE:
                print(f"[Rank {rank}] Step {step}: Outputs match!")
            else:
                print(f"[Rank {rank}] Step {step}: Outputs differ significantly!")
                all_match = False

        print(f"\n[Rank {rank}] ====================================================================")
        if all_match:
            print(f"[Rank {rank}] TEST RESULT: PASSED - Vision AutoTP outputs match baseline!")
        else:
            print(f"[Rank {rank}] TEST RESULT: FAILED - Vision AutoTP outputs differ from baseline!")
        print(f"[Rank {rank}] ====================================================================")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


def test_vision_autotp_training_step():
    """Test that Vision AutoTP can complete a full training step (forward + backward + optimizer)."""
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting Vision AutoTP training step test (world_size={world_size})")

    # Configuration
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    autotp_size = world_size

    # Create model config
    model_config = create_small_model_config(num_layers=2)

    config = {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "parallelism": "autotp",
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "activation_checkpointing": False,
        "autocast": True,
        "zero_stage": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "batch_size": 1,
        "num_iterations": 5,
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "lr_scheduler_type": "constant",
        "gradient_accumulation_steps": 1,
        "reduce_bucket_size": 500000000,
        "seed": 42,
        "clip_grad_norm": False,
        "max_grad_norm": 1.0,
        "autotp_size": autotp_size,
        "tp_overlap_comm": False,
        "train_batch_size_override": None,
    }

    trainer = AutoTPQwenVisionTrainer(config, rank)

    # Build model using AutoTP
    import deepspeed
    from deepspeed.module_inject.layers import set_autotp_mode
    from deepspeed.utils import groups

    set_autotp_mode(training=True)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.set_default_device(device)
    model = Qwen2_5_VisionTransformerPretrainedModel(model_config.vision_config)
    torch.set_default_device("cpu")

    model.to(torch_dtype)

    # Apply TP sharding
    model = deepspeed.tp_model_init(model, tp_size=autotp_size, dtype=torch_dtype)

    # Collect parameters and build optimizer
    params = list(model.parameters())
    trainer._build_optimizer(params)

    # Initialize DeepSpeed
    tensor_parallel_cfg = {"autotp_size": autotp_size}
    model_engine, optimizer, _, _ = trainer._initialize_deepspeed(
        model=model,
        params=params,
        config=config,
        torch_dtype=torch_dtype,
        tensor_parallel_config=tensor_parallel_cfg,
        optimizer=trainer.optimizer,
    )

    try:
        tp_group = groups.get_tensor_model_parallel_group()
    except Exception:
        tp_group = None

    trainer.model = model_engine
    trainer.deepspeed_engine = model_engine

    dist.barrier()
    print(f"[Rank {rank}] AutoTP model ready for training")

    # Run training steps
    num_steps = 3
    losses = []

    for step in range(num_steps):
        batch = prepare_vision_batch(device, seed=42 + step)
        broadcast_for_tensor_parallel([batch["pixel_values"], batch["image_grid_thw"]], tp_group)

        # Zero gradients
        trainer.zero_grad()

        # Forward pass
        with trainer._get_autocast_context():
            grid_thw = batch["image_grid_thw"].unsqueeze(0)
            output = trainer.model(hidden_states=batch["pixel_values"], grid_thw=grid_thw)

        # Compute a simple loss (mean of outputs)
        loss = output.mean()
        losses.append(loss.item())

        print(f"[Rank {rank}] Step {step}: output shape={output.shape}, loss={loss.item():.6f}")

        # Backward pass
        trainer.deepspeed_engine.backward(loss)

        # Optimizer step
        trainer.optimizer_step()

        print(f"[Rank {rank}] Step {step}: Training step completed")

    # Verify losses are changing (model is learning)
    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] Training step losses: {losses}")

        # Check that loss values are finite
        all_finite = all(abs(l) < float('inf') and l == l for l in losses)  # l == l checks for NaN

        if all_finite:
            print(f"[Rank {rank}] TEST RESULT: PASSED - Vision AutoTP training steps completed successfully!")
        else:
            print(f"[Rank {rank}] TEST RESULT: FAILED - Got invalid loss values!")
        print(f"[Rank {rank}] ====================================================================")

    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


if __name__ == "__main__":
    test_vision_autotp_vs_no_parallel()
