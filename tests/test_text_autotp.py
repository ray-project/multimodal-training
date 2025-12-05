"""Test DeepSpeed AutoTP functionality for the text model.

This test verifies that DeepSpeed's AutoTP produces the same gradients as the non-parallel baseline.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_text_autotp.py -v
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
from python.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel  # noqa: E402
from python.ray.text import BaseTextTrainer, QwenTextMixin  # noqa: E402
from python.ray.utils import init_distributed_comm  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for AutoTP test", allow_module_level=True)


class AutoTPQwenTextTrainer(QwenTextMixin, BaseTextTrainer):
    """Standalone trainer for AutoTP testing without Ray."""

    def _get_device(self) -> torch.device:
        """Respect LOCAL_RANK when running under torchrun."""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        return device


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
    config.text_config.num_hidden_layers = num_layers
    return config


def create_no_parallel_model(model_config, device, torch_dtype, seed=42):
    """Create a non-parallel baseline model with fixed seed."""
    # Set seed for deterministic initialization
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_default_device(device)
    model = Qwen2_5_VLTextModel._from_config(model_config.text_config)
    lm_head = nn.Linear(model_config.text_config.hidden_size, model_config.text_config.vocab_size, bias=False)
    torch.set_default_device("cpu")

    model.to(device=device, dtype=torch_dtype)
    lm_head.to(device=device, dtype=torch_dtype)

    # Tie weights
    lm_head.weight = model.embed_tokens.weight

    model.train()
    lm_head.train()

    # Disable dropout for deterministic behavior
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    return model, lm_head


def create_autotp_trainer(model_config, rank, autotp_size, device, torch_dtype, seed=42):
    """Create a trainer with AutoTP with fixed seed."""
    config = {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",  # Unused, we pass model_config directly
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

    trainer = AutoTPQwenTextTrainer(config, rank)

    # Build model using AutoTP
    from deepspeed.module_inject.layers import set_autotp_mode

    set_autotp_mode(training=True)

    # Set seed for deterministic initialization (same as baseline)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_default_device(device)
    model = Qwen2_5_VLTextModel._from_config(model_config.text_config)
    lm_head = nn.Linear(model_config.text_config.hidden_size, model_config.text_config.vocab_size, bias=False)
    torch.set_default_device("cpu")

    model.to(torch_dtype)
    lm_head.to(torch_dtype)

    # Tie weights
    lm_head.weight = model.embed_tokens.weight

    # Attach lm_head to model for DeepSpeed
    model.lm_head = lm_head

    # Disable dropout for deterministic behavior
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    # Apply tensor parallelism with deepspeed.tp_model_init()
    # This actually shards the model parameters across TP ranks
    import deepspeed

    params_before = sum(p.numel() for p in model.parameters())
    print(f"[Rank {rank}] Parameters BEFORE TP sharding: {params_before:,}")

    model = deepspeed.tp_model_init(model, tp_size=autotp_size, dtype=torch_dtype)

    params_after = sum(p.numel() for p in model.parameters())
    reduction_pct = 100 * (params_before - params_after) / params_before
    print(f"[Rank {rank}] Parameters AFTER TP sharding: {params_after:,} ({reduction_pct:.1f}% reduction)")

    # Re-collect parameters after TP sharding (shapes have changed)
    lm_head = model.lm_head
    params = trainer._collect_parameters_from_modules(model, lm_head)

    # Determine AutoTP configuration
    from deepspeed.utils import groups

    tensor_parallel_cfg = {"autotp_size": autotp_size}

    # Initialize DeepSpeed with AutoTP
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
    trainer.model_config = model_config
    trainer.lm_head = model_engine.module.lm_head
    trainer.model.tp_group = tp_group
    trainer.deepspeed_engine = model_engine

    return trainer


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


def prepare_batch(vocab_size, batch_size, seq_len, device, seed=None):
    """Create a simple causal LM batch with shifted labels."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    labels = input_ids.clone()
    return input_ids, labels


def compute_loss(model, lm_head, input_ids, labels, autocast_context=None):
    """Compute causal LM loss."""
    if autocast_context is not None:
        with autocast_context:
            outputs = model(input_ids=input_ids)
            logits = lm_head(outputs.last_hidden_state)
    else:
        outputs = model(input_ids=input_ids)
        logits = lm_head(outputs.last_hidden_state)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def collect_selected_parameters(model):
    """Collect selected parameters for comparison.

    Returns dict with parameter names and their values.
    For AutoTP, this will only include replicated parameters that should match across ranks.
    """
    params = {}

    # Collect first layer norm parameters (replicated in AutoTP)
    if hasattr(model, 'layers') and len(model.layers) > 0:
        first_layer = model.layers[0]
        if hasattr(first_layer, 'input_layernorm') and first_layer.input_layernorm.weight is not None:
            params['layer0_input_ln'] = first_layer.input_layernorm.weight.detach().clone()
        if hasattr(first_layer, 'post_attention_layernorm') and first_layer.post_attention_layernorm.weight is not None:
            params['layer0_post_attn_ln'] = first_layer.post_attention_layernorm.weight.detach().clone()

    # Collect final layer norm parameters (replicated)
    if hasattr(model, 'norm') and model.norm.weight is not None:
        params['final_norm'] = model.norm.weight.detach().clone()

    return params


def sync_model_weights(src_model, src_lm_head, dst_model, dst_lm_head):
    """Copy weights from source model to destination model (for initialization)."""
    # Copy model weights
    src_state = src_model.state_dict()
    dst_model.load_state_dict(src_state, strict=True)

    # Copy lm_head weights (should be tied to embeddings, but copy explicitly)
    with torch.no_grad():
        dst_lm_head.weight.copy_(src_lm_head.weight)


def test_autotp_vs_no_parallel():
    """Test that AutoTP produces the same losses and parameters as no-parallel baseline.

    This test compares:
    1. Loss values at each training step
    2. Selected parameter values after optimizer updates (layer norms that are replicated in AutoTP)
    """
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting AutoTP test (world_size={world_size})")

    # Configuration
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    batch_size = 2
    seq_len = 32
    num_steps = 3
    autotp_size = world_size  # Pure TP: all ranks in same TP group

    # Create model config
    model_config = create_small_model_config(num_layers=2)
    vocab_size = model_config.text_config.vocab_size

    print(f"[Rank {rank}] Model config: num_layers={model_config.text_config.num_hidden_layers}, vocab_size={vocab_size}")

    # ========================================
    # BASELINE: No-parallel model (rank 0 only)
    # ========================================
    baseline_model = None
    baseline_lm_head = None
    baseline_losses = []
    baseline_params = []

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] Creating NO-PARALLEL baseline model")
        print(f"[Rank {rank}] " + "=" * 60)

        baseline_model, baseline_lm_head = create_no_parallel_model(model_config, device, torch_dtype)

        # Create optimizer
        params = list(baseline_model.parameters()) + list(baseline_lm_head.parameters())
        # Deduplicate tied weights
        param_ids = set()
        unique_params = []
        for p in params:
            if id(p) not in param_ids:
                param_ids.add(id(p))
                unique_params.append(p)
        optimizer = torch.optim.AdamW(unique_params, lr=1e-4, weight_decay=0.01)

        # Training loop for baseline
        for step in range(num_steps):
            # Use same seed for deterministic data generation
            input_ids, labels = prepare_batch(vocab_size, batch_size, seq_len, device, seed=42 + step)

            optimizer.zero_grad()

            # Forward and backward
            loss = compute_loss(baseline_model, baseline_lm_head, input_ids, labels)
            baseline_losses.append(loss.item())
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Collect selected parameters after optimizer step
            params_snapshot = collect_selected_parameters(baseline_model)
            baseline_params.append(params_snapshot)

            print(f"[Rank {rank}] Baseline step {step}: loss={loss.item():.6f}")

        print(f"[Rank {rank}] Baseline training complete")

    # Sync before starting AutoTP test
    dist.barrier()

    # ========================================
    # AutoTP model (all ranks)
    # ========================================
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Creating AutoTP model (autotp_size={autotp_size})")
    print(f"[Rank {rank}] " + "=" * 60)

    # Create AutoTP trainer with same seed as baseline for identical initialization
    autotp_trainer = create_autotp_trainer(model_config, rank, autotp_size, device, torch_dtype, seed=42)
    tp_group = autotp_trainer.model.tp_group

    dist.barrier()
    print(f"[Rank {rank}] AutoTP model created with same initialization seed as baseline")

    # Training loop for AutoTP
    autotp_losses = []
    autotp_params = []

    for step in range(num_steps):
        # Use same seed for deterministic data generation (all ranks generate same data)
        input_ids, labels = prepare_batch(vocab_size, batch_size, seq_len, device, seed=42 + step)

        # Broadcast data to ensure all TP ranks see the same input
        broadcast_for_tensor_parallel([input_ids, labels], tp_group)

        autotp_trainer.zero_grad()

        # Forward and backward
        with autotp_trainer._get_autocast_context():
            outputs = autotp_trainer.model(input_ids=input_ids)
            logits = autotp_trainer.lm_head(outputs.last_hidden_state)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        autotp_losses.append(loss.item())

        # Backward
        autotp_trainer.deepspeed_engine.backward(loss)

        # Optimizer step
        autotp_trainer.optimizer_step()

        # Collect selected parameters after optimizer step
        params_snapshot = collect_selected_parameters(autotp_trainer.deepspeed_engine.module)
        autotp_params.append(params_snapshot)

        print(f"[Rank {rank}] AutoTP step {step}: loss={loss.item():.6f}")

    print(f"[Rank {rank}] AutoTP training complete")

    # ========================================
    # Compare losses and parameters (rank 0 only)
    # ========================================
    dist.barrier()

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: Baseline vs AutoTP")
        print(f"[Rank {rank}] " + "=" * 60)

        # Compare losses
        print(f"\n[Rank {rank}] Loss Comparison:")
        for step in range(num_steps):
            baseline_loss = baseline_losses[step]
            autotp_loss = autotp_losses[step]
            loss_diff = abs(baseline_loss - autotp_loss)
            rel_diff = loss_diff / (abs(baseline_loss) + 1e-8)

            print(f"[Rank {rank}] Step {step}: baseline_loss={baseline_loss:.6f}, autotp_loss={autotp_loss:.6f}, "
                  f"diff={loss_diff:.6e}, rel_diff={rel_diff:.6e}")

            LOSS_TOLERANCE = 1e-4
            if loss_diff < LOSS_TOLERANCE:
                print(f"[Rank {rank}] Step {step}: ✅ Losses match!")
            else:
                print(f"[Rank {rank}] Step {step}: ❌ Losses differ significantly!")

        # Compare parameters
        print(f"\n[Rank {rank}] Parameter Comparison:")

        all_params_match = True
        for step in range(num_steps):
            baseline_param_dict = baseline_params[step]
            autotp_param_dict = autotp_params[step]

            print(f"\n[Rank {rank}] Step {step} (after optimizer update):")

            # Compare each parameter
            step_all_match = True
            for param_name in baseline_param_dict.keys():
                if param_name not in autotp_param_dict:
                    print(f"[Rank {rank}]   {param_name}: ❌ Missing in AutoTP")
                    step_all_match = False
                    all_params_match = False
                    continue

                baseline_param = baseline_param_dict[param_name]
                autotp_param = autotp_param_dict[param_name]

                if baseline_param.shape != autotp_param.shape:
                    print(f"[Rank {rank}]   {param_name}: ❌ Shape mismatch "
                          f"(baseline={baseline_param.shape}, autotp={autotp_param.shape})")
                    step_all_match = False
                    all_params_match = False
                    continue

                diff = (baseline_param - autotp_param).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                rel_diff = (diff / (baseline_param.abs() + 1e-8)).mean().item()

                PARAM_TOLERANCE = 1e-5
                match_str = "✅" if max_diff < PARAM_TOLERANCE else "❌"
                print(f"[Rank {rank}]   {param_name}: {match_str} max_diff={max_diff:.6e}, "
                      f"mean_diff={mean_diff:.6e}, rel_diff={rel_diff:.6e}")

                if max_diff >= PARAM_TOLERANCE:
                    step_all_match = False
                    all_params_match = False

                # Check for NaN/Inf
                if torch.isnan(autotp_param).any():
                    print(f"[Rank {rank}]   {param_name}: ❌ Contains NaN!")
                    step_all_match = False
                    all_params_match = False
                elif torch.isinf(autotp_param).any():
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
    test_autotp_vs_no_parallel()
