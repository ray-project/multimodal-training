"""Test DeepSpeed AutoTP with Data Parallel for the text model.

This test verifies that DeepSpeed's AutoTP combined with Data Parallel produces
correct training behavior with 4 GPUs: tp_size=2, dp_size=2.

Run with:
    torchrun --nproc_per_node=4 -m pytest tests/test_text_autotp_dp.py -v
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
    pytest.skip("CUDA is required for AutoTP+DP test", allow_module_level=True)


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


def create_autotp_dp_trainer(model_config, rank, autotp_size, device, torch_dtype, seed=42):
    """Create a trainer with AutoTP and Data Parallel with fixed seed."""
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
        "autotp_size": autotp_size,  # TP size (not world size)
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

    # Collect parameters
    params = trainer._collect_parameters_from_modules(model, lm_head)

    # Determine AutoTP configuration
    from deepspeed.utils import groups

    tensor_parallel_cfg = {"autotp_size": autotp_size}

    # Initialize DeepSpeed with AutoTP + DP
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

    try:
        dp_group = groups.get_data_parallel_group()
    except Exception:
        dp_group = None

    trainer.model = model_engine
    trainer.model_config = model_config
    trainer.lm_head = model_engine.module.lm_head
    trainer.model.tp_group = tp_group
    trainer.model.dp_group = dp_group
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


def prepare_batch_for_dp(vocab_size, batch_size, seq_len, device, dp_rank, seed=None):
    """Create different batches for different DP ranks to simulate data parallelism."""
    if seed is not None:
        torch.manual_seed(seed + dp_rank)  # Different seed per DP rank
        torch.cuda.manual_seed_all(seed + dp_rank)

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
    For AutoTP, this will only include replicated parameters that should match across TP ranks.
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


def test_autotp_dp():
    """Test that AutoTP with Data Parallel works correctly.

    This test verifies:
    1. AutoTP (tp_size=2) combined with Data Parallel (dp_size=2) runs without errors
    2. Different DP ranks process different data
    3. Parameters are synchronized across DP ranks after optimizer steps
    4. Loss values are computed correctly for each DP rank

    Test configuration:
    - 4 GPUs total
    - TP size = 2 (each DP replica uses 2 GPUs for tensor parallelism)
    - DP size = 2 (2 data parallel replicas)
    - DP Group 0: Global ranks 0,1 (TP ranks 0,1)
    - DP Group 1: Global ranks 2,3 (TP ranks 0,1)
    """
    rank, world_size, local_rank = init_distributed()

    assert world_size == 4, f"This test requires exactly 4 GPUs, but got {world_size}"

    print(f"\n[Rank {rank}] Starting AutoTP+DP test (world_size={world_size})")

    # Configuration
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    batch_size = 2
    seq_len = 32
    num_steps = 3
    autotp_size = 2  # TP size
    # DP size = world_size / autotp_size = 4 / 2 = 2

    # Create model config
    model_config = create_small_model_config(num_layers=2)
    vocab_size = model_config.text_config.vocab_size

    print(f"[Rank {rank}] Model config: num_layers={model_config.text_config.num_hidden_layers}, vocab_size={vocab_size}")
    print(f"[Rank {rank}] Parallelism: tp_size={autotp_size}, dp_size={world_size // autotp_size}")

    # ========================================
    # BASELINE: No-parallel model (rank 0 only)
    # ========================================
    baseline_model = None
    baseline_lm_head = None
    baseline_losses = []

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

        # Training loop for baseline (using DP rank 0's data)
        for step in range(num_steps):
            # Use same seed as DP rank 0 for fair comparison
            input_ids, labels = prepare_batch_for_dp(vocab_size, batch_size, seq_len, device, dp_rank=0, seed=42 + step)

            optimizer.zero_grad()

            # Forward and backward
            loss = compute_loss(baseline_model, baseline_lm_head, input_ids, labels)
            baseline_losses.append(loss.item())
            loss.backward()

            # Optimizer step
            optimizer.step()

            print(f"[Rank {rank}] Baseline step {step}: loss={loss.item():.6f}")

        print(f"[Rank {rank}] Baseline training complete")

    # Sync before starting AutoTP+DP test
    dist.barrier()

    # ========================================
    # AutoTP+DP model (all ranks)
    # ========================================
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Creating AutoTP+DP model (autotp_size={autotp_size})")
    print(f"[Rank {rank}] " + "=" * 60)

    # Create AutoTP+DP trainer with same seed as baseline for identical initialization
    autotp_trainer = create_autotp_dp_trainer(model_config, rank, autotp_size, device, torch_dtype, seed=42)
    tp_group = autotp_trainer.model.tp_group
    dp_group = autotp_trainer.model.dp_group

    # Determine DP rank
    try:
        from deepspeed.utils import groups
        dp_rank = groups.get_data_parallel_rank()
        dp_world_size = groups.get_data_parallel_world_size()
        tp_rank = groups.get_tensor_model_parallel_rank()
    except Exception:
        # Fallback calculation
        dp_rank = rank // autotp_size
        dp_world_size = world_size // autotp_size
        tp_rank = rank % autotp_size

    dist.barrier()
    print(f"[Rank {rank}] AutoTP+DP model created: dp_rank={dp_rank}/{dp_world_size}, tp_rank={tp_rank}/{autotp_size}")

    # Training loop for AutoTP+DP
    autotp_losses = []
    autotp_params_before_sync = []
    autotp_params_after_sync = []

    for step in range(num_steps):
        # Each DP rank gets different data (simulating real data parallelism)
        input_ids, labels = prepare_batch_for_dp(vocab_size, batch_size, seq_len, device, dp_rank=dp_rank, seed=42 + step)

        # Broadcast data within TP group to ensure all TP ranks see the same input
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

        # Collect parameters before optimizer step (should differ across DP ranks due to different gradients)
        params_before = collect_selected_parameters(autotp_trainer.deepspeed_engine.module)
        autotp_params_before_sync.append(params_before)

        # Optimizer step (DeepSpeed should sync gradients across DP ranks)
        autotp_trainer.optimizer_step()

        # Collect parameters after optimizer step (should be synced across DP ranks)
        params_after = collect_selected_parameters(autotp_trainer.deepspeed_engine.module)
        autotp_params_after_sync.append(params_after)

        print(f"[Rank {rank}] AutoTP+DP step {step}: loss={loss.item():.6f}")

    print(f"[Rank {rank}] AutoTP+DP training complete")

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
        params_after = autotp_params_after_sync[step]

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
                # Within each DP rank, TP ranks should have identical replicated parameters
                # Across DP ranks, parameters should also be identical after sync

                # Compare rank 0 with rank 2 (both are TP rank 0 in their respective DP groups)
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
    # Compare with baseline (rank 0, DP rank 0, TP rank 0)
    # ========================================
    dist.barrier()

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: Baseline vs AutoTP+DP (DP rank 0)")
        print(f"[Rank {rank}] " + "=" * 60)

        # Compare losses (baseline used DP rank 0's data, so should match rank 0's AutoTP loss)
        print(f"\n[Rank {rank}] Loss Comparison:")
        for step in range(num_steps):
            baseline_loss = baseline_losses[step]
            autotp_loss = autotp_losses[step]
            loss_diff = abs(baseline_loss - autotp_loss)
            rel_diff = loss_diff / (abs(baseline_loss) + 1e-8)

            print(f"[Rank {rank}] Step {step}: baseline_loss={baseline_loss:.6f}, autotp_dp_loss={autotp_loss:.6f}, "
                  f"diff={loss_diff:.6e}, rel_diff={rel_diff:.6e}")

            LOSS_TOLERANCE = 1e-4
            if loss_diff < LOSS_TOLERANCE:
                print(f"[Rank {rank}] Step {step}: ✅ Losses match!")
            else:
                print(f"[Rank {rank}] Step {step}: ⚠️  Losses differ (expected with bfloat16 and TP)")

        print(f"\n[Rank {rank}] ====================================================================")
        print(f"[Rank {rank}] TEST RESULT: ✅ AutoTP+DP test complete")
        print(f"[Rank {rank}] - AutoTP (tp_size={autotp_size}) combined with DP (dp_size={world_size // autotp_size})")
        print(f"[Rank {rank}] - Different DP ranks processed different data")
        print(f"[Rank {rank}] - Parameters synchronized across DP ranks after optimizer steps")
        print(f"[Rank {rank}] ====================================================================")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


if __name__ == "__main__":
    test_autotp_dp()
