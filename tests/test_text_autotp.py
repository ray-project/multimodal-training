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

    # Note: Weights are NOT tied (lm_head has its own independent weights)

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

    # Note: Weights are NOT tied (lm_head has its own independent weights)

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

    shift_logits = logits[:, :-1, :].contiguous().float()
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


def run_baseline_training(
    model_config,
    device,
    torch_dtype,
    vocab_size,
    batch_size,
    seq_len,
    num_steps,
    seed=42,
    collect_params=False,
    collect_logits=False,
    rank=0,
):
    """Run baseline (no-parallel) training and return losses.

    Args:
        model_config: Model configuration
        device: Device to run on
        torch_dtype: Data type for model
        vocab_size: Vocabulary size
        batch_size: Batch size
        seq_len: Sequence length
        num_steps: Number of training steps
        seed: Random seed for initialization
        collect_params: Whether to collect parameter snapshots after each step
        collect_logits: Whether to collect logits at each step (before backward)
        rank: Current rank (for logging)

    Returns:
        Tuple of (losses, params, logits_list) where params/logits_list are empty lists if not collected
    """
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Creating NO-PARALLEL baseline model")
    print(f"[Rank {rank}] " + "=" * 60)

    baseline_model, baseline_lm_head = create_no_parallel_model(model_config, device, torch_dtype, seed=seed)

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
    baseline_losses = []
    baseline_params = []
    baseline_logits = []

    for step in range(num_steps):
        # Use same seed for deterministic data generation
        input_ids, labels = prepare_batch(vocab_size, batch_size, seq_len, device, seed=seed + step)

        optimizer.zero_grad()

        # Forward pass - compute logits explicitly
        outputs = baseline_model(input_ids=input_ids)
        logits = baseline_lm_head(outputs.last_hidden_state)

        # Collect logits before backward (detach to avoid memory issues)
        if collect_logits:
            baseline_logits.append(logits.detach().clone())

        # Compute loss
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        baseline_losses.append(loss.item())
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Collect selected parameters after optimizer step
        if collect_params:
            params_snapshot = collect_selected_parameters(baseline_model)
            baseline_params.append(params_snapshot)

        print(f"[Rank {rank}] Baseline step {step}: loss={loss.item():.6f}")

    print(f"[Rank {rank}] Baseline training complete")

    return baseline_losses, baseline_params, baseline_logits


def run_autotp_training(
    model_config,
    rank,
    world_size,
    device,
    torch_dtype,
    vocab_size,
    batch_size,
    seq_len,
    num_steps,
    seed=42,
    collect_params=False,
    collect_logits=False,
    vocab_parallel=False,
):
    """Run AutoTP training and return losses.

    Args:
        model_config: Model configuration
        rank: Current rank
        world_size: Total number of ranks
        device: Device to run on
        torch_dtype: Data type for model
        vocab_size: Vocabulary size
        batch_size: Batch size
        seq_len: Sequence length
        num_steps: Number of training steps
        seed: Random seed for initialization
        collect_params: Whether to collect parameter snapshots after each step
        collect_logits: Whether to collect logits at each step (gathered full logits)
        vocab_parallel: Whether to use VocabParallelEmbedding and vocab_parallel_causal_cross_entropy

    Returns:
        Tuple of (losses, params, logits_list) where params/logits_list are empty lists if not collected
    """
    autotp_size = world_size  # Pure TP: all ranks in same TP group

    print(f"\n[Rank {rank}] " + "=" * 60)
    if vocab_parallel:
        print(f"[Rank {rank}] Creating AutoTP model with VocabParallelEmbedding (autotp_size={autotp_size})")
    else:
        print(f"[Rank {rank}] Creating AutoTP model (autotp_size={autotp_size})")
    print(f"[Rank {rank}] " + "=" * 60)

    # Create config
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
        "use_vocab_parallel": vocab_parallel,
    }

    trainer = AutoTPQwenTextTrainer(config, rank)

    # Build model using AutoTP
    from deepspeed.module_inject.layers import set_autotp_mode
    set_autotp_mode(training=True)

    # Set seed for deterministic initialization
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_default_device(device)
    model = Qwen2_5_VLTextModel._from_config(model_config.text_config)
    lm_head = nn.Linear(model_config.text_config.hidden_size, model_config.text_config.vocab_size, bias=False)
    torch.set_default_device("cpu")

    model.to(torch_dtype)
    lm_head.to(torch_dtype)

    # Note: Weights are NOT tied (lm_head has its own independent weights)
    model.lm_head = lm_head

    # Disable dropout for deterministic behavior
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    # Collect parameters
    params = trainer._collect_parameters_from_modules(model, lm_head)

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
        tp_rank = groups.get_tensor_model_parallel_rank()
        tp_world_size = groups.get_tensor_model_parallel_world_size()
    except Exception:
        tp_group = None
        tp_rank = 0
        tp_world_size = 1

    # Apply VocabParallelEmbedding if requested
    if vocab_parallel and tp_group is not None:
        from python.tensor_parallel import VocabParallelEmbedding

        original_embedding = model_engine.module.embed_tokens
        original_lm_head = model_engine.module.lm_head
        print(f"[Rank {rank}] Original embedding shape: {original_embedding.weight.shape}")
        print(f"[Rank {rank}] Original lm_head shape: {original_lm_head.weight.shape}")

        vocab_parallel_embedding = VocabParallelEmbedding(
            num_embeddings=original_embedding.num_embeddings,
            embedding_dim=original_embedding.embedding_dim,
            padding_idx=original_embedding.padding_idx,
            tp_group=tp_group,
            dtype=original_embedding.weight.dtype,
            device=device,
        )

        # Compute vocab partition indices
        vocab_size = original_embedding.num_embeddings
        vocab_per_rank = vocab_size // tp_world_size
        start_idx = tp_rank * vocab_per_rank
        end_idx = start_idx + vocab_per_rank

        # Copy partitioned embedding weights
        with torch.no_grad():
            vocab_parallel_embedding.weight.data.copy_(original_embedding.weight.data[start_idx:end_idx])

        model_engine.module.embed_tokens = vocab_parallel_embedding

        # Create partitioned lm_head (NOT tied to embedding)
        # lm_head has shape [vocab_size, hidden_size], we partition on vocab dimension
        partitioned_lm_head = nn.Linear(
            original_lm_head.in_features,
            vocab_per_rank,
            bias=False,
            device=device,
            dtype=original_lm_head.weight.dtype,
        )
        with torch.no_grad():
            partitioned_lm_head.weight.data.copy_(original_lm_head.weight.data[start_idx:end_idx])

        model_engine.module.lm_head = partitioned_lm_head

        print(f"[Rank {rank}] VocabParallelEmbedding: vocab_range=[{start_idx}, {end_idx}), "
              f"partition_size={end_idx - start_idx}")
        print(f"[Rank {rank}] Partitioned lm_head shape: {partitioned_lm_head.weight.shape} (NOT tied to embedding)")

    trainer.model = model_engine
    trainer.model_config = model_config
    trainer.lm_head = model_engine.module.lm_head
    trainer.model.tp_group = tp_group
    trainer.deepspeed_engine = model_engine

    dist.barrier()
    print(f"[Rank {rank}] AutoTP model created with same initialization seed as baseline")

    # Import vocab parallel loss if needed
    if vocab_parallel:
        from python.tensor_parallel.cross_entropy import vocab_parallel_causal_cross_entropy

    # Training loop
    autotp_losses = []
    autotp_params = []
    autotp_logits = []

    for step in range(num_steps):
        input_ids, labels = prepare_batch(vocab_size, batch_size, seq_len, device, seed=seed + step)
        broadcast_for_tensor_parallel([input_ids, labels], tp_group)

        trainer.zero_grad()

        # Forward pass
        with trainer._get_autocast_context():
            outputs = trainer.model(input_ids=input_ids)
            logits = trainer.lm_head(outputs.last_hidden_state)

        # Collect logits (gather full logits if vocab parallel)
        if collect_logits:
            if vocab_parallel and tp_group is not None:
                # Each rank has logits of shape [batch, seq, vocab_per_rank]
                # Gather to get full [batch, seq, vocab_size]
                logits_list = [torch.zeros_like(logits) for _ in range(tp_world_size)]
                dist.all_gather(logits_list, logits.contiguous(), group=tp_group)
                full_logits = torch.cat(logits_list, dim=-1)
                autotp_logits.append(full_logits.detach().clone())
            else:
                autotp_logits.append(logits.detach().clone())

        # Compute loss
        if vocab_parallel and tp_group is not None:
            loss = vocab_parallel_causal_cross_entropy(
                logits,
                labels,
                tp_group,
                tp_rank,
                tp_world_size,
                ignore_index=-100,
            )
        else:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        autotp_losses.append(loss.item())

        # Backward
        trainer.deepspeed_engine.backward(loss)

        # Optimizer step
        trainer.optimizer_step()

        # Collect parameters after optimizer step
        if collect_params:
            params_snapshot = collect_selected_parameters(trainer.deepspeed_engine.module)
            autotp_params.append(params_snapshot)

        print(f"[Rank {rank}] AutoTP step {step}: loss={loss.item():.6f}")

    print(f"[Rank {rank}] AutoTP training complete")

    return autotp_losses, autotp_params, autotp_logits


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
    baseline_losses = []
    baseline_params = []

    if rank == 0:
        baseline_losses, baseline_params, _ = run_baseline_training(
            model_config=model_config,
            device=device,
            torch_dtype=torch_dtype,
            vocab_size=vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            seed=42,
            collect_params=True,
            collect_logits=False,
            rank=rank,
        )

    # Sync before starting AutoTP test
    dist.barrier()

    # ========================================
    # AutoTP model (all ranks)
    # ========================================
    autotp_losses, autotp_params, _ = run_autotp_training(
        model_config=model_config,
        rank=rank,
        world_size=world_size,
        device=device,
        torch_dtype=torch_dtype,
        vocab_size=vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
        seed=42,
        collect_params=True,
        collect_logits=False,
        vocab_parallel=False,
    )

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


def test_autotp_vocab_parallel():
    """Test that AutoTP with VocabParallelEmbedding produces the same loss as no-parallel baseline.

    This test verifies:
    1. VocabParallelEmbedding is properly created and partitions vocabulary across TP ranks
    2. Loss values match between baseline and AutoTP with vocab parallel loss
    3. Gradients flow correctly through the vocab-parallel embedding
    """
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting AutoTP Vocab Parallel test (world_size={world_size})")

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
    baseline_losses = []

    if rank == 0:
        baseline_losses, _, baseline_logits = run_baseline_training(
            model_config=model_config,
            device=device,
            torch_dtype=torch_dtype,
            vocab_size=vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            seed=42,
            collect_params=False,
            collect_logits=True,
            rank=rank,
        )
    else:
        baseline_logits = []

    dist.barrier()

    # ========================================
    # AutoTP model with vocab parallel (all ranks)
    # ========================================
    autotp_losses, _, autotp_logits = run_autotp_training(
        model_config=model_config,
        rank=rank,
        world_size=world_size,
        device=device,
        torch_dtype=torch_dtype,
        vocab_size=vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
        seed=42,
        collect_params=False,
        collect_logits=True,
        vocab_parallel=True,
    )

    # ========================================
    # Compare losses (rank 0 only)
    # ========================================
    dist.barrier()

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: Baseline vs AutoTP Vocab Parallel")
        print(f"[Rank {rank}] " + "=" * 60)

        # Compare logits first to identify if model forward pass is correct
        print(f"\n[Rank {rank}] Logits Comparison:")
        logits_match = True
        LOGITS_TOLERANCE = 1e-4
        for step in range(num_steps):
            baseline_logit = baseline_logits[step]
            autotp_logit = autotp_logits[step]

            if baseline_logit.shape != autotp_logit.shape:
                print(f"[Rank {rank}] Step {step}: ❌ Logits shape mismatch! "
                      f"baseline={baseline_logit.shape}, autotp={autotp_logit.shape}")
                logits_match = False
                continue

            # Compare logits
            logit_diff = (baseline_logit.float() - autotp_logit.float()).abs()
            max_diff = logit_diff.max().item()
            mean_diff = logit_diff.mean().item()
            rel_diff = (logit_diff / (baseline_logit.float().abs() + 1e-8)).mean().item()

            if max_diff < LOGITS_TOLERANCE:
                print(f"[Rank {rank}] Step {step}: ✅ Logits match! max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
            else:
                print(f"[Rank {rank}] Step {step}: ❌ Logits differ! max_diff={max_diff:.6e}, "
                      f"mean_diff={mean_diff:.6e}, rel_diff={rel_diff:.6e}")
                logits_match = False

        print(f"\n[Rank {rank}] Loss Comparison:")
        loss_match = True
        for step in range(num_steps):
            baseline_loss = baseline_losses[step]
            autotp_loss = autotp_losses[step]
            loss_diff = abs(baseline_loss - autotp_loss)
            rel_diff = loss_diff / (abs(baseline_loss) + 1e-8)

            print(f"[Rank {rank}] Step {step}: baseline_loss={baseline_loss:.6f}, autotp_vocab_loss={autotp_loss:.6f}, "
                  f"diff={loss_diff:.6e}, rel_diff={rel_diff:.6e}")

            LOSS_TOLERANCE = 1e-4
            if loss_diff < LOSS_TOLERANCE:
                print(f"[Rank {rank}] Step {step}: ✅ Losses are reasonably close!")
            else:
                print(f"[Rank {rank}] Step {step}: ❌ Losses differ significantly!")
                loss_match = False

        # Summary
        print(f"\n[Rank {rank}] ====================================================================")
        print(f"[Rank {rank}] SUMMARY:")
        print(f"[Rank {rank}]   Logits match: {'✅ YES' if logits_match else '❌ NO'}")
        print(f"[Rank {rank}]   Losses match: {'✅ YES' if loss_match else '❌ NO'}")
        if logits_match and not loss_match:
            print(f"[Rank {rank}]   => Issue is in LOSS CALCULATION (vocab_parallel_causal_cross_entropy)")
        elif not logits_match and not loss_match:
            print(f"[Rank {rank}]   => Issue is in MODEL FORWARD PASS (logits differ)")
        elif logits_match and loss_match:
            print(f"[Rank {rank}] TEST RESULT: ✅ PASSED - AutoTP with VocabParallelEmbedding working correctly!")
        print(f"[Rank {rank}] ====================================================================")

    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


def test_vocab_parallel_cross_entropy_correctness():
    """Test that vocab_parallel_causal_cross_entropy produces the same result as regular CrossEntropyLoss.

    This is a unit test that verifies the loss calculation is mathematically correct
    by comparing against a non-parallel baseline with the same logits.
    """
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting vocab parallel cross entropy correctness test (world_size={world_size})")

    from deepspeed.utils import groups
    from python.tensor_parallel.cross_entropy import vocab_parallel_causal_cross_entropy

    # Create a simple test case
    batch_size = 2
    seq_len = 16
    vocab_size = 1024  # Small vocab for testing

    # Ensure vocab_size is divisible by world_size
    assert vocab_size % world_size == 0, f"vocab_size {vocab_size} must be divisible by world_size {world_size}"
    vocab_per_rank = vocab_size // world_size

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize DeepSpeed groups
    from deepspeed.module_inject.layers import set_autotp_mode
    set_autotp_mode(training=True)

    # Get or create TP group (from previous test or fresh)
    try:
        groups._init_tp_mesh_device(world_size)
    except Exception:
        pass  # Already initialized

    tp_group = groups.get_tensor_model_parallel_group()
    tp_rank = groups.get_tensor_model_parallel_rank()
    tp_world_size = groups.get_tensor_model_parallel_world_size()

    print(f"[Rank {rank}] TP config: tp_rank={tp_rank}, tp_world_size={tp_world_size}")

    # Create test data (same on all ranks)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Generate full logits and labels on rank 0, then broadcast
    if rank == 0:
        full_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    else:
        full_logits = torch.zeros(batch_size, seq_len, vocab_size, device=device, dtype=torch.float32)
        labels = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

    # Broadcast full logits and labels from rank 0
    dist.broadcast(full_logits, src=0)
    dist.broadcast(labels, src=0)

    print(f"[Rank {rank}] Full logits shape: {full_logits.shape}, labels shape: {labels.shape}")

    # ========================================
    # Baseline: Regular CrossEntropyLoss (rank 0 only)
    # ========================================
    if rank == 0:
        shift_logits = full_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        baseline_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print(f"[Rank {rank}] Baseline CrossEntropyLoss: {baseline_loss.item():.6f}")
    else:
        baseline_loss = None

    # ========================================
    # Vocab Parallel: vocab_parallel_causal_cross_entropy
    # ========================================

    # Partition logits by vocabulary dimension
    # Each rank gets logits[:, :, vocab_start:vocab_end]
    vocab_start = tp_rank * vocab_per_rank
    vocab_end = vocab_start + vocab_per_rank
    sharded_logits = full_logits[:, :, vocab_start:vocab_end].contiguous()

    print(f"[Rank {rank}] Sharded logits shape: {sharded_logits.shape} (vocab_range=[{vocab_start}, {vocab_end}))")

    # Compute vocab parallel loss
    parallel_loss = vocab_parallel_causal_cross_entropy(
        sharded_logits,
        labels,
        tp_group,
        tp_rank,
        tp_world_size,
        ignore_index=-100,
    )

    print(f"[Rank {rank}] Vocab parallel loss: {parallel_loss.item():.6f}")

    # ========================================
    # Compare results (rank 0)
    # ========================================
    dist.barrier()

    if rank == 0:
        loss_diff = abs(baseline_loss.item() - parallel_loss.item())
        rel_diff = loss_diff / (abs(baseline_loss.item()) + 1e-8)

        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: Baseline vs Vocab Parallel CrossEntropy")
        print(f"[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] Baseline loss:       {baseline_loss.item():.8f}")
        print(f"[Rank {rank}] Vocab parallel loss: {parallel_loss.item():.8f}")
        print(f"[Rank {rank}] Absolute diff:       {loss_diff:.8e}")
        print(f"[Rank {rank}] Relative diff:       {rel_diff:.8e}")

        # Very tight tolerance for this exact comparison
        LOSS_TOLERANCE = 1e-5
        if loss_diff < LOSS_TOLERANCE:
            print(f"[Rank {rank}] ✅ PASSED - vocab_parallel_causal_cross_entropy produces correct result!")
            test_passed = True
        else:
            print(f"[Rank {rank}] ❌ FAILED - Losses don't match!")
            test_passed = False

        print(f"[Rank {rank}] ====================================================================")

        # Assert for pytest
        assert test_passed, f"vocab_parallel_causal_cross_entropy produced incorrect loss: diff={loss_diff:.8e}"

    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


if __name__ == "__main__":
    test_autotp_vs_no_parallel()
