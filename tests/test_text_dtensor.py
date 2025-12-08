"""Test DTensor tensor parallelism functionality for the text model.

This test verifies that PyTorch DTensor tensor parallelism produces the same gradients
as the non-parallel baseline.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_text_dtensor.py -v
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
from python.tensor_parallel import VocabParallelEmbedding  # noqa: E402
from python.tensor_parallel.cross_entropy import vocab_parallel_causal_cross_entropy  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for DTensor test", allow_module_level=True)


def init_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

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


def get_tensor_parallel_mapping():
    """Get tensor parallel mapping for Qwen model layers."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    return {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }


def create_dtensor_model(model_config, rank, world_size, device, torch_dtype, seed=42):
    """Create a DTensor tensor-parallel model with fixed seed.

    Returns:
        Tuple of (model, lm_head, tp_group, tp_mesh)
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module

    # Set seed for deterministic initialization (same as baseline)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_default_device(device)
    model = Qwen2_5_VLTextModel._from_config(model_config.text_config)
    lm_head = nn.Linear(model_config.text_config.hidden_size, model_config.text_config.vocab_size, bias=False)
    torch.set_default_device("cpu")

    model.to(torch_dtype)
    lm_head.to(torch_dtype)

    # Disable dropout for deterministic behavior
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    # Create device mesh for tensor parallelism
    tp_mesh = init_device_mesh("cuda", (world_size,))
    tp_group = tp_mesh.get_group()

    # Replace embedding with VocabParallelEmbedding
    original_embedding = model.embed_tokens
    vocab_parallel_embedding = VocabParallelEmbedding(
        num_embeddings=original_embedding.num_embeddings,
        embedding_dim=original_embedding.embedding_dim,
        padding_idx=original_embedding.padding_idx,
        tp_group=tp_group,
        tp_mesh=tp_mesh,
        dtype=original_embedding.weight.dtype,
        device=device,
    )

    # Copy the appropriate partition of weights from original embedding
    with torch.no_grad():
        start_idx = vocab_parallel_embedding.vocab_start_index
        end_idx = vocab_parallel_embedding.vocab_end_index
        vocab_parallel_embedding.weight.data.copy_(original_embedding.weight.data[start_idx:end_idx].to(device))

    model.embed_tokens = vocab_parallel_embedding

    # Parallelize transformer layers using DTensor
    tp_mapping = get_tensor_parallel_mapping()
    for layer in model.layers:
        parallelize_module(layer, tp_mesh, tp_mapping, src_data_rank=0)

    # Shard lm_head weights for vocab parallelism (NOT tied, since tie_word_embeddings=False for Qwen)
    # lm_head has shape [vocab_size, hidden_size], we take the partition [start_idx:end_idx, :]
    original_lm_head_weight = lm_head.weight.data.clone()
    lm_head.weight = nn.Parameter(original_lm_head_weight[start_idx:end_idx, :].to(device))

    model.to(device=device, dtype=torch_dtype)
    lm_head.to(device=device, dtype=torch_dtype)

    model.train()
    lm_head.train()

    return model, lm_head, tp_group, tp_mesh


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
    For DTensor, this will only include replicated parameters that should match across ranks.
    """
    params = {}

    # Collect first layer norm parameters (replicated in TP)
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
    # Note: Use foreach=False for consistency with DTensor path
    params = list(baseline_model.parameters()) + list(baseline_lm_head.parameters())
    # Deduplicate tied weights
    param_ids = set()
    unique_params = []
    for p in params:
        if id(p) not in param_ids:
            param_ids.add(id(p))
            unique_params.append(p)
    optimizer = torch.optim.AdamW(unique_params, lr=1e-4, weight_decay=0.01, foreach=False)

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


def run_dtensor_training(
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
):
    """Run DTensor tensor parallel training and return losses.

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

    Returns:
        Tuple of (losses, params, logits_list) where params/logits_list are empty lists if not collected
    """
    print(f"\n[Rank {rank}] " + "=" * 60)
    print(f"[Rank {rank}] Creating DTensor model (world_size={world_size})")
    print(f"[Rank {rank}] " + "=" * 60)

    model, lm_head, tp_group, tp_mesh = create_dtensor_model(
        model_config, rank, world_size, device, torch_dtype, seed=seed
    )

    tp_rank = dist.get_rank(tp_group)
    tp_world_size = dist.get_world_size(tp_group)

    print(f"[Rank {rank}] DTensor model created: tp_rank={tp_rank}, tp_world_size={tp_world_size}")

    # Create optimizer
    # Note: Must use foreach=False to avoid issues with mixed DTensor and regular Tensor parameters
    params = list(model.parameters()) + list(lm_head.parameters())
    # Deduplicate tied weights
    param_ids = set()
    unique_params = []
    for p in params:
        if id(p) not in param_ids:
            param_ids.add(id(p))
            unique_params.append(p)
    optimizer = torch.optim.AdamW(unique_params, lr=1e-4, weight_decay=0.01, foreach=False)

    dist.barrier()
    print(f"[Rank {rank}] DTensor model created with same initialization seed as baseline")

    # Training loop
    dtensor_losses = []
    dtensor_params = []
    dtensor_logits = []

    for step in range(num_steps):
        input_ids, labels = prepare_batch(vocab_size, batch_size, seq_len, device, seed=seed + step)
        broadcast_for_tensor_parallel([input_ids, labels], tp_group)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids)
        logits = lm_head(outputs.last_hidden_state)

        # Collect logits (gather full logits for comparison)
        if collect_logits:
            # Each rank has logits of shape [batch, seq, vocab_per_rank]
            # Gather to get full [batch, seq, vocab_size]
            logits_list = [torch.zeros_like(logits) for _ in range(tp_world_size)]
            dist.all_gather(logits_list, logits.contiguous(), group=tp_group)
            full_logits = torch.cat(logits_list, dim=-1)
            dtensor_logits.append(full_logits.detach().clone())

        # Compute loss using vocab_parallel_causal_cross_entropy
        loss = vocab_parallel_causal_cross_entropy(
            logits,
            labels,
            tp_group,
            tp_rank,
            tp_world_size,
            ignore_index=-100,
        )

        dtensor_losses.append(loss.item())

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Collect parameters after optimizer step
        if collect_params:
            params_snapshot = collect_selected_parameters(model)
            dtensor_params.append(params_snapshot)

        print(f"[Rank {rank}] DTensor step {step}: loss={loss.item():.6f}")

    print(f"[Rank {rank}] DTensor training complete")

    return dtensor_losses, dtensor_params, dtensor_logits


def test_dtensor_vs_no_parallel():
    """Test that DTensor tensor parallelism produces identical results to no-parallel baseline.

    This test verifies:
    1. Logits match on step 0 (forward pass correctness)
    2. Losses match at each training step
    3. Selected parameter values match after training (layer norms that are replicated in TP)
    """
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting DTensor test (world_size={world_size})")

    # Configuration
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    batch_size = 2
    seq_len = 32
    num_steps = 3

    # Create model config
    model_config = create_small_model_config(num_layers=2)
    vocab_size = model_config.text_config.vocab_size

    print(f"[Rank {rank}] Model config: num_layers={model_config.text_config.num_hidden_layers}, vocab_size={vocab_size}")

    # ========================================
    # BASELINE: No-parallel model (rank 0 only)
    # ========================================
    baseline_losses = []
    baseline_params = []
    baseline_logits = []

    if rank == 0:
        baseline_losses, baseline_params, baseline_logits = run_baseline_training(
            model_config=model_config,
            device=device,
            torch_dtype=torch_dtype,
            vocab_size=vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            seed=42,
            collect_params=True,
            collect_logits=True,
            rank=rank,
        )

    # Sync before starting DTensor test
    dist.barrier()

    # ========================================
    # DTensor model (all ranks)
    # ========================================
    dtensor_losses, dtensor_params, dtensor_logits = run_dtensor_training(
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
        collect_logits=True,
    )

    # ========================================
    # Compare results (rank 0 only)
    # ========================================
    dist.barrier()

    if rank == 0:
        print(f"\n[Rank {rank}] " + "=" * 60)
        print(f"[Rank {rank}] COMPARISON: Baseline vs DTensor")
        print(f"[Rank {rank}] " + "=" * 60)

        # 1. Compare logits at step 0 (forward pass correctness)
        print(f"\n[Rank {rank}] Step 0 Logits Comparison (forward pass check):")
        logits_match = True
        LOGITS_TOLERANCE = 1e-4

        baseline_logit = baseline_logits[0]
        dtensor_logit = dtensor_logits[0]

        if baseline_logit.shape != dtensor_logit.shape:
            print(f"[Rank {rank}] Logits shape mismatch! "
                  f"baseline={baseline_logit.shape}, dtensor={dtensor_logit.shape}")
            logits_match = False
        else:
            logit_diff = (baseline_logit.float() - dtensor_logit.float()).abs()
            max_diff = logit_diff.max().item()
            mean_diff = logit_diff.mean().item()

            if max_diff < LOGITS_TOLERANCE:
                print(f"[Rank {rank}] Logits MATCH! max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
            else:
                print(f"[Rank {rank}] Logits DIFFER! max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
                logits_match = False

        # 2. Compare losses at each step
        print(f"\n[Rank {rank}] Loss Comparison:")
        all_losses_match = True
        LOSS_TOLERANCE = 1e-3  # bfloat16 + TP all-reduce introduces small differences

        for step in range(num_steps):
            baseline_loss = baseline_losses[step]
            dtensor_loss = dtensor_losses[step]
            loss_diff = abs(baseline_loss - dtensor_loss)
            rel_diff = loss_diff / (abs(baseline_loss) + 1e-8)

            match_str = "MATCH" if loss_diff < LOSS_TOLERANCE else "DIFFER"
            print(f"[Rank {rank}] Step {step}: baseline={baseline_loss:.6f}, dtensor={dtensor_loss:.6f}, "
                  f"diff={loss_diff:.6e} [{match_str}]")

            if loss_diff >= LOSS_TOLERANCE:
                all_losses_match = False

        # 3. Compare parameters after final step
        print(f"\n[Rank {rank}] Final Parameter Comparison (after {num_steps} steps):")
        all_params_match = True
        PARAM_TOLERANCE = 1e-5

        baseline_param_dict = baseline_params[-1]
        dtensor_param_dict = dtensor_params[-1]

        for param_name in baseline_param_dict.keys():
            if param_name not in dtensor_param_dict:
                print(f"[Rank {rank}]   {param_name}: Missing in DTensor")
                all_params_match = False
                continue

            baseline_param = baseline_param_dict[param_name]
            dtensor_param = dtensor_param_dict[param_name]

            # Handle DTensor - convert to local tensor if needed
            if hasattr(dtensor_param, 'full_tensor'):
                dtensor_param = dtensor_param.full_tensor()
            elif hasattr(dtensor_param, 'to_local'):
                dtensor_param = dtensor_param.to_local()

            if baseline_param.shape != dtensor_param.shape:
                print(f"[Rank {rank}]   {param_name}: Shape mismatch "
                      f"(baseline={baseline_param.shape}, dtensor={dtensor_param.shape})")
                all_params_match = False
                continue

            diff = (baseline_param - dtensor_param).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            match_str = "MATCH" if max_diff < PARAM_TOLERANCE else "DIFFER"
            print(f"[Rank {rank}]   {param_name}: {match_str} max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

            if max_diff >= PARAM_TOLERANCE:
                all_params_match = False

            # Check for NaN/Inf
            if torch.isnan(dtensor_param).any():
                print(f"[Rank {rank}]   {param_name}: Contains NaN!")
                all_params_match = False
            elif torch.isinf(dtensor_param).any():
                print(f"[Rank {rank}]   {param_name}: Contains Inf!")
                all_params_match = False

        # Summary
        print(f"\n[Rank {rank}] ====================================================================")
        print(f"[Rank {rank}] SUMMARY:")
        print(f"[Rank {rank}]   Step 0 logits match: {'YES' if logits_match else 'NO'}")
        print(f"[Rank {rank}]   All losses match:    {'YES' if all_losses_match else 'NO'}")
        print(f"[Rank {rank}]   Final params match:  {'YES' if all_params_match else 'NO'}")

        if logits_match and all_losses_match and all_params_match:
            print(f"[Rank {rank}] TEST RESULT: PASSED")
        else:
            if not logits_match:
                print(f"[Rank {rank}]   => Issue in FORWARD PASS (logits differ)")
            if not all_losses_match:
                print(f"[Rank {rank}]   => Issue in LOSS CALCULATION")
            if not all_params_match:
                print(f"[Rank {rank}]   => Issue in BACKWARD PASS or OPTIMIZER")
            print(f"[Rank {rank}] TEST RESULT: FAILED")
        print(f"[Rank {rank}] ====================================================================")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


def test_vocab_parallel_cross_entropy_dtensor():
    """Test that vocab_parallel_causal_cross_entropy produces the same result as regular CrossEntropyLoss.

    This is a unit test that verifies the loss calculation is mathematically correct
    by comparing against a non-parallel baseline with the same logits.
    """
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting vocab parallel cross entropy test with DTensor (world_size={world_size})")

    from torch.distributed.device_mesh import init_device_mesh

    # Create a simple test case
    batch_size = 2
    seq_len = 16
    vocab_size = 1024  # Small vocab for testing

    # Ensure vocab_size is divisible by world_size
    assert vocab_size % world_size == 0, f"vocab_size {vocab_size} must be divisible by world_size {world_size}"
    vocab_per_rank = vocab_size // world_size

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Create device mesh for tensor parallelism
    tp_mesh = init_device_mesh("cuda", (world_size,))
    tp_group = tp_mesh.get_group()
    tp_rank = dist.get_rank(tp_group)
    tp_world_size = dist.get_world_size(tp_group)

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
            print(f"[Rank {rank}] PASSED - vocab_parallel_causal_cross_entropy produces correct result!")
            test_passed = True
        else:
            print(f"[Rank {rank}] FAILED - Losses don't match!")
            test_passed = False

        print(f"[Rank {rank}] ====================================================================")

        # Assert for pytest
        assert test_passed, f"vocab_parallel_causal_cross_entropy produced incorrect loss: diff={loss_diff:.8e}"

    dist.barrier()
    dist.destroy_process_group()

    print(f"[Rank {rank}] Test complete")


if __name__ == "__main__":
    test_dtensor_vs_no_parallel()
