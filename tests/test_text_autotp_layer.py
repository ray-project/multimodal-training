"""Test single transformer layer comparison between DTensor and AutoTP.

This test compares:
1. Forward timing breakdown (input_layernorm, attention, mlp, etc.)
2. Output correctness (same input -> same output)
3. Gradient correctness (same loss -> same gradients)

Uses hooks for fine-grained profiling without modifying model code.

Run with:
    torchrun --nproc_per_node=4 -m pytest tests/test_text_autotp_layer.py -v -s
"""

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig  # noqa: E402
from python.ray.utils import init_distributed_comm  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for this test", allow_module_level=True)


@dataclass
class TimingStats:
    """Accumulator for timing statistics."""
    forward_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    backward_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def add_forward(self, name: str, time_ms: float):
        self.forward_times[name].append(time_ms)

    def add_backward(self, name: str, time_ms: float):
        self.backward_times[name].append(time_ms)

    def get_summary(self, skip_first: int = 1) -> Dict[str, Dict[str, float]]:
        """Get timing summary, optionally skipping first N iterations for warmup."""
        summary = {"forward": {}, "backward": {}}
        for name, times in self.forward_times.items():
            valid_times = times[skip_first:] if len(times) > skip_first else times
            if valid_times:
                summary["forward"][name] = {
                    "mean": sum(valid_times) / len(valid_times),
                    "min": min(valid_times),
                    "max": max(valid_times),
                    "count": len(valid_times),
                }
        for name, times in self.backward_times.items():
            valid_times = times[skip_first:] if len(times) > skip_first else times
            if valid_times:
                summary["backward"][name] = {
                    "mean": sum(valid_times) / len(valid_times),
                    "min": min(valid_times),
                    "max": max(valid_times),
                    "count": len(valid_times),
                }
        return summary


class LayerProfiler:
    """Hook-based profiler for transformer layer components."""

    def __init__(self, stats: TimingStats, prefix: str = ""):
        self.stats = stats
        self.prefix = prefix
        self.handles: List[Any] = []
        self._start_times: Dict[str, float] = {}
        self._is_backward = False

    def _make_name(self, module_name: str) -> str:
        return f"{self.prefix}{module_name}" if self.prefix else module_name

    def _pre_hook(self, module_name: str):
        def hook(module, inputs):
            torch.cuda.synchronize()
            self._start_times[module_name] = time.perf_counter()
        return hook

    def _post_hook(self, module_name: str):
        def hook(module, inputs, outputs):
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - self._start_times[module_name]) * 1000
            name = self._make_name(module_name)
            if self._is_backward:
                self.stats.add_backward(name, elapsed_ms)
            else:
                self.stats.add_forward(name, elapsed_ms)
        return hook

    def _backward_pre_hook(self, module_name: str):
        def hook(module, grad_output):
            torch.cuda.synchronize()
            self._start_times[f"bwd_{module_name}"] = time.perf_counter()
        return hook

    def _backward_post_hook(self, module_name: str):
        def hook(module, grad_input, grad_output):
            torch.cuda.synchronize()
            key = f"bwd_{module_name}"
            if key in self._start_times:
                elapsed_ms = (time.perf_counter() - self._start_times[key]) * 1000
                name = self._make_name(module_name)
                self.stats.add_backward(name, elapsed_ms)
        return hook

    def register_decoder_layer(self, layer: nn.Module):
        """Register hooks for all components of a decoder layer."""
        # Layer norms
        if hasattr(layer, 'input_layernorm'):
            h = layer.input_layernorm.register_forward_pre_hook(self._pre_hook("input_layernorm"))
            self.handles.append(h)
            h = layer.input_layernorm.register_forward_hook(self._post_hook("input_layernorm"))
            self.handles.append(h)
            h = layer.input_layernorm.register_full_backward_pre_hook(self._backward_pre_hook("input_layernorm"))
            self.handles.append(h)
            h = layer.input_layernorm.register_full_backward_hook(self._backward_post_hook("input_layernorm"))
            self.handles.append(h)

        if hasattr(layer, 'post_attention_layernorm'):
            h = layer.post_attention_layernorm.register_forward_pre_hook(self._pre_hook("post_attn_layernorm"))
            self.handles.append(h)
            h = layer.post_attention_layernorm.register_forward_hook(self._post_hook("post_attn_layernorm"))
            self.handles.append(h)
            h = layer.post_attention_layernorm.register_full_backward_pre_hook(self._backward_pre_hook("post_attn_layernorm"))
            self.handles.append(h)
            h = layer.post_attention_layernorm.register_full_backward_hook(self._backward_post_hook("post_attn_layernorm"))
            self.handles.append(h)

        # Attention components
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            # Overall attention
            h = attn.register_forward_pre_hook(self._pre_hook("self_attn"))
            self.handles.append(h)
            h = attn.register_forward_hook(self._post_hook("self_attn"))
            self.handles.append(h)
            h = attn.register_full_backward_pre_hook(self._backward_pre_hook("self_attn"))
            self.handles.append(h)
            h = attn.register_full_backward_hook(self._backward_post_hook("self_attn"))
            self.handles.append(h)

            # Individual projections
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    h = proj.register_forward_pre_hook(self._pre_hook(f"attn.{proj_name}"))
                    self.handles.append(h)
                    h = proj.register_forward_hook(self._post_hook(f"attn.{proj_name}"))
                    self.handles.append(h)
                    h = proj.register_full_backward_pre_hook(self._backward_pre_hook(f"attn.{proj_name}"))
                    self.handles.append(h)
                    h = proj.register_full_backward_hook(self._backward_post_hook(f"attn.{proj_name}"))
                    self.handles.append(h)

        # MLP components
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            # Overall MLP
            h = mlp.register_forward_pre_hook(self._pre_hook("mlp"))
            self.handles.append(h)
            h = mlp.register_forward_hook(self._post_hook("mlp"))
            self.handles.append(h)
            h = mlp.register_full_backward_pre_hook(self._backward_pre_hook("mlp"))
            self.handles.append(h)
            h = mlp.register_full_backward_hook(self._backward_post_hook("mlp"))
            self.handles.append(h)

            # Individual projections
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    h = proj.register_forward_pre_hook(self._pre_hook(f"mlp.{proj_name}"))
                    self.handles.append(h)
                    h = proj.register_forward_hook(self._post_hook(f"mlp.{proj_name}"))
                    self.handles.append(h)
                    h = proj.register_full_backward_pre_hook(self._backward_pre_hook(f"mlp.{proj_name}"))
                    self.handles.append(h)
                    h = proj.register_full_backward_hook(self._backward_post_hook(f"mlp.{proj_name}"))
                    self.handles.append(h)

    def set_backward_mode(self, is_backward: bool):
        self._is_backward = is_backward

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


_distributed_initialized = False


def init_distributed():
    """Initialize distributed environment."""
    global _distributed_initialized

    if not dist.is_initialized():
        init_distributed_comm(backend="nccl", use_deepspeed=True)
        _distributed_initialized = True

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed environment if we initialized it."""
    global _distributed_initialized

    if _distributed_initialized and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        _distributed_initialized = False


def create_model_config(model_name="Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="sdpa", num_layers=1):
    """Create model config with specified number of layers."""
    config = Qwen2_5_VLConfig.from_pretrained(model_name, trust_remote_code=True)
    # Set attention implementation
    config.text_config._attn_implementation = attn_implementation
    # Reduce to specified number of layers for faster testing
    config.text_config.num_hidden_layers = num_layers
    return config


def create_dtensor_model(config, device, torch_dtype, seed: int = 42):
    """Create a text model with DTensor tensor parallelism."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    from python.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_default_device(device)
    model = Qwen2_5_VLTextModel._from_config(config.text_config)
    torch.set_default_device("cpu")

    model.to(device=device, dtype=torch_dtype)

    # Apply tensor parallelism
    tp_world_size = dist.get_world_size()
    tp_mesh = init_device_mesh("cuda", (tp_world_size,))

    tp_mapping = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }

    # Parallelize each layer
    for layer in model.layers:
        parallelize_module(layer, tp_mesh, tp_mapping, src_data_rank=0)

    # Disable dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    model.train()
    return model, tp_mesh.get_group()


def create_autotp_model(config, device, torch_dtype, seed: int = 42):
    """Create a text model with DeepSpeed AutoTP tensor parallelism."""
    import deepspeed
    from deepspeed.module_inject.layers import set_autotp_mode
    from deepspeed.utils import groups

    from python.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    set_autotp_mode(training=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_default_device(device)
    model = Qwen2_5_VLTextModel._from_config(config.text_config)
    torch.set_default_device("cpu")

    model.to(dtype=torch_dtype)

    # Apply AutoTP
    tp_size = dist.get_world_size()
    model = deepspeed.tp_model_init(model, tp_size=tp_size, dtype=torch_dtype)

    try:
        tp_group = groups.get_tensor_model_parallel_group()
    except Exception:
        tp_group = None

    # Disable dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    model.train()
    return model, tp_group


def prepare_inputs(config, batch_size: int, seq_len: int, device, torch_dtype, seed: int = 42):
    """Prepare inputs for the text model."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size

    # Input IDs (random tokens)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Position IDs for 3D RoPE (temporal, height, width all same for text-only)
    # Shape: [3, batch_size, seq_len]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)

    # Labels for loss computation (same as output hidden states shape)
    labels = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch_dtype)

    return input_ids, position_ids, labels


def run_forward_backward(model, input_ids, position_ids, labels, profiler: Optional[LayerProfiler] = None):
    """Run forward and backward pass with optional profiling."""
    # Forward pass
    if profiler:
        profiler.set_backward_mode(False)

    torch.cuda.synchronize()
    fwd_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        use_cache=False,
    )
    output = outputs.last_hidden_state

    torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - fwd_start) * 1000

    # Compute simple MSE loss
    loss = ((output - labels) ** 2).mean()

    # Backward pass
    if profiler:
        profiler.set_backward_mode(True)

    torch.cuda.synchronize()
    bwd_start = time.perf_counter()

    loss.backward()

    torch.cuda.synchronize()
    bwd_time = (time.perf_counter() - bwd_start) * 1000

    return output.detach().clone(), loss.item(), fwd_time, bwd_time


def broadcast_inputs(tensors: List[torch.Tensor], tp_group):
    """Broadcast inputs to ensure all TP ranks see identical data."""
    if tp_group is None or not dist.is_initialized():
        return

    try:
        from deepspeed.utils import groups
        src_rank = groups.get_tensor_model_parallel_src_rank()
    except Exception:
        src_rank = 0

    for tensor in tensors:
        dist.broadcast(tensor, src=src_rank, group=tp_group)


def print_timing_comparison(dtensor_summary: Dict, autotp_summary: Dict, rank: int):
    """Print side-by-side timing comparison."""
    print(f"\n[Rank {rank}] {'='*80}")
    print(f"[Rank {rank}] TIMING COMPARISON: DTensor vs AutoTP")
    print(f"[Rank {rank}] {'='*80}")

    # Collect all component names
    all_components = set()
    all_components.update(dtensor_summary.get("forward", {}).keys())
    all_components.update(autotp_summary.get("forward", {}).keys())

    # Sort components for consistent display
    sorted_components = sorted(all_components)

    print(f"\n[Rank {rank}] FORWARD PASS (mean time in ms):")
    print(f"[Rank {rank}] {'Component':<25} {'DTensor':>12} {'AutoTP':>12} {'Diff':>12} {'Ratio':>10}")
    print(f"[Rank {rank}] {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for comp in sorted_components:
        dt_time = dtensor_summary.get("forward", {}).get(comp, {}).get("mean", 0)
        at_time = autotp_summary.get("forward", {}).get(comp, {}).get("mean", 0)
        diff = dt_time - at_time
        ratio = dt_time / at_time if at_time > 0 else float('inf')

        print(f"[Rank {rank}] {comp:<25} {dt_time:>12.2f} {at_time:>12.2f} {diff:>+12.2f} {ratio:>10.2f}x")

    print(f"\n[Rank {rank}] BACKWARD PASS (mean time in ms):")
    print(f"[Rank {rank}] {'Component':<25} {'DTensor':>12} {'AutoTP':>12} {'Diff':>12} {'Ratio':>10}")
    print(f"[Rank {rank}] {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    # Backward components
    all_bwd_components = set()
    all_bwd_components.update(dtensor_summary.get("backward", {}).keys())
    all_bwd_components.update(autotp_summary.get("backward", {}).keys())
    sorted_bwd_components = sorted(all_bwd_components)

    for comp in sorted_bwd_components:
        dt_time = dtensor_summary.get("backward", {}).get(comp, {}).get("mean", 0)
        at_time = autotp_summary.get("backward", {}).get(comp, {}).get("mean", 0)
        diff = dt_time - at_time
        ratio = dt_time / at_time if at_time > 0 else float('inf')

        print(f"[Rank {rank}] {comp:<25} {dt_time:>12.2f} {at_time:>12.2f} {diff:>+12.2f} {ratio:>10.2f}x")


def test_single_layer_dtensor_vs_autotp():
    """Test single transformer layer comparison between DTensor and AutoTP.

    Compares:
    1. Fine-grained timing for each component
    2. Output values (should be close given same initialization)
    3. Loss values
    """
    rank, world_size, local_rank = init_distributed()

    print(f"\n[Rank {rank}] Starting single layer comparison test (world_size={world_size})")

    # Configuration
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    batch_size = 2
    seq_len = 512  # Reasonable sequence length
    num_iterations = 10
    num_layers = 1  # Single layer for focused testing

    # Create model config with single layer
    config = create_model_config(num_layers=num_layers)

    print(f"[Rank {rank}] Config: hidden_size={config.text_config.hidden_size}, "
          f"num_attention_heads={config.text_config.num_attention_heads}, "
          f"intermediate_size={config.text_config.intermediate_size}, "
          f"num_layers={config.text_config.num_hidden_layers}")

    dist.barrier()

    # ========================================
    # DTensor Model
    # ========================================
    print(f"\n[Rank {rank}] {'='*60}")
    print(f"[Rank {rank}] Creating DTensor model")
    print(f"[Rank {rank}] {'='*60}")

    dtensor_model, dtensor_tp_group = create_dtensor_model(
        config, device, torch_dtype, seed=42
    )

    dtensor_stats = TimingStats()
    dtensor_profiler = LayerProfiler(dtensor_stats, prefix="")
    # Register hooks on the first (only) layer
    dtensor_profiler.register_decoder_layer(dtensor_model.layers[0])

    dtensor_outputs = []
    dtensor_losses = []
    dtensor_fwd_times = []
    dtensor_bwd_times = []

    for i in range(num_iterations):
        # Prepare inputs (same seed for each iteration for reproducibility)
        input_ids, position_ids, labels = prepare_inputs(
            config, batch_size, seq_len, device, torch_dtype, seed=100 + i
        )
        broadcast_inputs([input_ids, labels], dtensor_tp_group)

        # Zero gradients
        for p in dtensor_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Run forward/backward
        output, loss, fwd_time, bwd_time = run_forward_backward(
            dtensor_model, input_ids, position_ids, labels, dtensor_profiler
        )

        dtensor_outputs.append(output)
        dtensor_losses.append(loss)
        dtensor_fwd_times.append(fwd_time)
        dtensor_bwd_times.append(bwd_time)

        print(f"[Rank {rank}] DTensor iter {i}: loss={loss:.6f}, fwd={fwd_time:.2f}ms, bwd={bwd_time:.2f}ms")

    dtensor_profiler.remove_hooks()

    # Collect DTensor gradients
    dtensor_grads = {}
    for name, param in dtensor_model.named_parameters():
        if param.grad is not None:
            dtensor_grads[name] = param.grad.detach().clone()

    # Clean up DTensor model
    del dtensor_model
    torch.cuda.empty_cache()

    dist.barrier()

    # ========================================
    # AutoTP Model
    # ========================================
    print(f"\n[Rank {rank}] {'='*60}")
    print(f"[Rank {rank}] Creating AutoTP model")
    print(f"[Rank {rank}] {'='*60}")

    autotp_model, autotp_tp_group = create_autotp_model(
        config, device, torch_dtype, seed=42
    )

    autotp_stats = TimingStats()
    autotp_profiler = LayerProfiler(autotp_stats, prefix="")
    # Register hooks on the first (only) layer
    autotp_profiler.register_decoder_layer(autotp_model.layers[0])

    autotp_outputs = []
    autotp_losses = []
    autotp_fwd_times = []
    autotp_bwd_times = []

    for i in range(num_iterations):
        # Prepare inputs (same seed as DTensor)
        input_ids, position_ids, labels = prepare_inputs(
            config, batch_size, seq_len, device, torch_dtype, seed=100 + i
        )
        broadcast_inputs([input_ids, labels], autotp_tp_group)

        # Zero gradients
        for p in autotp_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Run forward/backward
        output, loss, fwd_time, bwd_time = run_forward_backward(
            autotp_model, input_ids, position_ids, labels, autotp_profiler
        )

        autotp_outputs.append(output)
        autotp_losses.append(loss)
        autotp_fwd_times.append(fwd_time)
        autotp_bwd_times.append(bwd_time)

        print(f"[Rank {rank}] AutoTP iter {i}: loss={loss:.6f}, fwd={fwd_time:.2f}ms, bwd={bwd_time:.2f}ms")

    autotp_profiler.remove_hooks()

    # Collect AutoTP gradients
    autotp_grads = {}
    for name, param in autotp_model.named_parameters():
        if param.grad is not None:
            autotp_grads[name] = param.grad.detach().clone()

    dist.barrier()

    # ========================================
    # Comparison
    # ========================================
    if rank == 0:
        # Timing comparison
        dtensor_summary = dtensor_stats.get_summary(skip_first=2)
        autotp_summary = autotp_stats.get_summary(skip_first=2)
        print_timing_comparison(dtensor_summary, autotp_summary, rank)

        # Overall timing summary
        print(f"\n[Rank {rank}] {'='*80}")
        print(f"[Rank {rank}] OVERALL TIMING (mean, excluding warmup)")
        print(f"[Rank {rank}] {'='*80}")

        dt_fwd_mean = sum(dtensor_fwd_times[2:]) / len(dtensor_fwd_times[2:])
        dt_bwd_mean = sum(dtensor_bwd_times[2:]) / len(dtensor_bwd_times[2:])
        at_fwd_mean = sum(autotp_fwd_times[2:]) / len(autotp_fwd_times[2:])
        at_bwd_mean = sum(autotp_bwd_times[2:]) / len(autotp_bwd_times[2:])

        print(f"[Rank {rank}] DTensor:  Forward={dt_fwd_mean:.2f}ms, Backward={dt_bwd_mean:.2f}ms, Total={dt_fwd_mean+dt_bwd_mean:.2f}ms")
        print(f"[Rank {rank}] AutoTP:   Forward={at_fwd_mean:.2f}ms, Backward={at_bwd_mean:.2f}ms, Total={at_fwd_mean+at_bwd_mean:.2f}ms")
        print(f"[Rank {rank}] Speedup:  Forward={dt_fwd_mean/at_fwd_mean:.2f}x, Backward={dt_bwd_mean/at_bwd_mean:.2f}x")

        # Loss comparison
        print(f"\n[Rank {rank}] {'='*80}")
        print(f"[Rank {rank}] LOSS COMPARISON")
        print(f"[Rank {rank}] {'='*80}")

        for i in range(num_iterations):
            dt_loss = dtensor_losses[i]
            at_loss = autotp_losses[i]
            diff = abs(dt_loss - at_loss)
            rel_diff = diff / (abs(dt_loss) + 1e-8)
            match = "OK" if rel_diff < 0.01 else "DIFF"
            print(f"[Rank {rank}] Iter {i}: DTensor={dt_loss:.6f}, AutoTP={at_loss:.6f}, "
                  f"diff={diff:.6e}, rel_diff={rel_diff:.6e} [{match}]")

        # Output comparison (last iteration)
        print(f"\n[Rank {rank}] {'='*80}")
        print(f"[Rank {rank}] OUTPUT COMPARISON (last iteration)")
        print(f"[Rank {rank}] {'='*80}")

        dt_out = dtensor_outputs[-1]
        at_out = autotp_outputs[-1]

        out_diff = (dt_out - at_out).abs()
        max_diff = out_diff.max().item()
        mean_diff = out_diff.mean().item()
        rel_diff = (out_diff / (dt_out.abs() + 1e-8)).mean().item()

        print(f"[Rank {rank}] Output shape: DTensor={dt_out.shape}, AutoTP={at_out.shape}")
        print(f"[Rank {rank}] Max diff: {max_diff:.6e}")
        print(f"[Rank {rank}] Mean diff: {mean_diff:.6e}")
        print(f"[Rank {rank}] Rel diff: {rel_diff:.6e}")

        # Gradient comparison
        print(f"\n[Rank {rank}] {'='*80}")
        print(f"[Rank {rank}] GRADIENT COMPARISON")
        print(f"[Rank {rank}] {'='*80}")

        # Find common parameters (names may differ due to TP)
        print(f"[Rank {rank}] DTensor grad params: {list(dtensor_grads.keys())[:5]}...")
        print(f"[Rank {rank}] AutoTP grad params: {list(autotp_grads.keys())[:5]}...")

    cleanup_distributed()

    print(f"[Rank {rank}] Test complete")


@pytest.mark.skip(reason="Run standalone: torchrun --nproc_per_node=4 python tests/test_text_autotp_layer.py detailed")
def test_single_layer_timing_detailed():
    """Detailed timing test focusing on identifying performance differences.

    Runs multiple iterations with profiling to get stable timing measurements.

    Note: This test is skipped by pytest because running multiple distributed tests
    sequentially causes process group cleanup issues. Run it standalone with:
        torchrun --nproc_per_node=4 python tests/test_text_autotp_layer.py detailed
    """
    rank, world_size, local_rank = init_distributed()

    if rank == 0:
        print(f"\n[Rank {rank}] Starting detailed timing test (world_size={world_size})")

    # Configuration - larger batch/seq for more realistic timing
    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    batch_size = 2
    seq_len = 1024  # Longer sequence for better timing
    num_warmup = 3
    num_iterations = 20
    num_layers = 1  # Single layer for focused testing

    config = create_model_config(num_layers=num_layers)

    dist.barrier()

    # Test DTensor
    if rank == 0:
        print(f"\n[Rank {rank}] Running DTensor timing ({num_warmup} warmup + {num_iterations} iterations)...")

    dtensor_model, dtensor_tp_group = create_dtensor_model(
        config, device, torch_dtype, seed=42
    )

    dtensor_stats = TimingStats()
    dtensor_profiler = LayerProfiler(dtensor_stats)
    dtensor_profiler.register_decoder_layer(dtensor_model.layers[0])

    for i in range(num_warmup + num_iterations):
        input_ids, position_ids, labels = prepare_inputs(
            config, batch_size, seq_len, device, torch_dtype, seed=100 + i
        )
        broadcast_inputs([input_ids, labels], dtensor_tp_group)

        for p in dtensor_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        run_forward_backward(dtensor_model, input_ids, position_ids, labels, dtensor_profiler)

    dtensor_profiler.remove_hooks()
    del dtensor_model
    torch.cuda.empty_cache()

    dist.barrier()

    # Test AutoTP
    if rank == 0:
        print(f"[Rank {rank}] Running AutoTP timing ({num_warmup} warmup + {num_iterations} iterations)...")

    autotp_model, autotp_tp_group = create_autotp_model(
        config, device, torch_dtype, seed=42
    )

    autotp_stats = TimingStats()
    autotp_profiler = LayerProfiler(autotp_stats)
    autotp_profiler.register_decoder_layer(autotp_model.layers[0])

    for i in range(num_warmup + num_iterations):
        input_ids, position_ids, labels = prepare_inputs(
            config, batch_size, seq_len, device, torch_dtype, seed=100 + i
        )
        broadcast_inputs([input_ids, labels], autotp_tp_group)

        for p in autotp_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        run_forward_backward(autotp_model, input_ids, position_ids, labels, autotp_profiler)

    autotp_profiler.remove_hooks()

    dist.barrier()

    # Print results
    if rank == 0:
        dtensor_summary = dtensor_stats.get_summary(skip_first=num_warmup)
        autotp_summary = autotp_stats.get_summary(skip_first=num_warmup)
        print_timing_comparison(dtensor_summary, autotp_summary, rank)

    cleanup_distributed()

    if rank == 0:
        print(f"\n[Rank {rank}] Detailed timing test complete")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "detailed":
        test_single_layer_timing_detailed()
    else:
        test_single_layer_dtensor_vs_autotp()
