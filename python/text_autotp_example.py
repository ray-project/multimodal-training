"""
Minimal standalone training loop for experimenting with DeepSpeed AutoTP on the Qwen text model.

Run with multiple GPUs using torchrun or the DeepSpeed launcher, e.g.:

torchrun --nproc_per_node=2 -m python.text_autotp_example --model-name Qwen/Qwen2.5-7B-Instruct --autotp-size 2 --steps 5
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist

from .ray.text import BaseTextTrainer, QwenTextMixin
from .ray.utils import init_distributed_comm

logger = logging.getLogger(__name__)


class AutoTPQwenTextTrainer(QwenTextMixin, BaseTextTrainer):
    """Standalone trainer reusing BaseTextTrainer without Ray for AutoTP experiments."""

    def _get_device(self) -> torch.device:
        """Respect LOCAL_RANK when running under torchrun/DeepSpeed launcher."""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        return device


def _broadcast_for_tensor_parallel(tensors: list[torch.Tensor], tp_group) -> None:
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


def _parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed AutoTP text training example for Qwen.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HF model or local path")
    parser.add_argument("--autotp-size", type=int, default=None, help="Tensor parallel size (defaults to world size)")
    parser.add_argument("--zero-stage", type=int, default=1, choices=[0, 1, 2], help="DeepSpeed ZeRO stage")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank micro-batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for synthetic data")
    parser.add_argument("--steps", type=int, default=10, help="Number of training iterations to run")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for Adam")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16", help="Training dtype")
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="Attention implementation to use",
    )
    parser.add_argument("--activation-checkpointing", action="store_true", help="Enable activation checkpointing")
    parser.add_argument("--tp-overlap-comm", action="store_true", help="Enable TP communication overlap if supported")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Scheduler warmup steps")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--log-interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reduce-bucket-size", type=int, default=500000000, help="ZeRO reduce bucket size")
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", help="LR scheduler type")
    return parser.parse_args()


def _build_config(args) -> dict:
    """Construct a config dict compatible with BaseTextTrainer."""
    return {
        "model_name": args.model_name,
        "parallelism": "autotp",
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "activation_checkpointing": args.activation_checkpointing,
        "autocast": True,
        "zero_stage": args.zero_stage,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_iterations": args.steps,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": 0.0,
        "lr_scheduler_type": args.lr_scheduler_type,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "reduce_bucket_size": args.reduce_bucket_size,
        "seed": args.seed,
        "clip_grad_norm": False,
        "max_grad_norm": 1.0,
        "autotp_size": args.autotp_size,
        "tp_overlap_comm": args.tp_overlap_comm,
        # DeepSpeed batch_assert uses world_size even with tensor parallel; make it explicit here.
        "train_batch_size_override": None,
    }


def _init_distributed():
    """Initialize torch.distributed using the same helper as the Ray trainers."""
    if dist.is_initialized():
        return

    # Respect LOCAL_RANK for torchrun/DeepSpeed launchers
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env > 1 or os.environ.get("RANK") is not None:
        init_distributed_comm(backend="nccl", use_deepspeed=True)


def _prepare_batch(vocab_size: int, batch_size: int, seq_len: int, device: torch.device):
    """Create a simple causal LM batch with shifted labels."""
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    labels = input_ids.clone()
    return input_ids, labels


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")

    _init_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0

    config = _build_config(args)
    trainer = AutoTPQwenTextTrainer(config, rank)
    trainer.build_model()

    device = trainer._get_device()
    vocab_size = trainer.model_config.vocab_size
    tp_group = getattr(trainer.model, "tp_group", None)

    logger.info(
        f"[r{rank}] Starting AutoTP demo: autotp_size={config.get('autotp_size') or 'world'}, "
        f"zero_stage={config['zero_stage']}, dtype={config['dtype']}, seq_len={args.seq_len}"
    )

    for step in range(args.steps):
        input_ids, labels = _prepare_batch(vocab_size, args.batch_size, args.seq_len, device)
        _broadcast_for_tensor_parallel([input_ids, labels], tp_group)

        with trainer._get_autocast_context():
            outputs = trainer.model(input_ids=input_ids)
            logits = trainer.lm_head(outputs.last_hidden_state)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = trainer.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if trainer.use_deepspeed and trainer.deepspeed_engine is not None:
            trainer.deepspeed_engine.backward(loss)
        else:
            loss.backward()

        trainer.optimizer_step()
        trainer.zero_grad()

        if step % args.log_interval == 0 and (rank == 0 or not dist.is_initialized()):
            logger.info(f"[r{rank}] step {step}: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
