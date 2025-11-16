"""
Main entrypoint for training.
"""

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import hydra
import ray
import torch
from omegaconf import DictConfig
from ray.experimental.collective import create_collective_group

# Set log file path BEFORE importing logger module
if "RAY_TRAIN_LOG_FILE" not in os.environ:
    os.environ["RAY_TRAIN_LOG_FILE"] = os.path.abspath("logs/train.log")

from .checkpoint import find_latest_checkpoint, load_epoch_checkpoint, save_epoch_checkpoint
from .ray.actor_group import ActorGroup
from .ray.logger import setup_logging
from .ray.tensor_transfer import gather_gpu_ids
from .ray.text import QwenTextTrainer
from .ray.utils import initialize_ray
from .ray.vision import QwenVisionTrainer

# See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
mp.set_start_method("spawn", force=True)

logger = logging.getLogger(__name__)

config_dir = str(Path(__file__).parent.parent.parent / "configs")


def aggregate_grad_norms(vision_norms: list[dict], text_norms: list[dict]) -> float:
    """
    Aggregate gradient norm contributions from all actors to compute global gradient norm.

    Args:
        vision_norms: List of norm contribution dicts from vision actors
        text_norms: List of norm contribution dicts from text actors

    Returns:
        global_norm: The global gradient norm (scalar)
    """
    import math

    total_norm_sq = 0.0

    # Aggregate vision contributions
    if vision_norms:
        parallelism_type = vision_norms[0]["type"]

        if parallelism_type == "sequence":
            # For sequence parallelism, all actors have the same gradient after sync
            # Just use the first actor's norm (don't sum across actors)
            total_norm_sq += vision_norms[0]["norm_sq"]

        elif parallelism_type == "tensor":
            # For tensor parallelism:
            # - Replicated params: all actors have same gradient, use one actor's contribution
            # - Sharded params: each actor has different shard, already allreduced within TP group
            # Since we allreduce sharded_norm_sq within the TP group in compute_grad_norm_contribution,
            # all actors in the same TP group return the same sharded_norm_sq
            # So we just use the first actor's values
            replicated_norm_sq = vision_norms[0]["replicated_norm_sq"]
            sharded_norm_sq = vision_norms[0]["sharded_norm_sq"]
            total_norm_sq += replicated_norm_sq + sharded_norm_sq

        elif parallelism_type == "deepspeed":
            # For DeepSpeed (ZeRO without sequence parallelism):
            # Aggregate norm_sq from all actors
            total_norm_sq += sum(norm["norm_sq"] for norm in vision_norms)

        elif parallelism_type == "none":
            # No parallelism: sum contributions from all actors
            total_norm_sq += sum(norm["norm_sq"] for norm in vision_norms)

        else:
            logger.warning(f"Unknown vision parallelism type: {parallelism_type}, summing all contributions")
            if "norm_sq" in vision_norms[0]:
                total_norm_sq += sum(norm["norm_sq"] for norm in vision_norms)

    # Aggregate text contributions (similar logic)
    if text_norms:
        parallelism_type = text_norms[0]["type"]

        if parallelism_type == "tensor":
            # Same logic as vision tensor parallelism
            replicated_norm_sq = text_norms[0]["replicated_norm_sq"]
            sharded_norm_sq = text_norms[0]["sharded_norm_sq"]
            total_norm_sq += replicated_norm_sq + sharded_norm_sq

        elif parallelism_type == "deepspeed":
            # Aggregate norm_sq from all actors
            total_norm_sq += sum(norm["norm_sq"] for norm in text_norms)

        elif parallelism_type == "none":
            # No parallelism: sum contributions from all actors
            total_norm_sq += sum(norm["norm_sq"] for norm in text_norms)

        else:
            logger.warning(f"Unknown text parallelism type: {parallelism_type}, summing all contributions")
            if "norm_sq" in text_norms[0]:
                total_norm_sq += sum(norm["norm_sq"] for norm in text_norms)

    # Compute global norm as sqrt of total norm squared
    global_norm = math.sqrt(total_norm_sq)
    return global_norm


def get_vision_trainer_class(model_type: str):
    """Get the appropriate vision trainer class based on model type."""
    if model_type == "qwen2_5_vl":
        return QwenVisionTrainer
    else:
        raise ValueError(f"Unsupported vision model_type: {model_type}")


def get_text_trainer_class(model_type: str):
    """Get the appropriate text trainer class based on model type."""
    if model_type == "qwen2_5_vl":
        return QwenTextTrainer
    else:
        raise ValueError(f"Unsupported text model_type: {model_type}")


@hydra.main(config_path=config_dir, config_name="train", version_base=None)
def main(cfg: DictConfig):
    # Set up our custom logging - Hydra has already configured its own logging by this point
    setup_logging(force=True)

    logger.info(f"Configuration: {cfg}")

    initialize_ray()

    # Extract configs for vision and text trainers
    # Merge vision/text specific configs with training and data configs
    vision_config = dict(cfg.vision) if "vision" in cfg else {}
    text_config = dict(cfg.text) if "text" in cfg else {}

    # Add training, data, and deepspeed configs to both
    if "training" in cfg:
        vision_config.update(dict(cfg.training))
        text_config.update(dict(cfg.training))
    if "data" in cfg:
        vision_config.update(dict(cfg.data))
        text_config.update(dict(cfg.data))
    if "deepspeed" in cfg:
        vision_config.update(dict(cfg.deepspeed))
        text_config.update(dict(cfg.deepspeed))

    # Get number of actors and collocation setting from config
    parallel_size = cfg.training.parallel_size
    collocate = cfg.training.collocate

    logger.info(f"Creating {parallel_size} actors per model (vision and text)")
    logger.info(f"Collocation enabled: {collocate}")

    # Select appropriate trainer classes based on model type
    vision_model_type = vision_config["model_type"]
    text_model_type = text_config["model_type"]

    logger.info(f"Vision model type: {vision_model_type}")
    logger.info(f"Text model type: {text_model_type}")

    VisionTrainerClass = get_vision_trainer_class(vision_model_type)
    TextTrainerClass = get_text_trainer_class(text_model_type)

    # Create actor groups
    vision_trainer_group = ActorGroup(
        vision_config,
        VisionTrainerClass,
        num_actors=parallel_size,
        collocate=collocate,
    )

    # For text trainer group, reuse the placement group if collocating
    text_trainer_group = ActorGroup(
        text_config,
        TextTrainerClass,
        num_actors=parallel_size,
        collocate=collocate,
        placement_group_handle=vision_trainer_group.placement_group if collocate else None,
    )

    # Build models
    vision_trainer_group.execute_all("build_model")
    logger.info("Vision model built")
    text_trainer_group.execute_all("build_model")
    logger.info("Text model built")

    # Initialize trainers (creates datasets and dataloaders)
    vision_trainer_group.execute_all("initialize_trainer")
    text_trainer_group.execute_all("initialize_trainer")
    logger.info("Trainers initialized")

    # Set up CUDA IPC if collocating
    if collocate:
        logger.info("Setting up CUDA IPC for collocated actors...")

        # Gather GPU IDs from both groups
        vision_gpu_ids = gather_gpu_ids(vision_trainer_group._actors)
        text_gpu_ids = gather_gpu_ids(text_trainer_group._actors)

        logger.info(f"Vision GPU IDs: {vision_gpu_ids}")
        logger.info(f"Text GPU IDs: {text_gpu_ids}")

        # For each vision actor, set receiver info (which text actors it sends to)
        # Assuming 1-to-1 mapping: vision actor i sends to text actor i
        for i in range(parallel_size):
            receiver_gpu_id = text_gpu_ids[i]
            # Each vision actor sends to the corresponding text actor
            vision_trainer_group._actors[i].set_receiver_info.remote([receiver_gpu_id], use_ipc=True)

        # For each text actor, set receiver info (which vision actors receive gradients)
        for i in range(parallel_size):
            receiver_gpu_id = vision_gpu_ids[i]
            # Each text actor sends gradients back to the corresponding vision actor
            text_trainer_group._actors[i].set_receiver_info.remote([receiver_gpu_id], use_ipc=True)

        # Wait for all set_receiver_info calls to complete
        ray.get([actor.get_rank.remote() for actor in vision_trainer_group._actors])
        logger.info("CUDA IPC setup complete")
    else:
        # When not collocating, create collective group for NCCL communication
        logger.info("Creating NCCL collective group for cross-GPU communication...")
        create_collective_group(vision_trainer_group._actors + text_trainer_group._actors, backend="nccl")

    # Log parallelism strategies
    logger.info(f"Vision parallelism: {vision_config['parallelism']}, Text parallelism: {text_config['parallelism']}")

    # Get training configuration
    num_epochs = cfg.training.num_epochs
    num_iterations = cfg.training.num_iterations
    warmup_steps = cfg.training.warmup_steps
    no_checkpoint = cfg.training.no_checkpoint
    log_interval = cfg.training.log_interval
    gradient_accumulation_steps = cfg.training.gradient_accumulation_steps

    # Get checkpoint directory and convert to absolute path
    checkpoint_dir = None
    if not no_checkpoint:
        checkpoint_dir = cfg.training.checkpoint_dir
        if checkpoint_dir:
            import os

            checkpoint_dir = os.path.abspath(checkpoint_dir)
            logger.info(f"Checkpointing enabled. Directory (absolute): {checkpoint_dir}")
    else:
        logger.info("Checkpointing disabled (no_checkpoint=true)")

    # Check for existing checkpoint and auto-load if present
    start_epoch = 0
    if checkpoint_dir:
        latest_epoch = find_latest_checkpoint(checkpoint_dir)
        if latest_epoch is not None:
            logger.info(f"Found existing checkpoint at epoch {latest_epoch}, loading...")
            success = load_epoch_checkpoint(checkpoint_dir, latest_epoch, vision_trainer_group, text_trainer_group)
            if success:
                start_epoch = latest_epoch + 1
                logger.info(f"Resuming training from epoch {start_epoch}")
            else:
                logger.warning("Failed to load checkpoint, starting from scratch")
                start_epoch = 0

    # Training loop
    logger.info(
        f"Starting training: {num_epochs} epochs, {num_iterations} iterations/epoch, "
        f"warmup={warmup_steps} steps, gradient_accumulation_steps={gradient_accumulation_steps}"
    )

    iteration_times = []
    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info("=" * 60)

        epoch_start = time.perf_counter()
        epoch_loss = 0.0

        for iteration in range(num_iterations):
            # Check if this is a gradient accumulation boundary
            is_accumulation_boundary = (iteration + 1) % gradient_accumulation_steps == 0

            measure_metrics = global_step >= warmup_steps
            iteration_start = time.perf_counter() if measure_metrics else None

            # Zero gradients at the start of each accumulation cycle
            if iteration % gradient_accumulation_steps == 0:
                vision_trainer_group.execute_all("zero_grad")
                text_trainer_group.execute_all("zero_grad")

            # Vision forward pass
            vision_refs = vision_trainer_group.execute_all_async("forward_step", global_step)

            # Text forward pass
            iteration_list = [global_step] * len(vision_refs)
            text_refs = text_trainer_group.execute_all_async("forward_step", vision_refs, iteration_list)
            losses = ray.get(text_refs)

            # Extract loss values
            loss_values = [loss.item() if torch.is_tensor(loss) else loss for loss in losses]
            avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
            epoch_loss += avg_loss

            # Backward pass
            text_backward_refs = text_trainer_group.execute_all_async("backward_step")
            vision_backward_refs = vision_trainer_group.execute_all_async("backward_step", text_backward_refs)
            ray.get(vision_backward_refs)

            # Gradient clipping and optimizer step only at accumulation boundaries
            if is_accumulation_boundary:
                # Gradient clipping (if enabled)
                global_grad_norm = None
                if cfg.training.get("clip_grad_norm", False):
                    vision_norm_refs = vision_trainer_group.execute_all_async("compute_grad_norm_contribution")
                    text_norm_refs = text_trainer_group.execute_all_async("compute_grad_norm_contribution")
                    vision_norms = ray.get(vision_norm_refs)
                    text_norms = ray.get(text_norm_refs)
                    global_grad_norm = aggregate_grad_norms(vision_norms, text_norms)

                # Optimizer steps
                text_trainer_group.execute_all("optimizer_step", global_grad_norm)
                vision_trainer_group.execute_all("optimizer_step", global_grad_norm)

                # Increment global step only at accumulation boundaries
                global_step += 1

            if measure_metrics and iteration_start is not None:
                iteration_elapsed = time.perf_counter() - iteration_start
                iteration_times.append(iteration_elapsed)

            # Log at specified interval
            if (iteration + 1) % log_interval == 0 or iteration == 0:
                status = "warmup" if global_step < warmup_steps else "training"
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Iter {iteration + 1}/{num_iterations} "
                    f"({status}) - Loss: {avg_loss:.4f}, Global step: {global_step}"
                )

        # End of epoch
        epoch_elapsed = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / num_iterations
        logger.info("=" * 60)
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_elapsed:.2f}s - Avg Loss: {avg_epoch_loss:.4f}"
        )
        logger.info("=" * 60)

        # Save checkpoint at the end of each epoch (if checkpointing is enabled)
        if checkpoint_dir:
            save_epoch_checkpoint(checkpoint_dir, epoch, vision_trainer_group, text_trainer_group, cfg)

    # Log final summary statistics
    total_steps = num_epochs * num_iterations
    if iteration_times:
        total_time = sum(iteration_times)
        avg_time = total_time / len(iteration_times)
        logger.info("=" * 80)
        logger.info(
            f"Training completed! {num_epochs} epochs, {total_steps} total steps, "
            f"{len(iteration_times)} measured steps (avg {avg_time:.3f}s/step)"
        )
        logger.info("=" * 80)
    else:
        logger.info("Training completed!")

    ray.shutdown()


if __name__ == "__main__":
    main()
