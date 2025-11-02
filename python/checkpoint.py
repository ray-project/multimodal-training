"""
Checkpointing utilities for saving and loading model states during training.
"""

import json
import logging
import os
from datetime import datetime

import ray

logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: str) -> int | None:
    """
    Find the latest epoch checkpoint in the checkpoint directory.

    Args:
        checkpoint_dir: Base directory for checkpoints

    Returns:
        Latest epoch number, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Find all epoch directories
    epoch_dirs = []
    for entry in os.listdir(checkpoint_dir):
        if entry.startswith("epoch_"):
            try:
                epoch_num = int(entry.split("_")[1])
                epoch_dirs.append(epoch_num)
            except (IndexError, ValueError):
                continue

    if not epoch_dirs:
        return None

    return max(epoch_dirs)


def save_epoch_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    vision_group,
    text_group,
    config=None,
):
    """
    Save checkpoint for all actors at the end of an epoch.

    Args:
        checkpoint_dir: Base directory for checkpoints
        epoch: Current epoch number
        vision_group: Vision actor group
        text_group: Text actor group
        config: Optional config dict for metadata
    """
    logger.info(f"Saving checkpoint for epoch {epoch}...")

    # Execute save_checkpoint on all actors
    vision_refs = vision_group.execute_all_async("save_checkpoint", checkpoint_dir, epoch)
    text_refs = text_group.execute_all_async("save_checkpoint", checkpoint_dir, epoch)

    # Wait for all saves to complete
    vision_paths = ray.get(vision_refs)
    text_paths = ray.get(text_refs)

    logger.info(f"All actors saved checkpoints for epoch {epoch}")

    # Save metadata (only from main process)
    metadata = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "vision_checkpoints": vision_paths,
        "text_checkpoints": text_paths,
    }

    if config is not None:
        metadata["config_summary"] = {
            "vision_parallelism": config.vision.parallelism if "vision" in config else None,
            "text_parallelism": config.text.parallelism if "text" in config else None,
            "batch_size": config.training.batch_size if "training" in config else None,
            "parallel_size": config.training.parallel_size if "training" in config else None,
        }

    epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    # Ensure the directory exists before writing metadata (it should exist from actor saves, but be defensive)
    os.makedirs(epoch_dir, exist_ok=True)
    metadata_path = os.path.join(epoch_dir, "metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved checkpoint metadata to {metadata_path}")


def load_epoch_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    vision_group,
    text_group,
):
    """
    Load checkpoint for all actors from a specific epoch.

    Args:
        checkpoint_dir: Base directory for checkpoints
        epoch: Epoch number to load from
        vision_group: Vision actor group
        text_group: Text actor group

    Returns:
        True if all actors loaded successfully, False otherwise
    """
    logger.info(f"Loading checkpoint from epoch {epoch}...")

    # Execute load_checkpoint on all actors
    vision_refs = vision_group.execute_all_async("load_checkpoint", checkpoint_dir, epoch)
    text_refs = text_group.execute_all_async("load_checkpoint", checkpoint_dir, epoch)

    # Wait for all loads to complete
    vision_results = ray.get(vision_refs)
    text_results = ray.get(text_refs)

    # Check if all actors loaded successfully
    all_success = all(vision_results) and all(text_results)

    if all_success:
        logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
    else:
        logger.error(f"Failed to load checkpoint from epoch {epoch}")

    return all_success
