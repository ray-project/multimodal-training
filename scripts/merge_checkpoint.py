#!/usr/bin/env python3
"""
Script to merge separately saved vision and text checkpoints back into a full HuggingFace checkpoint.

This merges the disaggregated vision and text model checkpoints created during training
back into a single unified checkpoint that can be loaded with HuggingFace transformers.

Usage:
    python scripts/merge_checkpoint.py \
        --checkpoint-dir checkpoints/experiment/epoch_5 \
        --output-dir checkpoints/merged/epoch_5 \
        --model-name Qwen/Qwen2.5-VL-7B-Instruct \
        --parallel-size 2
"""

import argparse
import json
import logging
import os

import torch
from transformers import AutoConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_merge_sharded_weights(checkpoint_dir: str, num_ranks: int, component: str):
    """
    Load and merge sharded weights from multiple ranks.

    For tensor parallelism, weights are sharded across ranks and need to be merged.
    For other parallelism strategies, we can just use rank 0's weights.

    Args:
        checkpoint_dir: Directory containing the checkpoints (e.g., epoch_5/vision)
        num_ranks: Number of ranks (parallel_size)
        component: "vision" or "text"

    Returns:
        Merged state dict
    """
    logger.info(f"Loading {component} checkpoints from {checkpoint_dir}")

    # Load all rank checkpoints
    rank_state_dicts = []
    for rank in range(num_ranks):
        checkpoint_path = os.path.join(checkpoint_dir, f"rank_{rank}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        rank_state_dicts.append(checkpoint["model"])
        logger.info(f"Loaded {component} rank {rank}: {len(checkpoint['model'])} tensors")

    # If only one rank, return it directly
    if num_ranks == 1:
        return rank_state_dicts[0]

    # Merge sharded weights
    # For tensor parallelism, some layers are sharded along specific dimensions
    merged_state_dict = {}

    # Get all unique keys across ranks
    all_keys = set()
    for state_dict in rank_state_dicts:
        all_keys.update(state_dict.keys())

    for key in all_keys:
        tensors = [state_dict[key] for state_dict in rank_state_dicts if key in state_dict]

        if len(tensors) == 0:
            continue
        elif len(tensors) == 1:
            # Only one rank has this parameter (shouldn't happen if all ranks have same architecture)
            # Convert DTensor to local tensor if needed
            tensor = tensors[0]
            if hasattr(tensor, "to_local"):
                tensor = tensor.to_local()
            merged_state_dict[key] = tensor
            continue

        # Convert DTensors to local tensors for comparison and concatenation
        local_tensors = []
        for t in tensors:
            if hasattr(t, "to_local"):
                local_tensors.append(t.to_local())
            else:
                local_tensors.append(t)
        tensors = local_tensors

        # Check if tensors are identical across ranks (replicated parameters)
        all_same = all(torch.equal(tensors[0], t) for t in tensors[1:])
        if all_same:
            # Replicated parameter - just use first rank's copy
            merged_state_dict[key] = tensors[0]
            continue

        # Tensors differ - they are sharded, need to concatenate
        # Determine which dimension to concatenate along based on the layer type
        if "embed_tokens" in key or "tok_embeddings" in key:
            # Vocabulary embeddings are sharded along vocab dimension (dim 0)
            merged_state_dict[key] = torch.cat(tensors, dim=0)
            logger.debug(f"Merged {key} along dim 0 (vocab): {tensors[0].shape} -> {merged_state_dict[key].shape}")
        elif "lm_head" in key:
            # LM head is tied to embeddings, also sharded along vocab dimension
            merged_state_dict[key] = torch.cat(tensors, dim=0)
            logger.debug(f"Merged {key} along dim 0 (vocab): {tensors[0].shape} -> {merged_state_dict[key].shape}")
        elif any(name in key for name in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
            # Column-wise parallel layers: sharded along output dimension (dim 0 for weight)
            merged_state_dict[key] = torch.cat(tensors, dim=0)
            logger.debug(f"Merged {key} along dim 0 (output): {tensors[0].shape} -> {merged_state_dict[key].shape}")
        elif any(name in key for name in ["o_proj", "proj", "down_proj"]):
            # Row-wise parallel layers: sharded along input dimension (dim 1 for weight)
            merged_state_dict[key] = torch.cat(tensors, dim=1)
            logger.debug(f"Merged {key} along dim 1 (input): {tensors[0].shape} -> {merged_state_dict[key].shape}")
        else:
            # Default: assume replicated if shapes match, otherwise try to concatenate
            if all(t.shape == tensors[0].shape for t in tensors):
                # Same shape - check if actually identical
                if not all_same:
                    logger.warning(
                        f"Parameter {key} has same shape across ranks but different values. "
                        f"Averaging across ranks (this may not be correct!)"
                    )
                    merged_state_dict[key] = torch.stack(tensors).mean(dim=0)
                else:
                    merged_state_dict[key] = tensors[0]
            else:
                logger.warning(
                    f"Parameter {key} has different shapes across ranks. " f"Attempting to concatenate along dim 0"
                )
                merged_state_dict[key] = torch.cat(tensors, dim=0)

    logger.info(f"Merged {component} checkpoint: {len(merged_state_dict)} tensors")
    return merged_state_dict


def merge_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    model_name: str,
    parallel_size: int = 1,
    trust_remote_code: bool = True,
):
    """
    Merge separately saved vision and text checkpoints back into a full HuggingFace checkpoint.

    Args:
        checkpoint_dir: Base checkpoint directory containing vision/ and text/ subdirectories
        output_dir: Directory to save the merged checkpoint
        model_name: Original HuggingFace model name for config reference
        parallel_size: Number of parallel ranks (for merging sharded weights)
        trust_remote_code: Whether to trust remote code when loading config

    Returns:
        Path to merged checkpoint
    """
    logger.info(f"Merging checkpoint from {checkpoint_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parallel size: {parallel_size}")

    # Load vision checkpoints
    vision_dir = os.path.join(checkpoint_dir, "vision")
    vision_state_dict = load_and_merge_sharded_weights(vision_dir, parallel_size, "vision")

    # Load text checkpoints
    text_dir = os.path.join(checkpoint_dir, "text")
    text_state_dict = load_and_merge_sharded_weights(text_dir, parallel_size, "text")

    # Combine into full model state dict
    # Vision weights need "visual." prefix to match HuggingFace format
    # Text weights need "language_model." prefix (or "model." for older checkpoints)
    full_state_dict = {}

    for key, value in vision_state_dict.items():
        full_state_dict[f"visual.{key}"] = value

    for key, value in text_state_dict.items():
        # Keep lm_head weights without extra prefix to match HuggingFace format
        if key.startswith("lm_head."):
            full_state_dict[key] = value
            continue
        # Add language_model. prefix if not already present
        if not key.startswith("language_model.") and not key.startswith("model."):
            full_state_dict[f"language_model.{key}"] = value
        else:
            full_state_dict[key] = value

    logger.info(f"Full model state dict: {len(full_state_dict)} tensors")

    # Load model config
    logger.info(f"Loading config from {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save merged checkpoint in PyTorch format
    checkpoint_path = os.path.join(output_dir, "pytorch_model.bin")
    logger.info(f"Saving merged checkpoint to {checkpoint_path}")
    torch.save(full_state_dict, checkpoint_path)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    logger.info(f"Saving config to {config_path}")
    config.save_pretrained(output_dir)

    # Save metadata
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    merge_metadata = {
        "source_checkpoint": checkpoint_dir,
        "merged_checkpoint": checkpoint_path,
        "parallel_size": parallel_size,
        "num_vision_params": len(vision_state_dict),
        "num_text_params": len(text_state_dict),
        "num_total_params": len(full_state_dict),
        "original_metadata": metadata,
    }

    merge_metadata_path = os.path.join(output_dir, "merge_metadata.json")
    with open(merge_metadata_path, "w") as f:
        json.dump(merge_metadata, f, indent=2)

    logger.info("Merge complete!")
    logger.info(f"  Merged checkpoint: {checkpoint_path}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Metadata: {merge_metadata_path}")

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge vision and text checkpoints back into a full HuggingFace checkpoint"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Base checkpoint directory containing vision/ and text/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the merged checkpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Original HuggingFace model name for config reference",
    )
    parser.add_argument(
        "--parallel-size",
        type=int,
        default=1,
        help="Number of parallel ranks (default: 1)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Do not trust remote code when loading config",
    )

    args = parser.parse_args()

    merge_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        parallel_size=args.parallel_size,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    main()
