#!/usr/bin/env python3
"""
Script to split a HuggingFace Qwen2.5-VL checkpoint into separate vision and text model weights.

This is necessary because the training framework disaggregates the model into vision and text
components that can use different parallelism strategies.

Usage:
    python scripts/split_checkpoint.py \
        --model-name Qwen/Qwen2.5-VL-7B-Instruct \
        --output-dir checkpoints/split/qwen2.5-vl-7b
"""

import argparse
import logging
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Qwen2.5-VL is not yet wired into AutoModelForCausalLM, so import explicitly
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - handled at runtime
    Qwen2_5_VLForConditionalGeneration = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def split_checkpoint(model_name: str, output_dir: str, trust_remote_code: bool = True):
    """
    Load a Qwen2.5-VL checkpoint from HuggingFace and split it into vision and text components.

    Args:
        model_name: HuggingFace model name or path (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
        output_dir: Directory to save the split checkpoints
        trust_remote_code: Whether to trust remote code when loading the model

    Returns:
        Paths to saved vision and text checkpoints
    """
    logger.info(f"Loading model from HuggingFace: {model_name}")

    # Load model config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    logger.info(f"Model config loaded: {config.model_type}")

    # Load the full causal LM model so the lm_head weights are included
    logger.info("Loading full model (this may take a while)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
        )
    except ValueError:
        if Qwen2_5_VLForConditionalGeneration is None:
            raise
        logger.info("Falling back to Qwen2_5_VLForConditionalGeneration for loading.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )

    logger.info("Model loaded successfully")

    # Get model state dict
    state_dict = model.state_dict()
    logger.info(f"Total parameters in model: {len(state_dict)} tensors")

    # Split state dict into vision and text components
    vision_state_dict = {}
    text_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("visual.") or key.startswith("model.visual."):
            # Vision model weights - remove the visual prefix (with or without model.)
            if key.startswith("model.visual."):
                new_key = key[len("model.visual.") :]
            else:
                new_key = key[len("visual.") :]
            vision_state_dict[new_key] = value
        else:
            # Text model weights - remove common prefixes for cleaner keys
            new_key = key
            # Remove 'language_model.' prefix if present
            if new_key.startswith("language_model."):
                new_key = new_key[len("language_model.") :]
            # Remove 'model.' prefix if present (and not already removed above)
            elif new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            text_state_dict[new_key] = value

    logger.info(f"Vision model: {len(vision_state_dict)} tensors")
    logger.info(f"Text model: {len(text_state_dict)} tensors")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save vision checkpoint
    vision_path = os.path.join(output_dir, "vision_model.pt")
    logger.info(f"Saving vision checkpoint to {vision_path}")
    torch.save(
        {
            "model": vision_state_dict,
            "config": config.vision_config.to_dict() if hasattr(config, "vision_config") else {},
        },
        vision_path,
    )

    # Save text checkpoint
    text_path = os.path.join(output_dir, "text_model.pt")
    logger.info(f"Saving text checkpoint to {text_path}")
    torch.save(
        {
            "model": text_state_dict,
            "config": config.text_config.to_dict() if hasattr(config, "text_config") else {},
        },
        text_path,
    )

    # Save full config for reference
    config_path = os.path.join(output_dir, "config.json")
    logger.info(f"Saving full config to {config_path}")
    config.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "source_model": model_name,
        "vision_checkpoint": vision_path,
        "text_checkpoint": text_path,
        "vision_num_params": len(vision_state_dict),
        "text_num_params": len(text_state_dict),
    }

    metadata_path = os.path.join(output_dir, "split_metadata.json")
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Split complete!")
    logger.info(f"  Vision checkpoint: {vision_path}")
    logger.info(f"  Text checkpoint: {text_path}")
    logger.info(f"  Metadata: {metadata_path}")

    return vision_path, text_path


def main():
    parser = argparse.ArgumentParser(
        description="Split a HuggingFace Qwen2.5-VL checkpoint into vision and text components"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the split checkpoints",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Do not trust remote code when loading the model",
    )

    args = parser.parse_args()

    split_checkpoint(
        model_name=args.model_name,
        output_dir=args.output_dir,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    main()
