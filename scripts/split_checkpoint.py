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
from transformers import AutoConfig, AutoModelForVision2Seq

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

    # Load the full model with all weights
    # Use AutoModelForVision2Seq to include the lm_head weights (needed for untied embeddings)
    logger.info("Loading full model (this may take a while)...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
    )

    logger.info("Model loaded successfully")

    # Get model state dict
    state_dict = model.state_dict()
    logger.info(f"Total parameters in model: {len(state_dict)} tensors")

    # Split state dict into vision and text components
    # Keys from AutoModelForVision2Seq:
    #   - model.visual.* -> vision model
    #   - model.language_model.* -> text model
    #   - lm_head.* -> text model (separate from embeddings when tie_word_embeddings=False)
    #   - visual.* -> vision model (for some model versions)
    #   - language_model.* -> text model (for some model versions)
    vision_state_dict = {}
    text_state_dict = {}

    for key, value in state_dict.items():
        # Handle vision model weights
        if key.startswith("model.visual."):
            # Vision model weights - remove "model.visual." prefix
            new_key = key[len("model.visual.") :]
            vision_state_dict[new_key] = value
        elif key.startswith("visual."):
            # Alternative format: remove "visual." prefix
            new_key = key[len("visual.") :]
            vision_state_dict[new_key] = value
        # Handle text model and lm_head weights
        elif key.startswith("model.language_model."):
            # Text model weights - remove "model.language_model." prefix
            new_key = key[len("model.language_model.") :]
            text_state_dict[new_key] = value
        elif key.startswith("language_model."):
            # Alternative format: remove "language_model." prefix
            new_key = key[len("language_model.") :]
            text_state_dict[new_key] = value
        elif key.startswith("lm_head."):
            # lm_head weights - keep as is (important for untied embeddings)
            text_state_dict[key] = value
        elif key.startswith("model."):
            # Other model. prefixed keys go to text
            new_key = key[len("model.") :]
            text_state_dict[new_key] = value
        else:
            # Other keys go to text
            text_state_dict[key] = value

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
