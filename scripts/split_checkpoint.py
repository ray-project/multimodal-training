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
import sys
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForVision2Seq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _add_ms_swift_to_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ms_swift_path = os.path.join(repo_root, "ms-swift")
    if os.path.isdir(ms_swift_path) and ms_swift_path not in sys.path:
        sys.path.insert(0, ms_swift_path)


def _normalize_hf_state_dict(state_dict: Dict[str, torch.Tensor], mapping: Dict[str, str]) -> Dict[str, torch.Tensor]:
    if not mapping:
        return state_dict
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in mapping.items():
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix, 1)
                break
        normalized[new_key] = value
    return normalized


def _resolve_ms_swift_mapping(
    model_name: str, model_type: Optional[str], use_hf: bool
) -> Tuple[str, Dict[str, str], Dict[str, str], type]:
    _add_ms_swift_to_path()
    from swift.model import get_model_info_meta
    from swift.megatron.model.register import get_megatron_model_meta

    model_info, _ = get_model_info_meta(model_name, model_type=model_type, use_hf=use_hf, download_model=False)
    megatron_meta = get_megatron_model_meta(model_info.model_type)
    if megatron_meta is None:
        raise ValueError(f"No ms-swift Megatron metadata found for model_type='{model_info.model_type}'.")
    if megatron_meta.visual_cls is None or not hasattr(megatron_meta.visual_cls, "module_mapping"):
        raise ValueError(f"No visual module mapping found for model_type='{model_info.model_type}'.")
    module_mapping = megatron_meta.visual_cls.module_mapping
    bridge_cls = megatron_meta.bridge_cls
    hf_state_dict_mapping = getattr(bridge_cls, "hf_state_dict_mapping", {}) or {}
    return model_info.model_type, module_mapping, hf_state_dict_mapping, bridge_cls


def _get_text_prefix(bridge_cls: type) -> str:
    hf_layers_prefix = getattr(bridge_cls, "hf_layers_prefix", "model.layers")
    if "." in hf_layers_prefix:
        return hf_layers_prefix.rsplit(".", 1)[0]
    return ""


def _strip_text_prefixes(state_dict: Dict[str, torch.Tensor], text_prefix: str) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if text_prefix and new_key.startswith(f"{text_prefix}."):
            new_key = new_key[len(text_prefix) + 1 :]
        elif new_key.startswith("language_model."):
            new_key = new_key[len("language_model.") :]
        elif new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned[new_key] = value
    return cleaned


def _split_with_mapping(
    state_dict: Dict[str, torch.Tensor],
    module_mapping: Dict[str, str],
    text_prefix: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    vision_state_dict = {}
    text_state_dict = {}
    vision_prefixes = list(module_mapping.keys())

    for key, value in state_dict.items():
        matched_prefix = None
        for prefix in vision_prefixes:
            if key == prefix or key.startswith(f"{prefix}."):
                matched_prefix = prefix
                break
        if matched_prefix is not None:
            new_key = key[len(matched_prefix) :]
            if new_key.startswith("."):
                new_key = new_key[1:]
            vision_state_dict[new_key] = value
        else:
            text_state_dict[key] = value

    text_state_dict = _strip_text_prefixes(text_state_dict, text_prefix)
    return vision_state_dict, text_state_dict


def split_checkpoint(
    model_name: str,
    output_dir: str,
    trust_remote_code: bool = True,
    model_type: Optional[str] = None,
):
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

    # Split state dict using ms-swift canonical mapping
    resolved_model_type = None
    try:
        resolved_model_type, module_mapping, hf_state_dict_mapping, bridge_cls = _resolve_ms_swift_mapping(
            model_name, model_type=model_type, use_hf=True
        )
        text_prefix = _get_text_prefix(bridge_cls)
        normalized_state_dict = _normalize_hf_state_dict(state_dict, hf_state_dict_mapping)
        vision_state_dict, text_state_dict = _split_with_mapping(normalized_state_dict, module_mapping, text_prefix)
        logger.info(f"ms-swift model_type: {resolved_model_type}")
        logger.info(f"ms-swift visual mapping: {module_mapping}")
    except Exception as exc:
        logger.warning(f"Falling back to prefix-based split (ms-swift mapping unavailable): {exc}")
        vision_state_dict = {}
        text_state_dict = {}

        for key, value in state_dict.items():
            # Handle vision model weights
            if key.startswith("model.visual."):
                new_key = key[len("model.visual.") :]
                vision_state_dict[new_key] = value
            elif key.startswith("visual."):
                new_key = key[len("visual.") :]
                vision_state_dict[new_key] = value
            # Handle text model and lm_head weights
            elif key.startswith("model.language_model."):
                new_key = key[len("model.language_model.") :]
                text_state_dict[new_key] = value
            elif key.startswith("language_model."):
                new_key = key[len("language_model.") :]
                text_state_dict[new_key] = value
            elif key.startswith("lm_head."):
                text_state_dict[key] = value
            elif key.startswith("model."):
                new_key = key[len("model.") :]
                text_state_dict[new_key] = value
            else:
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
        "model_type": resolved_model_type or getattr(config, "model_type", None),
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
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Optional ms-swift model_type override (e.g., 'qwen2_5_vl')",
    )

    args = parser.parse_args()

    split_checkpoint(
        model_name=args.model_name,
        output_dir=args.output_dir,
        trust_remote_code=not args.no_trust_remote_code,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
