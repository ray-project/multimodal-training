#!/usr/bin/env python3
"""
Generate configuration files for sweeping various training conditions.
Creates YAML configs from a Jinja2 template for different combinations of:
- Model types and sizes
- Image sizes
- Parallelism strategies
- Attention backends
- Activation checkpointing
"""

import os
import sys
from itertools import product
from pathlib import Path

from jinja2 import Template

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Model settings: (model_type, model_name, warning)
MODEL_SETTINGS = [
    ("qwen2_5_vl", "Qwen/Qwen2.5-VL-7B-Instruct", ""),
    # ("qwen2_5_vl", "Qwen/Qwen2.5-VL-32B-Instruct", ""),
]

# Token configurations: (label, max_pixels, min_pixels)
# Labels indicate vision tokens (before spatial merging) passed to vision model
# All values ensure num_windows % 8 == 0 for sequence parallelism compatibility
TOKEN_CONFIGS = [
    ("1k_tokens", 200704, 200704),  # sqrt=448, 1024 vision tokens (closest to 1k)
    # ("4k_tokens", 802816, 802816),  # sqrt=896, 4096 vision tokens (exact)
    # ("8k_tokens", 1806336, 1806336),  # sqrt=1344, 9216 vision tokens (closest to 8k)
    # ("16k_tokens", 3211264, 3211264),  # sqrt=1792, 16384 vision tokens (closest to 16k)
    # ("32k_tokens", 5017600, 5017600),  # sqrt=2240, 25600 vision tokens (closest to 32k)
    # ("64k_tokens", 12845056, 12845056),  # sqrt=3584, 65536 vision tokens (closest to 64k)
]

# Parallelism strategies to test
# Note: "sequence" parallelism is only valid for vision models, not text models
VISION_PARALLELISM_OPTIONS = ["sequence"]
TEXT_PARALLELISM_OPTIONS = ["tensor"]

# Data parallel size options (1 = no data parallelism, >1 = replicate model across DP groups)
# Vision: total GPUs = vision_dp_size * vision_parallel_size
# Text: total GPUs = text_dp_size * text_parallel_size
VISION_DP_SIZE_OPTIONS = [2]  # Data parallel size for vision model
TEXT_DP_SIZE_OPTIONS = [2]  # Data parallel size for text model

# TP/SP size options per DP replica
# Vision: sequence parallel or tensor parallel size
# Text: tensor parallel size
# NOTE: vision and text parallel_size must be the same (required by training code)
VISION_PARALLEL_SIZE_OPTIONS = [4]  # TP/SP size for vision model
TEXT_PARALLEL_SIZE_OPTIONS = [4]  # TP size for text model (must match VISION_PARALLEL_SIZE_OPTIONS)


# DeepSpeed ZeRO stage
# NOTE: AutoTP (text.parallelism="autotp") is NOT compatible with ZeRO stage 3.
#       When using AutoTP, you must use zero_stage 1 or 2.
VISION_ZERO_STAGE_OPTIONS = [1]
TEXT_ZERO_STAGE_OPTIONS = [3]  # Change to [1] or [2] if using TEXT_PARALLELISM_OPTIONS = ["autotp"]

# Attention backends
# ATTENTION_BACKENDS = ["sdpa", "flash_attention_2"]
ATTENTION_BACKENDS = ["flash_attention_2"]

# Activation checkpointing
ACTIVATION_CHECKPOINTING_OPTIONS = [True]

# Autocast for mixed precision (enable torch.autocast)
AUTOCAST_OPTIONS = [True]

# Data types
DTYPE_OPTIONS = ["bfloat16"]

# Batch sizes to sweep
BATCH_SIZE_OPTIONS = [1, 2, 4, 8]

# Default training parameters
DEFAULT_TRAINING = {
    "learning_rate": 5e-05,
    "num_epochs": 1,  # Number of training epochs
    "num_iterations": 20,  # Number of iterations (batches) per epoch
    "warmup_steps": 10,
    "warmup_ratio": 0.03,  # Proportion of training steps for LR warmup (takes precedence over warmup_steps if > 0)
    "lr_scheduler_type": "cosine",  # Learning rate scheduler type: "cosine", "linear", "constant", etc.
    "weight_decay": 0.01,  # L2 regularization strength (weight decay)
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "collocate": True,  # Whether to collocate vision and text models on same GPUs
    "clip_grad_norm": False,  # Enable gradient clipping
    "max_grad_norm": 1.0,  # Maximum gradient norm for clipping
    "no_checkpoint": True,  # Disable checkpointing (set to True to disable)
    "checkpoint_dir": "/tmp/checkpoints",  # Directory to save/load checkpoints
    "log_interval": 1,  # Log training progress every N iterations
}

# AutoTP-specific default settings
DEFAULT_AUTOTP = {
    "autotp_size": "null",  # null = use parallel_size (set automatically)
    "tp_overlap_comm": "false",  # Enable TP communication overlap
}

# Default data parameters
# Note: Dataset path is resolved at runtime based on priority:
# 1. Command line argument (--data-path, --laion-data-path)
# 2. Environment variable (MSCOCO_DATA_PATH, LAION_POP_DATA_PATH)
# 3. Hardcoded default
DEFAULT_MSCOCO_PATH_FALLBACK = "__PATH_TO_MSCOCO2017__"
DEFAULT_LAION_POP_PATH_FALLBACK = "__PATH_TO_LAION_POP__"


def get_data_config(mscoco_data_path=None, laion_data_path=None):
    """
    Get data configuration with path resolution priority:
    1. Command line argument (mscoco_data_path, laion_data_path parameters)
    2. Environment variable (MSCOCO_DATA_PATH, LAION_POP_DATA_PATH)
    3. Hardcoded default

    Args:
        mscoco_data_path: Optional command line argument for MSCOCO data path
        laion_data_path: Optional command line argument for LAION data path

    Returns:
        Dictionary with data configuration
    """
    # Resolve MSCOCO path with priority: CLI arg > env var > default
    if mscoco_data_path:
        mscoco_base_path = mscoco_data_path
    else:
        mscoco_base_path = os.environ.get("MSCOCO_DATA_PATH", DEFAULT_MSCOCO_PATH_FALLBACK)

    # Resolve LAION path with priority: CLI arg > env var > default
    if laion_data_path:
        laion_base_path = laion_data_path
    else:
        laion_base_path = os.environ.get("LAION_POP_DATA_PATH", DEFAULT_LAION_POP_PATH_FALLBACK)

    # Warn if the configured paths don't exist
    if not os.path.exists(mscoco_base_path):
        import warnings

        warnings.warn(
            f"MSCOCO data path does not exist: {mscoco_base_path}\n"
            f"Set via --mscoco-data-path argument or MSCOCO_DATA_PATH environment variable.",
            UserWarning,
        )

    if not os.path.exists(laion_base_path):
        import warnings

        warnings.warn(
            f"LAION POP data path does not exist: {laion_base_path}\n"
            f"Set via --laion-data-path argument or LAION_POP_DATA_PATH environment variable.",
            UserWarning,
        )

    return {
        "force_fixed_size": True,
        # MSCOCO dataset paths
        "mscoco_data_path": mscoco_base_path,
        "mscoco_train_annotation_path": f"{mscoco_base_path}/annotations/coco2017_train_qwen.json",
        "mscoco_val_annotation_path": f"{mscoco_base_path}/annotations/coco2017_val_qwen.json",
        # LAION POP dataset paths
        "laion_pop_data_path": f"{laion_base_path}/images",
        "laion_pop_train_annotation_path": f"{laion_base_path}/laion_pop_train.jsonl",
        "laion_pop_val_annotation_path": f"{laion_base_path}/laion_pop_val.jsonl",
    }


def generate_config_name(params):
    """Generate a unique config name from parameters."""
    model_name = params["vision_model_name"].replace("/", "_").replace(".", "_")
    vision_par = params["vision_parallelism"]
    text_par = params["text_parallelism"]
    token_label = params["token_label"]
    batch_size = params["batch_size"]
    attn = params["vision_attention_backend"]
    # Check if activation_checkpointing is "true" string or True boolean
    ckpt_val = params["vision_activation_checkpointing"]
    ckpt = "ckpt" if (ckpt_val == "true" or ckpt_val is True) else "nockpt"
    dtype = params["vision_dtype"]

    # Get per-model dp_size and parallel_size
    vision_dp_size = params["vision_dp_size"]
    text_dp_size = params["text_dp_size"]
    vision_parallel_size = params["vision_parallel_size"]
    text_parallel_size = params["text_parallel_size"]

    # Build parallelism suffix with dp and tp/sp sizes
    # Format: vpar{type}{tp_size}[_dp{dp}] for vision, tpar{type}{tp_size}[_dp{dp}] for text
    vision_dp_suffix = f"dp{vision_dp_size}" if vision_dp_size > 1 else ""
    text_dp_suffix = f"dp{text_dp_size}" if text_dp_size > 1 else ""

    # Build vision parallelism string: e.g., "seq2" or "tp2" or "seq2dp2"
    vision_par_short = "seq" if vision_par == "sequence" else "tp"
    vision_par_str = f"{vision_par_short}{vision_parallel_size}{vision_dp_suffix}"

    # Build text parallelism string: e.g., "tp2" or "autotp2" or "tp2dp2"
    text_par_short = "autotp" if text_par == "autotp" else "tp"
    text_par_str = f"{text_par_short}{text_parallel_size}{text_dp_suffix}"

    return f"{model_name}_v{vision_par_str}_t{text_par_str}_{token_label}_bs{batch_size}_{attn}_{ckpt}_{dtype}"


def generate_configs(output_dir, mscoco_data_path=None, laion_data_path=None):
    """
    Generate configuration files for all combinations of parameters.

    Args:
        output_dir: Directory to save generated configs
        mscoco_data_path: Optional path to MSCOCO dataset (overrides env var and default)
        laion_data_path: Optional path to LAION dataset (overrides env var and default)
    """
    output_path = project_root / output_dir
    output_path.mkdir(exist_ok=True, parents=True)

    # Get data configuration with proper priority resolution
    data_config = get_data_config(mscoco_data_path, laion_data_path)

    # Load Jinja2 template
    template_path = project_root / "experiments" / "train.yaml.j2"
    with open(template_path) as f:
        template = Template(f.read())

    configs = []

    # Generate all combinations
    for (
        (model_type, model_name, warning),
        (token_label, max_pixels, min_pixels),
        vision_par,
        text_par,
        batch_size,
        attn_backend,
        act_ckpt,
        autocast,
        dtype,
        vision_zero_stage,
        text_zero_stage,
        vision_dp_size,
        text_dp_size,
        vision_parallel_size,
        text_parallel_size,
    ) in product(
        MODEL_SETTINGS,
        TOKEN_CONFIGS,
        VISION_PARALLELISM_OPTIONS,
        TEXT_PARALLELISM_OPTIONS,
        BATCH_SIZE_OPTIONS,
        ATTENTION_BACKENDS,
        ACTIVATION_CHECKPOINTING_OPTIONS,
        AUTOCAST_OPTIONS,
        DTYPE_OPTIONS,
        VISION_ZERO_STAGE_OPTIONS,
        TEXT_ZERO_STAGE_OPTIONS,
        VISION_DP_SIZE_OPTIONS,
        TEXT_DP_SIZE_OPTIONS,
        VISION_PARALLEL_SIZE_OPTIONS,
        TEXT_PARALLEL_SIZE_OPTIONS,
    ):

        # Validate: vision and text parallel_size must be the same (required by training code)
        if vision_parallel_size != text_parallel_size:
            raise ValueError(
                f"vision_parallel_size ({vision_parallel_size}) must equal text_parallel_size ({text_parallel_size}). "
                "The training code requires both models to use the same TP/SP size."
            )

        # Determine reduce_bucket_size based on token configuration
        # For 64k_tokens (12845056 pixels), use 100M elements; otherwise use 500M
        reduce_bucket_size = 100000000 if max_pixels > 2_000_000 else 500000000

        # Build parameter dict
        params = {
            # Vision model
            "vision_model_type": model_type,
            "vision_model_name": model_name,
            "vision_parallelism": vision_par,
            "vision_dtype": dtype,
            "vision_attention_backend": attn_backend,
            "vision_activation_checkpointing": str(act_ckpt).lower(),
            "vision_autocast": str(autocast).lower(),
            "vision_zero_stage": vision_zero_stage,
            "vision_dp_size": vision_dp_size,  # Data parallel size for vision
            "vision_parallel_size": vision_parallel_size,  # TP/SP size for vision
            # Text model (same as vision for now)
            "text_model_type": model_type,
            "text_model_name": model_name,
            "text_parallelism": text_par,
            "text_dtype": dtype,
            "text_attention_backend": attn_backend,
            "text_activation_checkpointing": str(act_ckpt).lower(),
            "text_autocast": str(autocast).lower(),
            "text_zero_stage": text_zero_stage,
            "text_autotp_size": DEFAULT_AUTOTP["autotp_size"],
            "text_tp_overlap_comm": DEFAULT_AUTOTP["tp_overlap_comm"],
            "text_dp_size": text_dp_size,  # Data parallel size for text
            "text_parallel_size": text_parallel_size,  # TP size for text
            # Training
            "batch_size": batch_size,
            "learning_rate": DEFAULT_TRAINING["learning_rate"],
            "num_epochs": DEFAULT_TRAINING["num_epochs"],
            "num_iterations": DEFAULT_TRAINING["num_iterations"],
            "warmup_steps": DEFAULT_TRAINING["warmup_steps"],
            "warmup_ratio": DEFAULT_TRAINING["warmup_ratio"],
            "lr_scheduler_type": DEFAULT_TRAINING["lr_scheduler_type"],
            "weight_decay": DEFAULT_TRAINING["weight_decay"],
            "gradient_accumulation_steps": DEFAULT_TRAINING["gradient_accumulation_steps"],
            "seed": DEFAULT_TRAINING["seed"],
            "collocate": str(DEFAULT_TRAINING["collocate"]).lower(),
            "clip_grad_norm": str(DEFAULT_TRAINING["clip_grad_norm"]).lower(),
            "max_grad_norm": DEFAULT_TRAINING["max_grad_norm"],
            "no_checkpoint": str(DEFAULT_TRAINING["no_checkpoint"]).lower(),
            "checkpoint_dir": DEFAULT_TRAINING["checkpoint_dir"],
            "log_interval": DEFAULT_TRAINING["log_interval"],
            # Data
            "force_fixed_size": str(data_config["force_fixed_size"]).lower(),
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "token_label": token_label,
            "mscoco_data_path": data_config["mscoco_data_path"],
            "mscoco_train_annotation_path": data_config["mscoco_train_annotation_path"],
            "mscoco_val_annotation_path": data_config["mscoco_val_annotation_path"],
            "laion_pop_data_path": data_config["laion_pop_data_path"],
            "laion_pop_train_annotation_path": data_config["laion_pop_train_annotation_path"],
            "laion_pop_val_annotation_path": data_config["laion_pop_val_annotation_path"],
            # DeepSpeed
            "reduce_bucket_size": reduce_bucket_size,
            "pretrained_checkpoint_dir": "/tmp/checkpoints/pretrained/qwen2.5-vl-7b",
        }

        # Generate config name and file
        config_name = generate_config_name(params)
        config_file = output_path / f"{config_name}.yaml"

        # Render template
        config_content = template.render(**params)

        # Write config file
        with open(config_file, "w") as f:
            f.write(config_content)

        configs.append(
            {
                "name": config_name,
                "file": str(config_file),
                "parallel_size": vision_parallel_size,  # TP/SP size (vision and text must be equal)
                "warning": warning,
            }
        )

        if warning:
            print(f"Warning for {config_name}: {warning}")

    print(f"Generated {len(configs)} configuration files in {output_path}")

    # Write manifest file
    manifest_path = output_path / "manifest.txt"
    with open(manifest_path, "w") as f:
        for cfg in configs:
            f.write(f"{cfg['name']}\t{cfg['file']}\t{cfg['parallel_size']}\t{cfg['warning']}\n")

    print(f"Manifest written to {manifest_path}")

    # Generate run_sweep.sh from template
    sweep_template_path = project_root / "experiments" / "run_sweep.sh.j2"
    if sweep_template_path.exists():
        from datetime import datetime

        with open(sweep_template_path) as f:
            sweep_template = Template(f.read())

        sweep_content = sweep_template.render(
            configs=configs,
            num_configs=len(configs),
            config_dir=output_dir,
            generation_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        sweep_script_path = project_root / "experiments" / "run_sweep.sh"
        with open(sweep_script_path, "w") as f:
            f.write(sweep_content)

        # Make the script executable
        sweep_script_path.chmod(0o755)

        print(f"Generated run_sweep.sh with {len(configs)} configs")
    else:
        print(f"Warning: run_sweep.sh.j2 template not found at {sweep_template_path}")

    return configs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate config files for sweeping experiments")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs",
        help="Output directory for generated configs (default: configs)",
    )
    parser.add_argument(
        "--mscoco-data-path",
        type=str,
        default=None,
        help="Path to MSCOCO dataset directory (overrides MSCOCO_DATA_PATH env var)",
    )
    parser.add_argument(
        "--laion-data-path",
        type=str,
        default=None,
        help="Path to LAION POP dataset directory (overrides LAION_POP_DATA_PATH env var)",
    )

    args = parser.parse_args()

    configs = generate_configs(
        args.output_dir,
        mscoco_data_path=args.mscoco_data_path,
        laion_data_path=args.laion_data_path,
    )
    print(f"\nGenerated {len(configs)} configs")
