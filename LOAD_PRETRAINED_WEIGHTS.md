# Loading Pretrained Weights

This document explains how to work with checkpoints when training vision-language models with disaggregated vision and text components.

## Overview

The training framework splits VLMs into separate vision and text components that can use different parallelism strategies. This requires special handling for checkpoints:

1. **Splitting HuggingFace checkpoints**: Convert a unified HF checkpoint into separate vision and text weights
2. **Training from pretrained weights**: Initialize training from split checkpoints
3. **Merging trained checkpoints**: Combine separately saved vision and text checkpoints back into a unified HF format

## Scripts

### 1. Split HuggingFace Checkpoint

**Script**: `scripts/split_checkpoint.py`

Loads a HuggingFace Qwen2.5-VL checkpoint and splits it into separate vision and text model weights (including the untied `lm_head`).

**Usage**:
```bash
python scripts/split_checkpoint.py \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --output-dir checkpoints/split/qwen2.5-vl-7b
```

**Arguments**:
- `--model-name`: HuggingFace model name or path (e.g., `Qwen/Qwen2.5-VL-7B-Instruct`)
- `--output-dir`: Directory to save the split checkpoints
- `--no-trust-remote-code`: Don't trust remote code when loading (optional)

**Output**:
```
checkpoints/split/qwen2.5-vl-7b/
├── vision_model.pt           # Vision encoder weights
├── text_model.pt             # Text decoder weights (plus lm_head)
├── config.json               # Full model config
└── split_metadata.json       # Split metadata
```

**Example**:
```bash
# Split Qwen2.5-VL-7B-Instruct (regenerates vision, text, and lm_head weights)
python scripts/split_checkpoint.py \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --output-dir checkpoints/pretrained/qwen2.5-vl-7b

# Split Qwen2.5-VL-3B-Instruct
python scripts/split_checkpoint.py \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --output-dir checkpoints/pretrained/qwen2.5-vl-3b
```

> **Note:** Existing split checkpoints created before this update are missing the `lm_head` weights. Re-run `scripts/split_checkpoint.py` using the commands above to regenerate the split directory before training.

### 2. Training from Pretrained Checkpoints

To initialize training from pretrained split checkpoints, specify the `pretrained_checkpoint_dir` in your training config:

**Method 1: Using config file**

Create or modify a config file (e.g., `configs/my_config.yaml`):

```yaml
vision:
  model_type: "qwen2_5_vl"
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  parallelism: "tensor"
  pretrained_checkpoint_dir: "checkpoints/pretrained/qwen2.5-vl-7b"  # Add this line

text:
  model_type: "qwen2_5_vl"
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  parallelism: "tensor"
  pretrained_checkpoint_dir: "checkpoints/pretrained/qwen2.5-vl-7b"  # Add this line

# ... rest of config
```

**Method 2: Using config generation**

Update `scripts/generate_configs.py` to include the `pretrained_checkpoint_dir` parameter:

```python
config_params = {
    # ... existing params
    "pretrained_checkpoint_dir": "checkpoints/pretrained/qwen2.5-vl-7b",
}
```

Then run:
```bash
python scripts/generate_configs.py --output-dir configs
```

**Training**:
```bash
# Run training with pretrained weights
CONFIG_NAME=my_config ./run.sh

# Or directly with python
python -m python.train_ray \
    --config-path=../configs \
    --config-name=my_config
```

The training script will automatically:
1. Load the vision model weights from `{pretrained_checkpoint_dir}/vision_model.pt`
2. Load the text model weights **and** untied `lm_head` from `{pretrained_checkpoint_dir}/text_model.pt`
3. Continue training from these pretrained weights

### 3. Merge Trained Checkpoints

**Script**: `scripts/merge_checkpoint.py`

Merges separately saved vision and text checkpoints (created during training) back into a unified HuggingFace format.

**Usage**:
```bash
python scripts/merge_checkpoint.py \
    --checkpoint-dir checkpoints/experiment/epoch_5 \
    --output-dir checkpoints/merged/epoch_5 \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --parallel-size 2
```

**Arguments**:
- `--checkpoint-dir`: Base checkpoint directory containing `vision/` and `text/` subdirectories
- `--output-dir`: Directory to save the merged checkpoint
- `--model-name`: Original HuggingFace model name for config reference
- `--parallel-size`: Number of parallel ranks (for merging sharded weights)
- `--no-trust-remote-code`: Don't trust remote code when loading config (optional)

**Input Structure**:
```
checkpoints/experiment/epoch_5/
├── vision/
│   ├── rank_0.pt
│   ├── rank_1.pt
│   └── ...
├── text/
│   ├── rank_0.pt
│   ├── rank_1.pt
│   └── ...
└── metadata.json
```

**Output**:
```
checkpoints/merged/epoch_5/
├── pytorch_model.bin         # Merged full model weights
├── config.json               # Full model config
└── merge_metadata.json       # Merge metadata
```

**Example**:
```bash
# Merge checkpoint from epoch 10 (with 4 parallel ranks)
python scripts/merge_checkpoint.py \
    --checkpoint-dir checkpoints/my_experiment_20250101/epoch_10 \
    --output-dir checkpoints/merged/my_experiment_epoch_10 \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --parallel-size 4

# Load the merged checkpoint in Python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "checkpoints/merged/my_experiment_epoch_10",
    trust_remote_code=True
)
```

## Complete Workflow Example

Here's a complete example of the checkpoint workflow:

```bash
# 1. Split the pretrained HuggingFace checkpoint
python scripts/split_checkpoint.py \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --output-dir checkpoints/pretrained/qwen2.5-vl-7b

# 2. Train from the split checkpoint
# Edit your config to include:
#   vision:
#     pretrained_checkpoint_dir: "checkpoints/pretrained/qwen2.5-vl-7b"
#   text:
#     pretrained_checkpoint_dir: "checkpoints/pretrained/qwen2.5-vl-7b"

CONFIG_NAME=my_finetune_config ./run.sh

# Training will save separate vision and text checkpoints to:
# checkpoints/my_experiment/epoch_N/vision/rank_*.pt
# checkpoints/my_experiment/epoch_N/text/rank_*.pt

# 3. After training, merge the final checkpoint
python scripts/merge_checkpoint.py \
    --checkpoint-dir checkpoints/my_experiment/epoch_5 \
    --output-dir checkpoints/final_model \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --parallel-size 2

# 4. Use the merged checkpoint with HuggingFace
python -c "
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('checkpoints/final_model', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
print('Model loaded successfully!')
"
```
