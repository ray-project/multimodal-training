# Hybrid Parallelism Training Experiments

This directory contains scripts for running systematic experiments with different parallelism strategies for multimodal model training.

## Overview

The experiment framework allows you to sweep across various training configurations including:
- Different model sizes (7B, 32B)
- Various image sizes (1k to 64k vision tokens)
- Multiple parallelism strategies (sequence, tensor, AutoTP)
- Different batch sizes
- Data parallel configurations (dp_size)

## Directory Structure

```
experiments/
├── EXPERIMENTS.md         # This file
├── generate_configs.py    # Script to generate config files and run_sweep.sh
├── run_sweep.sh          # Generated script to run training sweeps
├── run_sweep.sh.j2       # Jinja2 template for sweep script generation
└── train.yaml.j2         # Jinja2 template for config generation
```

## Prerequisites

1. **Set up data paths**: Configure environment variables or use command-line arguments for dataset paths:
   ```bash
   # Option 1: Environment variables
   export MSCOCO_DATA_PATH=/path/to/mscoco2017
   export LAION_POP_DATA_PATH=/path/to/laion_pop

   # Option 2: Pass as arguments to generate_configs.py (see below)
   ```

2. **Ensure Ray is running**: The training uses Ray for distributed execution. Make sure Ray is initialized.

3. **Install dependencies**: Ensure all required packages are installed (see main README.md)

## Usage

### Step 1: Generate Configuration Files

Run the config generation script from the **project root directory**:

```bash
# Generate configs with default paths (uses environment variables or prompts for paths)
python experiments/generate_configs.py

# Or specify data paths explicitly
python experiments/generate_configs.py \
    --mscoco-data-path /path/to/mscoco2017 \
    --laion-data-path /path/to/laion_pop

# Specify custom output directory (default: configs/)
python experiments/generate_configs.py --output-dir my_configs
```

This will generate:
- YAML config files in `configs/` directory
- `run_sweep.sh` script with a pre-populated list of all generated configs

#### Configuration Options

The script generates configs by combining these parameters:

**Model Settings:**
- Qwen2.5-VL-7B-Instruct (TP=4)
- Qwen2.5-VL-32B-Instruct (TP=8)

**Vision Token Configurations:**
- 1k tokens (448x448 px)
- 4k tokens (896x896 px)
- 8k tokens (1344x1344 px)
- 16k tokens (1792x1792 px)
- 32k tokens (2240x2240 px)
- 64k tokens (3584x3584 px)

**Parallelism Strategies:**
- Vision: sequence, tensor
- Text: tensor, autotp (DeepSpeed AutoTP)

**Data Parallelism:**
- dp_size: 1 (default, no data parallelism)
- Total GPUs = dp_size × parallel_size

**Batch Sizes:** 1, 2, 4, 8

**Other Settings:**
- Attention backend: flash_attention_2
- Activation checkpointing: enabled
- Mixed precision: bfloat16 with autocast
- Training iterations: 20 per epoch

### Step 2: Run Training Sweep

Run the sweep script from the **project root directory**:

```bash
# Run all generated configs
bash experiments/run_sweep.sh
```

The script will:
- Iterate through all config files in `configs/` directory
- Run training for each configuration
- Save detailed logs to `logs/sweep_<timestamp>/`
- Extract and report timing metrics
- Generate a summary report

#### Sweep Script Features

- **Auto-discovery**: Automatically finds all `.yaml` files in `configs/`
- **Detailed logging**: Each run's output is saved to a separate log file
- **Progress tracking**: Color-coded status messages (green=success, red=failed, yellow=skipped)
- **Timing metrics**: Extracts average iteration time and throughput
- **Summary report**: Generates `summary.txt` with results from all runs
- **Error handling**: Continues running even if some configs fail (configurable)

#### Selective Execution

To run only specific configs, edit `run_sweep.sh` and modify the `CONFIGS` array:

```bash
# Comment out the auto-discovery section
# CONFIGS=()
# for config_file in configs/*.yaml; do
#     config_name=$(basename "$config_file" .yaml)
#     CONFIGS+=("${config_name}:0")
# done

# Manually specify configs to run (format: "config_name:skip_flag")
# skip_flag: 0=run, 1=skip
CONFIGS=(
    "Qwen_Qwen2_5-VL-7B-Instruct_vparsequence_tpartensor_4k_tokens_bs4_flash_attention_2_ckpt_bfloat16:0"
    "Qwen_Qwen2_5-VL-7B-Instruct_vpartensor_tpartensor_4k_tokens_bs4_flash_attention_2_ckpt_bfloat16:0"
)
```

## Output Structure

After running experiments, you'll have:

```
multimodal-training/
├── configs/
│   ├── <config_1>.yaml
│   ├── <config_2>.yaml
│   ├── ...
│   └── manifest.txt
└── logs/
    └── sweep_<timestamp>/
        ├── <config_1>.log
        ├── <config_2>.log
        ├── ...
        └── summary.txt
```

### Summary Report

The `summary.txt` file provides:
- Overall sweep statistics (total runs, success/failure counts)
- Per-configuration timing metrics in table format
- Easy comparison of different configurations

Example summary output:
```
Configuration                                                        | Status     | Avg Iter Time (s)    | Iters/Sec
-----------------------------------------------------------------------------------------
model_name_vparseq_tpartensor_4k_tokens_bs4...                      | SUCCESS    | 2.345                | 0.4264
model_name_vpartensor_tpartensor_4k_tokens_bs4...                   | SUCCESS    | 2.567                | 0.3896
```
