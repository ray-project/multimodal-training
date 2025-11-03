#!/bin/bash

# Sweep script to run training with various Ray hybrid parallelism configurations
# This script will iterate through different configs in configs/ directory
# and save logs for each run to the logs/ directory

# set -e  # Exit on error (can be commented out if you want to continue after failures)

# Create logs directory if it doesn't exist
LOGS_DIR="./logs"
mkdir -p ${LOGS_DIR}

# Get timestamp for this sweep run
SWEEP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG_DIR="${LOGS_DIR}/sweep_${SWEEP_TIMESTAMP}"
mkdir -p ${SWEEP_LOG_DIR}

echo "=========================================="
echo "Starting training sweep at $(date)"
echo "Logs will be saved to: ${SWEEP_LOG_DIR}"
echo "=========================================="

# Define configurations to run
# Format: "config_name:skip_flag"
# skip_flag: 0=run, 1=skip
# Note: Config name should be without .yaml extension
# CONFIGS=(
    # "Qwen_Qwen2_5-VL-32B-Instruct_vparsequence_tpartensor_8k_tokens_bs8_flash_attention_2_ckpt_bfloat16:0"
    # "Qwen_Qwen2_5-VL-32B-Instruct_vparsequence_tpartensor_16k_tokens_bs8_flash_attention_2_ckpt_bfloat16:0"
    # "Qwen_Qwen2_5-VL-32B-Instruct_vparsequence_tpartensor_32k_tokens_bs2_flash_attention_2_ckpt_bfloat16:0"
    # "Qwen_Qwen2_5-VL-32B-Instruct_vparsequence_tpartensor_32k_tokens_bs4_flash_attention_2_ckpt_bfloat16:0"
# )

# Alternative: Auto-discover all configs (uncomment to use)
CONFIGS=()
for config_file in configs/*.yaml; do
    config_name=$(basename "$config_file" .yaml)
    CONFIGS+=("${config_name}:0")
done

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for successful and failed runs
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SKIPPED_RUNS=0

# Loop through each configuration
for config_entry in "${CONFIGS[@]}"; do
    # Parse configuration
    IFS=':' read -r config_name skip_flag <<< "$config_entry"

    # Check if this config should be skipped
    if [ "$skip_flag" -eq 1 ]; then
        echo -e "${YELLOW}[SKIP]${NC} Skipping configuration: ${config_name}"
        SKIPPED_RUNS=$((SKIPPED_RUNS + 1))
        continue
    fi

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # Extract readable info from config name
    # Format: Model_visionpar_textpar_tokens_bsX_...
    if [[ $config_name =~ ([0-9]+k_tokens).*bs([0-9]+) ]]; then
        tokens="${BASH_REMATCH[1]}"
        batch_size="${BASH_REMATCH[2]}"
        run_label="${tokens}_bs${batch_size}"
    else
        run_label="${config_name}"
    fi

    echo ""
    echo "=========================================="
    echo -e "${GREEN}[RUN ${TOTAL_RUNS}]${NC} Starting configuration: ${config_name}"
    echo "  Label: ${run_label}"
    echo "  Started at: $(date)"
    echo "=========================================="

    # Log file for this run
    log_file="${SWEEP_LOG_DIR}/${config_name}.log"

    # Write configuration info to log file
    echo "Configuration: ${config_name}" > ${log_file}
    echo "Started at: $(date)" >> ${log_file}
    echo "========================================" >> ${log_file}
    echo "" >> ${log_file}

    # Launch training and capture both stdout and stderr
    set +e  # Temporarily disable exit on error to capture failures
    python -m python.train_ray \
        --config-path=../configs \
        --config-name=${config_name} >> ${log_file} 2>&1

    exit_code=$?
    # set -e  # Re-enable exit on error

    # Extract iteration timing from the training script output
    avg_iter_time="N/A"
    total_time="N/A"
    num_measured_iters="N/A"

    # Look for the specific timing line: "Training completed! X epochs, Y total steps, Z measured steps (avg A.AAAss/step)"
    if grep -q "Training completed!" ${log_file}; then
        # Extract average iteration time from "(avg X.XXXs/step)"
        avg_iter_time=$(grep "Training completed!" ${log_file} | tail -1 | grep -oP 'avg \K[0-9]+\.[0-9]+(?=s/step)' || true)
        if [ -z "$avg_iter_time" ]; then
            echo "WARNING: Could not extract average iteration time from log"
            exit_code=1
        fi

        # Extract number of measured steps from "X measured steps"
        num_measured_iters=$(grep "Training completed!" ${log_file} | tail -1 | grep -oP '[0-9]+ (?=measured steps)' || true)
        if [ -z "$num_measured_iters" ]; then
            echo "WARNING: Could not extract number of measured steps from log"
            exit_code=1
        fi

        # Extract total time from "in X.XXXs" (if present) - optional, may not be in all formats
        total_time=$(grep "Training completed!" ${log_file} | tail -1 | grep -oP 'in \K[0-9]+\.[0-9]+(?=s)' || true)
        if [ -z "$total_time" ]; then
            total_time="N/A"
        fi
    else
        echo "WARNING: 'Training completed!' message not found in log"
        exit_code=1
    fi

    # Calculate throughput (iterations per second)
    if [ "$avg_iter_time" != "N/A" ] && [ "$avg_iter_time" != "0" ] && [ "$avg_iter_time" != "" ]; then
        iters_per_sec=$(awk "BEGIN {printf \"%.4f\", 1.0 / $avg_iter_time}")
    else
        iters_per_sec="N/A"
    fi

    # Record completion time and status
    echo "" >> ${log_file}
    echo "========================================" >> ${log_file}
    echo "Completed at: $(date)" >> ${log_file}
    echo "Exit code: ${exit_code}" >> ${log_file}
    echo "Extracted Timing Metrics:" >> ${log_file}
    echo "  Number of measured iterations: ${num_measured_iters}" >> ${log_file}
    echo "  Total time (measured iterations): ${total_time}s" >> ${log_file}
    echo "  Average iteration time: ${avg_iter_time}s" >> ${log_file}
    echo "  Iterations per second: ${iters_per_sec}" >> ${log_file}

    # Check if training was successful
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Configuration ${run_label} completed successfully"
        if [ "$avg_iter_time" != "N/A" ] && [ "$avg_iter_time" != "" ]; then
            echo "  Average iteration time: ${avg_iter_time}s"
            echo "  Iterations per second: ${iters_per_sec}"
        fi
        echo "Status: SUCCESS" >> ${log_file}
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    else
        echo -e "${RED}[FAILED]${NC} Configuration ${run_label} failed with exit code ${exit_code}"
        echo "Status: FAILED" >> ${log_file}
        FAILED_RUNS=$((FAILED_RUNS + 1))

        # Optionally, uncomment the next line to stop the sweep on first failure
        # exit 1
    fi

    echo "Log saved to: ${log_file}"

    # Optional: Add a delay between runs to allow system to stabilize
    # sleep 10
done

# Calculate total possible runs
TOTAL_POSSIBLE_RUNS=${#CONFIGS[@]}

# Print summary
echo ""
echo "=========================================="
echo "Sweep completed at $(date)"
echo "=========================================="
echo "Summary:"
echo "  Total configurations: ${TOTAL_POSSIBLE_RUNS}"
echo "  Executed: ${TOTAL_RUNS}"
echo -e "  ${GREEN}Successful: ${SUCCESSFUL_RUNS}${NC}"
echo -e "  ${RED}Failed: ${FAILED_RUNS}${NC}"
echo -e "  ${YELLOW}Skipped: ${SKIPPED_RUNS}${NC}"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "-----------------------------------------------------------------------------------------"
printf "%-70s | %-10s | %-20s | %-15s\n" "Configuration" "Status" "Avg Iter Time (s)" "Iters/Sec"
echo "-----------------------------------------------------------------------------------------"

for config_entry in "${CONFIGS[@]}"; do
    IFS=':' read -r config_name skip_flag <<< "$config_entry"

    if [ "$skip_flag" -eq 1 ]; then
        printf "%-70s | %-10s | %-20s | %-15s\n" "$config_name" "SKIPPED" "N/A" "N/A"
    elif [ -f "${SWEEP_LOG_DIR}/${config_name}.log" ]; then
        # Extract timing info from log
        avg_time=$(grep "Average iteration time:" "${SWEEP_LOG_DIR}/${config_name}.log" | tail -1 | grep -oP '\K[0-9.]+(?=s)' || echo "N/A")
        iters_per_sec=$(grep "Iterations per second:" "${SWEEP_LOG_DIR}/${config_name}.log" | tail -1 | awk '{print $NF}')

        if grep -q "Status: SUCCESS" "${SWEEP_LOG_DIR}/${config_name}.log"; then
            printf "%-70s | ${GREEN}%-10s${NC} | %-20s | %-15s\n" "$config_name" "SUCCESS" "${avg_time}" "$iters_per_sec"
        else
            printf "%-70s | ${RED}%-10s${NC} | %-20s | %-15s\n" "$config_name" "FAILED" "${avg_time}" "$iters_per_sec"
        fi
    fi
done
echo "=========================================="
echo "All logs saved to: ${SWEEP_LOG_DIR}"

# Create summary file
summary_file="${SWEEP_LOG_DIR}/summary.txt"
echo "Ray Hybrid Para Training Sweep Summary" > ${summary_file}
echo "======================================" >> ${summary_file}
echo "Started: ${SWEEP_TIMESTAMP}" >> ${summary_file}
echo "Completed: $(date)" >> ${summary_file}
echo "" >> ${summary_file}
echo "Total configurations: ${TOTAL_POSSIBLE_RUNS}" >> ${summary_file}
echo "Executed: ${TOTAL_RUNS}" >> ${summary_file}
echo "Successful: ${SUCCESSFUL_RUNS}" >> ${summary_file}
echo "Failed: ${FAILED_RUNS}" >> ${summary_file}
echo "Skipped: ${SKIPPED_RUNS}" >> ${summary_file}
echo "" >> ${summary_file}
echo "Configurations:" >> ${summary_file}
echo "Format: config_name | status | avg_iter_time | iters_per_sec" >> ${summary_file}
echo "--------------------------------------------------------------" >> ${summary_file}

for config_entry in "${CONFIGS[@]}"; do
    IFS=':' read -r config_name skip_flag <<< "$config_entry"

    if [ "$skip_flag" -eq 1 ]; then
        echo "  ${config_name}: SKIPPED" >> ${summary_file}
    elif [ -f "${SWEEP_LOG_DIR}/${config_name}.log" ]; then
        # Extract timing info from log
        avg_time=$(grep "Average iteration time:" "${SWEEP_LOG_DIR}/${config_name}.log" | tail -1 | grep -oP '\K[0-9.]+(?=s)' || echo "N/A")
        iters_per_sec=$(grep "Iterations per second:" "${SWEEP_LOG_DIR}/${config_name}.log" | tail -1 | awk '{print $NF}')

        if grep -q "Status: SUCCESS" "${SWEEP_LOG_DIR}/${config_name}.log"; then
            echo "  ${config_name}: SUCCESS | ${avg_time}s | ${iters_per_sec} iters/s" >> ${summary_file}
        else
            echo "  ${config_name}: FAILED | ${avg_time}s | ${iters_per_sec} iters/s" >> ${summary_file}
        fi
    fi
done

echo ""
echo "Summary saved to: ${summary_file}"

# Exit with error if any runs failed (optional, comment out if you don't want this)
if [ $FAILED_RUNS -gt 0 ]; then
    echo "WARNING: Some runs failed. Check logs for details."
    exit 1
fi
