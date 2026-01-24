#!/bin/bash

# run_all_tests.sh - Run all tests described in tests/README.md
# This script runs CPU tests first, then GPU tests individually

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "=========================================="
echo "Running Ray Hybrid Para Test Suite"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILED_TESTS=()
PASSED_TESTS=()
TEST_START_TIME="$(date)"
TEST_START_SECONDS="$(date +%s)"
TEST_DURATIONS=()
PYTHON_BIN="$(command -v python || true)"

get_python_version() {
    if [ -n "$PYTHON_BIN" ]; then
        python - <<'PY'
import sys
print(sys.version.replace("\n", " "))
PY
    else
        echo "python-not-found"
    fi
}

get_ray_version() {
    if [ -n "$PYTHON_BIN" ]; then
        python - <<'PY'
try:
    import ray
    print(ray.__version__)
except Exception as exc:
    print(f"unavailable: {exc}")
PY
    else
        echo "python-not-found"
    fi
}

collect_env_info() {
    local env_file="$1"
    local python_version
    local ray_version

    python_version="$(get_python_version)"
    ray_version="$(get_ray_version)"

    {
        echo "=========================================="
        echo "ENVIRONMENT DETAILS"
        echo "=========================================="
        echo ""
        echo "Python executable: ${PYTHON_BIN:-python-not-found}"
        echo "Python version: $python_version"
        echo "Ray version: $ray_version"
        echo ""
        echo "CUDA versions:"
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi 2>/dev/null | grep -m 1 "CUDA Version" || echo "  nvidia-smi output unavailable"
        else
            echo "  nvidia-smi not found"
        fi
        if command -v nvcc &> /dev/null; then
            nvcc --version 2>/dev/null || echo "  nvcc output unavailable"
        else
            echo "  nvcc not found"
        fi
        echo ""

        if command -v conda &> /dev/null; then
            echo "Conda environments:"
            conda info --envs 2>/dev/null || conda env list 2>/dev/null || echo "  unavailable"
            if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
                echo ""
                echo "Active conda env: $CONDA_DEFAULT_ENV"
            fi
            if [ -n "${CONDA_PREFIX:-}" ]; then
                echo "Conda prefix: $CONDA_PREFIX"
            fi
        elif [ -n "${CONDA_DEFAULT_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
            echo "Conda environment (from environment variables):"
            echo "Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
            echo "Conda prefix: ${CONDA_PREFIX:-unknown}"
        else
            echo "Conda: not in use"
        fi

        echo ""
        echo "Python package versions (pip freeze):"
        if [ -n "$PYTHON_BIN" ]; then
            python -m pip freeze 2>&1 || echo "pip freeze failed"
        else
            echo "python-not-found"
        fi
        echo ""
    } > "$env_file"
}

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local start_ts
    local end_ts
    local duration

    echo "=========================================="
    echo "Running: $test_name"
    echo "Command: $test_cmd"
    echo "=========================================="

    start_ts="$(date +%s)"
    if eval "$test_cmd"; then
        echo -e "${GREEN}✓ PASSED: $test_name${NC}"
        PASSED_TESTS+=("$test_name")
    else
        echo -e "${RED}✗ FAILED: $test_name${NC}"
        FAILED_TESTS+=("$test_name")
    fi
    end_ts="$(date +%s)"
    duration=$((end_ts - start_ts))
    TEST_DURATIONS+=("$test_name|$duration")
    echo ""
}

# Collect unique GPU test files via pytest collection
collect_gpu_test_files() {
    pytest -q --collect-only -m gpu tests 2>&1 | python -c '
import sys

paths = []
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("="):
        continue
    path = line.split("::", 1)[0]
    if path.endswith(".py"):
        paths.append(path)

seen = set()
for path in paths:
    if path not in seen:
        seen.add(path)
        print(path)
'
}

# Heuristic: DP tests generally require 4+ GPUs
required_gpus_for_file() {
    local file_path="$1"
    if [[ "$file_path" == *"_dp.py" ]]; then
        echo 4
    else
        echo 2
    fi
}

get_free_port() {
    python - <<'PY'
import socket

sock = socket.socket()
sock.bind(("", 0))
port = sock.getsockname()[1]
sock.close()
print(port)
PY
}

# CPU Tests
echo ""
echo "=========================================="
echo "PHASE 1: CPU-Only Tests"
echo "=========================================="
echo ""

run_test "CPU Tests (all non-gpu tests)" "pytest tests -m 'not gpu' -v"

# GPU Tests - Run individually to avoid process group cleanup issues
echo ""
echo "=========================================="
echo "PHASE 2: GPU Tests"
echo "=========================================="
echo ""
echo -e "${YELLOW}Note: GPU tests run individually to avoid process group cleanup issues${NC}"
echo ""

# Check if we have GPUs available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. Skipping GPU tests.${NC}"
    echo -e "${YELLOW}GPU tests require CUDA and at least 2 GPUs.${NC}"
else
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo -e "${YELLOW}Warning: Found $GPU_COUNT GPU(s), but 2+ required. Skipping GPU tests.${NC}"
    else
        echo "Found $GPU_COUNT GPU(s). Running GPU tests..."
        echo ""

        MEGATRON_SINGLE_TEST="tests/megatron/test_single_trainer.py"
        MEGATRON_MODEL="${MEGATRON_SINGLE_MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
        MEGATRON_MODEL_TYPE="${MEGATRON_SINGLE_MODEL_TYPE:-qwen2_moe}"
        MEGATRON_MATRIX="${MEGATRON_SINGLE_MATRIX:-1}"
        MEGATRON_LOAD_WEIGHTS="${MEGATRON_SINGLE_LOAD_WEIGHTS:-0}"

        if [ -f "$MEGATRON_SINGLE_TEST" ]; then
            if [ "$GPU_COUNT" -ge 4 ]; then
                run_test "GPU tests ($MEGATRON_SINGLE_TEST matrix)" \
                    "MEGATRON_SINGLE_MODEL=$MEGATRON_MODEL MEGATRON_SINGLE_MODEL_TYPE=$MEGATRON_MODEL_TYPE MEGATRON_SINGLE_MATRIX=$MEGATRON_MATRIX MEGATRON_SINGLE_LOAD_WEIGHTS=$MEGATRON_LOAD_WEIGHTS pytest $MEGATRON_SINGLE_TEST -m gpu"
            else
                run_test "GPU tests ($MEGATRON_SINGLE_TEST tp1/ep1)" \
                    "MEGATRON_SINGLE_MODEL=$MEGATRON_MODEL MEGATRON_SINGLE_MODEL_TYPE=$MEGATRON_MODEL_TYPE MEGATRON_SINGLE_TP_SIZE=1 MEGATRON_SINGLE_EP_SIZE=1 MEGATRON_SINGLE_LOAD_WEIGHTS=$MEGATRON_LOAD_WEIGHTS pytest $MEGATRON_SINGLE_TEST -m gpu"
            fi
        fi

        GPU_TEST_FILES=$(collect_gpu_test_files)
        if [ -z "$GPU_TEST_FILES" ]; then
            echo -e "${YELLOW}No tests collected with -m gpu. Skipping GPU tests.${NC}"
            echo ""
        else
            for test_file in $GPU_TEST_FILES; do
                if [ "$test_file" = "$MEGATRON_SINGLE_TEST" ]; then
                    continue
                fi
                REQUIRED_GPUS=$(required_gpus_for_file "$test_file")
                if [ "$GPU_COUNT" -lt "$REQUIRED_GPUS" ]; then
                    echo -e "${YELLOW}Skipping $test_file (requires $REQUIRED_GPUS+ GPUs, found $GPU_COUNT)${NC}"
                    echo ""
                    continue
                fi
                MASTER_PORT=$(get_free_port)
                run_test "GPU tests ($test_file)" \
                         "torchrun --master-port=$MASTER_PORT --nproc_per_node=$REQUIRED_GPUS -m pytest $test_file -m gpu -v"
            done
        fi
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""

SUMMARY_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR="$SCRIPT_DIR/summary"
SUMMARY_FILE="$SUMMARY_DIR/run_all_tests_summary_${SUMMARY_TIMESTAMP}.txt"
ENV_INFO_FILE="$SUMMARY_DIR/run_all_tests_env_${SUMMARY_TIMESTAMP}.txt"
SUMMARY_MARKDOWN="$SUMMARY_DIR/SUMMARY.md"
GIT_REVISION="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")"
TEST_END_TIME="$(date)"
TEST_END_SECONDS="$(date +%s)"
TOTAL_DURATION=$((TEST_END_SECONDS - TEST_START_SECONDS))

mkdir -p "$SUMMARY_DIR"
collect_env_info "$ENV_INFO_FILE"

{
    echo "=========================================="
    echo "TEST SUMMARY"
    echo "=========================================="
    echo ""
    echo "Start: $TEST_START_TIME"
    echo "End: $TEST_END_TIME"
    echo "Total duration: ${TOTAL_DURATION}s"
    echo "Git revision: $GIT_REVISION"
    echo "Environment details: $ENV_INFO_FILE"
    echo ""
    cat "$ENV_INFO_FILE"

    if [ ${#TEST_DURATIONS[@]} -gt 0 ]; then
        echo "Durations:"
        for entry in "${TEST_DURATIONS[@]}"; do
            name="${entry%%|*}"
            duration="${entry##*|}"
            echo "  - $name: ${duration}s"
        done
        echo ""
    fi

    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo "Failed (${#FAILED_TESTS[@]}):"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  ✗ $test"
        done
        echo ""
    fi

    if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
        echo "Passed (${#PASSED_TESTS[@]}):"
        for test in "${PASSED_TESTS[@]}"; do
            echo "  ✓ $test"
        done
        echo ""
    fi

    if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
        echo "All tests passed!"
        echo ""
    fi
} > "$SUMMARY_FILE"

{
    echo "# Test Summary"
    echo ""
    echo "- Start: $TEST_START_TIME"
    echo "- End: $TEST_END_TIME"
    echo "- Total duration: ${TOTAL_DURATION}s"
    echo "- Git revision: \`$GIT_REVISION\`"
    echo "- Latest run log: \`$(basename "$SUMMARY_FILE")\`"
    echo "- Environment log: \`$(basename "$ENV_INFO_FILE")\`"
    echo ""
    echo "## Results"
    echo ""
    echo "| Test | Status | Duration |"
    echo "| --- | --- | --- |"
    for entry in "${TEST_DURATIONS[@]}"; do
        name="${entry%%|*}"
        duration="${entry##*|}"
        status="✅ Passed"
        for failed in "${FAILED_TESTS[@]}"; do
            if [ "$failed" == "$name" ]; then
                status="❌ Failed"
                break
            fi
        done
        echo "| $name | $status | ${duration}s |"
    done
    echo ""
} > "$SUMMARY_MARKDOWN"

if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
    echo -e "${GREEN}Passed (${#PASSED_TESTS[@]}):${NC}"
    for test in "${PASSED_TESTS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $test"
    done
    echo ""
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "${RED}Failed (${#FAILED_TESTS[@]}):${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}✗${NC} $test"
    done
    echo ""
    echo -e "${RED}Summary written to: $SUMMARY_FILE${NC}"
    echo -e "${RED}Markdown summary: $SUMMARY_MARKDOWN${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}Summary written to: $SUMMARY_FILE${NC}"
    echo -e "${GREEN}Markdown summary: $SUMMARY_MARKDOWN${NC}"
    exit 0
fi
