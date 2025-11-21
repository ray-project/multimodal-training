#!/bin/bash

# run_all_tests.sh - Run all tests described in tests/README.md
# This script runs CPU tests first, then GPU tests individually

set -e  # Exit on error

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

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_cmd="$2"

    echo "=========================================="
    echo "Running: $test_name"
    echo "Command: $test_cmd"
    echo "=========================================="

    if eval "$test_cmd"; then
        echo -e "${GREEN}✓ PASSED: $test_name${NC}"
        PASSED_TESTS+=("$test_name")
    else
        echo -e "${RED}✗ FAILED: $test_name${NC}"
        FAILED_TESTS+=("$test_name")
    fi
    echo ""
}

# CPU Tests
echo ""
echo "=========================================="
echo "PHASE 1: CPU-Only Tests"
echo "=========================================="
echo ""

run_test "CPU Tests (all cpu_only markers)" "pytest -m cpu_only -v"

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

        run_test "Sequence-parallel collectives smoke" \
                 "torchrun --nproc_per_node=2 -m pytest tests/test_split_gather.py -m gpu -v"

        run_test "Vision model parity (test_vision_compare.py)" \
                 "torchrun --nproc_per_node=2 -m pytest tests/test_vision_compare.py -m gpu -v"

        run_test "Vision model detailed diagnostics (test_vision_detailed.py)" \
                 "torchrun --nproc_per_node=2 -m pytest tests/test_vision_detailed.py -m gpu -v"

        run_test "Text model with DeepSpeed AutoTP" \
                 "torchrun --nproc_per_node=2 -m pytest tests/test_text_autotp.py -v"

        # Run AutoTP + DP test only if 4+ GPUs available
        if [ "$GPU_COUNT" -ge 4 ]; then
            run_test "Text model with AutoTP + Data Parallel" \
                     "torchrun --nproc_per_node=4 -m pytest tests/test_text_autotp_dp.py -v"
        else
            echo -e "${YELLOW}Skipping test_text_autotp_dp.py (requires 4+ GPUs, found $GPU_COUNT)${NC}"
            echo ""
        fi
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""

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
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
