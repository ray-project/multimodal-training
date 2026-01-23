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

        GPU_TEST_FILES=$(collect_gpu_test_files)
        if [ -z "$GPU_TEST_FILES" ]; then
            echo -e "${YELLOW}No tests collected with -m gpu. Skipping GPU tests.${NC}"
            echo ""
        else
            for test_file in $GPU_TEST_FILES; do
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

{
    echo "=========================================="
    echo "TEST SUMMARY"
    echo "=========================================="
    echo ""
    echo "Date: $(date)"
    echo ""

    if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
        echo "Passed (${#PASSED_TESTS[@]}):"
        for test in "${PASSED_TESTS[@]}"; do
            echo "  ✓ $test"
        done
        echo ""
    fi

    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo "Failed (${#FAILED_TESTS[@]}):"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  ✗ $test"
        done
        echo ""
    else
        echo "All tests passed!"
        echo ""
    fi
} > "$SUMMARY_FILE"

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
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}Summary written to: $SUMMARY_FILE${NC}"
    exit 0
fi
