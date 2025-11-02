# Test Guide

The test suite is split into two categories using `pytest` markers:

- `cpu_only`: unit-level coverage that runs on a single process without CUDA/NCCL.
- `gpu` + `integration`: multi-GPU or end-to-end checks that rely on CUDA, NCCL, and in some cases optional datasets or pretrained weights.

## Quick CPU Regression Pass

```bash
pytest -m cpu_only
```

This runs the light-weight ragged utilities and dataset modality tests:

- `tests/test_sequence_parallel_ragged.py`
- `tests/test_ragged_sp_smoke.py`
- `tests/test_dataset_modalities.py`

These tests exercise the split/pad/unpad utilities and dataset preprocessing logic without requiring GPUs.

## GPU / Distributed Checks

The following scenarios require at least 2 GPUs with NCCL enabled. Launch them with `torchrun` (or another launcher that sets up the process group) so each rank runs the test module.

**IMPORTANT**: Run each GPU test file individually, not all together. Running multiple distributed test files in a single pytest invocation can cause hangs due to process group cleanup issues.

### Sequence-parallel collectives smoke

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_split_gather.py -m gpu -v
```

Verifies the ragged-safe all-gather across ranks.

### Vision model parity / diagnostics

```bash
# Tests vision sequence parallelism with pretrained weights
torchrun --nproc_per_node=2 -m pytest tests/test_vision_compare.py -m gpu -v

# Detailed diagnostic test for vision sequence parallelism
torchrun --nproc_per_node=2 -m pytest tests/test_vision_detailed.py -m gpu -v
```

These tests load the Qwen vision encoder (optionally with pretrained weights) and compare outputs across SP configurations. They expect GPUs with BF16 support and access to the `Qwen/Qwen2.5-VL-3B-Instruct` weights (cached locally via Hugging Face).

## Dataset alignment test

`tests/test_dataset_modalities.py::test_real_dataset_alignment` automatically skips if the COCO validation set defined in `DEFAULT_DATA_REGISTRY` is not present. To exercise it fully, download the dataset referenced in the registry before running `pytest -m cpu_only`.

## Tips

- **CPU tests**: Run with `pytest -m cpu_only` (26 tests, all passing)
- **GPU tests**: Run each file individually with torchrun (do NOT run all together)
- Set `HF_HOME` if you want weights/datasets cached at a specific path
- For verbose logging during GPU runs, add `-s` and export `CUDA_LAUNCH_BLOCKING=1`
- Use `-v` flag for more detailed test output

## Summary

| Test File | Marker | Status | Command |
|-----------|--------|--------|---------|
| `test_sequence_parallel_ragged.py` | cpu_only | ✅ Passing | `pytest tests/test_sequence_parallel_ragged.py -m cpu_only` |
| `test_ragged_sp_smoke.py` | cpu_only | ✅ Passing | `pytest tests/test_ragged_sp_smoke.py -m cpu_only` |
| `test_dataset_modalities.py` | cpu_only | ✅ Passing | `pytest tests/test_dataset_modalities.py -m cpu_only` |
| `test_split_gather.py` | gpu | ✅ Passing | `torchrun --nproc_per_node=2 -m pytest tests/test_split_gather.py -m gpu -v` |
| `test_vision_compare.py` | gpu | ✅ Passing | `torchrun --nproc_per_node=2 -m pytest tests/test_vision_compare.py -m gpu -v` |
| `test_vision_detailed.py` | gpu | ✅ Passing | `torchrun --nproc_per_node=2 -m pytest tests/test_vision_detailed.py -m gpu -v` |
