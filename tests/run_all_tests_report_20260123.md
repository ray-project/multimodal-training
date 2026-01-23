## Test run report

- Date: 2026-01-23
- Command: `bash tests/run_all_tests.sh`
- Working directory: `multimodal-training/`
- Result: failed during CPU test collection

## Summary

Pytest collected 38 items with 10 errors during collection. All errors share the same root cause:
`AttributeError: module 'deepspeed' has no attribute 'HAS_TRITON'`.

GPU phase was skipped because no tests were collected with `-m gpu`.

## Error details (representative)

```
E   AttributeError: module 'deepspeed' has no attribute 'HAS_TRITON'
```

One representative import stack:

```
tests/deepspeed/test_engine_deepspeed_prepp.py:13
  from python.ray.test_support import TinyTextTrainer, TinyVisionTrainer
python/ray/test_support.py:11
  from .vision import BaseVisionTrainer
python/ray/vision.py:25
  import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu
../../autotp/DeepSpeed/deepspeed/model_implementations/transformers/ds_transformer.py:17
  if deepspeed.HAS_TRITON and get_accelerator().is_triton_supported():
E AttributeError: module 'deepspeed' has no attribute 'HAS_TRITON'
```

## Affected tests (collection failed)

- `tests/deepspeed/test_engine_deepspeed_prepp.py`
- `tests/deepspeed/test_text_autotp.py`
- `tests/deepspeed/test_text_autotp_dp.py`
- `tests/deepspeed/test_vision_sp.py`
- `tests/deepspeed/test_vision_sp_dp.py`
- `tests/dtensor/test_text_dtensor.py`
- `tests/parallel/test_ragged_sp_smoke.py`
- `tests/parallel/test_split_gather.py`
- `tests/parallel/test_vision_compare.py`
- `tests/parallel/test_vision_detailed.py`
