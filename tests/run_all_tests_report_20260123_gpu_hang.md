## Test run report

- Date: 2026-01-23
- Command: `bash tests/run_all_tests.sh`
- Working directory: `multimodal-training/`
- Result: GPU tests hung during `test_text_autotp.py`

## Summary

CPU tests completed successfully (36 passed, 1 skipped).  
GPU tests started and the first GPU test (`test_engine_deepspeed_prepp.py`) passed using
`torchrun --master-port=<dynamic>`. The run then stalled during
`tests/deepspeed/test_text_autotp.py::test_autotp_vs_no_parallel` with no additional output
for more than 3 minutes. The process was terminated to avoid a prolonged hang.

## Last observed output

```
Running: GPU tests (tests/deepspeed/test_text_autotp.py)
Command: torchrun --master-port=50707 --nproc_per_node=2 -m pytest tests/deepspeed/test_text_autotp.py -m gpu -v
...
collecting ... collected 3 items

tests/deepspeed/test_text_autotp.py::test_autotp_vs_no_parallel
```

## Notes

- Terminated after 3+ minutes of silence per distributed timeout guidance.
