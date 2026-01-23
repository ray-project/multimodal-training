## Test run report

- Date: 2026-01-23
- Command: `bash tests/run_all_tests.sh`
- Working directory: `multimodal-training/`
- Result: GPU tests failed to start (rendezvous port conflict)

## Summary

All GPU test invocations failed at torchrun startup with:

```
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address.
port: 29500, ... EADDRINUSE, message: address already in use
```

CPU tests completed successfully (36 passed, 1 skipped).

## Affected tests

- `tests/deepspeed/test_engine_deepspeed_prepp.py`
- `tests/deepspeed/test_text_autotp.py`
- `tests/deepspeed/test_text_autotp_dp.py`
- `tests/deepspeed/test_vision_sp.py`
- `tests/deepspeed/test_vision_sp_dp.py`
- `tests/dtensor/test_text_dtensor.py`
- `tests/integration/test_phase4_functional.py`
- `tests/megatron/test_engine_megatron_prepp.py`
- `tests/parallel/test_rdt_non_collocated.py`
- `tests/parallel/test_split_gather.py`
- `tests/parallel/test_vision_compare.py`
- `tests/parallel/test_vision_detailed.py`

## Notes

The default torchrun rendezvous port (`29500`) appears to be in use by another process.
