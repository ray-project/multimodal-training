## Path Check Report (tests/run_all_tests.sh)

Checked the test paths referenced by `tests/run_all_tests.sh` against the current `tests/` layout.

### Expected failures (before path fixes)

- `tests/test_split_gather.py`
  - Status: missing path
  - Reason: test moved to `tests/parallel/test_split_gather.py`
  - Likely failure: pytest error "file not found"

- `tests/test_vision_compare.py`
  - Status: missing path
  - Reason: test moved to `tests/parallel/test_vision_compare.py`
  - Likely failure: pytest error "file not found"

- `tests/test_vision_detailed.py`
  - Status: missing path
  - Reason: test moved to `tests/parallel/test_vision_detailed.py`
  - Likely failure: pytest error "file not found"

### Fix cost estimate

Low: update three paths in `tests/run_all_tests.sh` to point at the `tests/parallel/` directory.
