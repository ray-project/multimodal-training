import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cpu_only: test does not require GPUs or distributed backends")
    config.addinivalue_line("markers", "gpu: test expects CUDA/NCCL and multi-GPU setup")
    config.addinivalue_line("markers", "integration: end-to-end or long-running scenario")
