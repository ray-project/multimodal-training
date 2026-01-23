from pathlib import Path
import sys

import pytest


def _ensure_local_deepspeed() -> None:
    """Prefer the repo DeepSpeed checkout and ensure HAS_TRITON exists."""
    repo_root = Path(__file__).resolve().parents[2]
    deepspeed_root = repo_root / "DeepSpeed"
    if deepspeed_root.exists():
        sys.path.insert(0, str(deepspeed_root))

    try:
        import deepspeed  # noqa: WPS433
    except Exception:
        return

    if not hasattr(deepspeed, "HAS_TRITON"):
        deepspeed.HAS_TRITON = False


_ensure_local_deepspeed()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cpu_only: test does not require GPUs or distributed backends")
    config.addinivalue_line("markers", "gpu: test expects CUDA/NCCL and multi-GPU setup")
    config.addinivalue_line("markers", "integration: end-to-end or long-running scenario")
