"""Pre-PP readiness test for Megatron engine initialization."""

import socket
import sys
from pathlib import Path

import pytest
import ray
import torch

pytestmark = [pytest.mark.gpu]

PROJECT_ROOT = Path(__file__).parent.parent.parent
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
MS_SWIFT_ROOT = PROJECT_ROOT / "ms-swift"

sys.path.insert(0, str(PROJECT_ROOT / "multimodal-training"))
sys.path.insert(0, str(MEGATRON_ROOT))
sys.path.insert(0, str(MS_SWIFT_ROOT))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@ray.remote(num_cpus=2, num_gpus=1)
class MegatronPreppActor:
    def initialize_megatron(self, master_port: int) -> dict:
        import os

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Megatron pre-PP readiness test.")

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        import torch.distributed as dist

        try:
            import megatron  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Megatron-LM import failed; ensure Megatron-LM is available.") from exc

        if not dist.is_initialized():
            torch.cuda.set_device(0)
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://127.0.0.1:{master_port}",
                rank=0,
                world_size=1,
            )

        from megatron.core import mpu

        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

        model = torch.nn.Linear(4, 4)
        inputs = torch.zeros(2, 4)
        loss = model(inputs).sum()
        loss.backward()

        return {
            "dist_initialized": dist.is_initialized(),
            "model_parallel_initialized": mpu.model_parallel_is_initialized(),
        }


def test_megatron_engine_prepp():
    import os

    arch_list = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_list = f"{major}.{minor}"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
    try:
        import megatron  # noqa: F401
    except Exception:
        pytest.skip("Megatron-LM is not available; skipping Megatron pre-PP readiness test.")

    master_port = _find_free_port()
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "working_dir": str(PROJECT_ROOT / "multimodal-training"),
            "py_modules": [str(MEGATRON_ROOT), str(MS_SWIFT_ROOT)],
            "excludes": [".git/**", "**/.git/**", "**/__pycache__/**"],
            "env_vars": {
                "PYTHONPATH": ":".join(
                    [
                        str(PROJECT_ROOT / "multimodal-training"),
                        str(MEGATRON_ROOT),
                        str(MS_SWIFT_ROOT),
                    ]
                ),
                **({"TORCH_CUDA_ARCH_LIST": arch_list} if arch_list else {}),
            },
        },
    )
    try:
        actor = MegatronPreppActor.remote()
        results = ray.get(actor.initialize_megatron.remote(master_port))
        assert results["dist_initialized"]
        assert results["model_parallel_initialized"]
    finally:
        ray.shutdown()
