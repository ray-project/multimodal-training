"""Pre-PP readiness test for Megatron engine initialization."""

import sys
from pathlib import Path

import pytest
import ray
import torch

from python.ray.actor_group import ActorGroup  # noqa: E402
from python.ray.megatron_trainer import (  # noqa: E402
    MegatronTextTrainer,
    MegatronVisionTrainer,
)

pytestmark = [pytest.mark.gpu]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
MS_SWIFT_ROOT = PROJECT_ROOT / "ms-swift"

sys.path.insert(0, str(PROJECT_ROOT / "multimodal-training"))
sys.path.insert(0, str(MEGATRON_ROOT))
sys.path.insert(0, str(MS_SWIFT_ROOT))


def _build_component_config(model_path: str):
    import os

    expert_model_parallel_size = int(os.environ.get("MEGATRON_TEST_EP_SIZE", "1"))
    num_experts_env = os.environ.get("MEGATRON_TEST_NUM_EXPERTS")
    load_weights_env = os.environ.get("MEGATRON_TEST_LOAD_WEIGHTS", "true").lower()
    load_weights = load_weights_env not in {"0", "false", "no"}
    config = {
        "model_name": model_path,
        "model_type": "qwen2_5_vl",
        "engine": "megatron",
        "engine_config": {
            "tensor_parallel_size": 1,
            "sequence_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "attention_backend": "unfused",
            "expert_model_parallel_size": expert_model_parallel_size,
            "load_weights": load_weights,
        },
        "parallelism": "tensor",
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "activation_checkpointing": False,
        "autocast": False,
        "seed": 123,
        "dp_size": 1,
        "parallel_size": 1,
        "text_seq_len": 4,
    }
    if num_experts_env is not None:
        config["engine_config"]["num_experts"] = int(num_experts_env)
    return config


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

    model_path = os.environ.get("MEGATRON_TEST_MODEL")
    if not model_path:
        pytest.skip("Set MEGATRON_TEST_MODEL to a HF model path for Megatron pre-PP readiness test.")

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
                "USE_HF": "1",
                "HF_HOME": os.environ.get("HF_HOME", "/mnt/local_storage/hf-cache"),
                **({"TORCH_CUDA_ARCH_LIST": arch_list} if arch_list else {}),
            },
        },
    )
    try:
        vision_config = _build_component_config(model_path)
        text_config = _build_component_config(model_path)

        vision_group = ActorGroup(vision_config, MegatronVisionTrainer, num_actors=1, num_cpus=2, num_gpus=1)
        text_group = ActorGroup(text_config, MegatronTextTrainer, num_actors=1, num_cpus=2, num_gpus=1)

        vision_group.execute_all("build_model")
        text_group.execute_all("build_model")
        vision_group.execute_all("initialize_trainer")
        text_group.execute_all("initialize_trainer")

        vision_pg = vision_group.execute_all("is_process_group_initialized")
        text_pg = text_group.execute_all("is_process_group_initialized")
        assert all(vision_pg), "Vision process group was not initialized"
        assert all(text_pg), "Text process group was not initialized"

        vision_outputs = vision_group.execute_all("forward_step", 0)
        text_group.execute_all("forward_step", vision_outputs, 0)

        text_backward = text_group.execute_all("backward_step")
        vision_group.execute_all("backward_step", text_backward)
    finally:
        ray.shutdown()
