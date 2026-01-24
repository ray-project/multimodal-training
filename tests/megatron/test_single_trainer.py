"""Single-trainer Megatron initialization test.

Env vars:
  - MEGATRON_SINGLE_MODEL: HF model path/name (required).
  - MEGATRON_SINGLE_MODEL_TYPE: model type (default: qwen2_5_vl).
  - MEGATRON_SINGLE_TRAINER: "text" or "vision" (default: text).
  - MEGATRON_SINGLE_NUM_ACTORS: Ray actors to launch (default: 1).
  - MEGATRON_SINGLE_TP_SIZE: tensor parallel size (default: 1).
  - MEGATRON_SINGLE_EP_SIZE: expert parallel size (default: 1).
  - MEGATRON_SINGLE_MATRIX: set to 1 to run (TP,EP) in {(1,1),(1,2),(2,1),(2,2)}.
  - MEGATRON_SINGLE_NUM_EXPERTS: number of experts (optional).
  - MEGATRON_SINGLE_LOAD_WEIGHTS: set to 0/false to skip weights (default: true).
"""

import sys
from pathlib import Path

import pytest
import ray
import torch

pytestmark = [pytest.mark.gpu]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
MS_SWIFT_ROOT = PROJECT_ROOT / "ms-swift"

sys.path.insert(0, str(PROJECT_ROOT / "multimodal-training"))
sys.path.insert(0, str(MEGATRON_ROOT))
sys.path.insert(0, str(MS_SWIFT_ROOT))

from python.ray.actor_group import ActorGroup  # noqa: E402
from python.ray.megatron_trainer import (  # noqa: E402
    MegatronTextTrainer,
    MegatronVisionTrainer,
)


def _get_tp_ep_matrix():
    import os

    matrix_enabled = os.environ.get("MEGATRON_SINGLE_MATRIX", "0").lower() in {"1", "true", "yes"}
    if matrix_enabled:
        return [(1, 1), (1, 2), (2, 1), (2, 2)]

    tp_size = int(os.environ.get("MEGATRON_SINGLE_TP_SIZE", "1"))
    ep_size = int(os.environ.get("MEGATRON_SINGLE_EP_SIZE", "1"))
    return [(tp_size, ep_size)]


def _build_component_config(model_path: str, model_type: str):
    import os

    tp_size = int(os.environ.get("MEGATRON_SINGLE_TP_SIZE", "1"))
    expert_model_parallel_size = int(os.environ.get("MEGATRON_SINGLE_EP_SIZE", "1"))
    num_experts_env = os.environ.get("MEGATRON_SINGLE_NUM_EXPERTS")
    load_weights_env = os.environ.get("MEGATRON_SINGLE_LOAD_WEIGHTS", "true").lower()
    load_weights = load_weights_env not in {"0", "false", "no"}

    config = {
        "model_name": model_path,
        "model_type": model_type,
        "engine": "megatron",
        "engine_config": {
            "tensor_parallel_size": tp_size,
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


@pytest.mark.parametrize("tp_size,ep_size", _get_tp_ep_matrix())
def test_megatron_single_trainer_init(tp_size: int, ep_size: int):
    import os

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the single-trainer Megatron test.")

    try:
        import megatron  # noqa: F401
    except Exception:
        pytest.skip("Megatron-LM is not available; skipping single-trainer Megatron test.")

    model_path = os.environ.get("MEGATRON_SINGLE_MODEL")
    if not model_path:
        pytest.skip("Set MEGATRON_SINGLE_MODEL to run the single-trainer Megatron test.")

    model_type = os.environ.get("MEGATRON_SINGLE_MODEL_TYPE", "qwen2_5_vl")
    trainer_name = os.environ.get("MEGATRON_SINGLE_TRAINER", "text").lower()
    if trainer_name == "vision":
        trainer_cls = MegatronVisionTrainer
    elif trainer_name == "text":
        trainer_cls = MegatronTextTrainer
    else:
        pytest.skip(f"Unsupported MEGATRON_SINGLE_TRAINER={trainer_name!r}; use 'text' or 'vision'.")

    arch_list = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_list = f"{major}.{minor}"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)

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
        os.environ["MEGATRON_SINGLE_TP_SIZE"] = str(tp_size)
        os.environ["MEGATRON_SINGLE_EP_SIZE"] = str(ep_size)

        config = _build_component_config(model_path, model_type)
        num_actors = int(os.environ.get("MEGATRON_SINGLE_NUM_ACTORS", str(tp_size * ep_size)))
        group = ActorGroup(config, trainer_cls, num_actors=num_actors, num_cpus=2, num_gpus=1)

        group.execute_all("build_model")
        group.execute_all("initialize_trainer")

        pg = group.execute_all("is_process_group_initialized")
        assert all(pg), "Process group was not initialized"
    finally:
        ray.shutdown()
