"""Pre-PP readiness test for DeepSpeed engine with Ray actors."""

import sys
from pathlib import Path
import pytest
import ray
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.ray.actor_group import ActorGroup  # noqa: E402
from python.ray.test_support import TinyTextTrainer, TinyVisionTrainer  # noqa: E402

pytestmark = [pytest.mark.gpu]




def _build_component_config():
    return {
        "model_name": "tiny",
        "model_type": "tiny",
        "engine": "deepspeed",
        "engine_config": {"zero_stage": 1, "reduce_bucket_size": 1000},
        "parallelism": "deepspeed",
        "dtype": "float32",
        "attention_backend": "sdpa",
        "activation_checkpointing": False,
        "autocast": False,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 1,
        "num_iterations": 1,
        "warmup_ratio": 0.0,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "gradient_accumulation_steps": 1,
        "seed": 123,
        "dp_size": 1,
        "parallel_size": 1,
        "datasets": ["dummy"],
        "data_registry": {},
        "num_workers": 0,
        "pin_memory": False,
        "data_flatten": False,
        "min_pixels": 1,
        "max_pixels": 1,
        "force_fixed_size": True,
        "hidden_size": 8,
        "vision_tokens": 2,
        "text_seq_len": 4,
        "vocab_size": 16,
        "image_token_id": 1,
    }


def test_deepspeed_engine_prepp():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepSpeed pre-PP readiness test.")
    pytest.importorskip("deepspeed")

    project_root = Path(__file__).parent.parent
    tests_root = Path(__file__).parent
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "working_dir": str(project_root),
            "py_modules": [str(tests_root)],
        },
    )
    try:
        vision_config = _build_component_config()
        text_config = _build_component_config()

        vision_group = ActorGroup(vision_config, TinyVisionTrainer, num_actors=1, num_cpus=2, num_gpus=1)
        text_group = ActorGroup(text_config, TinyTextTrainer, num_actors=1, num_cpus=2, num_gpus=1)

        vision_group.execute_all("build_model")
        text_group.execute_all("build_model")
        vision_group.execute_all("initialize_trainer")
        text_group.execute_all("initialize_trainer")

        vision_pg = vision_group.execute_all("is_process_group_initialized")
        text_pg = text_group.execute_all("is_process_group_initialized")
        assert all(vision_pg), "Vision process group was not initialized"
        assert all(text_pg), "Text process group was not initialized"

        vision_results = vision_group.execute_all("forward_step_no_return", 0)
        dummy_vision = {
            "vision_embeddings": torch.zeros(
                1,
                vision_config["vision_tokens"],
                vision_config["hidden_size"],
            ),
            "sample_index": 0,
            "iteration": 0,
        }
        text_refs = text_group.execute_all_async("forward_step", [dummy_vision], [0])
        text_results = ray.get(text_refs)
        assert isinstance(text_results[0], dict)
        assert "loss" in text_results[0]

        text_group.execute_all("backward_step")
        vision_group.execute_all("backward_step_with_dummy_grad")
    finally:
        ray.shutdown()
