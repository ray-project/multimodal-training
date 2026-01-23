import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_phase4_deepspeed_megatron_pipeline():
    config_path = os.environ.get("PHASE4_FUNCTIONAL_CONFIG")
    if not config_path:
        pytest.skip("Set PHASE4_FUNCTIONAL_CONFIG to run the phase 4 functional test.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the phase 4 functional test.")

    config_path = Path(config_path)
    if not config_path.exists():
        pytest.fail(f"PHASE4_FUNCTIONAL_CONFIG does not exist: {config_path}")

    project_root = Path(__file__).parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    cmd = [
        sys.executable,
        "-m",
        "python.train_ray",
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
        "training.num_iterations=2",
        "training.batch_size=1",
        "training.no_checkpoint=true",
        "training.log_interval=1",
    ]

    subprocess.run(cmd, cwd=project_root, env=env, check=True)
