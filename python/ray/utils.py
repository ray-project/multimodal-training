import logging
import os
import time

import ray

logger = logging.getLogger(__name__)


def _unique_log_path(archive_dir: str, timestamp: str, suffix: str = "") -> str:
    base_name = f"train_{timestamp}{suffix}.log"
    candidate = os.path.join(archive_dir, base_name)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(archive_dir, f"train_{timestamp}{suffix}_{counter}.log")
        counter += 1
    return candidate


def _archive_existing_log(symlink_path: str, archive_dir: str) -> None:
    if not os.path.exists(symlink_path) or os.path.islink(symlink_path):
        return
    mtime = time.localtime(os.path.getmtime(symlink_path))
    stamp = time.strftime("%Y%m%d_%H%M%S", mtime)
    os.makedirs(archive_dir, exist_ok=True)
    archived_path = _unique_log_path(archive_dir, stamp, suffix="_legacy")
    os.replace(symlink_path, archived_path)


def ensure_run_log_file(
    log_dir: str = "logs",
    archive_dir: str = "logs/archive",
    symlink_name: str = "logs/train.log",
) -> str:
    log_dir_abs = os.path.abspath(log_dir)
    archive_dir_abs = os.path.abspath(archive_dir)

    os.makedirs(log_dir_abs, exist_ok=True)
    os.makedirs(archive_dir_abs, exist_ok=True)
    if symlink_name:
        symlink_path = os.path.abspath(symlink_name)
        _archive_existing_log(symlink_path, archive_dir_abs)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = _unique_log_path(archive_dir_abs, timestamp)
    os.environ["RAY_TRAIN_LOG_FILE"] = log_file

    if symlink_name:
        if os.path.lexists(symlink_path):
            os.unlink(symlink_path)
        os.makedirs(os.path.dirname(symlink_path), exist_ok=True)
        rel_target = os.path.relpath(log_file, start=os.path.dirname(symlink_path))
        os.symlink(rel_target, symlink_path)

    return log_file


def get_physical_gpu_id() -> str:
    """Get the physical GPU UUID for the current device.

    Returns:
        str: Physical GPU UUID, or "cpu" if CUDA is not available
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def prepare_runtime_environment() -> dict[str, str]:
    env_vars = {}

    if os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    if os.environ.get("PYTHONPATH"):
        env_vars["PYTHONPATH"] = os.environ["PYTHONPATH"]

    if os.environ.get("MODELSCOPE_CACHE"):
        env_vars["MODELSCOPE_CACHE"] = os.environ["MODELSCOPE_CACHE"]

    # Set log file path for Ray actors to use
    if "RAY_TRAIN_LOG_FILE" not in os.environ:
        log_file = ensure_run_log_file()
        env_vars["RAY_TRAIN_LOG_FILE"] = log_file
    else:
        env_vars["RAY_TRAIN_LOG_FILE"] = os.environ["RAY_TRAIN_LOG_FILE"]

    return env_vars


def initialize_ray():
    env_vars = prepare_runtime_environment()
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": env_vars})


def init_distributed_comm(backend: str = "nccl", use_deepspeed: bool = False):
    """
    Initialize distributed communication for PyTorch.

    This function checks if DeepSpeed is available and uses its initialization
    when use_deepspeed=True (required for sequence parallelism). Otherwise,
    it falls back to PyTorch's standard dist.init_process_group.

    Args:
        backend: Communication backend to use (default: "nccl")
        use_deepspeed: Whether to use DeepSpeed's initialization (required for sequence parallelism)

    Returns:
        bool: True if initialization succeeded, False otherwise
    """
    import torch.distributed as dist

    if dist.is_initialized():
        logger.info("Distributed communication already initialized")
        return True

    if use_deepspeed:
        try:
            import deepspeed

            logger.info("Initializing distributed communication with DeepSpeed")
            deepspeed.init_distributed()
            return True
        except ImportError:
            logger.warning(
                "DeepSpeed not available but use_deepspeed=True. "
                "Falling back to PyTorch dist.init_process_group. "
                "Note: This may cause issues with sequence parallelism."
            )
            # Fall through to standard initialization

    logger.info(f"Initializing distributed communication with backend={backend}")
    dist.init_process_group(backend=backend)
    return True
