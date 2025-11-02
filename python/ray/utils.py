import logging
import os

import ray

logger = logging.getLogger(__name__)


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

    # Set log file path for Ray actors to use
    if "RAY_TRAIN_LOG_FILE" not in os.environ:
        log_file = os.path.abspath("logs/train.log")
        os.environ["RAY_TRAIN_LOG_FILE"] = log_file
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
