import os
from dataclasses import dataclass

import ray
from ray.air._internal.util import find_free_port

from .logger import setup_logging
from .utils import get_physical_gpu_id


@dataclass
class TorchDistributedConfig:
    backend: str
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    master_addr: str
    master_port: str


class RayActor:
    rank: int

    def __init__(self, rank: int):
        # Explicitly set up logging in this actor process
        setup_logging()
        self.rank = rank

    def setup_env_variables(
        self,
        distributed_config: TorchDistributedConfig,
    ) -> bool:
        """
        Mimics torchrun and setups up the process group. After, builds the device mesh.
        """
        # init distributed uses env variables to initialize the process group
        # we manually set these for each. We should just pass this to init_distributed,
        # but this keeps compat with the rest of the codebase
        os.environ["LOCAL_RANK"] = str(distributed_config.local_rank)
        os.environ["WORLD_SIZE"] = str(distributed_config.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["MASTER_ADDR"] = distributed_config.master_addr
        os.environ["MASTER_PORT"] = distributed_config.master_port
        return True

    def get_node_ip_addr(self) -> str:
        return ray.util.get_node_ip_address()

    def get_free_port(self) -> int:
        return find_free_port()

    def get_rank(self) -> int:
        return self.rank

    def get_physical_gpu_id(self) -> str:
        """Get the physical GPU UUID for this actor's device.

        Returns:
            str: Physical GPU UUID, or "cpu" if CUDA is not available
        """
        return get_physical_gpu_id()
