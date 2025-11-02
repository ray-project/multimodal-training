import logging

import ray
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from .actor import RayActor, TorchDistributedConfig

logger = logging.getLogger(__name__)


class ActorGroup:
    actors: list[RayActor]

    def __init__(
        self,
        config,
        actor_cls: type[RayActor],
        num_actors: int,
        num_cpus: int = 6,
        num_gpus: int = 1,
        collocate: bool = False,
        placement_group_handle=None,
    ):
        """Initialize ActorGroup.

        Args:
            config: Configuration dictionary
            actor_cls: Actor class to instantiate
            num_actors: Number of actors to create
            num_cpus: CPUs per actor
            num_gpus: GPUs per actor (fractional if collocating)
            collocate: Whether actors should be collocated with another group on same GPUs
            placement_group_handle: Existing placement group to use (for collocation)
        """
        self.num_actors = num_actors
        self.collocate = collocate

        # Calculate GPU allocation per actor
        # When collocating, use fractional GPUs (e.g., 0.5 per actor)
        gpus_per_actor = num_gpus / 2 if collocate else num_gpus

        logger.info(f"Creating ActorGroup with {num_actors} actors")
        logger.info(f"Collocation: {collocate}, GPUs per actor: {gpus_per_actor}")

        # Support both plain Python classes and already-remote Ray actor classes.
        if hasattr(actor_cls, "remote"):
            remote_actor_cls = actor_cls
        else:
            remote_actor_cls = ray.remote(num_cpus=num_cpus, num_gpus=gpus_per_actor)(actor_cls)

        # Create or reuse placement group for collocation
        if collocate:
            if placement_group_handle is None:
                # Create new placement group
                bundles = [{"GPU": num_gpus, "CPU": num_cpus} for _ in range(num_actors)]
                self.placement_group = placement_group(bundles, strategy="PACK")
                ray.get(self.placement_group.ready())
                logger.info(f"Created new placement group for {num_actors} actors")
            else:
                # Reuse existing placement group (for second model group)
                self.placement_group = placement_group_handle
                logger.info("Reusing existing placement group")
        else:
            self.placement_group = None

        # Create actors
        self._actors = []
        for i in range(num_actors):
            if self.placement_group is not None:
                # Use placement group scheduling
                actor = remote_actor_cls.options(
                    num_cpus=num_cpus / 2,  # Share CPUs too
                    num_gpus=gpus_per_actor,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=i,
                    ),
                ).remote(config, i)
            else:
                # Standard scheduling
                actor = remote_actor_cls.options(
                    num_cpus=num_cpus,
                    num_gpus=gpus_per_actor,
                ).remote(config, i)
            self._actors.append(actor)

        self._setup_process_group()

    def _setup_process_group(
        self,
    ):
        """
        Sets up distributed training configuration for all actors in the group.

        This method:
        1. Collects network information (node IP addresses, free ports) from all actors
        2. Creates TorchDistributedConfig objects for each actor with proper rank assignment
        3. Configures the distributed process group for SPMD (Single Program Multiple Data) training
        4. Sets up environment variables on each actor for distributed communication

        The configuration uses NCCL backend and designates the first actor as the master node.
        """
        node_addresses = ray.get([actor.get_node_ip_addr.remote() for actor in self._actors])  # type: ignore
        free_ports = ray.get([actor.get_free_port.remote() for actor in self._actors])  # type: ignore
        ranks = ray.get([actor.get_rank.remote() for actor in self._actors])  # type: ignore

        assert len(node_addresses) == len(free_ports), "Number of node addresses and free ports must match"

        # resolve the common args irrespective of worker info
        world_size: int = len(node_addresses)
        backend: str = "nccl"  # always use nccl - this will break local dev

        # resolve common args based on rank 0
        master_addr: str = node_addresses[0]
        master_port: str = str(free_ports[0])

        # make distributed configs
        distributed_configs: list[TorchDistributedConfig] = []

        for rank in ranks:
            # When collocating, each actor only sees one GPU (cuda:0) due to Ray's CUDA_VISIBLE_DEVICES
            # so local_rank should always be 0. Otherwise, local_rank equals the global rank.
            local_rank = 0 if self.collocate else rank

            distributed_configs.append(
                TorchDistributedConfig(
                    backend=backend,
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    local_world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                )
            )

        refs = [actor.setup_env_variables.remote(distributed_configs[i]) for i, actor in enumerate(self._actors)]
        assert all(ray.get(refs)), "Failed to setup process group"

    def _execute_single_actor(self, actor: RayActor, method_name: str, *args, **kwargs):
        """Execute a method on a single actor remotely.

        Args:
            actor: The actor handle
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """

        remote_call = getattr(actor, method_name)
        return remote_call.remote(*args, **kwargs)

    def _should_distribute_arguments(self, args: tuple, kwargs: dict) -> bool:
        """Check if arguments should be distributed across actors.

        Returns True if all args and kwargs are lists with length matching actor count.
        """
        length = len(self._actors)

        # Check if all positional arguments are lists with correct length
        args_are_distributable = all(isinstance(arg, list) and len(arg) == length for arg in args)

        # Check if all keyword arguments are lists with correct length
        kwargs_are_distributable = all(isinstance(value, list) and len(value) == length for value in kwargs.values())

        return args_are_distributable and kwargs_are_distributable

    def _distribute_arguments(self, args: tuple, kwargs: dict, actor_index: int):
        """Distribute arguments for a specific actor index.

        Args:
            args: Original positional arguments (list of lists)
            kwargs: Original keyword arguments (dict of lists)
            actor_index: Index of the actor to get arguments for

        Returns:
            Tuple of (distributed_args, distributed_kwargs)
        """
        distributed_args = tuple(arg[actor_index] for arg in args)
        distributed_kwargs = {k: v[actor_index] for k, v in kwargs.items()}
        return distributed_args, distributed_kwargs

    def execute_all_async(self, method_name: str, *args, **kwargs):
        """Execute a method on all actors asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        if self._should_distribute_arguments(args, kwargs):
            # Distribute arguments across actors
            results = []
            for i, actor in enumerate(self._actors):
                distributed_args, distributed_kwargs = self._distribute_arguments(args, kwargs, i)
                result = self._execute_single_actor(actor, method_name, *distributed_args, **distributed_kwargs)
                results.append(result)
            return results
        else:
            # Use same arguments for all actors
            return [self._execute_single_actor(actor, method_name, *args, **kwargs) for actor in self._actors]

    def execute_all(self, method_name: str, *args, **kwargs):
        """Execute a method on all actors synchronously and wait for completion.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of results from all actors
        """
        refs = self.execute_all_async(method_name, *args, **kwargs)
        return ray.get(refs)
