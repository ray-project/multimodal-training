from __future__ import annotations

from .base import BackendStrategy


class DeepSpeedStrategy(BackendStrategy):
    name = "deepspeed"

    def validate_parallelism(self, parallelism: str, component_name: str = "component"):
        if parallelism not in {"sequence", "deepspeed", "autotp"}:
            raise ValueError(
                f"{component_name} backend 'deepspeed' only supports parallelism "
                f"'sequence', 'deepspeed', or 'autotp' (got '{parallelism}')."
            )

    def is_deepspeed(self) -> bool:
        return True

    def initialize_engine(
        self,
        model,
        params,
        config,
        torch_dtype,
        mpu=None,
        tensor_parallel_config=None,
        optimizer=None,
    ):
        return self.trainer._initialize_deepspeed(
            model=model,
            params=params,
            config=config,
            torch_dtype=torch_dtype,
            mpu=mpu,
            tensor_parallel_config=tensor_parallel_config,
            optimizer=optimizer,
        )
