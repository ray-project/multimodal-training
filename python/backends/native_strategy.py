from __future__ import annotations

from .base import BackendStrategy


class NativeStrategy(BackendStrategy):
    name = "native"

    def validate_parallelism(self, parallelism: str, component_name: str = "component"):
        if parallelism in {"sequence", "deepspeed", "autotp"}:
            raise ValueError(
                f"{component_name} backend 'native' does not support parallelism '{parallelism}'. "
                "Use engine='deepspeed' for DeepSpeed-based modes."
            )
