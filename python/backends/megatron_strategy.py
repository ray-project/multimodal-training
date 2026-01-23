from __future__ import annotations

import importlib.util

from .base import BackendStrategy


class MegatronStrategy(BackendStrategy):
    name = "megatron"

    def ensure_available(self):
        megatron_spec = importlib.util.find_spec("megatron")
        if megatron_spec is None:
            raise RuntimeError(
                "Megatron engine requested but Megatron-LM is not available. "
                "Install Megatron-LM or add it to PYTHONPATH."
            )

    def validate_parallelism(self, parallelism: str, component_name: str = "component"):
        if parallelism not in {"tensor", "none"}:
            raise ValueError(
                f"{component_name} backend 'megatron' only supports parallelism "
                f"'tensor' or 'none' (got '{parallelism}')."
            )
