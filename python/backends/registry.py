from __future__ import annotations

from .deepspeed_strategy import DeepSpeedStrategy
from .megatron_strategy import MegatronStrategy
from .native_strategy import NativeStrategy

_STRATEGIES = {
    "native": NativeStrategy,
    "deepspeed": DeepSpeedStrategy,
    "megatron": MegatronStrategy,
}


def get_backend_strategy(engine: str | None):
    if engine is None:
        engine = "native"
    engine = engine.lower()
    if engine not in _STRATEGIES:
        raise ValueError(
            f"Unsupported backend engine '{engine}'. "
            "Supported engines: native, deepspeed, megatron."
        )
    return _STRATEGIES[engine]
