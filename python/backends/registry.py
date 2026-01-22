from __future__ import annotations

from .deepspeed_strategy import DeepSpeedStrategy
from .native_strategy import NativeStrategy

_STRATEGIES = {
    "native": NativeStrategy,
    "deepspeed": DeepSpeedStrategy,
}


def get_backend_strategy(engine: str | None):
    if engine is None:
        engine = "native"
    engine = engine.lower()
    if engine not in _STRATEGIES:
        raise ValueError(
            f"Unsupported backend engine '{engine}'. "
            "Supported engines: native, deepspeed."
        )
    return _STRATEGIES[engine]
