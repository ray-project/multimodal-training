from __future__ import annotations

from abc import ABC, abstractmethod


class BackendStrategy(ABC):
    name = "base"

    def __init__(self, trainer, config: dict):
        self.trainer = trainer
        self.config = config

    @property
    def engine_config(self) -> dict:
        return self.config.get("engine_config", {})

    def get_config_value(self, key: str, default=None):
        if key in self.engine_config:
            return self.engine_config.get(key, default)
        return self.config.get(key, default)

    def ensure_available(self):
        """Check backend dependencies and raise on missing requirements."""
        return None

    @abstractmethod
    def validate_parallelism(self, parallelism: str, component_name: str = "component"):
        pass

    def is_deepspeed(self) -> bool:
        return False
