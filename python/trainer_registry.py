from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TrainerRegistration:
    trainer_cls: type
    init_builder: Callable[[dict], dict] | None = None


_REGISTRY: dict[tuple[str, str, str], TrainerRegistration] = {}


def register_trainer(
    component_type: str,
    engine: str,
    model_type: str,
    trainer_cls: type,
    init_builder: Callable[[dict], dict] | None = None,
) -> None:
    key = (_normalize_component(component_type), _normalize_engine(engine), _normalize_model_type(model_type))
    _REGISTRY[key] = TrainerRegistration(trainer_cls=trainer_cls, init_builder=init_builder)


def resolve_trainer(component_type: str, engine: str | None, model_type: str, config: dict | None = None):
    component = _normalize_component(component_type)
    if engine is None:
        raise ValueError(
            f"Missing engine for component '{component}'. Set {component}.engine in config to "
            "'native', 'deepspeed', or 'megatron'."
        )
    engine = _normalize_engine(engine)
    model_type = _normalize_model_type(model_type)

    key = (component, engine, model_type)
    registration = _REGISTRY.get(key)
    if registration is None:
        registration = _resolve_default_trainer(component, engine, model_type)

    if registration is None:
        supported = ", ".join(_list_supported_combinations(component_type=component, engine=engine))
        if not supported:
            supported = ", ".join(_list_supported_combinations())
        raise ValueError(
            f"Unsupported trainer combination: component='{component}', engine='{engine}', model_type='{model_type}'. "
            f"Supported combinations: {supported}. "
            "Register a custom trainer with register_trainer(...) to override."
        )

    init_kwargs = registration.init_builder(config) if registration.init_builder else {}
    return registration.trainer_cls, init_kwargs


def _resolve_default_trainer(component_type: str, engine: str, model_type: str) -> TrainerRegistration | None:
    if model_type == "qwen2_5_vl" and engine in {"native", "deepspeed", "megatron"}:
        if component_type == "vision":
            if engine == "megatron":
                from .ray.megatron_trainer import MegatronVisionTrainer

                return TrainerRegistration(trainer_cls=MegatronVisionTrainer)
            from .ray.vision import QwenVisionTrainer

            return TrainerRegistration(trainer_cls=QwenVisionTrainer)
        if component_type == "text":
            if engine == "megatron":
                from .ray.megatron_trainer import MegatronTextTrainer

                return TrainerRegistration(trainer_cls=MegatronTextTrainer)
            from .ray.text import QwenTextTrainer

            return TrainerRegistration(trainer_cls=QwenTextTrainer)
        return None

    if model_type == "qwen3_vl" and engine == "megatron":
        if component_type == "vision":
            from .ray.megatron_trainer import MegatronVisionTrainer

            return TrainerRegistration(trainer_cls=MegatronVisionTrainer)
        if component_type == "text":
            from .ray.megatron_trainer import MegatronTextTrainer

            return TrainerRegistration(trainer_cls=MegatronTextTrainer)
        return None

    return None


def _normalize_component(component_type: str) -> str:
    component = component_type.strip().lower()
    if component not in {"vision", "text"}:
        raise ValueError(f"Unknown component_type '{component_type}'. Expected 'vision' or 'text'.")
    return component


def _normalize_engine(engine: str) -> str:
    engine = engine.strip().lower()
    if engine not in {"native", "deepspeed", "megatron"}:
        raise ValueError(
            f"Unsupported engine '{engine}'. Supported engines: native, deepspeed, megatron."
        )
    return engine


def _normalize_model_type(model_type: str) -> str:
    return model_type.strip().lower()


def _list_supported_combinations(
    component_type: str | None = None, engine: str | None = None, model_type: str | None = None
) -> list[str]:
    defaults = {
        ("vision", "native", "qwen2_5_vl"),
        ("vision", "deepspeed", "qwen2_5_vl"),
        ("vision", "megatron", "qwen2_5_vl"),
        ("text", "native", "qwen2_5_vl"),
        ("text", "deepspeed", "qwen2_5_vl"),
        ("text", "megatron", "qwen2_5_vl"),
        ("vision", "megatron", "qwen3_vl"),
        ("text", "megatron", "qwen3_vl"),
    }
    combinations = set(_REGISTRY.keys()) | defaults

    def _match(value, target):
        return target is None or value == target

    filtered = [
        f"{component}/{eng}/{model}"
        for component, eng, model in sorted(combinations)
        if _match(component, component_type) and _match(eng, engine) and _match(model, model_type)
    ]
    return filtered
