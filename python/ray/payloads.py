from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VisionOutputs:
    """Stable vision -> text payload wrapper for Ray transport."""

    embeddings: Any
    attention_mask: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vision_embeddings": self.embeddings,
            "vision_attention_mask": self.attention_mask,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class TextBackwardOutputs:
    """Stable text -> vision payload wrapper for Ray transport."""

    grad: Any
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"grad": self.grad, "meta": dict(self.meta)}


def normalize_vision_outputs(payload: Any) -> VisionOutputs:
    """Normalize legacy dicts or dataclass payloads into VisionOutputs."""
    if isinstance(payload, VisionOutputs):
        return payload

    if isinstance(payload, dict):
        if "vision_embeddings" not in payload:
            raise RuntimeError("Vision payload dict missing 'vision_embeddings'.")
        meta = dict(payload.get("meta") or {})
        for key in ("sample_index", "iteration", "forward_time_ms"):
            if key in payload and key not in meta:
                meta[key] = payload[key]
        return VisionOutputs(
            embeddings=payload.get("vision_embeddings"),
            attention_mask=payload.get("vision_attention_mask"),
            meta=meta,
        )

    raise RuntimeError(f"Unsupported vision payload type: {type(payload)}")


def normalize_text_backward_outputs(payload: Any) -> TextBackwardOutputs:
    """Normalize legacy dicts or dataclass payloads into TextBackwardOutputs."""
    if isinstance(payload, TextBackwardOutputs):
        return payload

    if isinstance(payload, dict):
        if "grad" not in payload:
            raise RuntimeError("Text backward payload dict missing 'grad'.")
        meta = dict(payload.get("meta") or {})
        if "backward_time_ms" in payload and "backward_time_ms" not in meta:
            meta["backward_time_ms"] = payload["backward_time_ms"]
        return TextBackwardOutputs(grad=payload.get("grad"), meta=meta)

    raise RuntimeError(f"Unsupported text backward payload type: {type(payload)}")
