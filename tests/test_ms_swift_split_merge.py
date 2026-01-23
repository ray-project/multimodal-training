import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.merge_checkpoint import _build_full_state_dict
from scripts.split_checkpoint import _get_text_prefix, _normalize_hf_state_dict, _split_with_mapping


class _DummyBridge:
    hf_layers_prefix = "model.layers"
    hf_state_dict_mapping = {"model.language_model.": "model."}


def test_ms_swift_split_merge_mapping_roundtrip():
    state_dict = {
        "model.visual.patch_embed.weight": torch.ones(1),
        "model.visual.blocks.0.attn.q_proj.weight": torch.ones(1),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        "model.layers.1.mlp.up_proj.weight": torch.ones(1),
        "model.embed_tokens.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
    }

    normalized = _normalize_hf_state_dict(state_dict, _DummyBridge.hf_state_dict_mapping)
    text_prefix = _get_text_prefix(_DummyBridge)
    vision_state, text_state = _split_with_mapping(
        normalized, module_mapping={"model.visual": "visual"}, text_prefix=text_prefix
    )
    full_state = _build_full_state_dict(vision_state, text_state, "model.visual", text_prefix)

    assert len(full_state) == len(normalized)
    assert set(full_state.keys()) == set(normalized.keys())

    assert "patch_embed.weight" in vision_state
    assert "blocks.0.attn.q_proj.weight" in vision_state
    assert "layers.0.self_attn.q_proj.weight" in text_state
    assert "embed_tokens.weight" in text_state
    assert "lm_head.weight" in text_state
