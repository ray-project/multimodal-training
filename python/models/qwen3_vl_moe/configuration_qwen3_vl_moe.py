# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3-VL-MoE configuration for disaggregated training."""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class Qwen3VLMoeVisionConfig(PretrainedConfig):
    """Configuration for Qwen3-VL-MoE vision encoder.

    Same architecture as Qwen3-VL (non-MoE) vision encoder.
    Extended with sequence_parallel flag for disaggregated training.
    """

    model_type = "qwen3_vl_moe"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=2304,  # 48*48 grid for position interpolation
        out_hidden_size=2048,  # Output to MoE text decoder (smaller than Qwen3-VL-8B's 3584)
        deepstack_visual_indexes=[8, 16, 24],
        initializer_range=0.02,
        sequence_parallel=False,  # For disaggregated training with DeepSpeed SP
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.num_position_embeddings = num_position_embeddings
        self.out_hidden_size = out_hidden_size
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.initializer_range = initializer_range
        self.sequence_parallel = sequence_parallel


class Qwen3VLMoeTextConfig(PretrainedConfig):
    r"""
    Configuration for Qwen3-VL-MoE text decoder with Mixture of Experts.

    This is adapted from transformers Qwen3VLMoeTextConfig with TP/PP plans for
    distributed training.
    """

    model_type = "qwen3_vl_moe_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        # MoE layers - expert parallelism (each expert is parallel)
        "layers.*.block_sparse_moe.experts.*.gate_proj": "colwise",
        "layers.*.block_sparse_moe.experts.*.up_proj": "colwise",
        "layers.*.block_sparse_moe.experts.*.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        attention_bias=False,
        attention_dropout=0.0,
        rope_scaling=None,
        image_token_id=None,
        video_token_id=None,
        # MoE-specific parameters
        num_experts=128,
        num_experts_per_tok=8,
        expert_layer_offset=0,  # Offset for expert layer indices
        norm_topk_prob=False,  # Normalize top-k probabilities
        router_aux_loss_coef=0.01,  # Auxiliary loss coefficient for load balancing
        **kwargs,
    ):
        # Default rope_scaling for Qwen3-VL-MoE (required for rotary embedding)
        if rope_scaling is None:
            rope_scaling = {
                "rope_type": "default",
                "mrope_section": [16, 12, 12],  # Different from Qwen3-VL-8B
            }
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        # BC: handle rope_scaling type -> rope_type
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        # MoE-specific parameters
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_layer_offset = expert_layer_offset
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3VLMoeConfig(PretrainedConfig):
    r"""
    Combined configuration for Qwen3-VL-MoE model.

    This configuration class stores the configuration for both the vision encoder
    and text decoder components of Qwen3-VL-MoE.
    """

    model_type = "qwen3_vl_moe"
    sub_configs = {"vision_config": Qwen3VLMoeVisionConfig, "text_config": Qwen3VLMoeTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(**kwargs)


__all__ = ["Qwen3VLMoeConfig", "Qwen3VLMoeTextConfig", "Qwen3VLMoeVisionConfig"]
