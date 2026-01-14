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
"""PyTorch Qwen3-VL model for disaggregated training.

This module provides standalone vision and text models extracted from Qwen3-VL
for disaggregated training with different parallelism strategies.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

# Import Qwen3-VL components from transformers
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel as HFQwen3VLTextModel,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionMLP,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)

from .configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig

logger = logging.get_logger(__name__)


@dataclass
class Qwen3VLVisionOutput:
    """Output from Qwen3-VL vision encoder with DeepStack features."""

    last_hidden_state: torch.Tensor
    deepstack_features: Optional[List[torch.Tensor]] = None


class Qwen3VisionTransformerPretrainedModel(Qwen3VLPreTrainedModel):
    """Standalone Qwen3-VL vision encoder for disaggregated training.

    This model wraps the Qwen3VLVisionModel and adds:
    - Support for sequence parallelism (via sp_group attribute)
    - DeepStack feature extraction at specified layer indices
    - Proper return format for disaggregated training
    """

    config_class = Qwen3VLVisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config: Qwen3VLVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        # Patch embedding - takes config object
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)

        # Learnable position embedding (same as HF implementation)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        # Rotary position embedding
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        # Transformer blocks - take config object
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])

        # Final merger (projects to text model hidden size) - no postshuffle norm
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

        # DeepStack mergers - one for each intermediate layer - with postshuffle norm
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in config.deepstack_visual_indexes
        ])

        self.gradient_checkpointing = False
        self.sp_group = None  # Set by trainer for sequence parallelism

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute interpolated position embeddings for variable grid sizes."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h.item())
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w.item())

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t.item(), 1)
            pos_embed = (
                pos_embed.view(t.item(), h.item() // merge_size, merge_size, w.item() // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings for the vision tokens."""
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        return_deepstack: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, Qwen3VLVisionOutput]:
        """Forward pass through vision encoder.

        Args:
            hidden_states: Input pixel values [seq_len, hidden_size] or [batch, C, T, H, W]
            grid_thw: Tensor of shape [num_images, 3] with (T, H, W) for each image
            return_deepstack: Whether to return DeepStack intermediate features

        Returns:
            If return_deepstack=True: Qwen3VLVisionOutput with embeddings and DeepStack features
            If return_deepstack=False: Just the final embeddings tensor
        """
        # Patch embedding
        hidden_states = self.patch_embed(hidden_states)

        # Add interpolated position embeddings (key step from HF implementation)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # Compute rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Reshape hidden states and rotary embeddings
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Compute cumulative sequence lengths for flash attention
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Store intermediate features for DeepStack
        deepstack_features = [] if return_deepstack else None
        deepstack_idx = 0

        # Forward through transformer blocks
        for layer_idx, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Capture DeepStack features at specified layers
            if return_deepstack and layer_idx in self.deepstack_visual_indexes:
                # Apply corresponding merger to get DeepStack features
                ds_features = self.deepstack_merger_list[deepstack_idx](hidden_states)
                deepstack_features.append(ds_features)
                deepstack_idx += 1

        # Final merger
        hidden_states = self.merger(hidden_states)

        if return_deepstack:
            return Qwen3VLVisionOutput(
                last_hidden_state=hidden_states,
                deepstack_features=deepstack_features,
            )
        return hidden_states


class Qwen3VLTextModel(HFQwen3VLTextModel):
    """Qwen3-VL text model for disaggregated training.

    This class extends the HuggingFace Qwen3VLTextModel to work with
    disaggregated training where vision and text models run in separate
    Ray actors.
    """

    config_class = Qwen3VLTextConfig

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)

    @classmethod
    def _from_config(cls, config: Qwen3VLTextConfig, **kwargs):
        """Create model from config without loading pretrained weights."""
        return cls(config)


class Qwen3VLModelOutput:
    """Output from Qwen3-VL combined model."""

    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        hidden_states: Optional[Tuple[torch.Tensor]] = None,
        attentions: Optional[Tuple[torch.Tensor]] = None,
        rope_deltas: Optional[torch.Tensor] = None,
    ):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.rope_deltas = rope_deltas


__all__ = [
    "Qwen3VLConfig",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
    "Qwen3VisionTransformerPretrainedModel",
    "Qwen3VLTextModel",
    "Qwen3VLVisionOutput",
    "Qwen3VLForConditionalGeneration",
]
