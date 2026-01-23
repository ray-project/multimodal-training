"""Smoke tests for ragged sequence parallel utilities."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig  # noqa: E402
from python.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel  # noqa: E402
from python.models.qwen2_5_vl.sequence_parallel_utils import (  # noqa: E402
    build_local_rope,
    pad_pack_to_batch,
    rebuild_local_cu_seqlens,
    split_pack_by_sp,
    unpad_batch_to_pack,
)

pytestmark = pytest.mark.cpu_only


class TestRaggedSPSmokeTest:
    """Smoke tests for ragged sequence parallel without distributed setup."""

    def test_vision_transformer_forward_without_sp(self):
        """Test VisionTransformer forward pass without sequence parallelism."""
        # Create a minimal config
        config = Qwen2_5_VLVisionConfig(
            hidden_size=256,
            num_heads=8,
            depth=2,
            patch_size=14,
            in_channels=3,
            out_hidden_size=256,
            spatial_merge_size=2,
            window_size=28,
            fullatt_block_indexes=[1],  # Make second layer full attention
            sequence_parallel=False,  # Disable SP for this test
            _attn_implementation="eager",
        )

        # Create model
        model = Qwen2_5_VisionTransformerPretrainedModel(config)
        model.eval()

        # Create dummy input with variable lengths
        # Simulating 2 images with different sizes
        # Image 1: 224x224 -> 16x16 patches -> 256 tokens
        # Image 2: 448x224 -> 32x16 patches -> 512 tokens
        # After spatial merge (2x2): 64 and 128 tokens
        # After window reordering: still 64 and 128 tokens

        # For simplicity, create flattened patch inputs expected by patch_embed
        def make_patch_stream(grid, in_ch, temp_patch, patch_sz):
            patches = []
            for t, h, w in grid.tolist():
                num_tokens = int(t * h * w)
                patches.append(
                    torch.randn(
                        num_tokens,
                        in_ch * temp_patch * patch_sz * patch_sz,
                    )
                )
            return torch.cat(patches, dim=0)

        # grid_thw: [num_images, 3] where 3 = (temporal, height, width)
        # For images: temporal=1, height=patches_h, width=patches_w
        # 64 tokens = 1 * 8 * 8 (after merge), so patches = 16x16
        # 128 tokens = 1 * 8 * 16 (after merge), so patches = 16x32
        grid_thw = torch.tensor(
            [
                [1, 16, 16],  # Image 1
                [1, 16, 32],  # Image 2
            ],
            dtype=torch.int64,
        )

        hidden_states = make_patch_stream(
            grid_thw,
            in_ch=config.in_channels,
            temp_patch=config.temporal_patch_size,
            patch_sz=config.patch_size,
        )
        total_tokens = hidden_states.shape[0]
        expected_tokens = total_tokens // (config.spatial_merge_size**2)

        # Forward pass
        try:
            output = model(hidden_states, grid_thw)
            assert (
                output.shape[0] == expected_tokens
            ), f"Expected {expected_tokens} tokens after spatial merge, got {output.shape[0]}"
            assert output.shape[1] == config.out_hidden_size
            print(f"✓ Forward pass successful: output shape = {output.shape}")
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")

    def test_ragged_utilities_integration(self):
        """Test integration of ragged utilities end-to-end."""
        batch_size = 3
        global_lengths = [50, 100, 75]
        hidden_size = 128
        emb_dim = 64
        sp_size = 2

        # Create global pack and embeddings
        total_len = sum(global_lengths)
        global_pack = torch.randn(total_len, hidden_size)
        global_cos = torch.randn(total_len, emb_dim)
        global_sin = torch.randn(total_len, emb_dim)

        # Split for rank 0
        pack_r0, meta_r0 = split_pack_by_sp(global_pack, global_lengths, sp_rank=0, sp_size=sp_size)

        # Rebuild cu_seqlens
        cu_seqlens_r0 = rebuild_local_cu_seqlens(meta_r0, use_windows=False)
        assert cu_seqlens_r0.dtype == torch.int32
        assert cu_seqlens_r0[-1].item() == pack_r0.shape[0]

        # Build rank-local RoPE
        cos_r0, sin_r0, _ = build_local_rope(meta_r0, global_cos, global_sin)
        assert cos_r0.shape[0] == pack_r0.shape[0]
        assert sin_r0.shape[0] == pack_r0.shape[0]

        # Test padding and unpadding
        padded, mask = pad_pack_to_batch(pack_r0, meta_r0)
        assert padded.shape[0] == batch_size
        assert padded.shape[1] == meta_r0.max_local_len

        # Unpad should recover original
        recovered, _ = unpad_batch_to_pack(padded, mask)
        torch.testing.assert_close(recovered, pack_r0)

        print("✓ Ragged utilities integration test passed")
        print(f"  - Split: {total_len} -> {pack_r0.shape[0]} (rank 0)")
        print(f"  - cu_seqlens: {cu_seqlens_r0.tolist()}")
        print(f"  - Padded: {padded.shape}, mask: {mask.sum().item()} real tokens")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
