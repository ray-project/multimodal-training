#!/usr/bin/env python3
"""
Compute the number of vision tokens for each token configuration.

The process is:
1. Images are resized to fixed_size x fixed_size where fixed_size = sqrt(max_pixels)
2. Patchify with patch_size (default 14x14)
3. Number of patches = (fixed_size / patch_size)^2
4. These patches become vision tokens BEFORE spatial merging
"""

import math

# Token configurations from scripts/generate_configs.py
TOKEN_CONFIGS = [
    ("1k_tokens", 200704, 200704),  # sqrt=448, 1024 vision tokens (closest to 1k)
    ("4k_tokens", 802816, 802816),  # sqrt=896, 4096 vision tokens (exact)
    ("8k_tokens", 1806336, 1806336),  # sqrt=1344, 9216 vision tokens (closest to 8k)
    ("16k_tokens", 3211264, 3211264),  # sqrt=1792, 16384 vision tokens (closest to 16k)
    ("32k_tokens", 5017600, 5017600),  # sqrt=2240, 25600 vision tokens (closest to 32k)
    ("64k_tokens", 12845056, 12845056),  # sqrt=3584, 65536 vision tokens (closest to 64k)
]

# Default Qwen2.5-VL vision config parameters
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2

print("=" * 80)
print("Vision Token Calculation (Before Spatial Merging)")
print("=" * 80)
print("\nParameters:")
print(f"  patch_size: {PATCH_SIZE}")
print(f"  spatial_merge_size: {SPATIAL_MERGE_SIZE}")
print(f"  temporal_patch_size: {TEMPORAL_PATCH_SIZE}")
print()

print(
    f"{'Config Name':<20} {'Max Pixels':<15} {'Image Size':<15} {'Patches/Dim':<15} {'Total Patches':<15} {'After Merge'}"
)
print("-" * 100)

for config_name, min_pixels, max_pixels in TOKEN_CONFIGS:
    # Step 1: Calculate fixed image size
    fixed_size = int(math.sqrt(max_pixels))

    # Step 2: Calculate number of patches per dimension
    patches_per_dim = fixed_size // PATCH_SIZE

    # Step 3: Total patches (vision tokens before merge)
    total_patches = patches_per_dim * patches_per_dim

    # Step 4: After spatial merging (for reference)
    after_merge = total_patches // (SPATIAL_MERGE_SIZE**2)

    print(
        f"{config_name:<20} {max_pixels:<15,} {fixed_size}x{fixed_size:<10} {patches_per_dim:<15} {total_patches:<15,} {after_merge:,}"
    )

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("\nThe number of vision tokens BEFORE spatial merging (input to vision model):")
for config_name, _, max_pixels in TOKEN_CONFIGS:
    fixed_size = int(math.sqrt(max_pixels))
    patches_per_dim = fixed_size // PATCH_SIZE
    total_patches = patches_per_dim * patches_per_dim
    print(f"  {config_name:<20} → {total_patches:>6,} tokens")

print("\nNote: These are the tokens that go into the vision encoder.")
print("      After the PatchMerger (spatial_merge_size=2), they become 4x smaller.")
