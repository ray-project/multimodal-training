"""Utilities for ragged (variable-length) sequence parallelism in the Qwen-VL vision encoder.

This module centralises the data-shuffling helpers that VisionTransformer/VisionAttention
use to support ragged batches under sequence parallelism. The helpers work with the
"ragged pack" representation (concatenated tokens of length ``sum_b S_b``) as well as
its padded ``[B, S_max, H]`` form required by collective communication.

Key invariants:
- All metadata (`RaggedSequenceMeta`) reflects the current rank-local token layout.
- ``cu_seqlens``/``cu_window_seqlens`` must be rebuilt whenever the layout changes and
  always use ``torch.int32`` (Flash Attention 2 requirement).
- Collectives (all-gather/all-to-all/reduce-scatter) work on uniformly padded buffers.
- Pad/unpad round-trips are exact: ``unpad_batch_to_pack(*pad_pack_to_batch(x)) == x``.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class RaggedSequenceMeta:
    """Metadata describing the rank-local ragged pack layout."""

    batch_size: int
    local_lengths: List[int]
    global_lengths: List[int]
    sp_rank: int
    sp_size: int
    device: torch.device
    global_sample_offsets: List[int]
    sample_global_slices: List[Tuple[int, int]]
    local_token_offsets: List[Tuple[int, int]]
    window_lengths_per_sample: Optional[List[List[int]]]
    window_index_ranges: Optional[List[Tuple[int, int]]]

    def __post_init__(self) -> None:
        self.total_local_len = sum(self.local_lengths)
        self.max_local_len = max(self.local_lengths) if self.local_lengths else 0
        self.min_local_len = min(self.local_lengths) if self.local_lengths else 0
        self.requires_dummy_pad = self.total_local_len == 0

    def clone_with(
        self,
        *,
        local_lengths: Optional[List[int]] = None,
        sample_global_slices: Optional[List[Tuple[int, int]]] = None,
        local_token_offsets: Optional[List[Tuple[int, int]]] = None,
        window_index_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> "RaggedSequenceMeta":
        """Create a new meta object with updated token/window layout."""

        new_local_lengths = local_lengths if local_lengths is not None else list(self.local_lengths)

        if local_token_offsets is None:
            offset = 0
            new_local_offsets: List[Tuple[int, int]] = []
            for length in new_local_lengths:
                new_local_offsets.append((offset, offset + length))
                offset += length
        else:
            new_local_offsets = list(local_token_offsets)

        new_sample_slices = (
            list(sample_global_slices) if sample_global_slices is not None else list(self.sample_global_slices)
        )

        if window_index_ranges is None and self.window_index_ranges is not None:
            new_window_ranges = list(self.window_index_ranges)
        else:
            new_window_ranges = list(window_index_ranges) if window_index_ranges is not None else None

        return RaggedSequenceMeta(
            batch_size=self.batch_size,
            local_lengths=new_local_lengths,
            global_lengths=list(self.global_lengths),
            sp_rank=self.sp_rank,
            sp_size=self.sp_size,
            device=self.device,
            global_sample_offsets=list(self.global_sample_offsets),
            sample_global_slices=new_sample_slices,
            local_token_offsets=new_local_offsets,
            window_lengths_per_sample=self.window_lengths_per_sample,
            window_index_ranges=new_window_ranges,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            "RaggedSequenceMeta("
            f"B={self.batch_size}, "
            f"local={self.local_lengths}, "
            f"rank={self.sp_rank}/{self.sp_size}, "
            f"total={self.total_local_len})"
        )


def _compute_token_cut_positions(sample_len: int, sp_size: int) -> List[int]:
    """Compute cumulative token cut positions (length sp_size+1)."""

    base = sample_len // sp_size
    remainder = sample_len % sp_size
    cuts = [0]
    for rank in range(sp_size):
        increment = base + (1 if rank < remainder else 0)
        cuts.append(cuts[-1] + increment)
    return cuts


def _compute_window_cut_positions(
    sample_len: int,
    sp_size: int,
    window_lengths: List[int],
) -> Tuple[List[int], List[int]]:
    """Compute token-aligned cut positions that respect window boundaries."""

    prefix = [0]
    for length in window_lengths:
        prefix.append(prefix[-1] + length)

    if prefix[-1] != sample_len:
        raise ValueError("Window lengths do not sum to sample length: " f"{prefix[-1]} vs expected {sample_len}")

    cut_positions = [0]
    cut_indices = [0]
    last_idx = 0
    for step in range(1, sp_size):
        remaining = sp_size - step
        desired = (sample_len * step) // sp_size
        idx = bisect_left(prefix, desired, lo=last_idx)
        idx = max(idx, last_idx)
        max_idx = len(prefix) - remaining
        idx = min(idx, max_idx)
        cut_positions.append(prefix[idx])
        cut_indices.append(idx)
        last_idx = idx

    cut_positions.append(prefix[-1])
    cut_indices.append(len(prefix) - 1)
    return cut_positions, cut_indices


def split_pack_by_sp(
    tokens: torch.Tensor,
    lengths_per_sample: List[int],
    sp_rank: int,
    sp_size: int,
    cu_window_seqlens: Optional[torch.Tensor] = None,
    debug: bool = False,
) -> Tuple[torch.Tensor, RaggedSequenceMeta]:
    """Split the global pack across SP ranks while keeping per-sample order."""

    batch_size = len(lengths_per_sample)
    device = tokens.device

    global_sample_offsets: List[int] = []
    offset = 0
    for sample_len in lengths_per_sample:
        global_sample_offsets.append(offset)
        offset += sample_len

    window_lengths_per_sample: Optional[List[List[int]]] = None
    if cu_window_seqlens is not None:
        window_lengths = torch.diff(cu_window_seqlens, dim=0).tolist()
        window_lengths_per_sample = []
        cursor = 0
        for sample_idx, sample_len in enumerate(lengths_per_sample):
            consumed = 0
            sample_windows: List[int] = []
            while consumed < sample_len and cursor < len(window_lengths):
                wlen = int(window_lengths[cursor])
                sample_windows.append(wlen)
                consumed += wlen
                cursor += 1
            if consumed != sample_len:
                raise ValueError(f"Window lengths do not cover sample {sample_idx}: " f"{consumed} vs {sample_len}")
            window_lengths_per_sample.append(sample_windows)
        if cursor != len(window_lengths):
            raise ValueError("Unused window lengths remain after per-sample split")

    local_lengths: List[int] = []
    sample_global_slices: List[Tuple[int, int]] = []
    local_token_offsets: List[Tuple[int, int]] = []
    window_index_ranges: Optional[List[Tuple[int, int]]] = (
        [None] * batch_size if window_lengths_per_sample is None else []
    )

    # Collect slices to materialise the rank-local tensor
    rank_slices: List[Tuple[int, int]] = []
    local_offset = 0
    for sample_idx, sample_len in enumerate(lengths_per_sample):
        sample_start = global_sample_offsets[sample_idx]

        if window_lengths_per_sample is not None:
            cut_positions, cut_indices = _compute_window_cut_positions(
                sample_len,
                sp_size,
                window_lengths_per_sample[sample_idx],
            )
        else:
            cut_positions = _compute_token_cut_positions(sample_len, sp_size)
            cut_indices = None

        start_rel = cut_positions[sp_rank]
        end_rel = cut_positions[sp_rank + 1]
        local_len = end_rel - start_rel

        local_lengths.append(local_len)

        global_start = sample_start + start_rel
        global_end = sample_start + end_rel
        sample_global_slices.append((global_start, global_end))

        local_token_offsets.append((local_offset, local_offset + local_len))
        local_offset += local_len

        if local_len > 0:
            rank_slices.append((global_start, global_end))

        if window_lengths_per_sample is not None and window_index_ranges is not None:
            assert cut_indices is not None
            window_index_ranges.append((cut_indices[sp_rank], cut_indices[sp_rank + 1]))

    if window_lengths_per_sample is None:
        window_lengths_list = None
        window_index_ranges = None
    else:
        window_lengths_list = window_lengths_per_sample
        assert window_index_ranges is not None

    if not rank_slices:
        hidden_size = tokens.shape[-1]
        pack_rank = torch.empty(0, hidden_size, device=device, dtype=tokens.dtype)
    else:
        pack_rank = torch.cat([tokens[start:end] for start, end in rank_slices], dim=0)

    meta = RaggedSequenceMeta(
        batch_size=batch_size,
        local_lengths=local_lengths,
        global_lengths=lengths_per_sample,
        sp_rank=sp_rank,
        sp_size=sp_size,
        device=device,
        global_sample_offsets=global_sample_offsets,
        sample_global_slices=sample_global_slices,
        local_token_offsets=local_token_offsets,
        window_lengths_per_sample=window_lengths_list,
        window_index_ranges=window_index_ranges,
    )

    if debug:
        mean_len = meta.total_local_len / meta.batch_size if meta.batch_size > 0 else 0
        padding_ratio = meta.max_local_len / mean_len if mean_len > 0 else 0.0
        logger.info(
            "[Rank %d] split_pack_by_sp: global=%d -> local=%d, " "local_lens=%s, padding_ratio=%.3f",
            sp_rank,
            tokens.shape[0],
            pack_rank.shape[0],
            local_lengths,
            padding_ratio,
        )

    return pack_rank, meta


def pad_pack_to_batch(
    pack_rank: torch.Tensor,
    meta: RaggedSequenceMeta,
    pad_value: float = 0.0,
    pad_to: Optional[int] = None,
    enforce_min_pad: int = 0,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """Convert the ragged pack to ``[B, S_max, H]`` for collectives."""

    batch_size = meta.batch_size
    hidden_size = pack_rank.shape[-1] if pack_rank.ndim > 1 else (pack_rank.shape[-1] if meta.total_local_len else 0)
    requested = meta.max_local_len if pad_to is None else pad_to
    if requested < meta.max_local_len:
        raise ValueError(f"Requested pad length {requested} < local max {meta.max_local_len}")
    max_len = max(requested, enforce_min_pad)

    padded = pack_rank.new_full((batch_size, max_len, hidden_size), pad_value)
    mask = torch.zeros(batch_size, max_len, device=pack_rank.device, dtype=torch.bool)

    offset = 0
    for sample_idx, local_len in enumerate(meta.local_lengths):
        if local_len == 0:
            continue
        padded[sample_idx, :local_len] = pack_rank[offset : offset + local_len]
        mask[sample_idx, :local_len] = True
        offset += local_len

    return padded, mask


def unpad_batch_to_pack(
    padded: torch.Tensor,
    mask: torch.BoolTensor,
    *,
    return_lengths: bool = False,
) -> Tuple[torch.Tensor, Optional[List[int]]]:
    """Convert padded ``[B, S_max, H]`` back to the ragged pack representation."""

    device = padded.device
    hidden_size = padded.shape[-1]
    lengths = mask.sum(dim=1).tolist()

    slices = [padded[idx, :length] for idx, length in enumerate(lengths) if length > 0]
    if slices:
        pack_rank = torch.cat(slices, dim=0)
    else:
        pack_rank = torch.empty(0, hidden_size, device=device, dtype=padded.dtype)

    if return_lengths:
        return pack_rank, lengths
    return pack_rank, None


def rebuild_local_cu_seqlens(
    meta: RaggedSequenceMeta,
    *,
    use_windows: bool = False,
) -> torch.Tensor:
    """Create rank-local cumulative sequence lengths matching ``meta``."""

    device = meta.device

    if not use_windows:
        prefix = [0]
        total = 0
        for length in meta.local_lengths:
            total += length
            prefix.append(total)
        return torch.tensor(prefix, device=device, dtype=torch.int32)

    if meta.window_lengths_per_sample is None or meta.window_index_ranges is None:
        raise ValueError("Window metadata not available in RaggedSequenceMeta")

    cu = [0]
    total = 0
    for sample_idx, window_lengths in enumerate(meta.window_lengths_per_sample):
        window_range = meta.window_index_ranges[sample_idx]
        if window_range is None:
            continue
        start_idx, end_idx = window_range
        for length in window_lengths[start_idx:end_idx]:
            total += length
            cu.append(total)

    return torch.tensor(cu, device=device, dtype=torch.int32)


def build_local_rope(
    meta: RaggedSequenceMeta,
    global_pos_emb_cos: torch.Tensor,
    global_pos_emb_sin: torch.Tensor,
    pad_to: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.BoolTensor]]:
    """Slice global RoPE embeddings to match the rank-local token order."""

    cos_slices = []
    sin_slices = []
    for start, end in meta.sample_global_slices:
        if end > start:
            cos_slices.append(global_pos_emb_cos[start:end])
            sin_slices.append(global_pos_emb_sin[start:end])

    if cos_slices:
        cos_rank = torch.cat(cos_slices, dim=0)
        sin_rank = torch.cat(sin_slices, dim=0)
    else:
        emb_dim = global_pos_emb_cos.shape[-1]
        cos_rank = torch.empty(0, emb_dim, device=meta.device, dtype=global_pos_emb_cos.dtype)
        sin_rank = torch.empty(0, emb_dim, device=meta.device, dtype=global_pos_emb_sin.dtype)

    if pad_to is None:
        return cos_rank, sin_rank, None

    emb_dim = cos_rank.shape[-1] if cos_rank.numel() else global_pos_emb_cos.shape[-1]
    cos_padded = cos_rank.new_zeros((meta.batch_size, pad_to, emb_dim))
    sin_padded = sin_rank.new_zeros((meta.batch_size, pad_to, emb_dim))
    mask = torch.zeros(meta.batch_size, pad_to, device=meta.device, dtype=torch.bool)

    offset = 0
    for idx, length in enumerate(meta.local_lengths):
        if length == 0:
            continue
        cos_padded[idx, :length] = cos_rank[offset : offset + length]
        sin_padded[idx, :length] = sin_rank[offset : offset + length]
        mask[idx, :length] = True
        offset += length

    return cos_padded, sin_padded, mask


def log_ragged_stats(meta: RaggedSequenceMeta, prefix: str = "") -> None:
    """Log basic distribution statistics for debugging."""

    if meta.batch_size == 0:
        return
    mean_len = meta.total_local_len / meta.batch_size if meta.batch_size > 0 else 0.0
    padding_ratio = meta.max_local_len / mean_len if mean_len > 0 else 0.0
    logger.info(
        "%s[Rank %d] Ragged stats: B=%d lengths=%s total=%d padding_ratio=%.3f",
        prefix,
        meta.sp_rank,
        meta.batch_size,
        meta.local_lengths,
        meta.total_local_len,
        padding_ratio,
    )


def compute_max_length_across_ranks(
    local_max_len: int,
    group: Optional[dist.ProcessGroup] = None,
) -> int:
    """All-reduce the maximum local padded length across the SP group."""

    if group is None or not dist.is_initialized() or dist.get_world_size(group=group) == 1:
        return local_max_len

    tensor = torch.tensor(local_max_len, dtype=torch.int64, device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return int(tensor.item())
