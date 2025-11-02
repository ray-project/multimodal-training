"""Tensor parallelism utilities for vocabulary parallel embeddings."""

from .embedding import VocabParallelEmbedding, vocab_range_from_global_vocab_size

__all__ = [
    "VocabParallelEmbedding",
    "vocab_range_from_global_vocab_size",
]
