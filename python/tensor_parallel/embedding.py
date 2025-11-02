"""
Vocabulary-parallel layers for tensor parallelism.

Based on Megatron-LM implementation but simplified for our use case.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> tuple[int, int]:
    """
    Calculate the vocabulary range [start, end) for a given rank.

    Args:
        global_vocab_size: Total vocabulary size
        rank: Rank of the current process
        world_size: Total number of processes in the TP group

    Returns:
        Tuple of (vocab_start_index, vocab_end_index)
    """
    assert global_vocab_size % world_size == 0, (
        f"Vocabulary size ({global_vocab_size}) must be divisible by " f"tensor parallel world size ({world_size})"
    )

    vocab_size_per_partition = global_vocab_size // world_size
    vocab_start_index = rank * vocab_size_per_partition
    vocab_end_index = vocab_start_index + vocab_size_per_partition

    return vocab_start_index, vocab_end_index


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer parallelized in the vocabulary dimension.

    This is adapted from Megatron-LM's VocabParallelEmbedding, simplified
    for our use case. The vocabulary is partitioned across tensor parallel
    ranks.

    Args:
        num_embeddings: Total vocabulary size (will be partitioned)
        embedding_dim: Size of embedding vectors
        padding_idx: Index for padding token (default: None)
        tp_group: Tensor parallel process group
        dtype: Data type for the embedding weights
        device: Device to place the embedding weights
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_mesh=None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super(VocabParallelEmbedding, self).__init__()

        # Store configuration
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.tp_group = tp_group
        self.tp_mesh = tp_mesh

        # Get tensor parallel info
        if tp_group is not None and dist.is_initialized():
            self.tp_world_size = dist.get_world_size(tp_group)
            self.tp_rank = dist.get_rank(tp_group)
        else:
            self.tp_world_size = 1
            self.tp_rank = 0

        # Calculate vocabulary partition for this rank
        self.vocab_start_index, self.vocab_end_index = vocab_range_from_global_vocab_size(
            self.num_embeddings,
            self.tp_rank,
            self.tp_world_size,
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Create the embedding weight parameter (partitioned vocabulary)
        # Use a simple partitioned Tensor (not DTensor) for compatibility with LM head
        # The all-reduce in forward() will combine contributions from all ranks
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                dtype=dtype,
                device=device,
            )
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with vocabulary parallelism.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]

        Returns:
            Embeddings [batch_size, seq_length, embedding_dim]
        """
        if self.tp_world_size > 1:
            # Build mask for tokens outside this rank's vocabulary range
            input_mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)

            # Adjust input IDs to be relative to this partition's vocabulary
            masked_input = input_ids.clone() - self.vocab_start_index

            # Set out-of-range indices to 0 to avoid index errors
            masked_input[input_mask] = 0
        else:
            masked_input = input_ids
            input_mask = None

        # Perform embedding lookup using the local weight partition
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            padding_idx=None,  # We handle padding through the mask
        )

        # Zero out embeddings for tokens outside this rank's vocabulary range
        if self.tp_world_size > 1:
            output_parallel = output_parallel.clone()
            output_parallel[input_mask] = 0.0

            # All-reduce across tensor parallel group to sum contributions
            # Only the rank that owns a token ID will have non-zero embeddings
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.tp_group)

        return output_parallel

    def extra_repr(self) -> str:
        """String representation of this module."""
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        s += f", vocab_range=[{self.vocab_start_index}, {self.vocab_end_index})"
        s += f", tp_rank={self.tp_rank}/{self.tp_world_size}"
        return s


def create_vocab_parallel_embedding(
    original_embedding: nn.Embedding,
    tp_group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> VocabParallelEmbedding:
    """
    Create a VocabParallelEmbedding from an existing nn.Embedding,
    copying and sharding the weights appropriately.

    Args:
        original_embedding: The original nn.Embedding to convert
        tp_group: Tensor parallel process group
        device: Device to place the new embedding

    Returns:
        VocabParallelEmbedding with weights copied from original
    """
    # Create new vocab parallel embedding
    vocab_parallel_emb = VocabParallelEmbedding(
        num_embeddings=original_embedding.num_embeddings,
        embedding_dim=original_embedding.embedding_dim,
        padding_idx=original_embedding.padding_idx,
        tp_group=tp_group,
        dtype=original_embedding.weight.dtype,
        device=device or original_embedding.weight.device,
    )

    # Copy the appropriate partition of weights
    with torch.no_grad():
        # Extract the vocabulary partition for this rank
        start_idx = vocab_parallel_emb.vocab_start_index
        end_idx = vocab_parallel_emb.vocab_end_index

        # Copy weights to the new embedding
        original_weight = original_embedding.weight.data
        if device is not None and original_weight.device != device:
            original_weight = original_weight.to(device)

        vocab_parallel_emb.weight.data.copy_(original_weight[start_idx:end_idx])

    return vocab_parallel_emb
