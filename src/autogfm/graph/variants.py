"""Controlled graph variants for ablation experiments."""

import torch
from torch import Tensor


def _validate_square_graph(adjacency: Tensor) -> None:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square rank-2 tensor.")


def make_identity_graph(adjacency: Tensor) -> Tensor:
    """Return a graph containing only self-relations."""
    _validate_square_graph(adjacency)
    return torch.eye(
        adjacency.shape[0],
        device=adjacency.device,
        dtype=adjacency.dtype,
    )


def make_permuted_graph(adjacency: Tensor, seed: int) -> Tensor:
    """Break graph-to-vehicle alignment while preserving graph statistics.

    The same node permutation is applied to rows and columns:
    `A_random = P A P^T`.
    """
    _validate_square_graph(adjacency)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(
        adjacency.shape[0],
        generator=generator,
    ).to(adjacency.device)

    return adjacency[permutation][:, permutation]
