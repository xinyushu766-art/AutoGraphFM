"""Small graph operations used by the readable baseline."""

import torch
from torch import Tensor


def normalize_adjacency(adjacency: Tensor) -> Tensor:
    """Apply symmetric GCN normalization.

    Computes `D^(-1/2) (A + I) D^(-1/2)`.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square rank-2 tensor.")
    if torch.any(adjacency < 0):
        raise ValueError("adjacency weights must be non-negative.")

    num_nodes = adjacency.shape[0]
    identity = torch.eye(
        num_nodes,
        device=adjacency.device,
        dtype=adjacency.dtype,
    )
    adjacency_with_loops = adjacency + identity
    degree = adjacency_with_loops.sum(dim=1)
    inverse_sqrt_degree = degree.clamp_min(1e-12).pow(-0.5)

    return (
        inverse_sqrt_degree[:, None]
        * adjacency_with_loops
        * inverse_sqrt_degree[None, :]
    )
