"""Leakage-safe edge screening and stability selection for sparse graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilitySelectionDiagnostics:
    mean_pairwise_jaccard: float
    stable_edge_ratio: float
    stable_edge_count: int
    candidate_edge_count: int


def top_k_edge_mask(adjacency: np.ndarray, top_k: int) -> np.ndarray:
    """Return a directed row-wise Top-K edge mask without self loops."""
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    num_nodes = matrix.shape[0]
    if not 1 <= top_k < num_nodes:
        raise ValueError("top_k must be in [1, num_nodes).")
    mask = np.zeros_like(matrix, dtype=bool)
    for row in range(num_nodes):
        scores = matrix[row].copy()
        scores[row] = -np.inf
        indices = np.argsort(scores)[-top_k:]
        mask[row, indices] = True
    return mask


def edge_frequency(
    graphs: list[np.ndarray],
    top_k: int,
) -> tuple[np.ndarray, StabilitySelectionDiagnostics]:
    """Compute edge-selection frequency and cross-run stability diagnostics."""
    if not graphs:
        raise ValueError("at least one graph is required.")
    shape = np.asarray(graphs[0]).shape
    if any(np.asarray(graph).shape != shape for graph in graphs):
        raise ValueError("all graphs must have the same shape.")
    masks = [top_k_edge_mask(graph, top_k) for graph in graphs]
    frequency = np.mean(np.stack(masks, axis=0), axis=0).astype(np.float32)

    pairwise: list[float] = []
    edge_sets: list[set[tuple[int, int]]] = []
    for mask in masks:
        rows, columns = np.nonzero(mask)
        edge_sets.append(set(zip(rows.tolist(), columns.tolist())))
    for left_index in range(len(edge_sets)):
        for right_index in range(left_index + 1, len(edge_sets)):
            left = edge_sets[left_index]
            right = edge_sets[right_index]
            pairwise.append(len(left & right) / max(len(left | right), 1))

    union = frequency > 0.0
    stable = frequency >= (2.0 / 3.0 if len(graphs) >= 3 else 1.0)
    diagnostics = StabilitySelectionDiagnostics(
        mean_pairwise_jaccard=float(np.mean(pairwise)) if pairwise else 1.0,
        stable_edge_ratio=float(stable.sum() / max(union.sum(), 1)),
        stable_edge_count=int(stable.sum()),
        candidate_edge_count=int(union.sum()),
    )
    return frequency, diagnostics


def lagged_ridge_scores(
    values: np.ndarray,
    train_end_index: int,
    *,
    ridge_alpha: float = 1.0,
    use_differences: bool = True,
) -> np.ndarray:
    """Estimate directed one-step predictive strength using train-only ridge.

    The returned matrix is indexed as ``score[target, source]``.  Only values at
    or before ``train_end_index`` are used.  This is a screening statistic, not
    a causal claim.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError("values must have shape [time, nodes].")
    if not 2 <= train_end_index < series.shape[0]:
        raise ValueError("train_end_index leaves insufficient training data.")
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative.")

    train = series[: train_end_index + 1]
    train = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0)
    if use_differences:
        transformed = np.diff(train, axis=0)
    else:
        transformed = train
    if transformed.shape[0] < 3:
        raise ValueError("insufficient rows for lagged screening.")

    x = transformed[:-1]
    y = transformed[1:]
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std

    num_nodes = series.shape[1]
    gram = xz.T @ xz + ridge_alpha * np.eye(num_nodes)
    coefficients = np.linalg.solve(gram, xz.T @ yz)
    scores = np.abs(coefficients.T)
    np.fill_diagonal(scores, 0.0)
    row_max = scores.max(axis=1, keepdims=True)
    scores = scores / np.maximum(row_max, 1e-8)
    return scores.astype(np.float32)


def select_candidate_mask(
    frequency: np.ndarray,
    lagged_scores: np.ndarray,
    *,
    top_k: int,
    static_prior: np.ndarray | None = None,
    frequency_weight: float = 0.55,
    lagged_weight: float = 0.35,
    prior_weight: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a fixed row-wise candidate edge set from train-only evidence."""
    frequency = np.asarray(frequency, dtype=np.float64)
    lagged_scores = np.asarray(lagged_scores, dtype=np.float64)
    if frequency.shape != lagged_scores.shape:
        raise ValueError("frequency and lagged_scores must have equal shape.")
    if frequency.ndim != 2 or frequency.shape[0] != frequency.shape[1]:
        raise ValueError("edge score matrices must be square.")
    num_nodes = frequency.shape[0]
    if not 1 <= top_k < num_nodes:
        raise ValueError("top_k must be in [1, num_nodes).")
    if min(frequency_weight, lagged_weight, prior_weight) < 0.0:
        raise ValueError("score weights must be non-negative.")

    prior_score = np.zeros_like(frequency)
    if static_prior is not None:
        prior = np.asarray(static_prior, dtype=np.float64)
        if prior.shape != frequency.shape:
            raise ValueError("static_prior has the wrong shape.")
        prior_score = np.maximum(prior, 0.0)
        prior_score /= np.maximum(prior_score.max(axis=1, keepdims=True), 1e-8)

    combined = (
        frequency_weight * frequency
        + lagged_weight * lagged_scores
        + prior_weight * prior_score
    )
    np.fill_diagonal(combined, -np.inf)
    mask = np.zeros_like(combined, dtype=bool)
    for row in range(num_nodes):
        indices = np.argsort(combined[row])[-top_k:]
        mask[row, indices] = True
    finite_score = np.where(np.isfinite(combined), combined, 0.0)
    return mask, finite_score.astype(np.float32)


def consensus_adjacency(
    graphs: list[np.ndarray],
    candidate_mask: np.ndarray,
    *,
    minimum_weight: float = 1e-6,
) -> np.ndarray:
    """Average learned weights on a fixed stability-selected edge set."""
    if not graphs:
        raise ValueError("at least one graph is required.")
    mean_graph = np.mean(np.stack(graphs, axis=0), axis=0)
    mask = np.asarray(candidate_mask, dtype=bool)
    if mean_graph.shape != mask.shape:
        raise ValueError("candidate_mask has the wrong shape.")
    consensus = np.where(mask, np.maximum(mean_graph, minimum_weight), 0.0)
    row_sum = consensus.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 1e-12
    if np.any(empty):
        fallback = np.asarray(graphs[0], dtype=np.float64)
        for row in np.flatnonzero(empty):
            scores = fallback[row].copy()
            scores[row] = -np.inf
            column = int(np.argmax(scores))
            consensus[row, column] = minimum_weight
    consensus /= np.maximum(consensus.sum(axis=1, keepdims=True), 1e-8)
    return consensus.astype(np.float32)
