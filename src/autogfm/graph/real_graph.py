"""Train-only automotive relation graph construction and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autogfm.realdata import PanelData, RealDataError, load_panel
from autogfm.realdata.common import sha256_file, stable_payload_id, write_csv_rows, write_json


@dataclass(frozen=True)
class GraphSpec:
    """Controls static and demand-correlation graph construction."""

    top_k: int = 5
    static_weight: float = 0.55
    correlation_weight: float = 0.45
    min_positive_correlation: float = 0.05
    correlation_shrinkage: float = 12.0
    include_brand_similarity: bool = True
    seed: int = 1042

    def validate(self) -> None:
        if self.top_k < 1:
            raise RealDataError("top_k must be positive.")
        if self.static_weight < 0 or self.correlation_weight < 0:
            raise RealDataError("graph weights cannot be negative.")
        if self.static_weight + self.correlation_weight <= 0:
            raise RealDataError("at least one graph weight must be positive.")
        if not -1.0 <= self.min_positive_correlation <= 1.0:
            raise RealDataError("min_positive_correlation must be in [-1, 1].")
        if self.correlation_shrinkage < 0:
            raise RealDataError("correlation_shrinkage cannot be negative.")


@dataclass(frozen=True)
class GraphBuildResult:
    output_dir: Path
    graph_id: str
    manifest: dict[str, Any]


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _static_similarity(panel: PanelData, include_brand: bool) -> np.ndarray:
    n = len(panel.series_keys)
    adjacency = np.zeros((n, n), dtype=np.float32)
    prices = []
    for row in panel.catalog:
        low = _as_float(row.get("min_price", ""))
        high = _as_float(row.get("max_price", ""))
        if low is None and high is None:
            prices.append(np.nan)
        elif low is None:
            prices.append(high)
        elif high is None:
            prices.append(low)
        else:
            prices.append((low + high) / 2.0)
    price_array = np.array(prices, dtype=np.float64)
    median = float(np.nanmedian(price_array)) if np.isfinite(price_array).any() else 0.0
    price_array = np.where(np.isfinite(price_array), price_array, median)
    scale = float(np.std(price_array))
    if scale < 1e-8:
        scale = 1.0

    for i in range(n):
        left = panel.catalog[i]
        for j in range(i + 1, n):
            right = panel.catalog[j]
            score = 0.0
            weight = 0.0
            comparisons = (
                ("segment", 0.35),
                ("body_type", 0.20),
                ("energy_type", 0.25),
            )
            if include_brand:
                comparisons = comparisons + (("brand_name", 0.10),)
            for field, field_weight in comparisons:
                left_value = str(left.get(field, "") or "unknown")
                right_value = str(right.get(field, "") or "unknown")
                if left_value != "unknown" and right_value != "unknown":
                    weight += field_weight
                    if left_value == right_value:
                        score += field_weight
            price_weight = 0.20
            weight += price_weight
            score += price_weight * np.exp(-abs(price_array[i] - price_array[j]) / scale)
            similarity = score / max(weight, 1e-8)
            adjacency[i, j] = adjacency[j, i] = float(similarity)
    return adjacency


def _train_correlation(panel: PanelData) -> tuple[np.ndarray, int]:
    train_end = int(panel.manifest["boundaries"]["train_end_index"])
    train_log_sales = np.log1p(np.clip(panel.sales[: train_end + 1], 0.0, None))
    differences = np.diff(train_log_sales, axis=0)
    n = differences.shape[1]
    if differences.shape[0] < 3:
        return np.zeros((n, n), dtype=np.float32), differences.shape[0]
    corr = np.corrcoef(differences, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32), differences.shape[0]


def _sparsify(matrix: np.ndarray, top_k: int) -> np.ndarray:
    n = matrix.shape[0]
    if n < 2:
        raise RealDataError("Graph requires at least two nodes.")
    k = min(top_k, n - 1)
    sparse = np.zeros_like(matrix, dtype=np.float32)
    for i in range(n):
        candidates = np.argsort(matrix[i])[-k:]
        for j in candidates:
            if i != j and matrix[i, j] > 0:
                sparse[i, j] = matrix[i, j]
    sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 0.0)
    return sparse


def _components(adjacency: np.ndarray) -> list[list[int]]:
    n = adjacency.shape[0]
    unseen = set(range(n))
    groups: list[list[int]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        group: list[int] = []
        while stack:
            node = stack.pop()
            group.append(node)
            neighbors = np.flatnonzero(adjacency[node] > 0).tolist()
            for neighbor in neighbors:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: (-len(group), group[0]))


def _permute_graph(adjacency: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(adjacency.shape[0])
    return adjacency[permutation][:, permutation].astype(np.float32)


def build_relation_graph(
    panel_dir: str | Path,
    output_dir: str | Path,
    spec: GraphSpec | None = None,
    *,
    overwrite: bool = False,
) -> GraphBuildResult:
    """Build hybrid graph and identity/permuted controls from a panel."""
    spec = spec or GraphSpec()
    spec.validate()
    panel = load_panel(panel_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    static = _static_similarity(panel, spec.include_brand_similarity)
    correlation, correlation_months = _train_correlation(panel)
    positive_correlation = np.where(
        correlation >= spec.min_positive_correlation, correlation, 0.0
    )
    shrinkage = correlation_months / (
        correlation_months + spec.correlation_shrinkage
    ) if correlation_months > 0 else 0.0
    positive_correlation = positive_correlation * float(shrinkage)

    total_weight = spec.static_weight + spec.correlation_weight
    combined = (
        spec.static_weight * static
        + spec.correlation_weight * positive_correlation
    ) / total_weight
    adjacency = _sparsify(combined.astype(np.float32), spec.top_k)
    identity = np.eye(adjacency.shape[0], dtype=np.float32)
    permuted = _permute_graph(adjacency, spec.seed)

    np.save(destination / "adjacency.npy", adjacency)
    np.save(destination / "static_similarity.npy", static)
    np.save(destination / "train_correlation.npy", correlation)
    np.save(destination / "identity_adjacency.npy", identity)
    np.save(destination / "permuted_adjacency.npy", permuted)

    edges: list[dict[str, Any]] = []
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[1]):
            if adjacency[i, j] > 0:
                edges.append(
                    {
                        "source_index": i,
                        "target_index": j,
                        "source_series_key": panel.series_keys[i],
                        "target_series_key": panel.series_keys[j],
                        "weight": float(adjacency[i, j]),
                        "static_similarity": float(static[i, j]),
                        "train_correlation": float(correlation[i, j]),
                    }
                )
    write_csv_rows(
        destination / "edges.csv",
        [
            "source_index", "target_index", "source_series_key", "target_series_key",
            "weight", "static_similarity", "train_correlation",
        ],
        edges,
    )

    degrees = (adjacency > 0).sum(axis=1)
    weighted_degrees = adjacency.sum(axis=1)
    groups = _components(adjacency)
    diagnostics = {
        "nodes": adjacency.shape[0],
        "undirected_edges": len(edges),
        "density": float((adjacency > 0).sum() / max(adjacency.shape[0] * (adjacency.shape[0] - 1), 1)),
        "degree_min": int(degrees.min()),
        "degree_mean": float(degrees.mean()),
        "degree_max": int(degrees.max()),
        "weighted_degree_mean": float(weighted_degrees.mean()),
        "components": len(groups),
        "largest_component_nodes": len(groups[0]),
        "largest_component_ratio": len(groups[0]) / adjacency.shape[0],
        "isolated_nodes": int((degrees == 0).sum()),
        "correlation_train_differences": correlation_months,
        "correlation_shrinkage_factor": float(shrinkage),
    }
    write_json(destination / "diagnostics.json", diagnostics)

    identity_payload = {
        "schema_version": "autogfm-relation-graph-v2",
        "panel_id": panel.manifest["panel_id"],
        "panel_sha256": sha256_file(Path(panel_dir) / "panel.npz"),
        "spec": asdict(spec),
        "adjacency_sha256": sha256_file(destination / "adjacency.npy"),
    }
    graph_id = stable_payload_id(identity_payload)
    manifest = {
        **identity_payload,
        "graph_id": graph_id,
        "diagnostics": diagnostics,
        "files": {
            name: {
                "path": name,
                "sha256": sha256_file(destination / name),
            }
            for name in (
                "adjacency.npy", "static_similarity.npy", "train_correlation.npy",
                "identity_adjacency.npy", "permuted_adjacency.npy", "edges.csv",
                "diagnostics.json",
            )
        },
    }
    write_json(destination / "manifest.json", manifest)
    return GraphBuildResult(destination, graph_id, manifest)
