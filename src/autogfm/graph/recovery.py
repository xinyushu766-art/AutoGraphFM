"""Synthetic graph-recovery benchmark for adaptive adjacency learners."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from autogfm.graph.adaptive import AdaptiveGraphLearner
from autogfm.graph.sparse_residual import DirectSparseResidualGraphLearner
from autogfm.realdata.common import stable_payload_id, write_csv_rows, write_json


@dataclass(frozen=True)
class GraphRecoverySpec:
    num_nodes: int = 12
    top_k: int = 2
    embedding_dim: int = 8
    steps: int = 240
    train_steps: int = 160
    validation_steps: int = 40
    max_epochs: int = 400
    patience: int = 50
    learning_rate: float = 2e-2
    noise_std: float = 0.025
    seeds: tuple[int, ...] = (42, 43, 44)

    def validate(self) -> None:
        if self.num_nodes < 4:
            raise ValueError("num_nodes must be at least 4.")
        if not 1 <= self.top_k < self.num_nodes:
            raise ValueError("top_k must be in [1, num_nodes).")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        if self.train_steps + self.validation_steps + 2 >= self.steps:
            raise ValueError("steps must leave a non-empty test interval.")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive.")
        if self.learning_rate <= 0 or self.noise_std < 0:
            raise ValueError("invalid learning rate or noise_std.")
        if not self.seeds:
            raise ValueError("at least one seed is required.")


@dataclass(frozen=True)
class GraphRecoveryResult:
    output_dir: Path
    benchmark_id: str
    manifest: dict[str, Any]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def directed_ring_graph(num_nodes: int, top_k: int) -> np.ndarray:
    """Create a deterministic sparse directed graph with no self loops."""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    offsets = tuple(range(1, top_k + 1))
    for target in range(num_nodes):
        for rank, offset in enumerate(offsets, start=1):
            source = (target - offset * 2 + 1) % num_nodes
            adjacency[target, source] = float(top_k - rank + 1)
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    return adjacency




def corrupted_prior_graph(
    truth: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Build a partial prior with one true and one false edge per row."""
    rng = np.random.default_rng(seed)
    num_nodes = truth.shape[0]
    prior = np.zeros_like(truth, dtype=np.float32)
    for row in range(num_nodes):
        true_indices = np.flatnonzero(truth[row] > 0)
        keep = int(true_indices[np.argmax(truth[row, true_indices])])
        candidates = [
            column
            for column in range(num_nodes)
            if column != row and column not in true_indices
        ]
        false_edge = int(rng.choice(candidates))
        prior[row, keep] = 1.0
        prior[row, false_edge] = 1.0
    prior /= prior.sum(axis=1, keepdims=True)
    return prior


def simulate_graph_series(
    spec: GraphRecoverySpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a stable log-sales panel with known directed spillovers."""
    rng = np.random.default_rng(seed)
    graph = directed_ring_graph(spec.num_nodes, spec.top_k)
    values = np.zeros((spec.steps, spec.num_nodes), dtype=np.float32)
    values[0] = rng.normal(0.0, 0.4, size=spec.num_nodes)
    node_phase = np.linspace(0.0, 2.0 * np.pi, spec.num_nodes, endpoint=False)
    for time_index in range(1, spec.steps):
        common = 0.015 * np.sin(2.0 * np.pi * time_index / 12.0)
        node_season = 0.025 * np.sin(
            2.0 * np.pi * time_index / 6.0 + node_phase
        )
        shock = np.zeros(spec.num_nodes, dtype=np.float32)
        if time_index in {72, 144, 204}:
            shock[(time_index // 12) % spec.num_nodes] = 1.2
        values[time_index] = (
            0.20 * values[time_index - 1]
            + 0.72 * (graph @ values[time_index - 1])
            + common
            + node_season
            + shock
            + rng.normal(0.0, spec.noise_std, size=spec.num_nodes)
        )
    mean = values[: spec.train_steps].mean(axis=0, keepdims=True)
    std = values[: spec.train_steps].std(axis=0, keepdims=True)
    values = (values - mean) / np.maximum(std, 1e-4)
    return values.astype(np.float32), graph


class _AdaptiveOneStepModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        mode: str,
        top_k: int,
    ) -> None:
        super().__init__()
        self.graph_learner = AdaptiveGraphLearner(
            num_nodes,
            embedding_dim,
            mode=mode,
            top_k=top_k,
        )
        self.self_weight = nn.Parameter(torch.tensor(0.2))
        self.graph_weight = nn.Parameter(torch.tensor(0.7))
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    def forward(self, x: Tensor) -> Tensor:
        adjacency = self.graph_learner()
        return (
            self.self_weight * x
            + self.graph_weight * torch.einsum("ij,bj->bi", adjacency, x)
            + self.bias
        )




class _DirectGraphLearner(nn.Module):
    """Unconstrained row-wise Top-K learner used as an identifiability control."""

    def __init__(self, num_nodes: int, top_k: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.logits = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.05)

    def forward(self) -> Tensor:
        scores = self.logits.clone()
        scores = scores.masked_fill(
            torch.eye(self.num_nodes, dtype=torch.bool, device=scores.device),
            -1e9,
        )
        values, indices = torch.topk(scores, self.top_k, dim=-1)
        sparse = torch.full_like(scores, -1e9)
        sparse.scatter_(1, indices, values)
        return torch.softmax(sparse, dim=-1)


class _DirectAdaptiveOneStepModel(nn.Module):
    def __init__(self, num_nodes: int, top_k: int) -> None:
        super().__init__()
        self.graph_learner = _DirectGraphLearner(num_nodes, top_k)
        self.self_weight = nn.Parameter(torch.tensor(0.2))
        self.graph_weight = nn.Parameter(torch.tensor(0.7))
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    def forward(self, x: Tensor) -> Tensor:
        adjacency = self.graph_learner()
        return (
            self.self_weight * x
            + self.graph_weight * torch.einsum("ij,bj->bi", adjacency, x)
            + self.bias
        )




class _SparseResidualOneStepModel(nn.Module):
    """Direct sparse residual graph around a supplied partial prior."""

    def __init__(
        self,
        num_nodes: int,
        top_k: int,
        prior: np.ndarray,
    ) -> None:
        super().__init__()
        self.graph_learner = DirectSparseResidualGraphLearner(
            num_nodes,
            top_k,
            initial_residual_weight=0.35,
            l1_strength=5e-5,
            deviation_strength=2e-4,
            gate_strength=5e-5,
        )
        self.register_buffer(
            "prior", torch.from_numpy(prior.astype(np.float32))
        )
        self.self_weight = nn.Parameter(torch.tensor(0.2))
        self.graph_weight = nn.Parameter(torch.tensor(0.7))
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    def learned_adjacency(self) -> Tensor:
        return self.graph_learner(self.prior)

    def regularization_loss(self) -> Tensor:
        return self.graph_learner.regularization_loss(self.prior)

    def forward(self, x: Tensor) -> Tensor:
        adjacency = self.learned_adjacency()
        return (
            self.self_weight * x
            + self.graph_weight * torch.einsum(
                "ij,bj->bi", adjacency, x
            )
            + self.bias
        )


class _FixedOneStepModel(nn.Module):
    def __init__(self, adjacency: np.ndarray) -> None:
        super().__init__()
        self.register_buffer(
            "adjacency",
            torch.from_numpy(adjacency.astype(np.float32)),
        )
        self.self_weight = nn.Parameter(torch.tensor(0.2))
        self.graph_weight = nn.Parameter(torch.tensor(0.7))
        self.bias = nn.Parameter(torch.zeros(adjacency.shape[0]))

    def forward(self, x: Tensor) -> Tensor:
        return (
            self.self_weight * x
            + self.graph_weight * torch.einsum("ij,bj->bi", self.adjacency, x)
            + self.bias
        )


def _fit_model(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    train_end: int,
    validation_end: int,
    spec: GraphRecoverySpec,
) -> tuple[int, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    best_state: dict[str, Tensor] | None = None
    best_validation = float("inf")
    stale = 0
    best_epoch = 0
    for epoch in range(1, spec.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[:train_end])
        loss = torch.mean((prediction - y[:train_end]) ** 2)
        regularizer = getattr(model, "regularization_loss", None)
        if regularizer is not None:
            loss = loss + regularizer()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = torch.mean(
                (model(x[train_end:validation_end]) - y[train_end:validation_end]) ** 2
            ).item()
        if validation < best_validation - 1e-8:
            best_validation = validation
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    if best_state is None:
        raise RuntimeError("Graph recovery training produced no checkpoint.")
    model.load_state_dict(best_state)
    return best_epoch, best_validation


def _edge_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    predicted_edges: set[tuple[int, int]] = set()
    true_edges: set[tuple[int, int]] = set()
    for row in range(truth.shape[0]):
        true_indices = np.flatnonzero(truth[row] > 0)
        for column in true_indices:
            true_edges.add((row, int(column)))
        scores = predicted[row].copy()
        scores[row] = -np.inf
        indices = np.argsort(scores)[-top_k:]
        for column in indices:
            predicted_edges.add((row, int(column)))
    overlap = predicted_edges & true_edges
    precision = len(overlap) / max(len(predicted_edges), 1)
    recall = len(overlap) / max(len(true_edges), 1)
    return {
        "edge_precision_at_k": float(precision),
        "edge_recall_at_k": float(recall),
        "edge_f1_at_k": float(
            2.0 * precision * recall / max(precision + recall, 1e-9)
        ),
    }


def run_graph_recovery_benchmark(
    output_dir: str | Path,
    spec: GraphRecoverySpec | None = None,
    *,
    overwrite: bool = False,
) -> GraphRecoveryResult:
    """Train adaptive graphs on a known synthetic dependency system."""
    spec = spec or GraphRecoverySpec()
    spec.validate()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    graph_dir = destination / "learned_graphs"
    graph_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in spec.seeds:
        _set_seed(seed)
        values, true_graph = simulate_graph_series(spec, seed)
        x = torch.from_numpy(values[:-1])
        y = torch.from_numpy(values[1:])
        train_end = spec.train_steps - 1
        validation_end = train_end + spec.validation_steps
        rng = np.random.default_rng(seed + 1000)
        permutation = rng.permutation(spec.num_nodes)
        permuted = true_graph[permutation][:, permutation]
        identity = np.eye(spec.num_nodes, dtype=np.float32)
        partial_prior = corrupted_prior_graph(true_graph, seed + 2000)
        models: dict[str, nn.Module] = {
            "identity_control": _FixedOneStepModel(identity),
            "permuted_control": _FixedOneStepModel(permuted),
            "partial_prior_control": _FixedOneStepModel(partial_prior),
            "oracle_graph": _FixedOneStepModel(true_graph),
            "agcrn_adaptive": _AdaptiveOneStepModel(
                spec.num_nodes,
                spec.embedding_dim,
                "agcrn",
                spec.top_k,
            ),
            "mtgnn_adaptive": _AdaptiveOneStepModel(
                spec.num_nodes,
                spec.embedding_dim,
                "mtgnn",
                spec.top_k,
            ),
            "direct_sparse_adaptive": _DirectAdaptiveOneStepModel(
                spec.num_nodes, spec.top_k
            ),
            "regularized_sparse_residual": _SparseResidualOneStepModel(
                spec.num_nodes, spec.top_k, partial_prior
            ),
        }
        for model_name, model in models.items():
            best_epoch, validation_mse = _fit_model(
                model, x, y, train_end, validation_end, spec
            )
            model.eval()
            with torch.no_grad():
                test_prediction = model(x[validation_end:])
                test_target = y[validation_end:]
                test_mse = torch.mean((test_prediction - test_target) ** 2).item()
                test_mae = torch.mean(
                    torch.abs(test_prediction - test_target)
                ).item()
            learned_method = getattr(model, "learned_adjacency", None)
            if learned_method is None:
                graph_learner = getattr(model, "graph_learner", None)
                learned_method = graph_learner
            if learned_method is not None:
                with torch.no_grad():
                    learned = learned_method().detach().cpu().numpy()
                np.save(
                    graph_dir / f"{model_name}_seed{seed}.npy",
                    learned.astype(np.float32),
                )
                edge_metrics = _edge_metrics(learned, true_graph, spec.top_k)
                graph_correlation = float(
                    np.corrcoef(learned.ravel(), true_graph.ravel())[0, 1]
                )
            else:
                edge_metrics = {
                    "edge_precision_at_k": 0.0,
                    "edge_recall_at_k": 0.0,
                    "edge_f1_at_k": 0.0,
                }
                graph_correlation = 0.0
            rows.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "best_epoch": best_epoch,
                    "validation_mse": validation_mse,
                    "test_mse": test_mse,
                    "test_mae": test_mae,
                    "graph_correlation": graph_correlation,
                    **edge_metrics,
                }
            )
        np.save(destination / f"true_graph_seed{seed}.npy", true_graph)
        np.save(destination / f"partial_prior_seed{seed}.npy", partial_prior)

    write_csv_rows(destination / "per_seed_metrics.csv", list(rows[0]), rows)
    aggregate: list[dict[str, Any]] = []
    for model_name in sorted({str(row["model_name"]) for row in rows}):
        selected = [row for row in rows if row["model_name"] == model_name]
        aggregate.append(
            {
                "model_name": model_name,
                "runs": len(selected),
                "mean_test_mse": float(
                    np.mean([row["test_mse"] for row in selected])
                ),
                "mean_test_mae": float(
                    np.mean([row["test_mae"] for row in selected])
                ),
                "mean_edge_precision_at_k": float(
                    np.mean([row["edge_precision_at_k"] for row in selected])
                ),
                "mean_edge_recall_at_k": float(
                    np.mean([row["edge_recall_at_k"] for row in selected])
                ),
                "mean_graph_correlation": float(
                    np.mean([row["graph_correlation"] for row in selected])
                ),
            }
        )
    aggregate.sort(key=lambda row: row["mean_test_mse"])
    write_csv_rows(destination / "aggregate_metrics.csv", list(aggregate[0]), aggregate)
    by_name = {row["model_name"]: row for row in aggregate}
    recovery_checks = {
        "direct_control_recovers_edges": (
            by_name["direct_sparse_adaptive"]["mean_edge_precision_at_k"] >= 0.5
        ),
        "mtgnn_beats_identity_mse": (
            by_name["mtgnn_adaptive"]["mean_test_mse"]
            < by_name["identity_control"]["mean_test_mse"]
        ),
        "mtgnn_beats_permuted_mse": (
            by_name["mtgnn_adaptive"]["mean_test_mse"]
            < by_name["permuted_control"]["mean_test_mse"]
        ),
        "mtgnn_edge_precision_at_k_ge_0_5": (
            by_name["mtgnn_adaptive"]["mean_edge_precision_at_k"] >= 0.5
        ),
        "mtgnn_graph_correlation_positive": (
            by_name["mtgnn_adaptive"]["mean_graph_correlation"] > 0.0
        ),
        "sparse_residual_recovers_edges": (
            by_name["regularized_sparse_residual"][
                "mean_edge_precision_at_k"
            ]
            >= 0.5
        ),
        "sparse_residual_beats_partial_prior_mse": (
            by_name["regularized_sparse_residual"]["mean_test_mse"]
            < by_name["partial_prior_control"]["mean_test_mse"]
        ),
    }
    low_rank_check_names = {
        "mtgnn_beats_identity_mse",
        "mtgnn_beats_permuted_mse",
        "mtgnn_edge_precision_at_k_ge_0_5",
        "mtgnn_graph_correlation_positive",
    }
    sparse_check_names = {
        "direct_control_recovers_edges",
        "sparse_residual_recovers_edges",
        "sparse_residual_beats_partial_prior_mse",
    }
    low_rank_go = all(recovery_checks[name] for name in low_rank_check_names)
    sparse_residual_go = all(
        recovery_checks[name] for name in sparse_check_names
    )
    recovery_go = all(recovery_checks.values())
    identity_payload = {
        "schema_version": "autogfm-graph-recovery-v2",
        "spec": asdict(spec),
        "aggregate": aggregate,
    }
    benchmark_id = stable_payload_id(identity_payload)
    manifest = {
        **identity_payload,
        "benchmark_id": benchmark_id,
        "engineering_status": "pass",
        "evidence_scope": "synthetic_known_graph_only",
        "recovery_checks": recovery_checks,
        "graph_recovery_status": "go" if recovery_go else "no-go",
        "low_rank_recovery_status": "go" if low_rank_go else "no-go",
        "sparse_residual_recovery_status": (
            "go" if sparse_residual_go else "no-go"
        ),
        "vehicle_graph_claim_authorized": False,
    }
    write_json(destination / "manifest.json", manifest)
    lines = [
        "# Synthetic Adaptive-Graph Recovery Benchmark",
        "",
        f"- Benchmark ID: `{benchmark_id}`",
        f"- Recovery status: `{'GO' if recovery_go else 'NO-GO'}`",
        "- Evidence scope: `synthetic_known_graph_only`",
        "",
        "| Rank | Model | Test MSE | Edge P@K | Edge R@K | Graph corr. |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(aggregate, start=1):
        lines.append(
            f"| {rank} | {row['model_name']} | {row['mean_test_mse']:.5f} | "
            f"{row['mean_edge_precision_at_k']:.3f} | "
            f"{row['mean_edge_recall_at_k']:.3f} | "
            f"{row['mean_graph_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Passing this benchmark only proves that the implementation can "
            "recover dependencies in a controlled synthetic system. It does "
            "not authorize a graph-effect claim on vehicle sales.",
            "",
        ]
    )
    (destination / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return GraphRecoveryResult(destination, benchmark_id, manifest)
