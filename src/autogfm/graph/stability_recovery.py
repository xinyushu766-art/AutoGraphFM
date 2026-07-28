"""Stage 12A stability-selected graph recovery on a known dependency system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from autogfm.graph.recovery import (
    GraphRecoverySpec,
    _DirectAdaptiveOneStepModel,
    _FixedOneStepModel,
    _edge_metrics,
    _fit_model,
    _set_seed,
    corrupted_prior_graph,
    simulate_graph_series,
)
from autogfm.graph.sparse_residual import DirectSparseResidualGraphLearner
from autogfm.graph.stability_selection import (
    consensus_adjacency,
    edge_frequency,
    lagged_ridge_scores,
    select_candidate_mask,
)
from autogfm.realdata.common import stable_payload_id, write_csv_rows, write_json


@dataclass(frozen=True)
class StabilityRecoverySpec:
    base: GraphRecoverySpec = GraphRecoverySpec()
    ridge_alpha: float = 1.0
    frequency_weight: float = 0.50
    lagged_weight: float = 0.45
    prior_weight: float = 0.05
    initial_residual_weight: float = 0.35
    l1_strength: float = 5e-5
    deviation_strength: float = 2e-4
    gate_strength: float = 5e-5
    minimum_edge_precision: float = 0.70
    maximum_mse_ratio_to_direct: float = 1.10

    def validate(self) -> None:
        self.base.validate()
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative.")
        weights = (
            self.frequency_weight,
            self.lagged_weight,
            self.prior_weight,
        )
        if min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError("candidate score weights are invalid.")
        if not 0.0 < self.minimum_edge_precision <= 1.0:
            raise ValueError("minimum_edge_precision must be in (0, 1].")
        if self.maximum_mse_ratio_to_direct <= 0.0:
            raise ValueError("maximum_mse_ratio_to_direct must be positive.")


@dataclass(frozen=True)
class StabilityRecoveryResult:
    output_dir: Path
    benchmark_id: str
    manifest: dict[str, Any]


class _MaskedSparseResidualOneStepModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        top_k: int,
        candidate_mask: np.ndarray,
        prior: np.ndarray,
        spec: StabilityRecoverySpec,
    ) -> None:
        super().__init__()
        self.graph_learner = DirectSparseResidualGraphLearner(
            num_nodes,
            top_k,
            initial_residual_weight=spec.initial_residual_weight,
            l1_strength=spec.l1_strength,
            deviation_strength=spec.deviation_strength,
            gate_strength=spec.gate_strength,
            candidate_mask=torch.from_numpy(candidate_mask),
        )
        self.register_buffer("prior", torch.from_numpy(prior.astype(np.float32)))
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
            + self.graph_weight * torch.einsum("ij,bj->bi", adjacency, x)
            + self.bias
        )


def _evaluate_model(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    validation_end: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(x[validation_end:])
        target = y[validation_end:]
        mse = torch.mean((prediction - target) ** 2).item()
        mae = torch.mean(torch.abs(prediction - target)).item()
    return float(mse), float(mae)


def run_stability_recovery_benchmark(
    output_dir: str | Path,
    spec: StabilityRecoverySpec | None = None,
    *,
    overwrite: bool = False,
) -> StabilityRecoveryResult:
    """Discover edges across seeds, then retrain on a fixed consensus support."""
    spec = spec or StabilityRecoverySpec()
    spec.validate()
    base = spec.base
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    graph_dir = destination / "learned_graphs"
    graph_dir.mkdir(exist_ok=True)

    discovery_graphs: list[np.ndarray] = []
    lagged_scores: list[np.ndarray] = []
    true_graph: np.ndarray | None = None
    cached: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    discovery_rows: list[dict[str, Any]] = []

    for seed in base.seeds:
        _set_seed(seed)
        values, truth = simulate_graph_series(base, seed)
        true_graph = truth
        cached[seed] = (values, truth)
        x = torch.from_numpy(values[:-1])
        y = torch.from_numpy(values[1:])
        train_end = base.train_steps - 1
        validation_end = train_end + base.validation_steps
        model = _DirectAdaptiveOneStepModel(base.num_nodes, base.top_k)
        best_epoch, validation_mse = _fit_model(
            model,
            x,
            y,
            train_end,
            validation_end,
            base,
        )
        with torch.no_grad():
            learned = model.graph_learner().detach().cpu().numpy()
        discovery_graphs.append(learned.astype(np.float32))
        lagged_scores.append(
            lagged_ridge_scores(
                values,
                base.train_steps - 1,
                ridge_alpha=spec.ridge_alpha,
                use_differences=False,
            )
        )
        edge_metrics = _edge_metrics(learned, truth, base.top_k)
        test_mse, test_mae = _evaluate_model(model, x, y, validation_end)
        discovery_rows.append(
            {
                "seed": seed,
                "model_name": "direct_sparse_discovery",
                "best_epoch": best_epoch,
                "validation_mse": validation_mse,
                "test_mse": test_mse,
                "test_mae": test_mae,
                **edge_metrics,
            }
        )
        np.save(graph_dir / f"discovery_seed{seed}.npy", learned)

    assert true_graph is not None
    frequency, discovery_stability = edge_frequency(
        discovery_graphs,
        base.top_k,
    )
    mean_lagged = np.mean(np.stack(lagged_scores, axis=0), axis=0)
    candidate_mask, candidate_scores = select_candidate_mask(
        frequency,
        mean_lagged,
        top_k=base.top_k,
        static_prior=None,
        frequency_weight=spec.frequency_weight,
        lagged_weight=spec.lagged_weight,
        prior_weight=spec.prior_weight,
    )
    consensus = consensus_adjacency(discovery_graphs, candidate_mask)
    np.save(destination / "edge_frequency.npy", frequency)
    np.save(destination / "lagged_ridge_scores.npy", mean_lagged)
    np.save(destination / "candidate_scores.npy", candidate_scores)
    np.save(destination / "candidate_mask.npy", candidate_mask)
    np.save(destination / "consensus_adjacency.npy", consensus)

    candidate_metrics = _edge_metrics(
        candidate_mask.astype(np.float32),
        true_graph,
        base.top_k,
    )
    final_rows: list[dict[str, Any]] = []
    final_graphs: dict[str, list[np.ndarray]] = {
        "fixed_consensus_graph": [],
        "stability_selected_sparse_residual": [],
    }
    for seed in base.seeds:
        _set_seed(seed + 10000)
        values, truth = cached[seed]
        x = torch.from_numpy(values[:-1])
        y = torch.from_numpy(values[1:])
        train_end = base.train_steps - 1
        validation_end = train_end + base.validation_steps
        models: dict[str, nn.Module] = {
            "fixed_consensus_graph": _FixedOneStepModel(consensus),
            "stability_selected_sparse_residual": (
                _MaskedSparseResidualOneStepModel(
                    base.num_nodes,
                    base.top_k,
                    candidate_mask,
                    consensus,
                    spec,
                )
            ),
        }
        for model_name, model in models.items():
            best_epoch, validation_mse = _fit_model(
                model,
                x,
                y,
                train_end,
                validation_end,
                base,
            )
            test_mse, test_mae = _evaluate_model(model, x, y, validation_end)
            learned_method = getattr(model, "learned_adjacency", None)
            if learned_method is None:
                learned = consensus
            else:
                with torch.no_grad():
                    learned = learned_method().detach().cpu().numpy()
            final_graphs[model_name].append(learned.astype(np.float32))
            edge_metrics = _edge_metrics(learned, truth, base.top_k)
            final_rows.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "best_epoch": best_epoch,
                    "validation_mse": validation_mse,
                    "test_mse": test_mse,
                    "test_mae": test_mae,
                    **edge_metrics,
                }
            )
            np.save(graph_dir / f"{model_name}_seed{seed}.npy", learned)

    all_rows = discovery_rows + final_rows
    write_csv_rows(destination / "per_seed_metrics.csv", list(all_rows[0]), all_rows)
    aggregate: list[dict[str, Any]] = []
    for model_name in sorted({str(row["model_name"]) for row in all_rows}):
        rows = [row for row in all_rows if row["model_name"] == model_name]
        aggregate.append(
            {
                "model_name": model_name,
                "runs": len(rows),
                "mean_test_mse": float(np.mean([row["test_mse"] for row in rows])),
                "mean_test_mae": float(np.mean([row["test_mae"] for row in rows])),
                "mean_edge_precision_at_k": float(
                    np.mean([row["edge_precision_at_k"] for row in rows])
                ),
                "mean_edge_recall_at_k": float(
                    np.mean([row["edge_recall_at_k"] for row in rows])
                ),
            }
        )
    aggregate.sort(key=lambda row: row["mean_test_mse"])
    write_csv_rows(destination / "aggregate_metrics.csv", list(aggregate[0]), aggregate)

    _, fixed_stability = edge_frequency(
        final_graphs["fixed_consensus_graph"], base.top_k
    )
    _, selected_stability = edge_frequency(
        final_graphs["stability_selected_sparse_residual"], base.top_k
    )
    by_name = {row["model_name"]: row for row in aggregate}
    direct_mse = by_name["direct_sparse_discovery"]["mean_test_mse"]
    selected_mse = by_name["stability_selected_sparse_residual"]["mean_test_mse"]
    checks = {
        "candidate_precision_pass": (
            candidate_metrics["edge_precision_at_k"] >= spec.minimum_edge_precision
        ),
        "selected_precision_pass": (
            by_name["stability_selected_sparse_residual"][
                "mean_edge_precision_at_k"
            ]
            >= spec.minimum_edge_precision
        ),
        "selected_mse_within_tolerance": (
            selected_mse <= direct_mse * spec.maximum_mse_ratio_to_direct
        ),
        "fixed_support_is_seed_stable": (
            selected_stability.mean_pairwise_jaccard >= 0.95
        ),
    }
    status = "go" if all(checks.values()) else "no-go"
    identity_payload = {
        "schema_version": "autogfm-stage12a-stability-recovery-v1",
        "spec": asdict(spec),
        "aggregate": aggregate,
        "candidate_edge_metrics": candidate_metrics,
    }
    benchmark_id = stable_payload_id(identity_payload)
    manifest = {
        **identity_payload,
        "benchmark_id": benchmark_id,
        "engineering_status": "pass",
        "evidence_scope": "synthetic_known_graph_only",
        "stability_selection_status": status,
        "checks": checks,
        "discovery_stability": asdict(discovery_stability),
        "fixed_consensus_stability": asdict(fixed_stability),
        "final_stability": asdict(selected_stability),
        "vehicle_graph_claim_authorized": False,
    }
    write_json(destination / "manifest.json", manifest)
    lines = [
        "# Stage 12A Stability-Selected Recovery",
        "",
        f"- Benchmark ID: `{benchmark_id}`",
        f"- Status: **{status.upper()}**",
        f"- Candidate edge precision@K: "
        f"`{candidate_metrics['edge_precision_at_k']:.3f}`",
        f"- Discovery edge Jaccard: "
        f"`{discovery_stability.mean_pairwise_jaccard:.3f}`",
        f"- Final edge Jaccard: "
        f"`{selected_stability.mean_pairwise_jaccard:.3f}`",
        "",
        "The support is selected only from training-period predictive evidence "
        "and repeated discovery runs. Passing this benchmark does not authorize "
        "a real vehicle-sales graph claim.",
        "",
    ]
    (destination / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return StabilityRecoveryResult(destination, benchmark_id, manifest)
