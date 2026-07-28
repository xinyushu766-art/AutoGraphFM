"""Stage 11B sparse residual graph experiments and edge-stability gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from autogfm.experiments.adaptive_graph_runner import (
    _graph_diagnostics,
    _paired_bootstrap,
)
from autogfm.experiments.real_panel_runner import (
    RealPanelExperimentSpec,
    _PanelWindowDataset,
    _evaluate,
    _graph_scale,
    _parameter_count,
    _set_seed,
    _train,
)
from autogfm.models import (
    AdaptiveGraphResidualGRU,
    LastValueForecaster,
    ResidualTemporalGRU,
    SafeGraphResidualGRU,
    SparseResidualGraphGRU,
)
from autogfm.realdata import RealDataError, build_window_split, load_panel
from autogfm.realdata.common import (
    sha256_file,
    stable_payload_id,
    write_csv_rows,
    write_json,
)


@dataclass(frozen=True)
class SparseGraphExperimentSpec:
    history: int = 12
    horizon: int = 3
    hidden_dim: int = 32
    embedding_dim: int = 8
    top_k: int = 5
    batch_size: int = 8
    max_epochs: int = 35
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seeds: tuple[int, ...] = (42, 43, 44)
    bootstrap_samples: int = 1000
    initial_residual_weight: float = 0.15
    l1_strength: float = 2e-4
    deviation_strength: float = 2e-3
    gate_strength: float = 2e-4
    consensus_ratio: float = 2.0 / 3.0
    min_pairwise_jaccard: float = 0.30
    min_stable_edge_ratio: float = 0.45

    def validate(self) -> None:
        RealPanelExperimentSpec(
            history=self.history,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            seeds=self.seeds,
        ).validate()
        if self.embedding_dim < 1 or self.top_k < 1:
            raise RealDataError("embedding_dim and top_k must be positive.")
        if self.bootstrap_samples < 100:
            raise RealDataError("bootstrap_samples must be at least 100.")
        if not 0.0 < self.initial_residual_weight < 1.0:
            raise RealDataError("initial_residual_weight must be in (0, 1).")
        if min(
            self.l1_strength,
            self.deviation_strength,
            self.gate_strength,
        ) < 0.0:
            raise RealDataError("regularization strengths must be non-negative.")
        if not 0.0 < self.consensus_ratio <= 1.0:
            raise RealDataError("consensus_ratio must be in (0, 1].")
        if not 0.0 <= self.min_pairwise_jaccard <= 1.0:
            raise RealDataError("min_pairwise_jaccard must be in [0, 1].")
        if not 0.0 <= self.min_stable_edge_ratio <= 1.0:
            raise RealDataError("min_stable_edge_ratio must be in [0, 1].")

    def base_spec(self) -> RealPanelExperimentSpec:
        return RealPanelExperimentSpec(
            history=self.history,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            seeds=self.seeds,
        )


@dataclass(frozen=True)
class SparseGraphExperimentResult:
    output_dir: Path
    experiment_id: str
    manifest: dict[str, Any]


def _learned_graph(model: nn.Module, prior: Tensor | None) -> Tensor | None:
    method = getattr(model, "learned_adjacency", None)
    if method is None:
        return None
    with torch.no_grad():
        return method(prior).detach().cpu()


def _residual_weight(model: nn.Module) -> float:
    method = getattr(model, "residual_graph_weight", None)
    if method is None:
        return 0.0
    return float(method())


def _top_k_edges(adjacency: np.ndarray, top_k: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for row in range(adjacency.shape[0]):
        scores = adjacency[row].copy()
        scores[row] = -np.inf
        indices = np.argsort(scores)[-top_k:]
        for column in indices:
            edges.add((row, int(column)))
    return edges


def _pairwise_jaccard(edge_sets: list[set[tuple[int, int]]]) -> float:
    values: list[float] = []
    for left_index in range(len(edge_sets)):
        for right_index in range(left_index + 1, len(edge_sets)):
            left = edge_sets[left_index]
            right = edge_sets[right_index]
            union = left | right
            values.append(len(left & right) / max(len(union), 1))
    return float(np.mean(values)) if values else 1.0


def _edge_consensus(
    graphs: list[np.ndarray],
    top_k: int,
    consensus_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not graphs:
        raise RealDataError("No learned graphs were available for consensus.")
    num_nodes = graphs[0].shape[0]
    frequency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edge_sets: list[set[tuple[int, int]]] = []
    for graph in graphs:
        edges = _top_k_edges(graph, top_k)
        edge_sets.append(edges)
        for row, column in edges:
            frequency[row, column] += 1.0
    frequency /= float(len(graphs))

    mean_graph = np.mean(np.stack(graphs, axis=0), axis=0)
    stable_mask = frequency >= consensus_ratio
    np.fill_diagonal(stable_mask, False)
    consensus = np.zeros_like(mean_graph, dtype=np.float32)
    consensus[stable_mask] = mean_graph[stable_mask]
    for row in range(num_nodes):
        if consensus[row].sum() <= 0.0:
            scores = frequency[row].copy()
            scores[row] = -np.inf
            column = int(np.argmax(scores))
            consensus[row, column] = max(mean_graph[row, column], 1e-6)
    consensus /= np.maximum(consensus.sum(axis=1, keepdims=True), 1e-8)

    union_mask = frequency > 0.0
    stable_edges = int(stable_mask.sum())
    candidate_edges = int(union_mask.sum())
    diagnostics = {
        "mean_pairwise_edge_jaccard": _pairwise_jaccard(edge_sets),
        "stable_edge_ratio": float(
            stable_edges / max(candidate_edges, 1)
        ),
        "stable_edge_count": float(stable_edges),
        "candidate_edge_count": float(candidate_edges),
    }
    return frequency, consensus, diagnostics


def run_sparse_graph_experiments(
    panel_dir: str | Path,
    graph_dir: str | Path,
    output_dir: str | Path,
    spec: SparseGraphExperimentSpec | None = None,
    *,
    evidence_scope: str | None = None,
    overwrite: bool = False,
) -> SparseGraphExperimentResult:
    """Compare direct sparse residual graphs with strict falsification controls."""
    spec = spec or SparseGraphExperimentSpec()
    spec.validate()
    base_spec = spec.base_spec()
    panel = load_panel(panel_dir)
    split = build_window_split(panel, spec.history, spec.horizon)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    graph_output = destination / "learned_graphs"
    graph_output.mkdir(exist_ok=True)
    consensus_output = destination / "consensus_graphs"
    consensus_output.mkdir(exist_ok=True)

    train_dataset = _PanelWindowDataset(
        panel, split.train_origins, spec.history, spec.horizon
    )
    validation_dataset = _PanelWindowDataset(
        panel, split.validation_origins, spec.history, spec.horizon
    )
    test_dataset = _PanelWindowDataset(
        panel, split.test_origins, spec.history, spec.horizon
    )
    static = torch.from_numpy(
        np.load(Path(graph_dir) / "adjacency.npy").astype(np.float32)
    )
    identity = torch.from_numpy(
        np.load(Path(graph_dir) / "identity_adjacency.npy").astype(np.float32)
    )
    permuted = torch.from_numpy(
        np.load(Path(graph_dir) / "permuted_adjacency.npy").astype(np.float32)
    )
    num_features = panel.features.shape[-1]
    num_nodes = panel.features.shape[1]
    top_k = min(spec.top_k, max(num_nodes - 1, 1))

    sparse_kwargs = {
        "num_features": num_features,
        "hidden_dim": spec.hidden_dim,
        "horizon": spec.horizon,
        "num_nodes": num_nodes,
        "top_k": top_k,
        "initial_residual_weight": spec.initial_residual_weight,
        "l1_strength": spec.l1_strength,
        "deviation_strength": spec.deviation_strength,
        "gate_strength": spec.gate_strength,
    }
    factories: dict[str, tuple[Callable[[], nn.Module], Tensor | None, bool]] = {
        "last_value": (
            lambda: LastValueForecaster(horizon=spec.horizon),
            None,
            False,
        ),
        "residual_temporal_gru": (
            lambda: ResidualTemporalGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            None,
            True,
        ),
        "safe_static_graph_gru": (
            lambda: SafeGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            static,
            True,
        ),
        "safe_identity_graph_gru": (
            lambda: SafeGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            identity,
            True,
        ),
        "safe_permuted_graph_gru": (
            lambda: SafeGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            permuted,
            True,
        ),
        "adaptive_mtgnn_graph_gru": (
            lambda: AdaptiveGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
                num_nodes=num_nodes,
                graph_mode="mtgnn",
                embedding_dim=spec.embedding_dim,
                top_k=top_k,
            ),
            None,
            True,
        ),
        "direct_sparse_graph_gru": (
            lambda: SparseResidualGraphGRU(
                **sparse_kwargs,
                use_prior=False,
            ),
            None,
            True,
        ),
        "residual_sparse_static_graph_gru": (
            lambda: SparseResidualGraphGRU(
                **sparse_kwargs,
                use_prior=True,
            ),
            static,
            True,
        ),
    }

    resolved_evidence_scope = evidence_scope or panel.manifest.get(
        "evidence_scope", "unverified_vehicle_panel"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    learned_by_model: dict[str, list[np.ndarray]] = {
        "direct_sparse_graph_gru": [],
        "residual_sparse_static_graph_gru": [],
    }

    for seed in spec.seeds:
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=spec.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
        )
        for model_name, (factory, adjacency, trainable) in factories.items():
            _set_seed(seed)
            model = factory()
            if trainable:
                best_epoch, best_validation = _train(
                    model,
                    train_loader,
                    validation_loader,
                    adjacency,
                    base_spec,
                    device,
                )
            else:
                model.to(device)
                validation_metrics, _, _, _ = _evaluate(
                    model, validation_loader, adjacency, device
                )
                best_epoch = 0
                best_validation = validation_metrics["log_mse"]
            test_metrics, predictions, targets, masks = _evaluate(
                model, test_loader, adjacency, device
            )
            learned = _learned_graph(
                model,
                adjacency.to(device) if adjacency is not None else None,
            )
            diagnostics = _graph_diagnostics(learned)
            if learned is not None:
                learned_array = learned.numpy().astype(np.float32)
                np.save(
                    graph_output / f"{model_name}_seed{seed}.npy",
                    learned_array,
                )
                if model_name in learned_by_model:
                    learned_by_model[model_name].append(learned_array)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "trainable_parameters": _parameter_count(model),
                    "best_epoch": best_epoch,
                    "best_validation_log_mse": best_validation,
                    "graph_scale_mean_abs": _graph_scale(model),
                    "residual_graph_weight": _residual_weight(model),
                    **diagnostics,
                    **test_metrics,
                }
            )
            for sample_index, origin in enumerate(split.test_origins):
                for horizon_index in range(spec.horizon):
                    indices = np.flatnonzero(
                        masks[sample_index, horizon_index] > 0.5
                    )
                    for series_index in indices:
                        prediction_rows.append(
                            {
                                "seed": seed,
                                "model_name": model_name,
                                "origin_index": origin,
                                "origin_month": panel.months[origin],
                                "horizon": horizon_index + 1,
                                "target_month": panel.months[
                                    origin + horizon_index + 1
                                ],
                                "series_index": int(series_index),
                                "series_key": panel.series_keys[int(series_index)],
                                "target": float(
                                    np.expm1(
                                        targets[
                                            sample_index,
                                            horizon_index,
                                            series_index,
                                        ]
                                    )
                                ),
                                "prediction": float(
                                    np.expm1(
                                        np.clip(
                                            predictions[
                                                sample_index,
                                                horizon_index,
                                                series_index,
                                            ],
                                            0.0,
                                            20.0,
                                        )
                                    )
                                ),
                            }
                        )

    write_csv_rows(
        destination / "per_seed_metrics.csv",
        list(per_seed_rows[0]),
        per_seed_rows,
    )
    write_csv_rows(
        destination / "test_predictions.csv",
        list(prediction_rows[0]),
        prediction_rows,
    )

    aggregate_rows: list[dict[str, Any]] = []
    for model_name in factories:
        rows = [row for row in per_seed_rows if row["model_name"] == model_name]
        wape = np.asarray([row["wape_percent"] for row in rows], dtype=float)
        aggregate_rows.append(
            {
                "model_name": model_name,
                "runs": len(rows),
                "mean_test_wape_percent": float(wape.mean()),
                "std_test_wape_percent": float(wape.std(ddof=1))
                if len(wape) > 1
                else 0.0,
                "mean_test_mae": float(np.mean([row["mae"] for row in rows])),
                "mean_test_rmse": float(np.mean([row["rmse"] for row in rows])),
                "mean_bias": float(np.mean([row["bias"] for row in rows])),
                "mean_graph_scale_abs": float(
                    np.mean([row["graph_scale_mean_abs"] for row in rows])
                ),
                "mean_residual_graph_weight": float(
                    np.mean([row["residual_graph_weight"] for row in rows])
                ),
            }
        )
    aggregate_rows.sort(key=lambda row: row["mean_test_wape_percent"])
    write_csv_rows(
        destination / "aggregate_metrics.csv",
        list(aggregate_rows[0]),
        aggregate_rows,
    )

    stability_rows: list[dict[str, Any]] = []
    stability_by_model: dict[str, dict[str, float]] = {}
    for model_name, graphs in learned_by_model.items():
        frequency, consensus, diagnostics = _edge_consensus(
            graphs,
            top_k,
            spec.consensus_ratio,
        )
        np.save(consensus_output / f"{model_name}_frequency.npy", frequency)
        np.save(consensus_output / f"{model_name}_consensus.npy", consensus)
        row = {"model_name": model_name, **diagnostics}
        stability_rows.append(row)
        stability_by_model[model_name] = diagnostics
    write_csv_rows(
        destination / "edge_stability.csv",
        list(stability_rows[0]),
        stability_rows,
    )

    sparse_names = {
        "direct_sparse_graph_gru",
        "residual_sparse_static_graph_gru",
    }
    best_sparse = min(
        (row for row in aggregate_rows if row["model_name"] in sparse_names),
        key=lambda row: row["mean_test_wape_percent"],
    )["model_name"]
    controls = [
        "residual_temporal_gru",
        "safe_static_graph_gru",
        "safe_identity_graph_gru",
        "safe_permuted_graph_gru",
        "adaptive_mtgnn_graph_gru",
    ]
    bootstrap_rows = [
        _paired_bootstrap(
            prediction_rows,
            best_sparse,
            control,
            spec.bootstrap_samples,
        )
        for control in controls
    ]
    write_csv_rows(
        destination / "paired_bootstrap.csv",
        list(bootstrap_rows[0]),
        bootstrap_rows,
    )

    by_name = {row["model_name"]: row for row in aggregate_rows}
    performance_checks = {
        f"beats_{control}": (
            by_name[best_sparse]["mean_test_wape_percent"]
            < by_name[control]["mean_test_wape_percent"]
        )
        for control in controls
    }
    significance_checks = {
        f"significant_vs_{row['control']}": row[
            "challenger_significantly_better"
        ]
        for row in bootstrap_rows
    }
    stability = stability_by_model[best_sparse]
    stability_checks = {
        "pairwise_jaccard_pass": (
            stability["mean_pairwise_edge_jaccard"]
            >= spec.min_pairwise_jaccard
        ),
        "stable_edge_ratio_pass": (
            stability["stable_edge_ratio"] >= spec.min_stable_edge_ratio
        ),
        "nontrivial_residual_weight": (
            by_name[best_sparse]["mean_residual_graph_weight"] > 0.02
        ),
        "nontrivial_graph_scale": (
            by_name[best_sparse]["mean_graph_scale_abs"] > 1e-4
        ),
    }
    experiment_supports_claim = (
        all(performance_checks.values())
        and all(significance_checks.values())
        and all(stability_checks.values())
    )

    summary_lines = [
        "# Stage 11B Sparse Residual Graph Experiment",
        "",
        f"- Device: `{device}`",
        f"- Evidence scope: `{resolved_evidence_scope}`",
        f"- Best sparse model: `{best_sparse}`",
        f"- Experiment graph status: "
        f"`{'go' if experiment_supports_claim else 'no-claim'}`",
        "",
        "| Rank | Model | Mean WAPE | MAE | RMSE | Residual weight |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(aggregate_rows, start=1):
        summary_lines.append(
            f"| {rank} | {row['model_name']} | "
            f"{row['mean_test_wape_percent']:.3f}% | "
            f"{row['mean_test_mae']:.3f} | "
            f"{row['mean_test_rmse']:.3f} | "
            f"{row['mean_residual_graph_weight']:.4f} |"
        )
    summary_lines.extend(
        [
            "",
            "The sparse graph is promoted only when it beats temporal, static, "
            "identity, permuted and low-rank controls, passes paired bootstrap "
            "tests, and produces stable edges across seeds.",
            "",
        ]
    )
    (destination / "summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    identity_payload = {
        "schema_version": "autogfm-stage11b-sparse-graph-v1",
        "panel_id": panel.manifest["panel_id"],
        "panel_sha256": sha256_file(Path(panel_dir) / "panel.npz"),
        "static_graph_sha256": sha256_file(Path(graph_dir) / "adjacency.npy"),
        "spec": asdict(spec),
        "aggregate_sha256": sha256_file(destination / "aggregate_metrics.csv"),
        "stability_sha256": sha256_file(destination / "edge_stability.csv"),
    }
    experiment_id = stable_payload_id(identity_payload)
    manifest = {
        **identity_payload,
        "experiment_id": experiment_id,
        "engineering_status": "pass",
        "evidence_scope": resolved_evidence_scope,
        "best_sparse_model": best_sparse,
        "winner": aggregate_rows[0]["model_name"],
        "performance_checks": performance_checks,
        "significance_checks": significance_checks,
        "stability_checks": stability_checks,
        "experiment_supports_graph_claim": experiment_supports_claim,
        "graph_research_claim": (
            "go" if experiment_supports_claim else "no-claim"
        ),
        "files": {
            name: {"path": name, "sha256": sha256_file(destination / name)}
            for name in (
                "per_seed_metrics.csv",
                "aggregate_metrics.csv",
                "test_predictions.csv",
                "paired_bootstrap.csv",
                "edge_stability.csv",
                "summary.md",
            )
        },
    }
    write_json(destination / "manifest.json", manifest)
    return SparseGraphExperimentResult(destination, experiment_id, manifest)
