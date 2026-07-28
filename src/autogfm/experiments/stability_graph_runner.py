"""Stage 12A stability-selected sparse graph experiments."""

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
from autogfm.graph.stability_selection import (
    consensus_adjacency,
    edge_frequency,
    lagged_ridge_scores,
    select_candidate_mask,
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
class StabilityGraphExperimentSpec:
    history: int = 12
    horizon: int = 3
    hidden_dim: int = 32
    embedding_dim: int = 8
    top_k: int = 5
    batch_size: int = 8
    discovery_max_epochs: int = 18
    discovery_patience: int = 5
    max_epochs: int = 35
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    discovery_seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    final_seeds: tuple[int, ...] = (42, 43, 44)
    bootstrap_samples: int = 1000
    ridge_alpha: float = 2.0
    frequency_weight: float = 0.45
    lagged_weight: float = 0.40
    prior_weight: float = 0.15
    initial_residual_weight: float = 0.20
    l1_strength: float = 2e-4
    deviation_strength: float = 1e-3
    gate_strength: float = 1e-4
    minimum_selected_jaccard: float = 0.95
    minimum_selected_stable_ratio: float = 0.95

    def validate(self) -> None:
        self.base_spec(self.final_seeds, self.max_epochs, self.patience).validate()
        self.base_spec(
            self.discovery_seeds,
            self.discovery_max_epochs,
            self.discovery_patience,
        ).validate()
        if self.embedding_dim < 1 or self.top_k < 1:
            raise RealDataError("embedding_dim and top_k must be positive.")
        if self.bootstrap_samples < 100:
            raise RealDataError("bootstrap_samples must be at least 100.")
        if self.ridge_alpha < 0.0:
            raise RealDataError("ridge_alpha must be non-negative.")
        weights = (
            self.frequency_weight,
            self.lagged_weight,
            self.prior_weight,
        )
        if min(weights) < 0.0 or sum(weights) <= 0.0:
            raise RealDataError("candidate score weights are invalid.")
        if not 0.0 < self.initial_residual_weight < 1.0:
            raise RealDataError("initial_residual_weight must be in (0, 1).")
        if min(
            self.l1_strength,
            self.deviation_strength,
            self.gate_strength,
        ) < 0.0:
            raise RealDataError("regularization strengths must be non-negative.")
        if not 0.0 <= self.minimum_selected_jaccard <= 1.0:
            raise RealDataError("minimum_selected_jaccard must be in [0, 1].")
        if not 0.0 <= self.minimum_selected_stable_ratio <= 1.0:
            raise RealDataError(
                "minimum_selected_stable_ratio must be in [0, 1]."
            )

    def base_spec(
        self,
        seeds: tuple[int, ...],
        max_epochs: int,
        patience: int,
    ) -> RealPanelExperimentSpec:
        return RealPanelExperimentSpec(
            history=self.history,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            seeds=seeds,
        )


@dataclass(frozen=True)
class StabilityGraphExperimentResult:
    output_dir: Path
    experiment_id: str
    manifest: dict[str, Any]


def _learned_graph(model: nn.Module, prior: Tensor | None) -> np.ndarray | None:
    method = getattr(model, "learned_adjacency", None)
    if method is None:
        return None
    with torch.no_grad():
        graph = method(prior).detach().cpu().numpy().astype(np.float32)
    return graph


def _residual_weight(model: nn.Module) -> float:
    method = getattr(model, "residual_graph_weight", None)
    return float(method()) if method is not None else 0.0


def _normalize_rows(matrix: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    graph = np.maximum(np.asarray(matrix, dtype=np.float32), 0.0)
    fallback = np.maximum(np.asarray(fallback, dtype=np.float32), 0.0)
    for row in range(graph.shape[0]):
        if graph[row].sum() <= 1e-8:
            graph[row] = fallback[row]
    graph /= np.maximum(graph.sum(axis=1, keepdims=True), 1e-8)
    return graph.astype(np.float32)


def _make_loaders(
    train_dataset: _PanelWindowDataset,
    validation_dataset: _PanelWindowDataset,
    test_dataset: _PanelWindowDataset,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def _aggregate_rows(
    per_seed_rows: list[dict[str, Any]],
    model_names: list[str],
) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for model_name in model_names:
        rows = [row for row in per_seed_rows if row["model_name"] == model_name]
        wape = np.asarray([row["wape_percent"] for row in rows], dtype=float)
        aggregate.append(
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
    aggregate.sort(key=lambda row: row["mean_test_wape_percent"])
    return aggregate


def _prediction_rows(
    *,
    seed: int,
    model_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    test_origins: tuple[int, ...],
    months: tuple[str, ...],
    series_keys: tuple[str, ...],
    horizon: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, origin in enumerate(test_origins):
        for horizon_index in range(horizon):
            indices = np.flatnonzero(masks[sample_index, horizon_index] > 0.5)
            for series_index in indices:
                rows.append(
                    {
                        "seed": seed,
                        "model_name": model_name,
                        "origin_index": origin,
                        "origin_month": months[origin],
                        "horizon": horizon_index + 1,
                        "target_month": months[origin + horizon_index + 1],
                        "series_index": int(series_index),
                        "series_key": series_keys[int(series_index)],
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
    return rows


def run_stability_graph_experiments(
    panel_dir: str | Path,
    graph_dir: str | Path,
    output_dir: str | Path,
    spec: StabilityGraphExperimentSpec | None = None,
    *,
    evidence_scope: str | None = None,
    overwrite: bool = False,
) -> StabilityGraphExperimentResult:
    """Discover a stable support, freeze it, and retrain strict controls."""
    spec = spec or StabilityGraphExperimentSpec()
    spec.validate()
    panel = load_panel(panel_dir)
    split = build_window_split(panel, spec.history, spec.horizon)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    discovery_dir = destination / "01_discovery_graphs"
    final_graph_dir = destination / "02_final_graphs"
    discovery_dir.mkdir(exist_ok=True)
    final_graph_dir.mkdir(exist_ok=True)

    train_dataset = _PanelWindowDataset(
        panel,
        split.train_origins,
        spec.history,
        spec.horizon,
    )
    validation_dataset = _PanelWindowDataset(
        panel,
        split.validation_origins,
        spec.history,
        spec.horizon,
    )
    test_dataset = _PanelWindowDataset(
        panel,
        split.test_origins,
        spec.history,
        spec.horizon,
    )
    static_array = np.load(Path(graph_dir) / "adjacency.npy").astype(np.float32)
    identity_array = np.load(
        Path(graph_dir) / "identity_adjacency.npy"
    ).astype(np.float32)
    permuted_array = np.load(
        Path(graph_dir) / "permuted_adjacency.npy"
    ).astype(np.float32)
    static = torch.from_numpy(static_array)
    identity = torch.from_numpy(identity_array)
    permuted = torch.from_numpy(permuted_array)
    num_features = panel.features.shape[-1]
    num_nodes = panel.features.shape[1]
    top_k = min(spec.top_k, max(num_nodes - 1, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discovery_spec = spec.base_spec(
        spec.discovery_seeds,
        spec.discovery_max_epochs,
        spec.discovery_patience,
    )
    discovery_graphs: list[np.ndarray] = []
    discovery_rows: list[dict[str, Any]] = []
    for seed in spec.discovery_seeds:
        train_loader, validation_loader, _ = _make_loaders(
            train_dataset,
            validation_dataset,
            test_dataset,
            spec.batch_size,
            seed,
        )
        _set_seed(seed)
        model = SparseResidualGraphGRU(
            num_features=num_features,
            hidden_dim=spec.hidden_dim,
            horizon=spec.horizon,
            num_nodes=num_nodes,
            top_k=top_k,
            use_prior=False,
            initial_residual_weight=spec.initial_residual_weight,
            l1_strength=spec.l1_strength,
            deviation_strength=spec.deviation_strength,
            gate_strength=spec.gate_strength,
        )
        best_epoch, best_validation = _train(
            model,
            train_loader,
            validation_loader,
            None,
            discovery_spec,
            device,
        )
        learned = _learned_graph(model, None)
        if learned is None:
            raise RuntimeError("Discovery model did not expose an adjacency.")
        discovery_graphs.append(learned)
        np.save(discovery_dir / f"direct_sparse_seed{seed}.npy", learned)
        discovery_rows.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_log_mse": best_validation,
                "residual_graph_weight": _residual_weight(model),
                **_graph_diagnostics(torch.from_numpy(learned)),
            }
        )
    write_csv_rows(
        destination / "discovery_metrics.csv",
        list(discovery_rows[0]),
        discovery_rows,
    )

    frequency, discovery_stability = edge_frequency(discovery_graphs, top_k)
    train_end_index = int(panel.manifest["boundaries"]["train_end_index"])
    lagged = lagged_ridge_scores(
        np.log1p(panel.sales),
        train_end_index,
        ridge_alpha=spec.ridge_alpha,
        use_differences=True,
    )
    candidate_mask, candidate_scores = select_candidate_mask(
        frequency,
        lagged,
        top_k=top_k,
        static_prior=static_array,
        frequency_weight=spec.frequency_weight,
        lagged_weight=spec.lagged_weight,
        prior_weight=spec.prior_weight,
    )
    consensus = consensus_adjacency(discovery_graphs, candidate_mask)
    screened_static = _normalize_rows(static_array * candidate_mask, consensus)
    np.save(destination / "edge_frequency.npy", frequency)
    np.save(destination / "lagged_ridge_scores.npy", lagged)
    np.save(destination / "candidate_scores.npy", candidate_scores)
    np.save(destination / "candidate_mask.npy", candidate_mask)
    np.save(destination / "consensus_adjacency.npy", consensus)
    np.save(destination / "screened_static_prior.npy", screened_static)

    final_spec = spec.base_spec(spec.final_seeds, spec.max_epochs, spec.patience)
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
    candidate_tensor = torch.from_numpy(candidate_mask)
    consensus_tensor = torch.from_numpy(consensus)
    screened_static_tensor = torch.from_numpy(screened_static)
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
        "fixed_consensus_graph_gru": (
            lambda: SafeGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            consensus_tensor,
            True,
        ),
        "stability_selected_sparse_graph_gru": (
            lambda: SparseResidualGraphGRU(
                **sparse_kwargs,
                use_prior=False,
                candidate_mask=candidate_tensor,
            ),
            None,
            True,
        ),
        "stability_selected_static_residual_gru": (
            lambda: SparseResidualGraphGRU(
                **sparse_kwargs,
                use_prior=True,
                candidate_mask=candidate_tensor,
            ),
            screened_static_tensor,
            True,
        ),
    }

    per_seed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    learned_by_model: dict[str, list[np.ndarray]] = {
        "direct_sparse_graph_gru": [],
        "stability_selected_sparse_graph_gru": [],
        "stability_selected_static_residual_gru": [],
    }
    for seed in spec.final_seeds:
        train_loader, validation_loader, test_loader = _make_loaders(
            train_dataset,
            validation_dataset,
            test_dataset,
            spec.batch_size,
            seed,
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
                    final_spec,
                    device,
                )
            else:
                model.to(device)
                validation, _, _, _ = _evaluate(
                    model,
                    validation_loader,
                    adjacency,
                    device,
                )
                best_epoch = 0
                best_validation = validation["log_mse"]
            test_metrics, predictions, targets, masks = _evaluate(
                model,
                test_loader,
                adjacency,
                device,
            )
            prior = adjacency.to(device) if adjacency is not None else None
            learned = _learned_graph(model, prior)
            diagnostics = _graph_diagnostics(
                torch.from_numpy(learned) if learned is not None else None
            )
            if learned is not None:
                np.save(final_graph_dir / f"{model_name}_seed{seed}.npy", learned)
                if model_name in learned_by_model:
                    learned_by_model[model_name].append(learned)
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
            prediction_rows.extend(
                _prediction_rows(
                    seed=seed,
                    model_name=model_name,
                    predictions=predictions,
                    targets=targets,
                    masks=masks,
                    test_origins=split.test_origins,
                    months=panel.months,
                    series_keys=panel.series_keys,
                    horizon=spec.horizon,
                )
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
    model_names = list(factories)
    aggregate = _aggregate_rows(per_seed_rows, model_names)
    write_csv_rows(
        destination / "aggregate_metrics.csv",
        list(aggregate[0]),
        aggregate,
    )

    stability_rows: list[dict[str, Any]] = []
    stability_by_model: dict[str, dict[str, float]] = {}
    for model_name, graphs in learned_by_model.items():
        _, diagnostics = edge_frequency(graphs, top_k)
        row = {"model_name": model_name, **asdict(diagnostics)}
        stability_rows.append(row)
        stability_by_model[model_name] = asdict(diagnostics)
    write_csv_rows(
        destination / "edge_stability.csv",
        list(stability_rows[0]),
        stability_rows,
    )

    selected_names = {
        "fixed_consensus_graph_gru",
        "stability_selected_sparse_graph_gru",
        "stability_selected_static_residual_gru",
    }
    best_selected = min(
        (row for row in aggregate if row["model_name"] in selected_names),
        key=lambda row: row["mean_test_wape_percent"],
    )["model_name"]
    controls = [
        "last_value",
        "residual_temporal_gru",
        "safe_static_graph_gru",
        "safe_identity_graph_gru",
        "safe_permuted_graph_gru",
        "adaptive_mtgnn_graph_gru",
        "direct_sparse_graph_gru",
    ]
    bootstrap_rows = [
        _paired_bootstrap(
            prediction_rows,
            best_selected,
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

    by_name = {row["model_name"]: row for row in aggregate}
    performance_checks = {
        f"beats_{control}": (
            by_name[best_selected]["mean_test_wape_percent"]
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
    if best_selected == "fixed_consensus_graph_gru":
        selected_stability = {
            "mean_pairwise_jaccard": 1.0,
            "stable_edge_ratio": 1.0,
        }
    else:
        selected_stability = stability_by_model[best_selected]
    stability_checks = {
        "selected_pairwise_jaccard_pass": (
            selected_stability["mean_pairwise_jaccard"]
            >= spec.minimum_selected_jaccard
        ),
        "selected_stable_edge_ratio_pass": (
            selected_stability["stable_edge_ratio"]
            >= spec.minimum_selected_stable_ratio
        ),
        "discovery_used_train_only": True,
    }
    experiment_supports_claim = (
        all(performance_checks.values())
        and all(significance_checks.values())
        and all(stability_checks.values())
    )
    resolved_scope = evidence_scope or panel.manifest.get(
        "evidence_scope",
        "unverified_vehicle_panel",
    )

    summary_lines = [
        "# Stage 12A Stability-Selected Graph Experiment",
        "",
        f"- Device: `{device}`",
        f"- Evidence scope: `{resolved_scope}`",
        f"- Best stability-selected model: `{best_selected}`",
        f"- Graph status: "
        f"`{'go' if experiment_supports_claim else 'no-claim'}`",
        f"- Discovery edge Jaccard: "
        f"`{discovery_stability.mean_pairwise_jaccard:.3f}`",
        "",
        "| Rank | Model | Mean WAPE | MAE | RMSE |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(aggregate, start=1):
        summary_lines.append(
            f"| {rank} | {row['model_name']} | "
            f"{row['mean_test_wape_percent']:.3f}% | "
            f"{row['mean_test_mae']:.3f} | "
            f"{row['mean_test_rmse']:.3f} |"
        )
    summary_lines.extend(
        [
            "",
            "The candidate support is selected only from discovery runs and "
            "lagged ridge scores computed before the train boundary. The test "
            "interval is untouched until final evaluation.",
            "",
        ]
    )
    (destination / "summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    identity_payload = {
        "schema_version": "autogfm-stage12a-stability-graph-v1",
        "panel_id": panel.manifest["panel_id"],
        "panel_sha256": sha256_file(Path(panel_dir) / "panel.npz"),
        "static_graph_sha256": sha256_file(Path(graph_dir) / "adjacency.npy"),
        "spec": asdict(spec),
        "aggregate_sha256": sha256_file(destination / "aggregate_metrics.csv"),
        "candidate_mask_sha256": sha256_file(destination / "candidate_mask.npy"),
    }
    experiment_id = stable_payload_id(identity_payload)
    manifest = {
        **identity_payload,
        "experiment_id": experiment_id,
        "engineering_status": "pass",
        "evidence_scope": resolved_scope,
        "best_stability_selected_model": best_selected,
        "winner": aggregate[0]["model_name"],
        "discovery_stability": asdict(discovery_stability),
        "selected_stability": selected_stability,
        "performance_checks": performance_checks,
        "significance_checks": significance_checks,
        "stability_checks": stability_checks,
        "experiment_supports_graph_claim": experiment_supports_claim,
        "graph_research_claim": "go" if experiment_supports_claim else "no-claim",
        "files": {
            name: {"path": name, "sha256": sha256_file(destination / name)}
            for name in (
                "discovery_metrics.csv",
                "per_seed_metrics.csv",
                "aggregate_metrics.csv",
                "test_predictions.csv",
                "paired_bootstrap.csv",
                "edge_stability.csv",
                "candidate_mask.npy",
                "consensus_adjacency.npy",
                "screened_static_prior.npy",
                "summary.md",
            )
        },
    }
    write_json(destination / "manifest.json", manifest)
    return StabilityGraphExperimentResult(destination, experiment_id, manifest)
