"""Stage 11 vehicle-panel graph falsification and adaptive graph experiments."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

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
)
from autogfm.realdata import RealDataError, build_window_split, load_panel
from autogfm.realdata.common import (
    sha256_file,
    stable_payload_id,
    write_csv_rows,
    write_json,
)


@dataclass(frozen=True)
class AdaptiveGraphExperimentSpec:
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
class AdaptiveGraphExperimentResult:
    output_dir: Path
    experiment_id: str
    manifest: dict[str, Any]


def _learned_graph(model: nn.Module, prior: Tensor | None) -> Tensor | None:
    method = getattr(model, "learned_adjacency", None)
    if method is None:
        return None
    with torch.no_grad():
        return method(prior).detach().cpu()


def _graph_diagnostics(adjacency: Tensor | None) -> dict[str, float]:
    if adjacency is None:
        return {
            "graph_entropy_mean": 0.0,
            "graph_diagonal_mass_mean": 0.0,
            "graph_asymmetry_mean_abs": 0.0,
            "graph_nonzero_mean": 0.0,
        }
    graph = adjacency.clamp_min(0.0)
    rows = graph / graph.sum(-1, keepdim=True).clamp_min(1e-8)
    entropy = -(rows * rows.clamp_min(1e-8).log()).sum(-1)
    return {
        "graph_entropy_mean": float(entropy.mean()),
        "graph_diagonal_mass_mean": float(torch.diagonal(rows).mean()),
        "graph_asymmetry_mean_abs": float((rows - rows.T).abs().mean()),
        "graph_nonzero_mean": float((rows > 1e-8).sum(-1).float().mean()),
    }


def _prior_weight(model: nn.Module) -> float:
    value = getattr(model, "prior_logit", None)
    if value is None:
        return 0.0
    return float(torch.sigmoid(value.detach()).cpu())


def _paired_bootstrap(
    predictions: list[dict[str, Any]],
    challenger: str,
    control: str,
    samples: int,
    seed: int = 2026,
) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in predictions:
        if row["model_name"] not in {challenger, control}:
            continue
        key = (
            int(row["seed"]),
            int(row["origin_index"]),
            int(row["horizon"]),
            int(row["series_index"]),
        )
        by_key.setdefault(key, {})[str(row["model_name"])] = float(row["prediction"])
        by_key[key]["target"] = float(row["target"])
    matched = [
        values
        for values in by_key.values()
        if challenger in values and control in values
    ]
    if not matched:
        raise RealDataError("No paired prediction cells were available.")
    target = np.asarray([row["target"] for row in matched], dtype=float)
    challenger_error = np.abs(
        np.asarray([row[challenger] for row in matched], dtype=float) - target
    )
    control_error = np.abs(
        np.asarray([row[control] for row in matched], dtype=float) - target
    )
    denominator = max(np.abs(target).sum(), 1e-9)
    observed = 100.0 * (challenger_error.sum() - control_error.sum()) / denominator
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(target), size=len(target))
        denom = max(np.abs(target[selected]).sum(), 1e-9)
        estimates[index] = 100.0 * (
            challenger_error[selected].sum() - control_error[selected].sum()
        ) / denom
    return {
        "challenger": challenger,
        "control": control,
        "paired_cells": len(target),
        "wape_difference_percentage_points": float(observed),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "challenger_significantly_better": bool(np.quantile(estimates, 0.975) < 0.0),
    }


def run_adaptive_graph_experiments(
    panel_dir: str | Path,
    graph_dir: str | Path,
    output_dir: str | Path,
    spec: AdaptiveGraphExperimentSpec | None = None,
    *,
    evidence_scope: str | None = None,
    overwrite: bool = False,
) -> AdaptiveGraphExperimentResult:
    """Run temporal, fixed-graph and adaptive-graph controls on one panel."""
    spec = spec or AdaptiveGraphExperimentSpec()
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
        "adaptive_agcrn_graph_gru": (
            lambda: AdaptiveGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
                num_nodes=num_nodes,
                graph_mode="agcrn",
                embedding_dim=spec.embedding_dim,
                top_k=top_k,
            ),
            None,
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
        "hybrid_agcrn_static_graph_gru": (
            lambda: AdaptiveGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
                num_nodes=num_nodes,
                graph_mode="agcrn",
                embedding_dim=spec.embedding_dim,
                top_k=top_k,
                use_prior=True,
            ),
            static,
            True,
        ),
        "hybrid_mtgnn_static_graph_gru": (
            lambda: AdaptiveGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
                num_nodes=num_nodes,
                graph_mode="mtgnn",
                embedding_dim=spec.embedding_dim,
                top_k=top_k,
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
                np.save(
                    graph_output / f"{model_name}_seed{seed}.npy",
                    learned.numpy().astype(np.float32),
                )
            per_seed_rows.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "trainable_parameters": _parameter_count(model),
                    "best_epoch": best_epoch,
                    "best_validation_log_mse": best_validation,
                    "graph_scale_mean_abs": _graph_scale(model),
                    "learned_graph_weight": _prior_weight(model),
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
                "mean_learned_graph_weight": float(
                    np.mean([row["learned_graph_weight"] for row in rows])
                ),
                "mean_graph_asymmetry": float(
                    np.mean([row["graph_asymmetry_mean_abs"] for row in rows])
                ),
            }
        )
    aggregate_rows.sort(key=lambda row: row["mean_test_wape_percent"])
    write_csv_rows(
        destination / "aggregate_metrics.csv",
        list(aggregate_rows[0]),
        aggregate_rows,
    )

    adaptive_names = {
        name for name in factories if name.startswith(("adaptive_", "hybrid_"))
    }
    best_adaptive = min(
        (row for row in aggregate_rows if row["model_name"] in adaptive_names),
        key=lambda row: row["mean_test_wape_percent"],
    )["model_name"]
    controls = [
        "residual_temporal_gru",
        "safe_static_graph_gru",
        "safe_identity_graph_gru",
        "safe_permuted_graph_gru",
    ]
    bootstrap_rows = [
        _paired_bootstrap(
            prediction_rows,
            best_adaptive,
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
    claim_checks = {
        f"beats_{control}": (
            by_name[best_adaptive]["mean_test_wape_percent"]
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
    claim_authorized = all(claim_checks.values()) and all(
        significance_checks.values()
    )

    summary_lines = [
        "# Stage 11 Adaptive Graph Experiment",
        "",
        f"- Device: `{device}`",
        f"- Evidence scope: `{resolved_evidence_scope}`",
        f"- Best adaptive model: `{best_adaptive}`",
        f"- Graph claim: `{'authorized' if claim_authorized else 'no-claim'}`",
        "",
        "| Rank | Model | Mean WAPE | MAE | RMSE | Graph scale |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(aggregate_rows, start=1):
        summary_lines.append(
            f"| {rank} | {row['model_name']} | "
            f"{row['mean_test_wape_percent']:.3f}% | "
            f"{row['mean_test_mae']:.3f} | {row['mean_test_rmse']:.3f} | "
            f"{row['mean_graph_scale_abs']:.4f} |"
        )
    summary_lines.extend(
        [
            "",
            "The adaptive graph claim requires lower mean WAPE and a paired "
            "bootstrap upper confidence bound below zero against temporal, "
            "static, identity, and permuted controls.",
            "",
        ]
    )
    (destination / "summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    identity_payload = {
        "schema_version": "autogfm-stage11-adaptive-graph-v1",
        "panel_id": panel.manifest["panel_id"],
        "panel_sha256": sha256_file(Path(panel_dir) / "panel.npz"),
        "static_graph_sha256": sha256_file(Path(graph_dir) / "adjacency.npy"),
        "spec": asdict(spec),
        "aggregate_sha256": sha256_file(destination / "aggregate_metrics.csv"),
    }
    experiment_id = stable_payload_id(identity_payload)
    manifest = {
        **identity_payload,
        "experiment_id": experiment_id,
        "engineering_status": "pass",
        "evidence_scope": resolved_evidence_scope,
        "best_adaptive_model": best_adaptive,
        "winner": aggregate_rows[0]["model_name"],
        "claim_checks": claim_checks,
        "significance_checks": significance_checks,
        "graph_claim_authorized": claim_authorized,
        "graph_research_claim": "go" if claim_authorized else "no-claim",
        "files": {
            name: {"path": name, "sha256": sha256_file(destination / name)}
            for name in (
                "per_seed_metrics.csv",
                "aggregate_metrics.csv",
                "test_predictions.csv",
                "paired_bootstrap.csv",
                "summary.md",
            )
        },
    }
    write_json(destination / "manifest.json", manifest)
    return AdaptiveGraphExperimentResult(destination, experiment_id, manifest)
