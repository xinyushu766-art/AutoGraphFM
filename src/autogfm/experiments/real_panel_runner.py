"""Multi-seed residual GRU experiments on leakage-safe real-data panels."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from autogfm.models import LastValueForecaster, ResidualTemporalGRU, SafeGraphResidualGRU
from autogfm.realdata import PanelData, RealDataError, build_window_split, load_panel
from autogfm.realdata.common import sha256_file, stable_payload_id, write_csv_rows, write_json


@dataclass(frozen=True)
class RealPanelExperimentSpec:
    history: int = 12
    horizon: int = 3
    hidden_dim: int = 32
    batch_size: int = 8
    max_epochs: int = 40
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seeds: tuple[int, ...] = (42, 43, 44)

    def validate(self) -> None:
        if self.history < 2 or self.horizon < 1:
            raise RealDataError("history must be >=2 and horizon must be positive.")
        if self.hidden_dim < 1 or self.batch_size < 1:
            raise RealDataError("hidden_dim and batch_size must be positive.")
        if self.max_epochs < 1 or self.patience < 1:
            raise RealDataError("max_epochs and patience must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise RealDataError("invalid optimizer configuration.")
        if self.gradient_clip <= 0:
            raise RealDataError("gradient_clip must be positive.")
        if not self.seeds:
            raise RealDataError("at least one seed is required.")


@dataclass(frozen=True)
class RealPanelExperimentResult:
    output_dir: Path
    experiment_id: str
    manifest: dict[str, Any]


class _PanelWindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        panel: PanelData,
        origins: tuple[int, ...],
        history: int,
        horizon: int,
    ) -> None:
        self.panel = panel
        self.origins = origins
        self.history = history
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        origin = self.origins[index]
        x = self.panel.features[origin - self.history + 1 : origin + 1]
        target_sales = self.panel.sales[origin + 1 : origin + self.horizon + 1]
        y = np.log1p(np.clip(target_sales, 0.0, None)).astype(np.float32)
        mask = self.panel.observed_mask[
            origin + 1 : origin + self.horizon + 1
        ].astype(np.float32)
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y),
            torch.from_numpy(mask),
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def _metrics(
    prediction_log: np.ndarray,
    target_log: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    selected = mask > 0.5
    prediction = np.expm1(np.clip(prediction_log[selected], 0.0, 20.0))
    target = np.expm1(np.clip(target_log[selected], 0.0, 20.0))
    if prediction.size == 0:
        raise RealDataError("Evaluation mask contains no observed targets.")
    error = prediction - target
    absolute = np.abs(error)
    return {
        "log_mse": float(np.mean((prediction_log[selected] - target_log[selected]) ** 2)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_percent": float(100.0 * absolute.sum() / max(np.abs(target).sum(), 1e-8)),
        "smape_percent": float(
            100.0 * np.mean(2.0 * absolute / (np.abs(prediction) + np.abs(target) + 1e-8))
        ),
        "mean_prediction_sales": float(prediction.mean()),
        "mean_target_sales": float(target.mean()),
        "bias": float(error.mean()),
        "forecast_cells": int(prediction.size),
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    adjacency: Tensor | None,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    graph = adjacency.to(device) if adjacency is not None else None
    for x, y, mask in loader:
        x = x.to(device)
        prediction = model(x, graph)
        predictions.append(prediction.cpu().numpy())
        targets.append(y.numpy())
        masks.append(mask.numpy())
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    mask_array = np.concatenate(masks, axis=0)
    return (
        _metrics(prediction_array, target_array, mask_array),
        prediction_array,
        target_array,
        mask_array,
    )


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    adjacency: Tensor | None,
    spec: RealPanelExperimentSpec,
    device: torch.device,
) -> tuple[int, float]:
    model.to(device)
    graph = adjacency.to(device) if adjacency is not None else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    best_state: dict[str, Tensor] | None = None
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, spec.max_epochs + 1):
        model.train()
        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, graph)
            loss = _masked_mse(prediction, y, mask)
            regularizer = getattr(model, "regularization_loss", None)
            if regularizer is not None:
                loss = loss + regularizer(graph)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            optimizer.step()

        validation, _, _, _ = _evaluate(
            model, validation_loader, adjacency, device
        )
        validation_loss = validation["log_mse"]
        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return best_epoch, best_validation


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _graph_scale(model: nn.Module) -> float:
    value = getattr(model, "graph_scale", None)
    if value is None:
        return 0.0
    return float(torch.tanh(value.detach()).abs().mean().cpu().item())


def run_real_panel_experiments(
    panel_dir: str | Path,
    graph_dir: str | Path,
    output_dir: str | Path,
    spec: RealPanelExperimentSpec | None = None,
    *,
    overwrite: bool = False,
) -> RealPanelExperimentResult:
    """Run residual temporal and graph controls over multiple model seeds."""
    spec = spec or RealPanelExperimentSpec()
    spec.validate()
    panel = load_panel(panel_dir)
    split = build_window_split(panel, spec.history, spec.horizon)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    train_dataset = _PanelWindowDataset(panel, split.train_origins, spec.history, spec.horizon)
    validation_dataset = _PanelWindowDataset(
        panel, split.validation_origins, spec.history, spec.horizon
    )
    test_dataset = _PanelWindowDataset(panel, split.test_origins, spec.history, spec.horizon)
    adjacency_map = {
        "safe_static_graph_gru": torch.from_numpy(
            np.load(Path(graph_dir) / "adjacency.npy").astype(np.float32)
        ),
        "safe_identity_graph_gru": torch.from_numpy(
            np.load(Path(graph_dir) / "identity_adjacency.npy").astype(np.float32)
        ),
        "safe_permuted_graph_gru": torch.from_numpy(
            np.load(Path(graph_dir) / "permuted_adjacency.npy").astype(np.float32)
        ),
    }
    num_features = panel.features.shape[-1]
    factories: dict[str, tuple[Callable[[], nn.Module], bool]] = {
        "last_value": (
            lambda: LastValueForecaster(horizon=spec.horizon),
            False,
        ),
        "residual_temporal_gru": (
            lambda: ResidualTemporalGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            True,
        ),
        "safe_static_graph_gru": (
            lambda: SafeGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            True,
        ),
        "safe_identity_graph_gru": (
            lambda: SafeGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            True,
        ),
        "safe_permuted_graph_gru": (
            lambda: SafeGraphResidualGRU(
                num_features=num_features,
                hidden_dim=spec.hidden_dim,
                horizon=spec.horizon,
            ),
            True,
        ),
    }

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
            validation_dataset, batch_size=spec.batch_size, shuffle=False
        )
        test_loader = DataLoader(test_dataset, batch_size=spec.batch_size, shuffle=False)
        for model_name, (factory, trainable) in factories.items():
            _set_seed(seed)
            model = factory()
            adjacency = adjacency_map.get(model_name)
            if trainable:
                best_epoch, best_validation = _train(
                    model,
                    train_loader,
                    validation_loader,
                    adjacency,
                    spec,
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
            per_seed_rows.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "trainable_parameters": _parameter_count(model),
                    "best_epoch": best_epoch,
                    "best_validation_log_mse": best_validation,
                    "graph_scale_mean_abs": _graph_scale(model),
                    **test_metrics,
                }
            )
            for sample_index, origin in enumerate(split.test_origins):
                for horizon_index in range(spec.horizon):
                    for series_index in np.flatnonzero(masks[sample_index, horizon_index] > 0.5):
                        prediction_rows.append(
                            {
                                "seed": seed,
                                "model_name": model_name,
                                "origin_index": origin,
                                "origin_month": panel.months[origin],
                                "horizon": horizon_index + 1,
                                "target_month": panel.months[origin + horizon_index + 1],
                                "series_index": int(series_index),
                                "series_key": panel.series_keys[int(series_index)],
                                "target": float(np.expm1(targets[sample_index, horizon_index, series_index])),
                                "prediction": float(np.expm1(np.clip(predictions[sample_index, horizon_index, series_index], 0.0, 20.0))),
                            }
                        )

    write_csv_rows(
        destination / "per_seed_metrics.csv",
        list(per_seed_rows[0].keys()),
        per_seed_rows,
    )
    write_csv_rows(
        destination / "test_predictions.csv",
        list(prediction_rows[0].keys()),
        prediction_rows,
    )

    aggregate_rows: list[dict[str, Any]] = []
    for model_name in factories:
        rows = [row for row in per_seed_rows if row["model_name"] == model_name]
        values = np.array([row["wape_percent"] for row in rows], dtype=np.float64)
        aggregate_rows.append(
            {
                "model_name": model_name,
                "runs": len(rows),
                "mean_test_wape_percent": float(values.mean()),
                "std_test_wape_percent": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low_test_wape_percent": float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float(values.mean()),
                "ci95_high_test_wape_percent": float(values.mean() + 1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float(values.mean()),
                "mean_test_mae": float(np.mean([row["mae"] for row in rows])),
                "mean_test_rmse": float(np.mean([row["rmse"] for row in rows])),
                "mean_bias": float(np.mean([row["bias"] for row in rows])),
                "mean_best_epoch": float(np.mean([row["best_epoch"] for row in rows])),
                "mean_graph_scale_abs": float(np.mean([row["graph_scale_mean_abs"] for row in rows])),
            }
        )
    aggregate_rows.sort(key=lambda row: row["mean_test_wape_percent"])
    write_csv_rows(
        destination / "aggregate_metrics.csv",
        list(aggregate_rows[0].keys()),
        aggregate_rows,
    )

    lines = [
        "# Stage 2.3 Real-Panel Multi-Seed Experiment",
        "",
        f"- Device: `{device}`",
        f"- Seeds: `{list(spec.seeds)}`",
        f"- Train/validation/test windows: `{len(split.train_origins)}/{len(split.validation_origins)}/{len(split.test_origins)}`",
        "",
        "| Rank | Model | Mean WAPE | 95% CI | MAE | RMSE | Bias | Graph scale |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(aggregate_rows, start=1):
        lines.append(
            f"| {rank} | {row['model_name']} | {row['mean_test_wape_percent']:.3f}% | "
            f"[{row['ci95_low_test_wape_percent']:.3f}, {row['ci95_high_test_wape_percent']:.3f}] | "
            f"{row['mean_test_mae']:.3f} | {row['mean_test_rmse']:.3f} | "
            f"{row['mean_bias']:.3f} | {row['mean_graph_scale_abs']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim gate",
            "",
            "A graph-effect claim is allowed only when the static graph beats the temporal, identity, and permuted controls across seeds.",
            "",
        ]
    )
    (destination / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    by_name = {row["model_name"]: row for row in aggregate_rows}
    static_wape = by_name["safe_static_graph_gru"]["mean_test_wape_percent"]
    claim_gate = {
        "beats_last_value": static_wape < by_name["last_value"]["mean_test_wape_percent"],
        "beats_temporal": static_wape < by_name["residual_temporal_gru"]["mean_test_wape_percent"],
        "beats_identity": static_wape < by_name["safe_identity_graph_gru"]["mean_test_wape_percent"],
        "beats_permuted": static_wape < by_name["safe_permuted_graph_gru"]["mean_test_wape_percent"],
    }
    identity_payload = {
        "schema_version": "autogfm-real-panel-experiment-v2",
        "panel_id": panel.manifest["panel_id"],
        "graph_sha256": sha256_file(Path(graph_dir) / "adjacency.npy"),
        "spec": asdict(spec),
        "aggregate_sha256": sha256_file(destination / "aggregate_metrics.csv"),
    }
    experiment_id = stable_payload_id(identity_payload)
    manifest = {
        **identity_payload,
        "experiment_id": experiment_id,
        "device": str(device),
        "claim_gate": claim_gate,
        "winner": aggregate_rows[0]["model_name"],
        "files": {
            name: {"path": name, "sha256": sha256_file(destination / name)}
            for name in (
                "per_seed_metrics.csv", "aggregate_metrics.csv",
                "test_predictions.csv", "summary.md",
            )
        },
    }
    write_json(destination / "manifest.json", manifest)
    return RealPanelExperimentResult(destination, experiment_id, manifest)
