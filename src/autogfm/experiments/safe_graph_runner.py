"""Early-stopped safe graph comparison for Stage 0.2E."""

from __future__ import annotations

import copy
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.data import SyntheticAutomotiveDataset
from autogfm.experiments.runner import count_trainable_parameters
from autogfm.graph import make_identity_graph, make_permuted_graph
from autogfm.metrics import compute_forecast_metrics, inverse_log_sales
from autogfm.models import (
    LastValueForecaster,
    ResidualTemporalGRU,
    SafeGraphResidualGRU,
)


@dataclass(frozen=True)
class ChronologicalSplit:
    """Purged train, validation, and test subsets."""

    train: Subset
    validation: Subset
    test: Subset


@dataclass(frozen=True)
class SafeGraphExperimentResult:
    """Final held-out test result."""

    model_name: str
    trainable_parameters: int
    best_epoch: int
    best_validation_log_mse: float
    test_log_mse: float
    test_mae: float
    test_rmse: float
    test_wape_percent: float
    test_smape_percent: float
    mean_prediction_sales: float
    mean_target_sales: float
    mean_bias_sales: float
    graph_scale_mean_abs: float


@dataclass(frozen=True)
class ModelSpecification:
    """Factory and graph input for one controlled model."""

    name: str
    factory: Callable[[], nn.Module]
    adjacency: Tensor | None
    trainable: bool


def set_seed(seed: int) -> None:
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def chronological_three_way_split(
    dataset: SyntheticAutomotiveDataset,
    train_ratio: float = 0.65,
    validation_ratio: float = 0.15,
) -> ChronologicalSplit:
    """Create purged chronological train, validation, and test windows.

    A gap of `horizon - 1` windows is inserted at each boundary so that target
    months never appear in two different subsets.
    """
    if train_ratio <= 0.0 or validation_ratio <= 0.0:
        raise ValueError("train_ratio and validation_ratio must be positive.")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "train_ratio + validation_ratio must be smaller than 1."
        )

    num_windows = len(dataset)
    purge = dataset.config.horizon - 1

    train_end = int(num_windows * train_ratio)
    validation_start = train_end + purge
    validation_end = int(
        num_windows * (train_ratio + validation_ratio)
    )
    test_start = validation_end + purge

    if train_end <= 0:
        raise ValueError("Training split is empty.")
    if validation_start >= validation_end:
        raise ValueError("Validation split is empty after purging.")
    if test_start >= num_windows:
        raise ValueError("Test split is empty after purging.")

    return ChronologicalSplit(
        train=Subset(dataset, list(range(0, train_end))),
        validation=Subset(
            dataset,
            list(range(validation_start, validation_end)),
        ),
        test=Subset(dataset, list(range(test_start, num_windows))),
    )


def _forward(
    model: nn.Module,
    x: Tensor,
    adjacency: Tensor | None,
) -> Tensor:
    return model(x, adjacency)


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    adjacency: Tensor | None,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    model.eval()
    graph = adjacency.to(device) if adjacency is not None else None

    predictions: list[Tensor] = []
    targets: list[Tensor] = []

    for x, y in loader:
        prediction = _forward(model, x.to(device), graph)
        predictions.append(prediction.cpu())
        targets.append(y)

    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )


@torch.no_grad()
def _validation_log_mse(
    model: nn.Module,
    loader: DataLoader,
    adjacency: Tensor | None,
    device: torch.device,
) -> float:
    prediction, target = _collect_predictions(
        model,
        loader,
        adjacency,
        device,
    )
    return float(torch.mean((prediction - target) ** 2).item())


def _train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    adjacency: Tensor | None,
    config: TrainingConfig,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> tuple[int, float, list[dict[str, float]]]:
    """Train and restore the best validation checkpoint."""
    if patience <= 0:
        raise ValueError("patience must be positive.")

    model.to(device)
    graph = adjacency.to(device) if adjacency is not None else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation = _validation_log_mse(
        model,
        validation_loader,
        adjacency,
        device,
    )
    epochs_without_improvement = 0
    history = [
        {
            "epoch": 0.0,
            "train_log_mse": float("nan"),
            "validation_log_mse": best_validation,
        }
    ]

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, x, graph)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        validation_loss = _validation_log_mse(
            model,
            validation_loader,
            adjacency,
            device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_log_mse": float(np.mean(train_losses)),
                "validation_log_mse": validation_loss,
            }
        )

        if validation_loss < best_validation - min_delta:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return best_epoch, best_validation, history


def _build_specifications(
    dataset: SyntheticAutomotiveDataset,
    market_config: SyntheticMarketConfig,
    training_config: TrainingConfig,
    graph_permutation_seed: int,
) -> list[ModelSpecification]:
    adjacency = dataset.adjacency

    def safe_graph_factory() -> SafeGraphResidualGRU:
        return SafeGraphResidualGRU(
            num_features=market_config.num_features,
            hidden_dim=training_config.hidden_dim,
            horizon=market_config.horizon,
        )

    return [
        ModelSpecification(
            name="last_value",
            factory=lambda: LastValueForecaster(
                horizon=market_config.horizon
            ),
            adjacency=None,
            trainable=False,
        ),
        ModelSpecification(
            name="residual_temporal_gru",
            factory=lambda: ResidualTemporalGRU(
                num_features=market_config.num_features,
                hidden_dim=training_config.hidden_dim,
                horizon=market_config.horizon,
            ),
            adjacency=None,
            trainable=True,
        ),
        ModelSpecification(
            name="safe_static_graph_gru",
            factory=safe_graph_factory,
            adjacency=adjacency,
            trainable=True,
        ),
        ModelSpecification(
            name="safe_identity_graph_gru",
            factory=safe_graph_factory,
            adjacency=make_identity_graph(adjacency),
            trainable=True,
        ),
        ModelSpecification(
            name="safe_permuted_graph_gru",
            factory=safe_graph_factory,
            adjacency=make_permuted_graph(
                adjacency,
                seed=graph_permutation_seed,
            ),
            trainable=True,
        ),
    ]


def _evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    adjacency: Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    prediction_log, target_log = _collect_predictions(
        model,
        loader,
        adjacency,
        device,
    )
    metrics = compute_forecast_metrics(
        prediction_log,
        target_log,
    )

    prediction_sales = inverse_log_sales(prediction_log)
    target_sales = inverse_log_sales(target_log)
    bias = prediction_sales - target_sales

    graph_scale = getattr(model, "graph_scale", None)
    graph_scale_mean_abs = (
        float(torch.tanh(graph_scale).abs().mean().item())
        if graph_scale is not None
        else 0.0
    )

    return {
        "test_log_mse": metrics.log_mse,
        "test_mae": metrics.mae,
        "test_rmse": metrics.rmse,
        "test_wape_percent": metrics.wape_percent,
        "test_smape_percent": metrics.smape_percent,
        "mean_prediction_sales": float(
            prediction_sales.mean().item()
        ),
        "mean_target_sales": float(target_sales.mean().item()),
        "mean_bias_sales": float(bias.mean().item()),
        "graph_scale_mean_abs": graph_scale_mean_abs,
    }


def _write_metrics(
    results: list[SafeGraphExperimentResult],
    path: Path,
) -> None:
    rows = [asdict(result) for result in results]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_learning_curves(
    rows: list[dict[str, float | str]],
    path: Path,
) -> None:
    fieldnames = [
        "model_name",
        "epoch",
        "train_log_mse",
        "validation_log_mse",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    results: list[SafeGraphExperimentResult],
    path: Path,
) -> None:
    ranked = sorted(
        results,
        key=lambda result: result.test_wape_percent,
    )

    lines = [
        "# Stage 0.2E Safe Graph Summary",
        "",
        "Early stopping uses validation data. Ranking uses untouched test WAPE.",
        "",
        (
            "| Rank | Model | Best epoch | Test WAPE (%) | Test MAE | "
            "Bias | Mean | Graph scale |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, result in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {result.model_name} | "
            f"{result.best_epoch} | "
            f"{result.test_wape_percent:.3f} | "
            f"{result.test_mae:.3f} | "
            f"{result.mean_bias_sales:.3f} | "
            f"{result.mean_prediction_sales:.3f} | "
            f"{result.graph_scale_mean_abs:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Decision rules",
            "",
            "- Safe static graph should beat residual temporal GRU.",
            "- Safe static graph should beat identity and permuted controls.",
            "- Graph scale near zero means the model ignored graph context.",
            "- This remains a one-seed diagnostic until multi-seed evaluation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage0_2e_safe_graph_experiments(
    market_config: SyntheticMarketConfig,
    training_config: TrainingConfig,
    output_dir: str | Path,
    patience: int = 8,
    min_delta: float = 1e-4,
    train_ratio: float = 0.65,
    validation_ratio: float = 0.15,
    model_seed: int | None = None,
    graph_permutation_seed: int | None = None,
) -> list[SafeGraphExperimentResult]:
    """Run early-stopped models and evaluate on held-out test windows.

    `market_config.seed` controls data generation. `model_seed` controls model
    initialization and training randomness. `graph_permutation_seed` controls
    only the incorrect permuted-graph control.
    """
    market_config.validate()
    training_config.validate()

    resolved_model_seed = (
        training_config.seed
        if model_seed is None
        else model_seed
    )
    resolved_graph_seed = (
        market_config.seed + 1000
        if graph_permutation_seed is None
        else graph_permutation_seed
    )
    set_seed(resolved_model_seed)

    dataset = SyntheticAutomotiveDataset(market_config)
    split = chronological_three_way_split(
        dataset,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    train_loader = DataLoader(
        split.train,
        batch_size=training_config.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        split.validation,
        batch_size=training_config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        split.test,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specifications = _build_specifications(
        dataset,
        market_config,
        training_config,
        graph_permutation_seed=resolved_graph_seed,
    )

    results: list[SafeGraphExperimentResult] = []
    curve_rows: list[dict[str, float | str]] = []

    for specification in specifications:
        set_seed(resolved_model_seed)
        model = specification.factory()
        # Model construction consumes random numbers. Reset again so shared
        # dropout masks are comparable across architectures.
        set_seed(resolved_model_seed)

        if specification.trainable:
            best_epoch, best_validation, history = (
                _train_with_early_stopping(
                    model=model,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                    adjacency=specification.adjacency,
                    config=training_config,
                    patience=patience,
                    min_delta=min_delta,
                    device=device,
                )
            )
        else:
            best_epoch = 0
            best_validation = _validation_log_mse(
                model,
                validation_loader,
                specification.adjacency,
                device,
            )
            history = [
                {
                    "epoch": 0.0,
                    "train_log_mse": float("nan"),
                    "validation_log_mse": best_validation,
                }
            ]

        for row in history:
            curve_rows.append(
                {
                    "model_name": specification.name,
                    **row,
                }
            )

        test_metrics = _evaluate_test(
            model,
            test_loader,
            specification.adjacency,
            device,
        )
        result = SafeGraphExperimentResult(
            model_name=specification.name,
            trainable_parameters=count_trainable_parameters(model),
            best_epoch=best_epoch,
            best_validation_log_mse=best_validation,
            **test_metrics,
        )
        results.append(result)

        print(
            f"{result.model_name:<28} "
            f"best_epoch={result.best_epoch:3d} "
            f"test_WAPE={result.test_wape_percent:8.3f}% "
            f"bias={result.mean_bias_sales:9.3f} "
            f"graph_scale={result.graph_scale_mean_abs:.4f}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _write_metrics(results, destination / "metrics.csv")
    _write_learning_curves(
        curve_rows,
        destination / "learning_curves.csv",
    )
    _write_summary(
        results,
        destination / "experiment_summary.md",
    )
    return results
