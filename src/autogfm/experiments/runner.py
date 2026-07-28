"""Unified training and comparison for Stage 0.2 baselines."""

from __future__ import annotations

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
from autogfm.graph import make_identity_graph, make_permuted_graph
from autogfm.metrics import ForecastMetrics, compute_forecast_metrics
from autogfm.models import LastValueForecaster, StaticGraphGRU, TemporalOnlyGRU


@dataclass(frozen=True)
class ExperimentResult:
    """One model's validation result."""

    model_name: str
    trainable_parameters: int
    log_mse: float
    mae: float
    rmse: float
    wape_percent: float
    smape_percent: float

    @classmethod
    def from_metrics(
        cls,
        model_name: str,
        trainable_parameters: int,
        metrics: ForecastMetrics,
    ) -> "ExperimentResult":
        """Combine metadata and computed metrics."""
        return cls(
            model_name=model_name,
            trainable_parameters=trainable_parameters,
            **metrics.to_dict(),
        )


@dataclass(frozen=True)
class ModelSpecification:
    """Factory and graph input for one controlled experiment."""

    name: str
    factory: Callable[[], nn.Module]
    adjacency: Tensor | None
    trainable: bool


def set_seed(seed: int) -> None:
    """Set deterministic seeds used by the small CPU experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def temporal_split(
    dataset: SyntheticAutomotiveDataset,
    train_ratio: float,
) -> tuple[Subset, Subset]:
    """Split windows chronologically, never randomly."""
    split_index = int(len(dataset) * train_ratio)

    if split_index <= 0 or split_index >= len(dataset):
        raise ValueError("train_ratio creates an empty train or validation set.")

    train_ids = list(range(split_index))
    validation_ids = list(range(split_index, len(dataset)))

    return Subset(dataset, train_ids), Subset(dataset, validation_ids)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that receive gradients."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    adjacency: Tensor | None,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    """Train one model using MSE in log-sales space."""
    model.to(device)
    graph = adjacency.to(device) if adjacency is not None else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    for _ in range(config.epochs):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, graph)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    adjacency: Tensor | None,
    device: torch.device,
) -> ForecastMetrics:
    """Evaluate one model using all validation windows."""
    model.to(device)
    model.eval()

    graph = adjacency.to(device) if adjacency is not None else None
    predictions: list[Tensor] = []
    targets: list[Tensor] = []

    for x, y in validation_loader:
        x = x.to(device)
        prediction = model(x, graph)

        predictions.append(prediction.cpu())
        targets.append(y.cpu())

    return compute_forecast_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )


def _build_model_specifications(
    dataset: SyntheticAutomotiveDataset,
    market_config: SyntheticMarketConfig,
    training_config: TrainingConfig,
) -> list[ModelSpecification]:
    adjacency = dataset.adjacency

    def graph_model_factory() -> StaticGraphGRU:
        return StaticGraphGRU(
            num_features=market_config.num_features,
            hidden_dim=training_config.hidden_dim,
            horizon=market_config.horizon,
        )

    return [
        ModelSpecification(
            name="last_value",
            factory=lambda: LastValueForecaster(horizon=market_config.horizon),
            adjacency=None,
            trainable=False,
        ),
        ModelSpecification(
            name="temporal_gru",
            factory=lambda: TemporalOnlyGRU(
                num_features=market_config.num_features,
                hidden_dim=training_config.hidden_dim,
                horizon=market_config.horizon,
            ),
            adjacency=None,
            trainable=True,
        ),
        ModelSpecification(
            name="static_graph_gru",
            factory=graph_model_factory,
            adjacency=adjacency,
            trainable=True,
        ),
        ModelSpecification(
            name="identity_graph_gru",
            factory=graph_model_factory,
            adjacency=make_identity_graph(adjacency),
            trainable=True,
        ),
        ModelSpecification(
            name="permuted_graph_gru",
            factory=graph_model_factory,
            adjacency=make_permuted_graph(
                adjacency,
                seed=training_config.seed + 1,
            ),
            trainable=True,
        ),
    ]


def _write_metrics_csv(
    results: list[ExperimentResult],
    output_path: Path,
) -> None:
    rows = [asdict(result) for result in results]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_summary(
    results: list[ExperimentResult],
    output_path: Path,
) -> None:
    sorted_results = sorted(results, key=lambda result: result.wape_percent)

    lines = [
        "# Stage 0.2 Experiment Summary",
        "",
        "Models are ranked by validation WAPE on the original sales scale.",
        "",
        (
            "| Rank | Model | Parameters | Log MSE | MAE | RMSE | "
            "WAPE (%) | sMAPE (%) |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, result in enumerate(sorted_results, start=1):
        lines.append(
            f"| {rank} | {result.model_name} | "
            f"{result.trainable_parameters} | "
            f"{result.log_mse:.4f} | "
            f"{result.mae:.2f} | "
            f"{result.rmse:.2f} | "
            f"{result.wape_percent:.2f} | "
            f"{result.smape_percent:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- The graph method must beat the temporal-only baseline.",
            "- The correct graph should beat the permuted graph.",
            "- The correct graph should differ from the identity graph.",
            "- One random seed is not enough for a final research claim.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_stage0_2_experiments(
    market_config: SyntheticMarketConfig,
    training_config: TrainingConfig,
    output_dir: str | Path,
) -> list[ExperimentResult]:
    """Run all Stage 0.2 baselines and write result files."""
    market_config.validate()
    training_config.validate()
    set_seed(training_config.seed)

    dataset = SyntheticAutomotiveDataset(market_config)
    train_set, validation_set = temporal_split(
        dataset,
        training_config.train_ratio,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=training_config.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specifications = _build_model_specifications(
        dataset,
        market_config,
        training_config,
    )

    results: list[ExperimentResult] = []

    for specification in specifications:
        set_seed(training_config.seed)
        model = specification.factory()

        if specification.trainable:
            train_model(
                model=model,
                train_loader=train_loader,
                adjacency=specification.adjacency,
                config=training_config,
                device=device,
            )

        metrics = evaluate_model(
            model=model,
            validation_loader=validation_loader,
            adjacency=specification.adjacency,
            device=device,
        )
        result = ExperimentResult.from_metrics(
            model_name=specification.name,
            trainable_parameters=count_trainable_parameters(model),
            metrics=metrics,
        )
        results.append(result)

        print(
            f"{result.model_name:<22} "
            f"WAPE={result.wape_percent:8.3f}% "
            f"MAE={result.mae:10.3f} "
            f"RMSE={result.rmse:10.3f}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(results, destination / "metrics.csv")
    _write_markdown_summary(
        results,
        destination / "experiment_summary.md",
    )

    return results
