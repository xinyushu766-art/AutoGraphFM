"""Residual baseline comparison for Stage 0.2D."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.data import SyntheticAutomotiveDataset
from autogfm.experiments.runner import (
    count_trainable_parameters,
    temporal_split,
)
from autogfm.graph import make_identity_graph, make_permuted_graph
from autogfm.metrics import (
    compute_forecast_metrics,
    inverse_log_sales,
)
from autogfm.models import (
    LastValueForecaster,
    ResidualStaticGraphGRU,
    ResidualTemporalGRU,
)


@dataclass(frozen=True)
class ResidualExperimentResult:
    """Validation result with scale and bias diagnostics."""

    model_name: str
    trainable_parameters: int
    log_mse: float
    mae: float
    rmse: float
    wape_percent: float
    smape_percent: float
    mean_prediction_sales: float
    mean_target_sales: float
    mean_bias_sales: float
    prediction_std_sales: float
    target_std_sales: float
    relative_wape_improvement_percent: float = 0.0


@dataclass(frozen=True)
class ModelSpecification:
    """Factory and graph input for one experiment."""

    name: str
    factory: Callable[[], nn.Module]
    adjacency: Tensor | None
    trainable: bool


def set_seed(seed: int) -> None:
    """Set deterministic seeds used by this small experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _forward(
    model: nn.Module,
    x: Tensor,
    adjacency: Tensor | None,
) -> Tensor:
    return model(x, adjacency)


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    adjacency: Tensor | None,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    """Train one residual model in log-sales space."""
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
            prediction = _forward(model, x, graph)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    validation_loader: DataLoader,
    adjacency: Tensor | None,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return forecasting metrics and sales-scale diagnostics."""
    model.to(device)
    model.eval()

    graph = adjacency.to(device) if adjacency is not None else None
    predictions: list[Tensor] = []
    targets: list[Tensor] = []

    for x, y in validation_loader:
        x = x.to(device)
        prediction = _forward(model, x, graph)

        predictions.append(prediction.cpu())
        targets.append(y.cpu())

    prediction_log = torch.cat(predictions, dim=0)
    target_log = torch.cat(targets, dim=0)

    metrics = compute_forecast_metrics(
        prediction_log,
        target_log,
    ).to_dict()

    prediction_sales = inverse_log_sales(prediction_log)
    target_sales = inverse_log_sales(target_log)
    bias = prediction_sales - target_sales

    diagnostics = {
        "mean_prediction_sales": float(prediction_sales.mean().item()),
        "mean_target_sales": float(target_sales.mean().item()),
        "mean_bias_sales": float(bias.mean().item()),
        "prediction_std_sales": float(
            prediction_sales.std(unbiased=False).item()
        ),
        "target_std_sales": float(
            target_sales.std(unbiased=False).item()
        ),
    }
    return metrics, diagnostics


def _build_specifications(
    dataset: SyntheticAutomotiveDataset,
    market_config: SyntheticMarketConfig,
    training_config: TrainingConfig,
) -> list[ModelSpecification]:
    adjacency = dataset.adjacency

    def residual_graph_factory() -> ResidualStaticGraphGRU:
        return ResidualStaticGraphGRU(
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
            name="residual_static_graph_gru",
            factory=residual_graph_factory,
            adjacency=adjacency,
            trainable=True,
        ),
        ModelSpecification(
            name="residual_identity_graph_gru",
            factory=residual_graph_factory,
            adjacency=make_identity_graph(adjacency),
            trainable=True,
        ),
        ModelSpecification(
            name="residual_permuted_graph_gru",
            factory=residual_graph_factory,
            adjacency=make_permuted_graph(
                adjacency,
                seed=training_config.seed + 1,
            ),
            trainable=True,
        ),
    ]


def _write_csv(
    results: list[ResidualExperimentResult],
    output_path: Path,
) -> None:
    rows = [asdict(result) for result in results]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    results: list[ResidualExperimentResult],
    output_path: Path,
) -> None:
    ranked = sorted(
        results,
        key=lambda result: result.wape_percent,
    )

    lines = [
        "# Stage 0.2D Residual Forecasting Summary",
        "",
        "Models are ranked by validation WAPE. Lower is better.",
        "",
        (
            "| Rank | Model | Parameters | WAPE (%) | MAE | RMSE | "
            "Mean prediction | Mean target | Bias | Improvement vs Last Value |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, result in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {result.model_name} | "
            f"{result.trainable_parameters} | "
            f"{result.wape_percent:.3f} | "
            f"{result.mae:.3f} | "
            f"{result.rmse:.3f} | "
            f"{result.mean_prediction_sales:.3f} | "
            f"{result.mean_target_sales:.3f} | "
            f"{result.mean_bias_sales:.3f} | "
            f"{result.relative_wape_improvement_percent:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive improvement means lower WAPE than Last Value.",
            "- Correct graph should beat identity and permuted graph controls.",
            "- One seed is diagnostic only; it is not a final research claim.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_stage0_2d_residual_experiments(
    market_config: SyntheticMarketConfig,
    training_config: TrainingConfig,
    output_dir: str | Path,
) -> list[ResidualExperimentResult]:
    """Run residual baselines and write CSV plus Markdown results."""
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
    specifications = _build_specifications(
        dataset,
        market_config,
        training_config,
    )

    raw_results: list[ResidualExperimentResult] = []

    for specification in specifications:
        set_seed(training_config.seed)
        model = specification.factory()

        if specification.trainable:
            _train_model(
                model=model,
                train_loader=train_loader,
                adjacency=specification.adjacency,
                config=training_config,
                device=device,
            )

        metrics, diagnostics = _evaluate(
            model=model,
            validation_loader=validation_loader,
            adjacency=specification.adjacency,
            device=device,
        )

        result = ResidualExperimentResult(
            model_name=specification.name,
            trainable_parameters=count_trainable_parameters(model),
            **metrics,
            **diagnostics,
        )
        raw_results.append(result)

    last_value_wape = raw_results[0].wape_percent
    results = [
        replace(
            result,
            relative_wape_improvement_percent=(
                100.0
                * (last_value_wape - result.wape_percent)
                / last_value_wape
            ),
        )
        for result in raw_results
    ]

    for result in results:
        print(
            f"{result.model_name:<31} "
            f"WAPE={result.wape_percent:8.3f}% "
            f"MAE={result.mae:10.3f} "
            f"bias={result.mean_bias_sales:10.3f} "
            f"improvement={result.relative_wape_improvement_percent:8.3f}%"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _write_csv(results, destination / "metrics.csv")
    _write_summary(
        results,
        destination / "experiment_summary.md",
    )
    return results
