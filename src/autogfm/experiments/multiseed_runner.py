"""Paired multi-seed stability evaluation for Stage 0.2F."""

from __future__ import annotations

import csv
import gc
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev

import torch

from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.experiments.safe_graph_runner import (
    SafeGraphExperimentResult,
    run_stage0_2e_safe_graph_experiments,
)


@dataclass(frozen=True)
class MultiSeedResult:
    """One model result under one model random seed."""

    data_seed: int
    model_seed: int
    graph_permutation_seed: int
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

    @classmethod
    def from_safe_result(
        cls,
        result: SafeGraphExperimentResult,
        data_seed: int,
        model_seed: int,
        graph_permutation_seed: int,
    ) -> "MultiSeedResult":
        """Attach explicit seed roles to one Stage 0.2E result."""
        return cls(
            data_seed=data_seed,
            model_seed=model_seed,
            graph_permutation_seed=graph_permutation_seed,
            **asdict(result),
        )


@dataclass(frozen=True)
class AggregateMetric:
    """Across-seed aggregate for one model."""

    model_name: str
    runs: int
    mean_test_wape_percent: float
    std_test_wape_percent: float
    sem_test_wape_percent: float
    ci95_low_test_wape_percent: float
    ci95_high_test_wape_percent: float
    mean_test_mae: float
    mean_test_rmse: float
    mean_bias_sales: float
    mean_best_epoch: float
    mean_graph_scale_abs: float


@dataclass(frozen=True)
class PairedComparison:
    """Paired WAPE comparison against the safe static graph model."""

    reference_model: str
    runs: int
    safe_static_wins: int
    safe_static_win_rate_percent: float
    mean_wape_delta_static_minus_reference: float
    std_wape_delta: float
    ci95_low_wape_delta: float
    ci95_high_wape_delta: float


@dataclass(frozen=True)
class MultiSeedExperimentBundle:
    """All Stage 0.2F outputs."""

    per_seed_results: list[MultiSeedResult]
    aggregate_metrics: list[AggregateMetric]
    paired_comparisons: list[PairedComparison]


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _validate_model_seeds(model_seeds: list[int]) -> None:
    if not model_seeds:
        raise ValueError("model_seeds must not be empty.")
    if len(set(model_seeds)) != len(model_seeds):
        raise ValueError("model_seeds must be unique.")


def _mean_std_ci95(values: list[float]) -> tuple[float, float, float, float, float]:
    """Return mean, sample std, standard error, and a t-based 95% CI."""
    if not values:
        raise ValueError("values must not be empty.")

    value_mean = mean(values)
    if len(values) == 1:
        return value_mean, 0.0, 0.0, value_mean, value_mean

    value_std = stdev(values)
    standard_error = value_std / math.sqrt(len(values))
    degrees_of_freedom = len(values) - 1
    critical = _T_CRITICAL_95.get(degrees_of_freedom, 1.96)
    half_width = critical * standard_error

    return (
        value_mean,
        value_std,
        standard_error,
        value_mean - half_width,
        value_mean + half_width,
    )


def aggregate_model_results(
    results: list[MultiSeedResult],
) -> list[AggregateMetric]:
    """Aggregate test metrics by model across model seeds."""
    model_names = list(dict.fromkeys(result.model_name for result in results))
    aggregates: list[AggregateMetric] = []

    for model_name in model_names:
        rows = [
            result
            for result in results
            if result.model_name == model_name
        ]
        wapes = [row.test_wape_percent for row in rows]
        (
            mean_wape,
            std_wape,
            sem_wape,
            ci_low,
            ci_high,
        ) = _mean_std_ci95(wapes)

        aggregates.append(
            AggregateMetric(
                model_name=model_name,
                runs=len(rows),
                mean_test_wape_percent=mean_wape,
                std_test_wape_percent=std_wape,
                sem_test_wape_percent=sem_wape,
                ci95_low_test_wape_percent=ci_low,
                ci95_high_test_wape_percent=ci_high,
                mean_test_mae=mean(row.test_mae for row in rows),
                mean_test_rmse=mean(row.test_rmse for row in rows),
                mean_bias_sales=mean(row.mean_bias_sales for row in rows),
                mean_best_epoch=mean(row.best_epoch for row in rows),
                mean_graph_scale_abs=mean(
                    row.graph_scale_mean_abs for row in rows
                ),
            )
        )

    return aggregates


def build_paired_comparisons(
    results: list[MultiSeedResult],
) -> list[PairedComparison]:
    """Compare safe static graph WAPE against each control per seed."""
    by_seed: dict[int, dict[str, MultiSeedResult]] = {}
    for result in results:
        by_seed.setdefault(result.model_seed, {})[
            result.model_name
        ] = result

    static_name = "safe_static_graph_gru"
    references = [
        "last_value",
        "residual_temporal_gru",
        "safe_identity_graph_gru",
        "safe_permuted_graph_gru",
    ]
    comparisons: list[PairedComparison] = []

    for reference_name in references:
        deltas: list[float] = []
        wins = 0

        for seed_rows in by_seed.values():
            if static_name not in seed_rows or reference_name not in seed_rows:
                raise ValueError(
                    f"Missing paired result for {reference_name}."
                )

            static_wape = seed_rows[static_name].test_wape_percent
            reference_wape = seed_rows[reference_name].test_wape_percent
            delta = static_wape - reference_wape
            deltas.append(delta)
            wins += int(delta < 0.0)

        (
            mean_delta,
            std_delta,
            _,
            ci_low,
            ci_high,
        ) = _mean_std_ci95(deltas)

        comparisons.append(
            PairedComparison(
                reference_model=reference_name,
                runs=len(deltas),
                safe_static_wins=wins,
                safe_static_win_rate_percent=100.0 * wins / len(deltas),
                mean_wape_delta_static_minus_reference=mean_delta,
                std_wape_delta=std_delta,
                ci95_low_wape_delta=ci_low,
                ci95_high_wape_delta=ci_high,
            )
        )

    return comparisons


def _write_dataclass_csv(rows: list[object], path: Path) -> None:
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(dictionaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(dictionaries)


def _write_summary(
    aggregate_metrics: list[AggregateMetric],
    comparisons: list[PairedComparison],
    path: Path,
    data_seed: int,
    model_seeds: list[int],
    graph_permutation_seed: int,
) -> None:
    ranked = sorted(
        aggregate_metrics,
        key=lambda row: row.mean_test_wape_percent,
    )

    lines = [
        "# Stage 0.2F Multi-Seed Stability Summary",
        "",
        f"- Fixed data seed: `{data_seed}`",
        f"- Model seeds: `{model_seeds}`",
        f"- Fixed graph permutation seed: `{graph_permutation_seed}`",
        "",
        "## Aggregate held-out test metrics",
        "",
        (
            "| Rank | Model | Runs | Mean WAPE (%) | Std | 95% CI | "
            "Mean MAE | Mean bias | Mean graph scale |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row.model_name} | {row.runs} | "
            f"{row.mean_test_wape_percent:.3f} | "
            f"{row.std_test_wape_percent:.3f} | "
            f"[{row.ci95_low_test_wape_percent:.3f}, "
            f"{row.ci95_high_test_wape_percent:.3f}] | "
            f"{row.mean_test_mae:.3f} | "
            f"{row.mean_bias_sales:.3f} | "
            f"{row.mean_graph_scale_abs:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            (
                "A negative delta means Safe Static Graph GRU has lower WAPE "
                "than the reference model."
            ),
            "",
            (
                "| Reference | Static wins | Win rate | Mean delta (pp) | "
                "95% CI of delta |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in comparisons:
        lines.append(
            f"| {row.reference_model} | "
            f"{row.safe_static_wins}/{row.runs} | "
            f"{row.safe_static_win_rate_percent:.1f}% | "
            f"{row.mean_wape_delta_static_minus_reference:.3f} | "
            f"[{row.ci95_low_wape_delta:.3f}, "
            f"{row.ci95_high_wape_delta:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            (
                "Proceed to the real-data database stage only if the safe "
                "static graph consistently beats the temporal and permuted "
                "controls, or if the experiment clearly identifies why it does "
                "not."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage0_2f_multiseed_experiments(
    market_config: SyntheticMarketConfig,
    training_template: TrainingConfig,
    model_seeds: list[int],
    graph_permutation_seed: int,
    output_dir: str | Path,
    patience: int = 12,
    min_delta: float = 1e-4,
    train_ratio: float = 0.65,
    validation_ratio: float = 0.15,
) -> MultiSeedExperimentBundle:
    """Run paired Stage 0.2E experiments over model random seeds."""
    market_config.validate()
    training_template.validate()
    _validate_model_seeds(model_seeds)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    per_seed_results: list[MultiSeedResult] = []

    for model_seed in model_seeds:
        training_config = replace(
            training_template,
            seed=model_seed,
        )
        seed_output = destination / f"seed_{model_seed}"

        safe_results = run_stage0_2e_safe_graph_experiments(
            market_config=market_config,
            training_config=training_config,
            output_dir=seed_output,
            patience=patience,
            min_delta=min_delta,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            model_seed=model_seed,
            graph_permutation_seed=graph_permutation_seed,
        )

        for result in safe_results:
            per_seed_results.append(
                MultiSeedResult.from_safe_result(
                    result=result,
                    data_seed=market_config.seed,
                    model_seed=model_seed,
                    graph_permutation_seed=graph_permutation_seed,
                )
            )

        # Long CPU runs repeatedly construct GRUs and optimizers. Explicitly
        # release cyclic objects between seeds to keep memory and backend state
        # stable in one process.
        del safe_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate_metrics = aggregate_model_results(per_seed_results)
    paired_comparisons = build_paired_comparisons(per_seed_results)

    _write_dataclass_csv(
        per_seed_results,
        destination / "per_seed_metrics.csv",
    )
    _write_dataclass_csv(
        aggregate_metrics,
        destination / "aggregate_metrics.csv",
    )
    _write_dataclass_csv(
        paired_comparisons,
        destination / "paired_comparisons.csv",
    )
    _write_summary(
        aggregate_metrics=aggregate_metrics,
        comparisons=paired_comparisons,
        path=destination / "experiment_summary.md",
        data_seed=market_config.seed,
        model_seeds=model_seeds,
        graph_permutation_seed=graph_permutation_seed,
    )

    print("\n=== Aggregate test WAPE across model seeds ===")
    for row in sorted(
        aggregate_metrics,
        key=lambda item: item.mean_test_wape_percent,
    ):
        print(
            f"{row.model_name:<28} "
            f"mean={row.mean_test_wape_percent:8.3f}% "
            f"std={row.std_test_wape_percent:7.3f} "
            f"95%CI=[{row.ci95_low_test_wape_percent:7.3f}, "
            f"{row.ci95_high_test_wape_percent:7.3f}]"
        )

    print("\n=== Paired safe-static comparisons ===")
    for row in paired_comparisons:
        print(
            f"vs {row.reference_model:<25} "
            f"wins={row.safe_static_wins}/{row.runs} "
            f"mean_delta={row.mean_wape_delta_static_minus_reference:7.3f}pp"
        )

    return MultiSeedExperimentBundle(
        per_seed_results=per_seed_results,
        aggregate_metrics=aggregate_metrics,
        paired_comparisons=paired_comparisons,
    )
