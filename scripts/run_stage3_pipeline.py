"""Run the full AutoGraphFM Stage 3 pipeline."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from autogfm.experiments import RealPanelExperimentSpec
from autogfm.forecasting import BacktestSpec
from autogfm.graph import GraphSpec
from autogfm.pipeline import Stage3PipelineSpec, run_stage3_pipeline
from autogfm.realdata import CanonicalizationConfig, PanelSpec


def _month(value: str) -> date:
    parsed = date.fromisoformat(value + "-01" if len(value) == 7 else value)
    return date(parsed.year, parsed.month, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--vehicles-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-month", type=_month)
    parser.add_argument("--train-end", type=_month, required=True)
    parser.add_argument("--validation-end", type=_month, required=True)
    parser.add_argument("--test-end", type=_month, required=True)
    parser.add_argument("--min-history-months", type=int, default=24)
    parser.add_argument("--max-missing-ratio", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--neural-horizon", type=int, default=3)
    parser.add_argument("--neural-epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.quick:
        origin_step = 3
        bootstrap_samples = 100
        neural_epochs = min(args.neural_epochs, 8)
        seeds = tuple(args.seeds[:1])
        patience = 3
    else:
        origin_step = 1
        bootstrap_samples = 500
        neural_epochs = args.neural_epochs
        seeds = tuple(args.seeds)
        patience = 7

    spec = Stage3PipelineSpec(
        canonicalization=CanonicalizationConfig(
            source_name="automotive_three_file",
            data_version="stage3",
            strict_unmatched=False,
            duplicate_policy="sum",
        ),
        panel=PanelSpec(
            start_month=args.start_month,
            train_end=args.train_end,
            validation_end=args.validation_end,
            test_end=args.test_end,
            min_history_months=args.min_history_months,
            max_missing_ratio=args.max_missing_ratio,
            max_trailing_missing_months=3,
            imputation="linear",
        ),
        graph=GraphSpec(top_k=args.top_k),
        backtest=BacktestSpec(
            history=args.history,
            horizons=tuple(sorted(set(args.horizons))),
            origin_step=origin_step,
            bootstrap_samples=bootstrap_samples,
        ),
        neural=RealPanelExperimentSpec(
            history=args.history,
            horizon=args.neural_horizon,
            max_epochs=neural_epochs,
            patience=patience,
            seeds=seeds,
        ),
        run_neural=not args.skip_neural,
    )
    result = run_stage3_pipeline(
        args.sales_csv,
        args.mapping_csv,
        args.vehicles_csv,
        args.output_dir,
        spec,
        overwrite=args.overwrite,
    )
    print(f"run_id={result.run_id}")
    print(f"report={result.output_dir / 'STAGE3_RUN_REPORT.md'}")


if __name__ == "__main__":
    main()
