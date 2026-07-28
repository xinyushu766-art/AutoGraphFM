"""Inspect SRNI-CAR, prepare its three-file contract, and run Stage 4."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from autogfm.experiments import RealPanelExperimentSpec
from autogfm.forecasting import BacktestSpec
from autogfm.graph import GraphSpec
from autogfm.monitoring import MonitoringSpec
from autogfm.pipeline import Stage3PipelineSpec, Stage4PipelineSpec, run_stage4_pipeline
from autogfm.realdata import (
    CanonicalizationConfig,
    PanelSpec,
    SrniCarPreparationConfig,
    inspect_srni_car_sales,
    prepare_srni_car_sales,
)
from autogfm.reporting import Stage4ReportSpec


def _month(value: str) -> date:
    parsed = date.fromisoformat(value + "-01" if len(value) == 7 else value)
    return date(parsed.year, parsed.month, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-end", type=_month, default=_month("2020-12"))
    parser.add_argument("--validation-end", type=_month, default=_month("2021-12"))
    parser.add_argument("--test-end", type=_month, default=_month("2022-12"))
    parser.add_argument("--start-month", type=_month)
    parser.add_argument("--min-history-months", type=int, default=36)
    parser.add_argument("--max-missing-ratio", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--neural-horizon", type=int, default=3)
    parser.add_argument("--neural-epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument(
        "--allow-nonofficial",
        action="store_true",
        help=(
            "Allow synthetic or profile-unverified files for engineering checks. "
            "Their report remains non-deployable and cannot support real-data claims."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inspection_dir = args.output_dir / "00_srni_inspection"
    prepared_dir = args.output_dir / "01_srni_three_file"
    run_dir = args.output_dir / "run"
    inspection = inspect_srni_car_sales(
        args.sales_csv,
        inspection_dir,
        overwrite=args.overwrite,
    )
    if (
        inspection.evidence_scope != "official_real_data"
        and not args.allow_nonofficial
    ):
        raise RuntimeError(
            "The candidate Sales CSV did not match the published SRNI-CAR "
            "profile. Modeling was stopped before training. Review "
            f"{inspection_dir / 'SRNI_AUTHENTICITY_REPORT.md'} or rerun with "
            "--allow-nonofficial only for an engineering check."
        )

    required_scope = None if args.allow_nonofficial else "official_real_data"
    prepared = prepare_srni_car_sales(
        args.sales_csv,
        prepared_dir,
        config=SrniCarPreparationConfig(
            required_evidence_scope=required_scope,
        ),
        overwrite=args.overwrite,
    )

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

    stage3 = Stage3PipelineSpec(
        canonicalization=CanonicalizationConfig(
            source_name="srni_car",
            data_version="2016-2022",
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
    result = run_stage4_pipeline(
        prepared.sales_csv,
        prepared.mapping_csv,
        prepared.vehicles_csv,
        run_dir,
        Stage4PipelineSpec(
            stage3=stage3,
            monitoring=MonitoringSpec(recent_months=6),
            report=Stage4ReportSpec(
                evidence_scope=inspection.evidence_scope,
            ),
        ),
        overwrite=args.overwrite,
    )
    print(f"evidence_scope={inspection.evidence_scope}")
    print(f"verification_level={inspection.verification_level}")
    print(f"srni_dataset_id={prepared.dataset_id}")
    print(f"run_id={result.run_id}")
    print(f"deployment_status={result.manifest['deployment_status']}")
    print(f"graph_claim_status={result.manifest['graph_research_claim']}")
    print(f"audit={result.output_dir / '08_report' / 'AUDIT_REPORT.html'}")


if __name__ == "__main__":
    main()
