from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from autogfm.forecasting import BacktestSpec, ProductionForecastSpec, build_production_forecast, run_rolling_backtest
from autogfm.graph import GraphSpec, build_relation_graph
from autogfm.realdata import CanonicalizationConfig, PanelSpec, build_monthly_panel, canonicalize_three_files
from autogfm.realdata.demo import generate_stage3_demo_data


@pytest.fixture(scope="session")
def stage3_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("stage3_artifacts")
    demo = generate_stage3_demo_data(root / "raw", num_series=12, num_months=60, seed=7)
    canonical = canonicalize_three_files(
        demo.sales_csv,
        demo.mapping_csv,
        demo.vehicles_csv,
        root / "canonical",
        config=CanonicalizationConfig(source_name="test", data_version="v1"),
        overwrite=True,
    )
    panel = build_monthly_panel(
        canonical.output_dir / "canonical_monthly_sales.csv",
        canonical.output_dir / "canonical_catalog.csv",
        root / "panel",
        PanelSpec(
            start_month=date(2020, 1, 1),
            train_end=date(2023, 12, 1),
            validation_end=date(2024, 6, 1),
            test_end=date(2024, 12, 1),
            min_history_months=36,
            max_missing_ratio=0.15,
            max_trailing_missing_months=1,
        ),
        overwrite=True,
    )
    graph = build_relation_graph(
        panel.output_dir,
        root / "graph",
        GraphSpec(top_k=3),
        overwrite=True,
    )
    backtest = run_rolling_backtest(
        panel.output_dir,
        graph.output_dir,
        root / "backtest",
        BacktestSpec(
            history=12,
            horizons=(1, 3),
            origin_step=2,
            bootstrap_samples=20,
            seed=11,
        ),
        overwrite=True,
    )
    production = build_production_forecast(
        panel.output_dir,
        graph.output_dir,
        backtest.output_dir,
        root / "production",
        ProductionForecastSpec(history=12),
        overwrite=True,
    )
    return {
        "root": root,
        "demo": demo,
        "canonical": canonical,
        "panel": panel,
        "graph": graph,
        "backtest": backtest,
        "production": production,
    }
