"""One-command Stage 3 pipeline from raw CSVs to audited model evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autogfm.experiments import RealPanelExperimentSpec, run_real_panel_experiments
from autogfm.forecasting import (
    BacktestSpec,
    ProductionForecastSpec,
    build_production_forecast,
    run_rolling_backtest,
)
from autogfm.graph import GraphSpec, build_relation_graph
from autogfm.realdata import (
    CanonicalizationConfig,
    PanelSpec,
    build_monthly_panel,
    canonicalize_three_files,
)
from autogfm.realdata.common import sha256_file, stable_payload_id, write_json


@dataclass(frozen=True)
class Stage3PipelineSpec:
    canonicalization: CanonicalizationConfig = field(default_factory=CanonicalizationConfig)
    panel: PanelSpec | None = None
    graph: GraphSpec = field(default_factory=GraphSpec)
    backtest: BacktestSpec = field(default_factory=BacktestSpec)
    neural: RealPanelExperimentSpec = field(default_factory=RealPanelExperimentSpec)
    production: ProductionForecastSpec = field(default_factory=ProductionForecastSpec)
    run_neural: bool = True


@dataclass(frozen=True)
class Stage3PipelineResult:
    output_dir: Path
    run_id: str
    manifest: dict[str, Any]


def run_stage3_pipeline(
    sales_csv: str | Path,
    mapping_csv: str | Path,
    vehicles_csv: str | Path,
    output_dir: str | Path,
    spec: Stage3PipelineSpec,
    *,
    overwrite: bool = False,
) -> Stage3PipelineResult:
    """Execute canonicalization, panel, graph, backtest, and neural controls."""
    if spec.panel is None:
        raise ValueError("Stage3PipelineSpec.panel is required.")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    canonical = canonicalize_three_files(
        sales_csv,
        mapping_csv,
        vehicles_csv,
        destination / "01_canonical",
        config=spec.canonicalization,
        overwrite=True,
    )
    panel = build_monthly_panel(
        canonical.output_dir / "canonical_monthly_sales.csv",
        canonical.output_dir / "canonical_catalog.csv",
        destination / "02_panel",
        spec.panel,
        overwrite=True,
    )
    graph = build_relation_graph(
        panel.output_dir,
        destination / "03_graph",
        spec.graph,
        overwrite=True,
    )
    backtest = run_rolling_backtest(
        panel.output_dir,
        graph.output_dir,
        destination / "04_backtest",
        spec.backtest,
        overwrite=True,
    )
    neural = None
    if spec.run_neural:
        neural = run_real_panel_experiments(
            panel.output_dir,
            graph.output_dir,
            destination / "05_neural",
            spec.neural,
            overwrite=True,
        )
    production = build_production_forecast(
        panel.output_dir,
        graph.output_dir,
        backtest.output_dir,
        destination / "06_forecast",
        spec.production,
        overwrite=True,
    )

    identity = {
        "schema_version": "autogfm-stage3-pipeline-v1",
        "source_sha256": {
            "sales": sha256_file(sales_csv),
            "mapping": sha256_file(mapping_csv),
            "vehicles": sha256_file(vehicles_csv),
        },
        "spec": asdict(spec),
        "dataset_id": canonical.dataset_id,
        "panel_id": panel.panel_id,
        "graph_id": graph.graph_id,
        "backtest_id": backtest.backtest_id,
        "neural_experiment_id": neural.experiment_id if neural else None,
        "production_forecast_id": production.forecast_id,
    }
    run_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "run_id": run_id,
        "quality": canonical.manifest["quality"],
        "panel_quality": panel.manifest["quality"],
        "graph_diagnostics": graph.manifest["diagnostics"],
        "backtest_selected_models": backtest.manifest["selected_models"],
        "backtest_winners": backtest.manifest["test_winners"],
        "backtest_graph_claim_checks": backtest.manifest["graph_claim_checks"],
        "production_selected_models": production.manifest["selected_models"],
        "neural_winner": neural.manifest["winner"] if neural else None,
        "neural_claim_gate": neural.manifest["claim_gate"] if neural else None,
    }
    write_json(destination / "manifest.json", manifest)

    lines = [
        "# AutoGraphFM Stage 3 Run Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Canonical dataset: `{canonical.dataset_id}`",
        f"- Panel: `{panel.panel_id}`",
        f"- Graph: `{graph.graph_id}`",
        f"- Rolling backtest: `{backtest.backtest_id}`",
        f"- Neural experiment: `{neural.experiment_id if neural else 'skipped'}`",
        f"- Production forecast: `{production.forecast_id}`",
        "",
        "## Data gate",
        "",
        f"- Canonical series: {canonical.manifest['quality']['canonical_series']}",
        f"- Canonical monthly rows: {canonical.manifest['quality']['canonical_monthly_rows']}",
        f"- Rejected sales rows: {canonical.manifest['quality']['rejected_sales_rows']}",
        f"- Duplicate monthly keys aggregated: {canonical.manifest['quality']['duplicate_monthly_keys_aggregated']}",
        f"- Included panel series: {panel.manifest['quality']['included_series']}",
        f"- Observed ratio: {panel.manifest['quality']['observed_ratio']:.4f}",
        "",
        "## Graph gate",
        "",
        f"- Nodes / edges: {graph.manifest['diagnostics']['nodes']} / {graph.manifest['diagnostics']['undirected_edges']}",
        f"- Largest component ratio: {graph.manifest['diagnostics']['largest_component_ratio']:.4f}",
        f"- Isolated nodes: {graph.manifest['diagnostics']['isolated_nodes']}",
        "",
        "## Backtest",
        "",
        f"- Validation-selected models: `{backtest.manifest['selected_models']}`",
        f"- Oracle held-out winners by horizon: `{backtest.manifest['test_winners']}`",
        f"- Graph control checks: `{backtest.manifest['graph_claim_checks']}`",
        "",
        "## Production forecast",
        "",
        f"- Selected models: `{production.manifest['selected_models']}`",
        f"- Forecast rows: `{production.manifest['forecast_rows']}`",
        "",
        "## Neural controls",
        "",
        f"- Winner: `{neural.manifest['winner'] if neural else 'skipped'}`",
        f"- Claim gate: `{neural.manifest['claim_gate'] if neural else 'skipped'}`",
        "",
        "## Interpretation rule",
        "",
        "The pipeline being executable is an engineering result. A graph-learning research claim is permitted only when the true graph beats temporal, identity, and permuted controls on held-out data.",
        "",
    ]
    (destination / "STAGE3_RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return Stage3PipelineResult(destination, run_id, manifest)
