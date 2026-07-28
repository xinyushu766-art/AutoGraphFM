"""Stage 4 pipeline: audited forecasting, monitoring, and release reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autogfm.monitoring import MonitoringSpec, run_monitoring
from autogfm.pipeline.stage3 import (
    Stage3PipelineResult,
    Stage3PipelineSpec,
    run_stage3_pipeline,
)
from autogfm.realdata.common import stable_payload_id, write_json
from autogfm.reporting import (
    Stage4ReportSpec,
    build_stage4_audit_report,
)


@dataclass(frozen=True)
class Stage4PipelineSpec:
    """Configuration for the complete Stage 4 release workflow."""

    stage3: Stage3PipelineSpec
    monitoring: MonitoringSpec = field(default_factory=MonitoringSpec)
    report: Stage4ReportSpec = field(default_factory=Stage4ReportSpec)


@dataclass(frozen=True)
class Stage4PipelineResult:
    output_dir: Path
    run_id: str
    stage3: Stage3PipelineResult
    manifest: dict[str, Any]


def run_stage4_pipeline(
    sales_csv: str | Path,
    mapping_csv: str | Path,
    vehicles_csv: str | Path,
    output_dir: str | Path,
    spec: Stage4PipelineSpec,
    *,
    overwrite: bool = False,
) -> Stage4PipelineResult:
    """Run Stage 3 evidence generation, operational gates, and audit reporting."""
    destination = Path(output_dir)
    stage3 = run_stage3_pipeline(
        sales_csv,
        mapping_csv,
        vehicles_csv,
        destination,
        spec.stage3,
        overwrite=overwrite,
    )
    neural_dir = destination / "05_neural"
    monitoring = run_monitoring(
        destination / "02_panel",
        destination / "04_backtest",
        destination / "06_forecast",
        destination / "07_monitoring",
        spec.monitoring,
        neural_dir=neural_dir if neural_dir.is_dir() else None,
        overwrite=True,
    )
    report = build_stage4_audit_report(
        destination,
        destination / "08_report",
        spec.report,
        overwrite=True,
    )

    identity = {
        "schema_version": "autogfm-stage4-pipeline-v1",
        "stage3_run_id": stage3.run_id,
        "monitoring_id": monitoring.monitoring_id,
        "report_id": report.report_id,
        "spec": asdict(spec),
    }
    run_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "run_id": run_id,
        "engineering_status": "pass",
        "deployment_status": monitoring.status,
        "graph_research_claim": report.decision["graph_research_claim"],
        "evidence_scope": report.decision["evidence_scope"],
        "release_ready": report.decision["release_ready"],
        "block_reasons": monitoring.manifest["block_reasons"],
        "warn_reasons": monitoring.manifest["warn_reasons"],
        "selected_models": stage3.manifest["backtest_selected_models"],
        "production_forecast_id": stage3.manifest["production_forecast_id"],
    }
    write_json(destination / "stage4_manifest.json", manifest)

    lines = [
        "# AutoGraphFM Stage 4 Run Report",
        "",
        f"- Stage 4 run: `{run_id}`",
        f"- Stage 3 evidence run: `{stage3.run_id}`",
        f"- Engineering status: **{manifest['engineering_status'].upper()}**",
        f"- Deployment status: **{manifest['deployment_status'].upper()}**",
        f"- Graph research claim: **{manifest['graph_research_claim'].upper()}**",
        f"- Evidence scope: **{manifest['evidence_scope'].upper()}**",
        f"- Release ready: **{manifest['release_ready']}**",
        "",
        "## Decisions",
        "",
        f"- Blocking reasons: `{manifest['block_reasons']}`",
        f"- Warning reasons: `{manifest['warn_reasons']}`",
        f"- Validation-selected models: `{manifest['selected_models']}`",
        "",
        "## Durable outputs",
        "",
        "- Operational monitoring: `07_monitoring/MONITORING_REPORT.md`",
        "- Standalone audit: `08_report/AUDIT_REPORT.html`",
        "- Forecast model card: `06_forecast/MODEL_CARD.md`",
        "",
        "A completed pipeline is not automatically deployment-ready. "
        "The monitoring gate and falsification controls are authoritative.",
        "",
    ]
    (destination / "STAGE4_RUN_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return Stage4PipelineResult(destination, run_id, stage3, manifest)
