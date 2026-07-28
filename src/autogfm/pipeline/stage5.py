"""Stage 5 pipeline: governed release candidate generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autogfm.governance import ReleaseSpec, prepare_release_candidate
from autogfm.pipeline.stage4 import (
    Stage4PipelineResult,
    Stage4PipelineSpec,
    run_stage4_pipeline,
)
from autogfm.realdata.common import stable_payload_id, write_json


@dataclass(frozen=True)
class Stage5PipelineSpec:
    """Complete pipeline configuration through governed release packaging."""

    stage4: Stage4PipelineSpec
    release: ReleaseSpec = field(default_factory=ReleaseSpec)


@dataclass(frozen=True)
class Stage5PipelineResult:
    output_dir: Path
    run_id: str
    stage4: Stage4PipelineResult
    manifest: dict[str, Any]


def run_stage5_pipeline(
    sales_csv: str | Path,
    mapping_csv: str | Path,
    vehicles_csv: str | Path,
    output_dir: str | Path,
    spec: Stage5PipelineSpec,
    *,
    overwrite: bool = False,
) -> Stage5PipelineResult:
    """Run Stage 4 and apply the configured release policy."""
    destination = Path(output_dir)
    stage4 = run_stage4_pipeline(
        sales_csv,
        mapping_csv,
        vehicles_csv,
        destination,
        spec.stage4,
        overwrite=overwrite,
    )
    release = prepare_release_candidate(
        destination,
        destination / "09_release",
        spec.release,
        overwrite=True,
    )
    identity = {
        "schema_version": "autogfm-stage5-pipeline-v1",
        "stage4_run_id": stage4.run_id,
        "release_id": release.release_id,
        "spec": asdict(spec),
    }
    run_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "run_id": run_id,
        "release_status": release.status,
        "deployable": release.deployable,
        "release_reasons": release.manifest["reasons"],
        "deployment_status": stage4.manifest["deployment_status"],
        "evidence_scope": stage4.manifest["evidence_scope"],
        "graph_research_claim": stage4.manifest["graph_research_claim"],
    }
    write_json(destination / "stage5_manifest.json", manifest)
    lines = [
        "# AutoGraphFM Stage 5 Release Report",
        "",
        f"- Stage 5 run: `{run_id}`",
        f"- Release status: **{release.status.upper()}**",
        f"- Deployable: **{release.deployable}**",
        f"- Reasons: `{release.manifest['reasons']}`",
        "",
        "The governed release policy is authoritative. A shadow bundle is never deployable.",
        "",
    ]
    (destination / "STAGE5_RELEASE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return Stage5PipelineResult(destination, run_id, stage4, manifest)
