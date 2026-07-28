"""Release governance and immutable forecast bundle creation."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autogfm.realdata.common import sha256_file, stable_payload_id, write_json


@dataclass(frozen=True)
class ReleaseSpec:
    """Policy controlling whether a deployable or shadow bundle is emitted."""

    mode: str = "production"
    allow_warn: bool = False
    require_user_data: bool = True
    require_graph_claim: bool = False

    def validate(self) -> None:
        if self.mode not in {"production", "shadow"}:
            raise ValueError("mode must be 'production' or 'shadow'.")


@dataclass(frozen=True)
class ReleaseResult:
    output_dir: Path
    release_id: str
    status: str
    deployable: bool
    manifest: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_release_candidate(
    run_dir: str | Path,
    output_dir: str | Path,
    spec: ReleaseSpec | None = None,
    *,
    overwrite: bool = False,
) -> ReleaseResult:
    """Apply release policy and create a checksummed production or shadow packet."""
    spec = spec or ReleaseSpec()
    spec.validate()
    root = Path(run_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    stage4 = _load_json(root / "stage4_manifest.json")
    reasons: list[str] = []
    if stage4["engineering_status"] != "pass":
        reasons.append("engineering_not_passed")
    deployment = stage4["deployment_status"]
    if deployment == "block":
        reasons.append("deployment_gate_blocked")
    elif deployment == "warn" and not spec.allow_warn:
        reasons.append("deployment_gate_warning_not_accepted")
    deployable_scopes = {"user_data", "official_real_data"}
    if spec.require_user_data and stage4["evidence_scope"] not in deployable_scopes:
        reasons.append("evidence_scope_is_not_verified_real_data")
    if spec.require_graph_claim and stage4["graph_research_claim"] != "go":
        reasons.append("graph_claim_requirement_not_met")

    if spec.mode == "shadow":
        status = "shadow" if stage4["engineering_status"] == "pass" else "blocked"
        deployable = False
    else:
        status = "approved" if not reasons else "blocked"
        deployable = status == "approved"

    identity = {
        "schema_version": "autogfm-release-candidate-v1",
        "stage4_run_id": stage4["run_id"],
        "spec": asdict(spec),
        "status": status,
        "deployable": deployable,
        "reasons": reasons,
    }
    release_id = stable_payload_id(identity)

    bundle_files: dict[str, dict[str, str]] = {}
    bundle_dir: Path | None = None
    if deployable or status == "shadow":
        bundle_dir = destination / (
            "deployment_bundle" if deployable else "shadow_bundle"
        )
        bundle_dir.mkdir(parents=True, exist_ok=True)
        sources = {
            "production_forecast.csv": root
            / "06_forecast"
            / "production_forecast.csv",
            "calibration.json": root / "06_forecast" / "calibration.json",
            "MODEL_CARD.md": root / "06_forecast" / "MODEL_CARD.md",
            "selected_model_test_metrics.csv": root
            / "04_backtest"
            / "selected_model_test_metrics.csv",
            "MONITORING_REPORT.md": root
            / "07_monitoring"
            / "MONITORING_REPORT.md",
            "AUDIT_REPORT.html": root / "08_report" / "AUDIT_REPORT.html",
            "AUDIT_SUMMARY.md": root / "08_report" / "AUDIT_SUMMARY.md",
            "stage4_manifest.json": root / "stage4_manifest.json",
        }
        for name, source in sources.items():
            target = bundle_dir / name
            shutil.copy2(source, target)
            bundle_files[name] = {
                "path": str(target.relative_to(destination)),
                "sha256": sha256_file(target),
            }
        checksum_lines = [
            f"{meta['sha256']}  {name}" for name, meta in sorted(bundle_files.items())
        ]
        checksum_path = bundle_dir / "SHA256SUMS.txt"
        checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        bundle_files["SHA256SUMS.txt"] = {
            "path": str(checksum_path.relative_to(destination)),
            "sha256": sha256_file(checksum_path),
        }

    decision = {
        **identity,
        "release_id": release_id,
        "source_decision": {
            "engineering_status": stage4["engineering_status"],
            "deployment_status": stage4["deployment_status"],
            "graph_research_claim": stage4["graph_research_claim"],
            "evidence_scope": stage4["evidence_scope"],
            "release_ready": stage4["release_ready"],
        },
        "bundle_directory": (
            str(bundle_dir.relative_to(destination)) if bundle_dir else None
        ),
        "files": bundle_files,
    }
    write_json(destination / "release_decision.json", decision)

    lines = [
        "# AutoGraphFM Release Readiness",
        "",
        f"- Release ID: `{release_id}`",
        f"- Mode: **{spec.mode.upper()}**",
        f"- Status: **{status.upper()}**",
        f"- Deployable: **{deployable}**",
        f"- Evidence scope: **{stage4['evidence_scope'].upper()}**",
        f"- Deployment gate: **{stage4['deployment_status'].upper()}**",
        "",
        "## Blocking reasons",
        "",
    ]
    lines.extend([f"- {reason}" for reason in reasons] or ["- None"])
    lines.extend(
        [
            "",
            "A shadow packet is explicitly non-deployable. A production packet is emitted only when every configured policy requirement is satisfied.",
            "",
        ]
    )
    readiness_path = destination / "RELEASE_READINESS.md"
    readiness_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        **decision,
        "files": {
            **bundle_files,
            "release_decision.json": {
                "path": "release_decision.json",
                "sha256": sha256_file(destination / "release_decision.json"),
            },
            "RELEASE_READINESS.md": {
                "path": "RELEASE_READINESS.md",
                "sha256": sha256_file(readiness_path),
            },
        },
    }
    write_json(destination / "manifest.json", manifest)
    return ReleaseResult(destination, release_id, status, deployable, manifest)
