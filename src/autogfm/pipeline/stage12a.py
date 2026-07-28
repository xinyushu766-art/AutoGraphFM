"""Stage 12A: train-only stability selection and consensus graph retraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autogfm.experiments import (
    StabilityGraphExperimentResult,
    StabilityGraphExperimentSpec,
    run_stability_graph_experiments,
)
from autogfm.graph import (
    StabilityRecoveryResult,
    StabilityRecoverySpec,
    run_stability_recovery_benchmark,
)
from autogfm.realdata.common import stable_payload_id, write_json


@dataclass(frozen=True)
class Stage12ASpec:
    recovery: StabilityRecoverySpec = StabilityRecoverySpec()
    vehicle_experiment: StabilityGraphExperimentSpec = (
        StabilityGraphExperimentSpec()
    )


@dataclass(frozen=True)
class Stage12AResult:
    output_dir: Path
    run_id: str
    manifest: dict[str, Any]
    recovery: StabilityRecoveryResult
    vehicle_experiment: StabilityGraphExperimentResult | None


def run_stage12a_pipeline(
    output_dir: str | Path,
    spec: Stage12ASpec | None = None,
    *,
    panel_dir: str | Path | None = None,
    graph_dir: str | Path | None = None,
    evidence_scope: str | None = None,
    overwrite: bool = False,
) -> Stage12AResult:
    """Run stability-selected recovery and optional vehicle-panel controls."""
    spec = spec or Stage12ASpec()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    recovery = run_stability_recovery_benchmark(
        destination / "01_stability_recovery",
        spec.recovery,
        overwrite=True,
    )
    if (panel_dir is None) != (graph_dir is None):
        raise ValueError("panel_dir and graph_dir must be supplied together.")
    vehicle: StabilityGraphExperimentResult | None = None
    if panel_dir is not None and graph_dir is not None:
        vehicle = run_stability_graph_experiments(
            panel_dir,
            graph_dir,
            destination / "02_vehicle_panel",
            spec.vehicle_experiment,
            evidence_scope=evidence_scope,
            overwrite=True,
        )

    recovery_status = recovery.manifest["stability_selection_status"]
    vehicle_scope = (
        vehicle.manifest["evidence_scope"] if vehicle is not None else "not_evaluated"
    )
    real_vehicle_scope = vehicle_scope in {
        "public_real_vehicle_panel",
        "user_real_vehicle_panel",
        "official_real_vehicle_panel",
    }
    vehicle_supports_claim = bool(
        vehicle is not None
        and vehicle.manifest["experiment_supports_graph_claim"]
    )
    graph_claim_authorized = bool(
        recovery_status == "go"
        and real_vehicle_scope
        and vehicle_supports_claim
    )
    architecture_decision = (
        "stability_selected_sparse_graph_ready_for_real_vehicle_challenge"
        if recovery_status == "go"
        else "refine_candidate_screening_and_consensus"
    )
    payload = {
        "schema_version": "autogfm-stage12a-v1",
        "spec": {
            "recovery": asdict(spec.recovery),
            "vehicle_experiment": asdict(spec.vehicle_experiment),
        },
        "recovery_benchmark_id": recovery.benchmark_id,
        "vehicle_experiment_id": vehicle.experiment_id if vehicle else None,
        "vehicle_evidence_scope": vehicle_scope,
        "architecture_decision": architecture_decision,
        "graph_claim_authorized": graph_claim_authorized,
    }
    run_id = stable_payload_id(payload)
    manifest = {
        **payload,
        "run_id": run_id,
        "engineering_status": "pass",
        "stability_recovery_status": recovery_status,
        "vehicle_stability_graph_status": (
            vehicle.manifest["graph_research_claim"]
            if vehicle is not None
            else "not_evaluated"
        ),
        "final_graph_research_claim": (
            "go" if graph_claim_authorized else "no-claim"
        ),
        "next_action": (
            "run_full_real_vehicle_panel_stability_selected_experiment"
            if architecture_decision
            == "stability_selected_sparse_graph_ready_for_real_vehicle_challenge"
            else "increase_discovery_repeats_or_adjust_screening_weights"
        ),
    }
    write_json(destination / "stage12a_manifest.json", manifest)

    lines = [
        "# AutoGraphFM Stage 12A Report",
        "",
        f"- Run ID: `{run_id}`",
        "- Engineering: **PASS**",
        f"- Stability recovery: **{recovery_status.upper()}**",
        f"- Vehicle-panel status: "
        f"**{manifest['vehicle_stability_graph_status'].upper()}**",
        f"- Final graph claim: "
        f"**{manifest['final_graph_research_claim'].upper()}**",
        f"- Architecture decision: `{architecture_decision}`",
        "",
        "Stage 12A uses repeated train-only discovery runs, lagged ridge "
        "screening, fixed candidate support, and consensus retraining. This "
        "removes random edge selection from the final model while preserving "
        "strict temporal and graph falsification controls.",
        "",
    ]
    if vehicle is None:
        lines.extend(
            [
                "No vehicle panel was supplied. Vehicle-level forecasting "
                "remains `NOT_EVALUATED`.",
                "",
            ]
        )
    elif not real_vehicle_scope:
        lines.extend(
            [
                f"The supplied evidence scope is `{vehicle_scope}`. It may "
                "validate engineering, stability and falsification controls, "
                "but cannot authorize a real graph-effect claim.",
                "",
            ]
        )
    (destination / "STAGE12A_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return Stage12AResult(destination, run_id, manifest, recovery, vehicle)
