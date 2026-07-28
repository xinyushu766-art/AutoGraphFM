"""Stage 11B: regularized direct sparse residual graph validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autogfm.experiments import (
    SparseGraphExperimentResult,
    SparseGraphExperimentSpec,
    run_sparse_graph_experiments,
)
from autogfm.graph import (
    GraphRecoveryResult,
    GraphRecoverySpec,
    run_graph_recovery_benchmark,
)
from autogfm.realdata.common import stable_payload_id, write_json


@dataclass(frozen=True)
class Stage11BSpec:
    recovery: GraphRecoverySpec = GraphRecoverySpec()
    vehicle_experiment: SparseGraphExperimentSpec = SparseGraphExperimentSpec()


@dataclass(frozen=True)
class Stage11BResult:
    output_dir: Path
    run_id: str
    manifest: dict[str, Any]
    recovery: GraphRecoveryResult
    vehicle_experiment: SparseGraphExperimentResult | None


def run_stage11b_pipeline(
    output_dir: str | Path,
    spec: Stage11BSpec | None = None,
    *,
    panel_dir: str | Path | None = None,
    graph_dir: str | Path | None = None,
    evidence_scope: str | None = None,
    overwrite: bool = False,
) -> Stage11BResult:
    """Validate sparse recovery, edge stability and vehicle-panel controls."""
    spec = spec or Stage11BSpec()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    recovery = run_graph_recovery_benchmark(
        destination / "01_sparse_recovery",
        spec.recovery,
        overwrite=True,
    )
    vehicle: SparseGraphExperimentResult | None = None
    if (panel_dir is None) != (graph_dir is None):
        raise ValueError("panel_dir and graph_dir must be supplied together.")
    if panel_dir is not None and graph_dir is not None:
        vehicle = run_sparse_graph_experiments(
            panel_dir,
            graph_dir,
            destination / "02_vehicle_panel",
            spec.vehicle_experiment,
            evidence_scope=evidence_scope,
            overwrite=True,
        )

    sparse_recovery_status = recovery.manifest[
        "sparse_residual_recovery_status"
    ]
    vehicle_scope = (
        vehicle.manifest["evidence_scope"] if vehicle is not None else "not_evaluated"
    )
    real_vehicle_scope = vehicle_scope in {
        "public_real_vehicle_panel",
        "user_real_vehicle_panel",
        "official_real_vehicle_panel",
    }
    experiment_supports_claim = bool(
        vehicle is not None
        and vehicle.manifest["experiment_supports_graph_claim"]
    )
    graph_claim_authorized = bool(
        sparse_recovery_status == "go"
        and real_vehicle_scope
        and experiment_supports_claim
    )
    architecture_decision = (
        "sparse_residual_graph_eligible_for_real_vehicle_challenge"
        if sparse_recovery_status == "go"
        else "refine_sparse_residual_regularization"
    )
    payload = {
        "schema_version": "autogfm-stage11b-v1",
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
        "sparse_recovery_status": sparse_recovery_status,
        "low_rank_recovery_status": recovery.manifest[
            "low_rank_recovery_status"
        ],
        "vehicle_sparse_graph_status": (
            vehicle.manifest["graph_research_claim"]
            if vehicle is not None
            else "not_evaluated"
        ),
        "final_graph_research_claim": (
            "go" if graph_claim_authorized else "no-claim"
        ),
        "next_action": (
            "run_full_real_vehicle_panel_multiseed_sparse_graph_experiment"
            if architecture_decision
            == "sparse_residual_graph_eligible_for_real_vehicle_challenge"
            else "tune_sparse_penalties_and_consensus_thresholds"
        ),
    }
    write_json(destination / "stage11b_manifest.json", manifest)

    lines = [
        "# AutoGraphFM Stage 11B Report",
        "",
        f"- Run ID: `{run_id}`",
        "- Engineering: **PASS**",
        f"- Sparse recovery: **{sparse_recovery_status.upper()}**",
        f"- Low-rank recovery: **{manifest['low_rank_recovery_status'].upper()}**",
        f"- Vehicle-panel status: "
        f"**{manifest['vehicle_sparse_graph_status'].upper()}**",
        f"- Final graph claim: **{manifest['final_graph_research_claim'].upper()}**",
        f"- Architecture decision: `{architecture_decision}`",
        "",
        "## Interpretation",
        "",
        "Stage 11B learns a row-sparse direct residual adjacency around the "
        "static automotive prior. Promotion requires known-graph recovery, "
        "lower forecast error against temporal and graph controls, paired "
        "bootstrap significance, and stable edges across model seeds.",
        "",
    ]
    if vehicle is None:
        lines.extend(
            [
                "No vehicle panel was supplied. The architecture gate can pass, "
                "but vehicle-level forecasting remains `NOT_EVALUATED`.",
                "",
            ]
        )
    elif not real_vehicle_scope:
        lines.extend(
            [
                f"The supplied panel evidence scope is `{vehicle_scope}`. It may "
                "validate engineering and falsification controls, but cannot "
                "authorize a real vehicle-sales graph claim.",
                "",
            ]
        )
    (destination / "STAGE11B_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return Stage11BResult(destination, run_id, manifest, recovery, vehicle)
