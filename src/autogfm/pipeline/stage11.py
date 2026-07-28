"""Stage 11: adaptive-graph recovery and vehicle-panel falsification gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autogfm.experiments import (
    AdaptiveGraphExperimentResult,
    AdaptiveGraphExperimentSpec,
    run_adaptive_graph_experiments,
)
from autogfm.graph import (
    GraphRecoveryResult,
    GraphRecoverySpec,
    run_graph_recovery_benchmark,
)
from autogfm.realdata.common import stable_payload_id, write_json


@dataclass(frozen=True)
class Stage11Spec:
    recovery: GraphRecoverySpec = GraphRecoverySpec()
    vehicle_experiment: AdaptiveGraphExperimentSpec = AdaptiveGraphExperimentSpec()


@dataclass(frozen=True)
class Stage11Result:
    output_dir: Path
    run_id: str
    manifest: dict[str, Any]
    recovery: GraphRecoveryResult
    vehicle_experiment: AdaptiveGraphExperimentResult | None


def run_stage11_pipeline(
    output_dir: str | Path,
    spec: Stage11Spec | None = None,
    *,
    panel_dir: str | Path | None = None,
    graph_dir: str | Path | None = None,
    evidence_scope: str | None = None,
    overwrite: bool = False,
) -> Stage11Result:
    """Run graph identifiability first, then optional vehicle-panel controls."""
    spec = spec or Stage11Spec()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    recovery = run_graph_recovery_benchmark(
        destination / "01_graph_recovery",
        spec.recovery,
        overwrite=True,
    )
    vehicle: AdaptiveGraphExperimentResult | None = None
    if (panel_dir is None) != (graph_dir is None):
        raise ValueError("panel_dir and graph_dir must be supplied together.")
    if panel_dir is not None and graph_dir is not None:
        vehicle = run_adaptive_graph_experiments(
            panel_dir,
            graph_dir,
            destination / "02_vehicle_panel",
            spec.vehicle_experiment,
            evidence_scope=evidence_scope,
            overwrite=True,
        )

    vehicle_scope = (
        vehicle.manifest["evidence_scope"] if vehicle is not None else "not_evaluated"
    )
    real_vehicle_scope = vehicle_scope in {
        "public_real_vehicle_panel",
        "user_real_vehicle_panel",
        "official_real_vehicle_panel",
    }
    graph_claim_authorized = bool(
        vehicle is not None
        and real_vehicle_scope
        and vehicle.manifest["graph_claim_authorized"]
    )
    architecture_decision = (
        "do_not_promote_low_rank_embedding_graphs"
        if recovery.manifest["graph_recovery_status"] != "go"
        else "eligible_for_vehicle_panel_challenge"
    )
    payload = {
        "schema_version": "autogfm-stage11-v1",
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
        "synthetic_recovery_status": recovery.manifest["graph_recovery_status"],
        "vehicle_graph_status": (
            vehicle.manifest["graph_research_claim"]
            if vehicle is not None
            else "not_evaluated"
        ),
        "final_graph_research_claim": "go" if graph_claim_authorized else "no-claim",
        "next_action": (
            "replace_low_rank_graph_parameterization_or_add_regularized_direct_graph"
            if architecture_decision == "do_not_promote_low_rank_embedding_graphs"
            else "run_full_real_vehicle_panel_multiseed_experiment"
        ),
    }
    write_json(destination / "stage11_manifest.json", manifest)
    lines = [
        "# AutoGraphFM Stage 11 Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Engineering: **PASS**",
        f"- Synthetic recovery: **{manifest['synthetic_recovery_status'].upper()}**",
        f"- Vehicle-panel status: **{manifest['vehicle_graph_status'].upper()}**",
        f"- Final graph claim: **{manifest['final_graph_research_claim'].upper()}**",
        f"- Architecture decision: `{architecture_decision}`",
        "",
        "## Interpretation",
        "",
        "The recovery benchmark is an architecture unit test: it checks whether "
        "the learned adjacency can recover a known dependency system before the "
        "model is trusted on automotive data. A synthetic pass is necessary but "
        "never sufficient for a vehicle-sales graph claim.",
        "",
    ]
    if recovery.manifest["graph_recovery_status"] != "go":
        lines.extend(
            [
                "The direct sparse identifiability control recovers the known "
                "edges, while the low-rank AGCRN/MTGNN embedding learners do not "
                "meet the configured edge-recovery threshold. They remain "
                "research challengers rather than promoted production models.",
                "",
            ]
        )
    if vehicle is None:
        lines.extend(
            [
                "No vehicle panel was supplied, so the vehicle-level graph "
                "experiment remains `NOT_EVALUATED`.",
                "",
            ]
        )
    elif not real_vehicle_scope:
        lines.extend(
            [
                f"The supplied panel evidence scope is `{vehicle_scope}`. It may "
                "validate engineering behavior but cannot authorize a real-data "
                "graph-effect claim.",
                "",
            ]
        )
    (destination / "STAGE11_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return Stage11Result(destination, run_id, manifest, recovery, vehicle)
