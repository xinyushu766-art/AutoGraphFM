"""Validate Stage 12A stability selection, consensus and evidence gates."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from autogfm.experiments import StabilityGraphExperimentSpec
from autogfm.graph import (
    GraphSpec,
    StabilityRecoverySpec,
    build_relation_graph,
)
from autogfm.graph.recovery import GraphRecoverySpec
from autogfm.pipeline import Stage12ASpec, run_stage12a_pipeline
from autogfm.realdata import (
    CanonicalizationConfig,
    PanelSpec,
    build_monthly_panel,
    canonicalize_three_files,
)
from autogfm.realdata.common import read_csv_rows, write_json
from autogfm.realdata.demo import generate_stage3_demo_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage12a_validation"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Validation output exists: {args.output_dir}. Use --overwrite."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    demo = generate_stage3_demo_data(
        args.output_dir / "demo_data",
        num_series=18,
        num_months=72,
        seed=31,
    )
    canonical = canonicalize_three_files(
        demo.sales_csv,
        demo.mapping_csv,
        demo.vehicles_csv,
        args.output_dir / "canonical",
        config=CanonicalizationConfig(
            source_name="stage12a_demo",
            data_version="v1",
        ),
        overwrite=True,
    )
    panel = build_monthly_panel(
        canonical.output_dir / "canonical_monthly_sales.csv",
        canonical.output_dir / "canonical_catalog.csv",
        args.output_dir / "panel",
        PanelSpec(
            start_month=date(2020, 1, 1),
            train_end=date(2023, 12, 1),
            validation_end=date(2024, 12, 1),
            test_end=date(2025, 12, 1),
            min_history_months=36,
            max_missing_ratio=0.12,
            max_trailing_missing_months=1,
        ),
        overwrite=True,
    )
    graph = build_relation_graph(
        panel.output_dir,
        args.output_dir / "graph",
        GraphSpec(top_k=4),
        overwrite=True,
    )

    if args.full:
        recovery = StabilityRecoverySpec(
            base=GraphRecoverySpec(
                max_epochs=360,
                patience=55,
                seeds=(42, 43, 44),
            )
        )
        vehicle = StabilityGraphExperimentSpec(
            hidden_dim=24,
            top_k=4,
            discovery_max_epochs=15,
            discovery_patience=5,
            max_epochs=22,
            patience=6,
            discovery_seeds=(101, 102, 103, 104, 105),
            final_seeds=(42, 43, 44),
            bootstrap_samples=500,
        )
    else:
        recovery = StabilityRecoverySpec(
            base=GraphRecoverySpec(
                max_epochs=220,
                patience=35,
                seeds=(42, 43, 44),
            )
        )
        vehicle = StabilityGraphExperimentSpec(
            hidden_dim=12,
            top_k=4,
            discovery_max_epochs=5,
            discovery_patience=2,
            max_epochs=6,
            patience=3,
            discovery_seeds=(101, 102, 103),
            final_seeds=(42, 43),
            bootstrap_samples=100,
        )

    result = run_stage12a_pipeline(
        args.output_dir / "run",
        Stage12ASpec(
            recovery=recovery,
            vehicle_experiment=vehicle,
        ),
        panel_dir=panel.output_dir,
        graph_dir=graph.output_dir,
        evidence_scope="demo_only",
        overwrite=True,
    )

    recovery_manifest = result.recovery.manifest
    vehicle_manifest = result.vehicle_experiment.manifest
    _, aggregate, _ = read_csv_rows(
        args.output_dir / "run/02_vehicle_panel/aggregate_metrics.csv"
    )
    model_names = {row["model_name"] for row in aggregate}
    _, stability_rows, _ = read_csv_rows(
        args.output_dir / "run/02_vehicle_panel/edge_stability.csv"
    )
    required_models = {
        "last_value",
        "residual_temporal_gru",
        "safe_static_graph_gru",
        "safe_identity_graph_gru",
        "safe_permuted_graph_gru",
        "adaptive_mtgnn_graph_gru",
        "direct_sparse_graph_gru",
        "fixed_consensus_graph_gru",
        "stability_selected_sparse_graph_gru",
        "stability_selected_static_residual_gru",
    }
    checks = {
        "stability_recovery_pass": (
            recovery_manifest["stability_selection_status"] == "go"
        ),
        "candidate_precision_ge_0_70": (
            recovery_manifest["candidate_edge_metrics"][
                "edge_precision_at_k"
            ]
            >= 0.70
        ),
        "recovery_final_edges_fully_stable": (
            recovery_manifest["final_stability"][
                "mean_pairwise_jaccard"
            ]
            >= 0.95
        ),
        "all_stability_controls_executed": required_models <= model_names,
        "candidate_artifacts_exist": all(
            (
                args.output_dir
                / "run/02_vehicle_panel"
                / name
            ).is_file()
            for name in (
                "candidate_mask.npy",
                "consensus_adjacency.npy",
                "screened_static_prior.npy",
            )
        ),
        "selected_graph_edges_stable": any(
            row["model_name"] == "stability_selected_sparse_graph_gru"
            and float(row["mean_pairwise_jaccard"]) >= 0.95
            for row in stability_rows
        ),
        "paired_bootstrap_has_seven_controls": (
            len(
                read_csv_rows(
                    args.output_dir
                    / "run/02_vehicle_panel/paired_bootstrap.csv"
                )[1]
            )
            == 7
        ),
        "demo_scope_cannot_authorize_claim": (
            result.manifest["graph_claim_authorized"] is False
            and result.manifest["final_graph_research_claim"] == "no-claim"
        ),
        "architecture_ready_for_real_challenge": (
            result.manifest["architecture_decision"]
            == "stability_selected_sparse_graph_ready_for_real_vehicle_challenge"
        ),
        "stage12a_reports_exist": (
            (args.output_dir / "run/STAGE12A_REPORT.md").is_file()
            and (args.output_dir / "run/stage12a_manifest.json").is_file()
            and vehicle_manifest["engineering_status"] == "pass"
        ),
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "run_id": result.run_id,
        "checks": checks,
        "stability_recovery_status": result.manifest[
            "stability_recovery_status"
        ],
        "architecture_decision": result.manifest["architecture_decision"],
        "vehicle_stability_graph_status": result.manifest[
            "vehicle_stability_graph_status"
        ],
        "final_graph_research_claim": result.manifest[
            "final_graph_research_claim"
        ],
        "candidate_edge_precision_at_k": recovery_manifest[
            "candidate_edge_metrics"
        ]["edge_precision_at_k"],
        "recovery_final_edge_jaccard": recovery_manifest[
            "final_stability"
        ]["mean_pairwise_jaccard"],
    }
    write_json(args.output_dir / "validation_result.json", payload)
    lines = [
        "# Stage 12A Validation Report",
        "",
        f"- Validation: **{payload['status'].upper()}**",
        f"- Run ID: `{payload['run_id']}`",
        f"- Stability recovery: "
        f"**{payload['stability_recovery_status'].upper()}**",
        f"- Architecture decision: `{payload['architecture_decision']}`",
        f"- Final graph claim: "
        f"**{payload['final_graph_research_claim'].upper()}**",
        "",
        "## Checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Recovery diagnosis",
            "",
            "- Candidate edge precision@K: "
            f"`{payload['candidate_edge_precision_at_k']:.3f}`",
            "- Final edge Jaccard: "
            f"`{payload['recovery_final_edge_jaccard']:.3f}`",
            "",
            "Passing Stage 12A authorizes a full real vehicle-panel challenge. "
            "It does not authorize a graph-effect claim by itself.",
            "",
        ]
    )
    (args.output_dir / "STAGE12A_VALIDATION_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
