"""Validate Stage 11B sparse recovery, stability and evidence gates."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from autogfm.experiments import SparseGraphExperimentSpec
from autogfm.graph import GraphRecoverySpec, GraphSpec, build_relation_graph
from autogfm.pipeline import Stage11BSpec, run_stage11b_pipeline
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
        default=Path("outputs/stage11b_validation"),
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
            source_name="stage11b_demo",
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
        recovery = GraphRecoverySpec(
            max_epochs=500,
            patience=70,
            embedding_dim=12,
            seeds=(42, 43, 44),
        )
        vehicle = SparseGraphExperimentSpec(
            hidden_dim=24,
            embedding_dim=8,
            top_k=4,
            max_epochs=25,
            patience=6,
            seeds=(42, 43, 44),
            bootstrap_samples=500,
        )
    else:
        recovery = GraphRecoverySpec(
            max_epochs=240,
            patience=40,
            embedding_dim=10,
            seeds=(42, 43),
        )
        vehicle = SparseGraphExperimentSpec(
            hidden_dim=12,
            embedding_dim=6,
            top_k=4,
            max_epochs=6,
            patience=3,
            seeds=(42, 43),
            bootstrap_samples=100,
            consensus_ratio=1.0,
            min_pairwise_jaccard=0.0,
            min_stable_edge_ratio=0.0,
        )
    result = run_stage11b_pipeline(
        args.output_dir / "run",
        Stage11BSpec(
            recovery=recovery,
            vehicle_experiment=vehicle,
        ),
        panel_dir=panel.output_dir,
        graph_dir=graph.output_dir,
        evidence_scope="demo_only",
        overwrite=True,
    )

    _, aggregate, _ = read_csv_rows(
        args.output_dir / "run/02_vehicle_panel/aggregate_metrics.csv"
    )
    model_names = {row["model_name"] for row in aggregate}
    _, stability_rows, _ = read_csv_rows(
        args.output_dir / "run/02_vehicle_panel/edge_stability.csv"
    )
    recovery_manifest = result.recovery.manifest
    sparse_row = next(
        row
        for row in recovery_manifest["aggregate"]
        if row["model_name"] == "regularized_sparse_residual"
    )
    required_models = {
        "last_value",
        "residual_temporal_gru",
        "safe_static_graph_gru",
        "safe_identity_graph_gru",
        "safe_permuted_graph_gru",
        "adaptive_mtgnn_graph_gru",
        "direct_sparse_graph_gru",
        "residual_sparse_static_graph_gru",
    }
    checks = {
        "sparse_recovery_architecture_pass": (
            result.manifest["sparse_recovery_status"] == "go"
        ),
        "sparse_residual_recovers_known_edges": (
            sparse_row["mean_edge_precision_at_k"] >= 0.5
        ),
        "all_sparse_graph_controls_executed": required_models <= model_names,
        "edge_stability_artifacts_exist": (
            len(stability_rows) == 2
            and (
                args.output_dir
                / "run/02_vehicle_panel/consensus_graphs/"
                "residual_sparse_static_graph_gru_consensus.npy"
            ).is_file()
        ),
        "paired_bootstrap_has_five_controls": (
            len(
                read_csv_rows(
                    args.output_dir
                    / "run/02_vehicle_panel/paired_bootstrap.csv"
                )[1]
            )
            == 5
        ),
        "demo_scope_cannot_authorize_claim": (
            result.manifest["graph_claim_authorized"] is False
            and result.manifest["final_graph_research_claim"] == "no-claim"
        ),
        "architecture_decision_is_sparse_challenge": (
            result.manifest["architecture_decision"]
            == "sparse_residual_graph_eligible_for_real_vehicle_challenge"
        ),
        "vehicle_experiment_engineering_pass": (
            result.vehicle_experiment is not None
            and result.vehicle_experiment.manifest["engineering_status"]
            == "pass"
        ),
        "stage11b_report_exists": (
            args.output_dir / "run/STAGE11B_REPORT.md"
        ).is_file(),
        "stage11b_manifest_exists": (
            args.output_dir / "run/stage11b_manifest.json"
        ).is_file(),
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "run_id": result.run_id,
        "checks": checks,
        "sparse_recovery_status": result.manifest["sparse_recovery_status"],
        "low_rank_recovery_status": result.manifest[
            "low_rank_recovery_status"
        ],
        "architecture_decision": result.manifest["architecture_decision"],
        "vehicle_sparse_graph_status": result.manifest[
            "vehicle_sparse_graph_status"
        ],
        "final_graph_research_claim": result.manifest[
            "final_graph_research_claim"
        ],
        "sparse_residual_edge_precision_at_k": sparse_row[
            "mean_edge_precision_at_k"
        ],
    }
    write_json(args.output_dir / "validation_result.json", payload)
    lines = [
        "# Stage 11B Validation Report",
        "",
        f"- Validation: **{payload['status'].upper()}**",
        f"- Run ID: `{payload['run_id']}`",
        f"- Sparse recovery: **{payload['sparse_recovery_status'].upper()}**",
        f"- Low-rank recovery: **{payload['low_rank_recovery_status'].upper()}**",
        f"- Architecture decision: `{payload['architecture_decision']}`",
        f"- Final graph claim: **{payload['final_graph_research_claim'].upper()}**",
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
            "- Sparse residual edge precision@K: "
            f"`{payload['sparse_residual_edge_precision_at_k']:.3f}`",
            "",
            "Passing Stage 11B authorizes a real vehicle-panel challenge, not "
            "a graph-effect claim. The latter still requires real evidence, "
            "performance controls, significance and cross-seed edge stability.",
            "",
        ]
    )
    (args.output_dir / "STAGE11B_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
