"""Validate Stage 11 adaptive-graph recovery and conservative claim gates."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from autogfm.experiments import AdaptiveGraphExperimentSpec
from autogfm.graph import GraphRecoverySpec, GraphSpec, build_relation_graph
from autogfm.pipeline import Stage11Spec, run_stage11_pipeline
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
        default=Path("outputs/stage11_validation"),
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
        seed=29,
    )
    canonical = canonicalize_three_files(
        demo.sales_csv,
        demo.mapping_csv,
        demo.vehicles_csv,
        args.output_dir / "canonical",
        config=CanonicalizationConfig(
            source_name="stage11_demo",
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
        vehicle = AdaptiveGraphExperimentSpec(
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
            max_epochs=220,
            patience=35,
            embedding_dim=10,
            seeds=(42, 43),
        )
        vehicle = AdaptiveGraphExperimentSpec(
            hidden_dim=12,
            embedding_dim=6,
            top_k=4,
            max_epochs=6,
            patience=3,
            seeds=(42,),
            bootstrap_samples=100,
        )
    result = run_stage11_pipeline(
        args.output_dir / "run",
        Stage11Spec(
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
    _, bootstrap_rows, _ = read_csv_rows(
        args.output_dir / "run/02_vehicle_panel/paired_bootstrap.csv"
    )
    recovery_manifest = result.recovery.manifest
    direct_row = next(
        row
        for row in recovery_manifest["aggregate"]
        if row["model_name"] == "direct_sparse_adaptive"
    )
    mtgnn_row = next(
        row
        for row in recovery_manifest["aggregate"]
        if row["model_name"] == "mtgnn_adaptive"
    )
    required_models = {
        "last_value",
        "residual_temporal_gru",
        "safe_static_graph_gru",
        "safe_identity_graph_gru",
        "safe_permuted_graph_gru",
        "adaptive_agcrn_graph_gru",
        "adaptive_mtgnn_graph_gru",
        "hybrid_agcrn_static_graph_gru",
        "hybrid_mtgnn_static_graph_gru",
    }
    checks = {
        "direct_graph_identifiability_control_recovers_edges": (
            direct_row["mean_edge_precision_at_k"] >= 0.5
        ),
        "low_rank_embedding_recovery_is_explicit": (
            mtgnn_row["mean_edge_precision_at_k"] >= 0.0
            and recovery_manifest["graph_recovery_status"] in {"go", "no-go"}
        ),
        "all_vehicle_graph_controls_executed": required_models <= model_names,
        "paired_bootstrap_has_four_controls": len(bootstrap_rows) == 4,
        "learned_graph_artifacts_exist": len(
            list(
                (args.output_dir / "run/02_vehicle_panel/learned_graphs").glob(
                    "*.npy"
                )
            )
        )
        >= 4,
        "demo_scope_cannot_authorize_graph_claim": (
            result.manifest["graph_claim_authorized"] is False
            and result.manifest["final_graph_research_claim"] == "no-claim"
        ),
        "architecture_decision_is_explicit": result.manifest[
            "architecture_decision"
        ]
        in {
            "do_not_promote_low_rank_embedding_graphs",
            "eligible_for_vehicle_panel_challenge",
        },
        "vehicle_experiment_engineering_pass": (
            result.vehicle_experiment is not None
            and result.vehicle_experiment.manifest["engineering_status"] == "pass"
        ),
        "stage11_report_exists": (
            args.output_dir / "run/STAGE11_REPORT.md"
        ).is_file(),
        "stage11_manifest_exists": (
            args.output_dir / "run/stage11_manifest.json"
        ).is_file(),
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "run_id": result.run_id,
        "checks": checks,
        "synthetic_recovery_status": result.manifest[
            "synthetic_recovery_status"
        ],
        "architecture_decision": result.manifest["architecture_decision"],
        "vehicle_graph_status": result.manifest["vehicle_graph_status"],
        "final_graph_research_claim": result.manifest[
            "final_graph_research_claim"
        ],
        "direct_sparse_edge_precision_at_k": direct_row[
            "mean_edge_precision_at_k"
        ],
        "mtgnn_edge_precision_at_k": mtgnn_row[
            "mean_edge_precision_at_k"
        ],
    }
    write_json(args.output_dir / "validation_result.json", payload)
    lines = [
        "# Stage 11 Validation Report",
        "",
        f"- Validation: **{payload['status'].upper()}**",
        f"- Run ID: `{payload['run_id']}`",
        f"- Synthetic recovery: **{payload['synthetic_recovery_status'].upper()}**",
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
            "- Direct sparse edge precision@K: "
            f"`{payload['direct_sparse_edge_precision_at_k']:.3f}`",
            "- MTGNN embedding edge precision@K: "
            f"`{payload['mtgnn_edge_precision_at_k']:.3f}`",
            "",
            "The validation intentionally accepts an explicit architecture NO-GO. "
            "A failed low-rank graph recovery test is a research finding, not an "
            "engineering failure, when the positive identifiability control passes.",
            "",
        ]
    )
    (args.output_dir / "STAGE11_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
