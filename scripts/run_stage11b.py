"""Run Stage 11B sparse residual graph recovery and panel experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.experiments import SparseGraphExperimentSpec
from autogfm.graph import GraphRecoverySpec
from autogfm.pipeline import Stage11BSpec, run_stage11b_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument(
        "--evidence-scope",
        default="unverified_vehicle_panel",
        choices=[
            "demo_only",
            "unverified_vehicle_panel",
            "public_real_vehicle_panel",
            "user_real_vehicle_panel",
            "official_real_vehicle_panel",
        ],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.quick:
        recovery = GraphRecoverySpec(
            max_epochs=140,
            patience=25,
            seeds=(42,),
        )
        vehicle = SparseGraphExperimentSpec(
            hidden_dim=16,
            embedding_dim=6,
            max_epochs=6,
            patience=3,
            seeds=(42, 43),
            bootstrap_samples=100,
            consensus_ratio=1.0,
            min_pairwise_jaccard=0.0,
            min_stable_edge_ratio=0.0,
        )
    else:
        recovery = GraphRecoverySpec()
        vehicle = SparseGraphExperimentSpec()
    result = run_stage11b_pipeline(
        args.output_dir,
        Stage11BSpec(recovery=recovery, vehicle_experiment=vehicle),
        panel_dir=args.panel_dir,
        graph_dir=args.graph_dir,
        evidence_scope=args.evidence_scope,
        overwrite=args.overwrite,
    )
    print(f"run_id={result.run_id}")
    print(f"sparse_recovery={result.manifest['sparse_recovery_status']}")
    print(
        "vehicle_sparse_graph="
        f"{result.manifest['vehicle_sparse_graph_status']}"
    )
    print(f"graph_claim={result.manifest['final_graph_research_claim']}")


if __name__ == "__main__":
    main()
