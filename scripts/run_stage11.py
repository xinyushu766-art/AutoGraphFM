"""Run AutoGraphFM Stage 11 graph recovery and optional panel experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.experiments import AdaptiveGraphExperimentSpec
from autogfm.graph import GraphRecoverySpec
from autogfm.pipeline import Stage11Spec, run_stage11_pipeline


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
            max_epochs=120,
            patience=25,
            seeds=(42,),
        )
        vehicle = AdaptiveGraphExperimentSpec(
            hidden_dim=16,
            embedding_dim=6,
            max_epochs=6,
            patience=3,
            seeds=(42,),
            bootstrap_samples=100,
        )
    else:
        recovery = GraphRecoverySpec()
        vehicle = AdaptiveGraphExperimentSpec()
    result = run_stage11_pipeline(
        args.output_dir,
        Stage11Spec(recovery=recovery, vehicle_experiment=vehicle),
        panel_dir=args.panel_dir,
        graph_dir=args.graph_dir,
        evidence_scope=args.evidence_scope,
        overwrite=args.overwrite,
    )
    print(f"run_id={result.run_id}")
    print(f"synthetic_recovery={result.manifest['synthetic_recovery_status']}")
    print(f"vehicle_graph={result.manifest['vehicle_graph_status']}")
    print(f"graph_claim={result.manifest['final_graph_research_claim']}")


if __name__ == "__main__":
    main()
