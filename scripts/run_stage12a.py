"""Run Stage 12A stability-selected graph recovery and panel experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.experiments import StabilityGraphExperimentSpec
from autogfm.graph import StabilityRecoverySpec
from autogfm.graph.recovery import GraphRecoverySpec
from autogfm.pipeline import Stage12ASpec, run_stage12a_pipeline


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
        recovery = StabilityRecoverySpec(
            base=GraphRecoverySpec(
                max_epochs=220,
                patience=35,
                seeds=(42, 43, 44),
            )
        )
        vehicle = StabilityGraphExperimentSpec(
            hidden_dim=16,
            top_k=4,
            discovery_max_epochs=6,
            discovery_patience=3,
            max_epochs=8,
            patience=3,
            discovery_seeds=(101, 102, 103),
            final_seeds=(42, 43),
            bootstrap_samples=100,
            minimum_selected_jaccard=0.95,
            minimum_selected_stable_ratio=0.95,
        )
    else:
        recovery = StabilityRecoverySpec()
        vehicle = StabilityGraphExperimentSpec()

    result = run_stage12a_pipeline(
        args.output_dir,
        Stage12ASpec(recovery=recovery, vehicle_experiment=vehicle),
        panel_dir=args.panel_dir,
        graph_dir=args.graph_dir,
        evidence_scope=args.evidence_scope,
        overwrite=args.overwrite,
    )
    print(f"run_id={result.run_id}")
    print(
        "stability_recovery="
        f"{result.manifest['stability_recovery_status']}"
    )
    print(
        "vehicle_stability_graph="
        f"{result.manifest['vehicle_stability_graph_status']}"
    )
    print(f"graph_claim={result.manifest['final_graph_research_claim']}")


if __name__ == "__main__":
    main()
