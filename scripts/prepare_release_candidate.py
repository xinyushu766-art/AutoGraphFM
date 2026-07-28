"""Apply release policy to an existing Stage 4 run."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.governance import ReleaseSpec, prepare_release_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--mode", choices=["production", "shadow"], default="production"
    )
    parser.add_argument("--allow-warn", action="store_true")
    parser.add_argument("--allow-non-user-data", action="store_true")
    parser.add_argument("--require-graph-claim", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "09_release"
    result = prepare_release_candidate(
        args.run_dir,
        output_dir,
        ReleaseSpec(
            mode=args.mode,
            allow_warn=args.allow_warn,
            require_user_data=not args.allow_non_user_data,
            require_graph_claim=args.require_graph_claim,
        ),
        overwrite=args.overwrite,
    )
    print(f"release_id={result.release_id}")
    print(f"status={result.status}")
    print(f"deployable={result.deployable}")


if __name__ == "__main__":
    main()
