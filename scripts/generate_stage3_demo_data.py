"""Generate high-fidelity three-file demo data for Stage 3."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.realdata.demo import generate_stage3_demo_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("examples/stage3/demo_data"))
    parser.add_argument("--num-series", type=int, default=24)
    parser.add_argument("--num-months", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = generate_stage3_demo_data(
        args.output_dir,
        num_series=args.num_series,
        num_months=args.num_months,
        seed=args.seed,
    )
    print(f"sales={result.sales_csv}")
    print(f"mapping={result.mapping_csv}")
    print(f"vehicles={result.vehicles_csv}")


if __name__ == "__main__":
    main()
