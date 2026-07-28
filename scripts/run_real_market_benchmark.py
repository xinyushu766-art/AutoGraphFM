"""Run the real aggregate China automotive market benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.forecasting.market_benchmark import run_market_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--value-column", default="EV_Sales")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_market_benchmark(
        args.input,
        args.output_dir,
        value_column=args.value_column,
        overwrite=args.overwrite,
    )
    print(result.output_dir / "REAL_MARKET_BENCHMARK.html")


if __name__ == "__main__":
    main()
