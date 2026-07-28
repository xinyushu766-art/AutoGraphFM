"""Prepare the public model-level China automobile sales CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.realdata.kaggle_china_auto import prepare_china_auto_sales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare_china_auto_sales(args.input, args.output_dir, overwrite=args.overwrite)
    print(f"dataset_id={result.dataset_id}")
    print(f"valid_rows={result.manifest['profile']['valid_rows']}")


if __name__ == "__main__":
    main()
