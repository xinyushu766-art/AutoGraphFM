"""Validate a candidate SRNI-CAR Sales CSV before any model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.realdata import inspect_srni_car_sales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Return success for engineering-only synthetic or unverified files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = inspect_srni_car_sales(
        args.sales_csv,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"evidence_scope={result.evidence_scope}")
    print(f"verification_level={result.verification_level}")
    print(f"manifest={args.output_dir / 'srni_authenticity_manifest.json'}")
    if result.evidence_scope != "official_real_data" and not args.allow_unverified:
        raise SystemExit(
            "Candidate file is not approved for official SRNI-CAR claims. "
            "See the authenticity report."
        )


if __name__ == "__main__":
    main()
