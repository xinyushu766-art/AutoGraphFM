"""Convert a candidate SRNI-CAR Sales CSV to the governed three-file contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.realdata import SrniCarPreparationConfig, prepare_srni_car_sales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name", default="srni_car")
    parser.add_argument("--data-version", default="2016-2022")
    parser.add_argument(
        "--require-official",
        action="store_true",
        help="Reject files that do not match the published SRNI-CAR profile.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required_scope = "official_real_data" if args.require_official else None
    result = prepare_srni_car_sales(
        args.sales_csv,
        args.output_dir,
        config=SrniCarPreparationConfig(
            source_name=args.source_name,
            data_version=args.data_version,
            required_evidence_scope=required_scope,
        ),
        overwrite=args.overwrite,
    )
    quality = result.manifest["quality"]
    source = result.manifest["source"]
    print(f"evidence_scope={source['evidence_scope']}")
    print(f"verification_level={source['verification_level']}")
    print(f"dataset_id={result.dataset_id}")
    print(f"valid_sales_rows={quality['valid_sales_rows']}")
    print(f"series={quality['series']}")
    print(f"month_range={quality['month_min']}..{quality['month_max']}")
    print(f"sales_csv={result.sales_csv}")
    print(f"mapping_csv={result.mapping_csv}")
    print(f"vehicles_csv={result.vehicles_csv}")


if __name__ == "__main__":
    main()
