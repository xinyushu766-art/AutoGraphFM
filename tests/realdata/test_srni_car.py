import csv
from pathlib import Path

import pytest

from autogfm.realdata import (
    CanonicalizationConfig,
    RealDataError,
    SrniCarAuthenticitySpec,
    SrniCarPreparationConfig,
    canonicalize_three_files,
    inspect_srni_car_sales,
    prepare_srni_car_sales,
)
from autogfm.realdata.common import read_csv_rows


_HEADERS = [
    "Car series",
    "Brand",
    "Year",
    "Month",
    "Car model type",
    "Brand energy type",
    "Size",
    "Brand country of origin",
    "Model launch date",
    "Brand establishment date",
    "Brand entered China date",
    "Sales",
]


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADERS)
        writer.writerows(rows)


def _write_srni(path: Path) -> None:
    _write_rows(
        path,
        [
            [
                "Alpha",
                "BrandA",
                "2019",
                "January",
                "SUV",
                "ICE",
                "Compact",
                "China",
                "2017",
                "2001",
                "2005",
                "100",
            ],
            [
                "Alpha",
                "BrandA",
                "2019",
                "2",
                "SUV",
                "ICE",
                "Compact",
                "China",
                "2017",
                "2001",
                "2005",
                "120",
            ],
            [
                "Beta",
                "BrandB",
                "2019",
                "1月",
                "Sedan",
                "NEV",
                "Mid-size",
                "China",
                "2018",
                "2010",
                "2012",
                "80",
            ],
            [
                "Beta",
                "BrandB",
                "2019",
                "02",
                "Sedan",
                "NEV",
                "Mid-size",
                "China",
                "2018",
                "2010",
                "2012",
                "90",
            ],
            [
                "Bad",
                "BrandB",
                "2019",
                "not-a-month",
                "Sedan",
                "NEV",
                "Small",
                "China",
                "2018",
                "2010",
                "2012",
                "10",
            ],
        ],
    )


def _write_clean_srni(path: Path) -> None:
    _write_rows(
        path,
        [
            [
                "Alpha",
                "MakerA",
                "2019",
                "1",
                "SUV",
                "ICE",
                "Compact",
                "China",
                "2017",
                "2001",
                "2005",
                "100",
            ],
            [
                "Alpha",
                "MakerA",
                "2019",
                "2",
                "SUV",
                "ICE",
                "Compact",
                "China",
                "2017",
                "2001",
                "2005",
                "120",
            ],
            [
                "Beta",
                "MakerB",
                "2019",
                "1",
                "Sedan",
                "NEV",
                "Mid-size",
                "China",
                "2018",
                "2010",
                "2012",
                "80",
            ],
            [
                "Beta",
                "MakerB",
                "2019",
                "2",
                "Sedan",
                "NEV",
                "Mid-size",
                "China",
                "2018",
                "2010",
                "2012",
                "90",
            ],
        ],
    )


def test_prepare_srni_car_builds_three_file_contract(tmp_path: Path) -> None:
    source = tmp_path / "Sales.csv"
    _write_srni(source)
    result = prepare_srni_car_sales(source, tmp_path / "prepared")

    assert result.manifest["quality"]["valid_sales_rows"] == 4
    assert result.manifest["quality"]["rejected_rows"] == 1
    assert result.manifest["quality"]["series"] == 2
    assert result.manifest["quality"]["month_min"] == "2019-01-01"
    assert result.manifest["quality"]["month_max"] == "2019-02-01"
    assert result.manifest["source"]["evidence_scope"] == "real_unverified"

    _, sales, _ = read_csv_rows(result.sales_csv)
    assert {(row["series_name"], row["month"]) for row in sales} == {
        ("Alpha", "2019-01-01"),
        ("Alpha", "2019-02-01"),
        ("Beta", "2019-01-01"),
        ("Beta", "2019-02-01"),
    }


def test_prepared_srni_files_enter_existing_canonicalizer(tmp_path: Path) -> None:
    source = tmp_path / "Sales.csv"
    _write_srni(source)
    prepared = prepare_srni_car_sales(source, tmp_path / "prepared")
    canonical = canonicalize_three_files(
        prepared.sales_csv,
        prepared.mapping_csv,
        prepared.vehicles_csv,
        tmp_path / "canonical",
        config=CanonicalizationConfig(source_name="srni_car", data_version="test"),
    )

    assert canonical.manifest["quality"]["canonical_series"] == 2
    assert canonical.manifest["quality"]["canonical_monthly_rows"] == 4
    assert canonical.manifest["quality"]["unmatched_rows"] == 0


def test_inspection_detects_synthetic_names(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.csv"
    _write_rows(
        source,
        [
            [
                "Series01",
                "Brand1",
                "2019",
                "1",
                "SUV",
                "ICE",
                "Compact",
                "China",
                "2017",
                "2001",
                "2005",
                "100",
            ],
            [
                "Series02",
                "Brand2",
                "2019",
                "1",
                "Sedan",
                "NEV",
                "Mid-size",
                "China",
                "2018",
                "2010",
                "2012",
                "80",
            ],
        ],
    )

    inspection = inspect_srni_car_sales(source, tmp_path / "inspection")

    assert inspection.evidence_scope == "synthetic_only"
    assert inspection.verification_level == "synthetic_detected"


def test_custom_official_profile_can_be_verified_without_large_fixture(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.csv"
    _write_clean_srni(source)
    spec = SrniCarAuthenticitySpec(
        valid_rows_min=4,
        valid_rows_max=4,
        series_min=2,
        series_max=2,
        brands_min=2,
        brands_max=2,
        expected_month_min="2019-01-01",
        expected_month_max="2019-02-01",
    )

    inspection = inspect_srni_car_sales(source, spec=spec)

    assert inspection.evidence_scope == "official_real_data"
    assert inspection.manifest["official_profile_checks"]["not_synthetic"] is True


def test_preparation_enforces_required_evidence_scope(tmp_path: Path) -> None:
    source = tmp_path / "candidate.csv"
    _write_clean_srni(source)

    with pytest.raises(RealDataError, match="evidence scope mismatch"):
        prepare_srni_car_sales(
            source,
            tmp_path / "prepared",
            config=SrniCarPreparationConfig(
                required_evidence_scope="official_real_data"
            ),
        )
