from pathlib import Path

from autogfm.realdata.common import read_csv_rows


def test_canonicalization_audits_duplicate_and_bad_row(stage3_artifacts: dict[str, object]) -> None:
    result = stage3_artifacts["canonical"]
    quality = result.manifest["quality"]
    assert quality["duplicate_monthly_keys_aggregated"] == 1
    assert quality["rejected_sales_rows"] == 1
    assert quality["canonical_series"] == 12


def test_canonical_outputs_have_unique_business_keys(stage3_artifacts: dict[str, object]) -> None:
    result = stage3_artifacts["canonical"]
    _, rows, _ = read_csv_rows(result.output_dir / "canonical_monthly_sales.csv")
    keys = [(row["series_key"], row["month"]) for row in rows]
    assert len(keys) == len(set(keys))


def test_manifest_hashes_point_to_existing_outputs(stage3_artifacts: dict[str, object]) -> None:
    result = stage3_artifacts["canonical"]
    for file_info in result.manifest["outputs"].values():
        assert (result.output_dir / file_info["path"]).is_file()
        assert len(file_info["sha256"]) == 64
