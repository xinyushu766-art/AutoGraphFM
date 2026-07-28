import csv
from pathlib import Path

from autogfm.realdata.kaggle_china_auto import prepare_china_auto_sales


def test_prepare_china_auto_sales(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    source.write_text(
        "model,units_sold,make,low_price,high_price,year_month,is_ev,body_type,brand,brand_country\n"
        "Model Y,100,Tesla China,20,30,2024-01-01,EV,SUV,Tesla,United States\n"
        "Model Y,110,Tesla China,20,30,2024-02-01,EV,SUV,Tesla,United States\n",
        encoding="utf-8",
    )
    result = prepare_china_auto_sales(source, tmp_path / "out")
    assert result.manifest["profile"]["valid_rows"] == 2
    assert result.manifest["profile"]["series_count"] == 1
    with result.sales_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["series_name"] == "Tesla China::Model Y"
