from pathlib import Path

from autogfm.forecasting.market_benchmark import run_market_benchmark


def test_market_benchmark(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    rows = ["year_month,EV_Sales,Total_Sales"]
    year, month = 2018, 1
    for index in range(76):
        rows.append(f"{year:04d}-{month:02d}-01,{100+index*5},{1000+index*7}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = run_market_benchmark(source, tmp_path / "out")
    assert result.manifest["source_rows"] == 76
    assert set(result.manifest["selected_models"]) == {"1", "3"}
    assert (result.output_dir / "REAL_MARKET_BENCHMARK.html").exists()
