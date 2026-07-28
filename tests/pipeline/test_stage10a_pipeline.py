from __future__ import annotations

from pathlib import Path

from autogfm.pipeline.stage10a import run_stage10a_real_market


def test_stage10a_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    rows = ["year_month,EV_Sales,Total_Sales"]
    year, month = 2018, 1
    for index in range(76):
        rows.append(
            f"{year:04d}-{month:02d}-01,{1000 + index * 25},{10000 + index * 40}"
        )
        month += 1
        if month == 13:
            year += 1
            month = 1
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = run_stage10a_real_market(source, tmp_path / "stage10a")
    assert result.manifest["engineering_status"] == "pass"
    assert result.manifest["graph_research_claim"] == "no-claim"
    assert (result.output_dir / "STAGE10A_REPORT.md").is_file()
