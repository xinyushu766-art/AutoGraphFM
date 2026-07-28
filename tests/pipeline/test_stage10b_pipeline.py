from __future__ import annotations

from pathlib import Path

from autogfm.pipeline.stage10b import run_stage10b_literature_market


def test_stage10b_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    rows = ["year_month,EV_Sales,Total_Sales"]
    year, month = 2018, 1
    for index in range(76):
        seasonal = -300 if month in {1, 2} else 300
        rows.append(
            f"{year:04d}-{month:02d}-01,"
            f"{max(1, 1000 + index * 35 + seasonal)},"
            f"{max(1, 10000 + index * 45 + seasonal)}"
        )
        month += 1
        if month == 13:
            year += 1
            month = 1
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = run_stage10b_literature_market(source, tmp_path / "stage10b")
    assert result.manifest["engineering_status"] == "pass"
    assert result.manifest["graph_research_claim"] == "no-claim"
    assert len(result.manifest["comparison"]) == 6
    assert (result.output_dir / "STAGE10B_REPORT.md").is_file()
