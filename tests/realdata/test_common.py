from datetime import date

import pytest

from autogfm.realdata.adapters import resolve_column
from autogfm.realdata.common import RealDataError, normalize_token, parse_month, stable_series_key


def test_parse_month_accepts_common_chinese_and_compact_formats() -> None:
    assert parse_month("2024年7月") == date(2024, 7, 1)
    assert parse_month("202407") == date(2024, 7, 1)
    assert parse_month("2024/07/19") == date(2024, 7, 1)


def test_parse_month_rejects_invalid_text() -> None:
    with pytest.raises(RealDataError):
        parse_month("not-a-month")


def test_scored_partial_resolution_separates_sales_from_rank() -> None:
    headers = ["车系名称", "月份", "月销量(辆)", "销量排名"]
    assert resolve_column(headers, "sales").column == "月销量(辆)"
    assert resolve_column(headers, "rank").column == "销量排名"


def test_stable_series_key_normalizes_punctuation() -> None:
    assert stable_series_key("北辰 汽车", "A-01") == "北辰汽车::a01"
    assert normalize_token("  月销量(辆) ") == "月销量辆"
