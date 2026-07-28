"""High-fidelity three-file demo data for end-to-end Stage 3 validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from autogfm.realdata.common import write_csv_rows


@dataclass(frozen=True)
class DemoDataResult:
    sales_csv: Path
    mapping_csv: Path
    vehicles_csv: Path


def _month_sequence(start: date, count: int) -> list[date]:
    values = []
    ordinal = start.year * 12 + start.month - 1
    for offset in range(count):
        value = ordinal + offset
        values.append(date(value // 12, value % 12 + 1, 1))
    return values


def generate_stage3_demo_data(
    output_dir: str | Path,
    *,
    num_series: int = 24,
    num_months: int = 72,
    seed: int = 42,
) -> DemoDataResult:
    """Generate realistic Chinese-header sales, mapping, and vehicle CSVs."""
    if num_series < 8:
        raise ValueError("num_series must be at least 8.")
    if num_months < 48:
        raise ValueError("num_months must be at least 48.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    brands = ["北辰", "远航", "凌风", "星越", "澜图", "启明"]
    segments = ["紧凑型SUV", "中型SUV", "紧凑型轿车", "中型轿车"]
    bodies = ["SUV", "三厢车"]
    energies = ["汽油", "纯电", "插电混动"]
    months = _month_sequence(date(2020, 1, 1), num_months)

    catalog_rows = []
    mapping_rows = []
    brand_ids = []
    segment_ids = []
    energy_ids = []
    prices = []
    aliases = []
    for index in range(num_series):
        brand = brands[index % len(brands)]
        segment = segments[index % len(segments)]
        body = bodies[0] if "SUV" in segment else bodies[1]
        energy = energies[(index // len(segments)) % len(energies)]
        canonical = f"{brand}{chr(65 + index % 26)}{index + 1:02d}"
        alias = f"{canonical}-2024款" if index % 3 == 0 else f"{canonical} 系列"
        low = 9.0 + 1.7 * (index % 8) + 3.0 * energy_ids.count(1)
        high = low + 3.0 + 0.5 * (index % 4)
        catalog_rows.append(
            {
                "品牌名称": brand,
                "车系名称": canonical,
                "级别": segment,
                "车身类型": body,
                "能源类型": energy,
                "最低指导价": f"{low:.2f}",
                "最高指导价": f"{high:.2f}",
                "在售状态": "在售",
                "上市时间": "2020-01",
            }
        )
        mapping_rows.append(
            {
                "原始车系": alias,
                "标准车系": canonical,
                "品牌名称": brand,
            }
        )
        aliases.append(alias)
        brand_ids.append(index % len(brands))
        segment_ids.append(index % len(segments))
        energy_ids.append((index // len(segments)) % len(energies))
        prices.append((low + high) / 2.0)

    brand_ids_array = np.array(brand_ids)
    segment_ids_array = np.array(segment_ids)
    energy_ids_array = np.array(energy_ids)
    prices_array = np.array(prices)
    graph = np.zeros((num_series, num_series), dtype=np.float64)
    price_scale = max(float(prices_array.std()), 1.0)
    for i in range(num_series):
        for j in range(num_series):
            if i == j:
                continue
            score = 0.15 * np.exp(-abs(prices_array[i] - prices_array[j]) / price_scale)
            score += 0.40 * (segment_ids_array[i] == segment_ids_array[j])
            score += 0.25 * (energy_ids_array[i] == energy_ids_array[j])
            score += 0.20 * (brand_ids_array[i] == brand_ids_array[j])
            graph[i, j] = score
    for i in range(num_series):
        keep = np.argsort(graph[i])[-5:]
        mask = np.ones(num_series, dtype=bool)
        mask[keep] = False
        graph[i, mask] = 0.0
    graph = np.maximum(graph, graph.T)
    graph /= np.maximum(graph.sum(axis=1, keepdims=True), 1e-8)

    base = rng.uniform(350.0, 2200.0, size=num_series)
    trend = rng.uniform(-1.0, 7.0, size=num_series)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=num_series)
    sales = np.zeros((num_months, num_series), dtype=np.float64)
    previous = base.copy()
    for month_index in range(num_months):
        seasonal = 0.12 * np.sin(2.0 * np.pi * month_index / 12.0 + phase)
        market_cycle = 1.0 + 0.06 * np.sin(2.0 * np.pi * month_index / 36.0)
        neighbor = graph @ previous
        energy_growth = 1.0 + (energy_ids_array > 0) * (0.006 * month_index)
        innovation = rng.normal(0.0, 45.0, size=num_series)
        current = (
            0.50 * previous
            + 0.24 * neighbor
            + 0.26 * base * (1.0 + seasonal) * market_cycle * energy_growth
            + trend
            + innovation
        )
        # A common shock and gradual recovery, visible to all methods.
        if month_index == 38:
            current *= 0.78
        if 39 <= month_index <= 42:
            current *= 0.90 + 0.03 * (month_index - 39)
        current = np.clip(current, 20.0, None)
        sales[month_index] = current
        previous = current

    sales_rows = []
    for month_index, month in enumerate(months):
        month_sales = sales[month_index]
        ranks = np.argsort(np.argsort(-month_sales)) + 1
        for series_index in range(num_series):
            # Introduce a small number of missing source observations.
            if (month_index + 5 * series_index) % 97 == 0 and month_index > 6:
                continue
            sales_rows.append(
                {
                    "车系名称": aliases[series_index],
                    "品牌": brands[brand_ids[series_index]],
                    "月份": month.strftime("%Y-%m"),
                    "月销量(辆)": int(round(month_sales[series_index])),
                    "销量排名": int(ranks[series_index]),
                }
            )
    # Exact business-key duplicate: canonicalizer should aggregate it and report it.
    sales_rows.append(
        {
            "车系名称": aliases[0],
            "品牌": brands[brand_ids[0]],
            "月份": months[-1].strftime("%Y-%m"),
            "月销量(辆)": 3,
            "销量排名": 1,
        }
    )
    # Deliberately malformed row: must be quarantined, never silently coerced.
    sales_rows.append(
        {
            "车系名称": aliases[1],
            "品牌": brands[brand_ids[1]],
            "月份": "not-a-month",
            "月销量(辆)": "bad",
            "销量排名": "",
        }
    )

    sales_path = destination / "sales_aligned_completed.csv"
    mapping_path = destination / "series_mapping_completed.csv"
    vehicles_path = destination / "vehicles_completed.csv"
    write_csv_rows(
        sales_path,
        ["车系名称", "品牌", "月份", "月销量(辆)", "销量排名"],
        sales_rows,
    )
    write_csv_rows(
        mapping_path,
        ["原始车系", "标准车系", "品牌名称"],
        mapping_rows,
    )
    write_csv_rows(
        vehicles_path,
        [
            "品牌名称", "车系名称", "级别", "车身类型", "能源类型",
            "最低指导价", "最高指导价", "在售状态", "上市时间",
        ],
        catalog_rows,
    )
    return DemoDataResult(sales_path, mapping_path, vehicles_path)
