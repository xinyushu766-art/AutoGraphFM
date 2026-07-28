"""Stage 10B: literature-informed regime models and adaptive-graph scaffold."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autogfm.forecasting.market_benchmark import (
    STAGE10A_MODEL_NAMES,
    STAGE10B_MODEL_NAMES,
    run_multiwindow_market_benchmark,
)
from autogfm.forecasting.regime import regime_snapshot
from autogfm.graph.adaptive import AdaptiveGraphLearner, blend_adjacencies
from autogfm.realdata.common import sha256_file, stable_payload_id, write_json


@dataclass(frozen=True)
class Stage10BResult:
    output_dir: Path
    run_id: str
    manifest: dict[str, Any]


def _read_market(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    months = [str(row["year_month"]) for row in rows]
    ev = np.asarray([float(row["EV_Sales"]) for row in rows], dtype=float)
    total = np.asarray([float(row["Total_Sales"]) for row in rows], dtype=float)
    return months, ev, total


def _metric_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["horizon"]): row
        for row in manifest["aggregate_selected_metrics"]
    }


def _comparison_rows(
    target: str,
    baseline: dict[str, Any],
    expanded: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_map = _metric_map(baseline)
    expanded_map = _metric_map(expanded)
    rows: list[dict[str, Any]] = []
    for horizon in sorted(baseline_map):
        old = float(baseline_map[horizon]["wape_percent"])
        new = float(expanded_map[horizon]["wape_percent"])
        rows.append(
            {
                "target": target,
                "horizon": horizon,
                "stage10a_wape_percent": old,
                "stage10b_wape_percent": new,
                "relative_improvement_percent": 100.0 * (old - new) / max(old, 1e-9),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def _aggregate_test_models(path: Path, target: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["horizon"]), str(row["model_name"]))
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (horizon, model), selected in sorted(grouped.items()):
        actual = np.asarray([float(row["target"]) for row in selected])
        predicted = np.asarray([float(row["prediction"]) for row in selected])
        error = predicted - actual
        absolute = np.abs(error)
        result.append(
            {
                "target": target,
                "horizon": horizon,
                "model_name": model,
                "n": len(selected),
                "wape_percent": (
                    100.0 * absolute.sum() / max(np.abs(actual).sum(), 1e-9)
                ),
                "mae": float(absolute.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(error.mean()),
                "evidence_role": (
                    "retrospective_all_fold_diagnostic_not_selection_policy"
                ),
            }
        )
    return result

def _regime_rows(
    months: list[str],
    values: np.ndarray,
    target: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin in range(12, len(values)):
        snapshot = regime_snapshot(values, origin)
        rows.append(
            {
                "target": target,
                "month": months[origin],
                "regime": snapshot.label,
                "change_score": snapshot.change_score,
                "mean_shift": snapshot.mean_shift,
                "slope_shift": snapshot.slope_shift,
                "volatility_ratio": snapshot.volatility_ratio,
                "yoy_log_change": snapshot.yoy_log_change,
                "recent_log_slope": snapshot.recent_log_slope,
            }
        )
    return rows


def _adaptive_graph_audit() -> dict[str, Any]:
    torch.manual_seed(42)
    agcrn = AdaptiveGraphLearner(12, 6, mode="agcrn")()
    mtgnn = AdaptiveGraphLearner(12, 6, mode="mtgnn", top_k=4)()
    prior = torch.eye(12) * 0.5 + torch.ones(12, 12) / 24.0
    blended = blend_adjacencies(mtgnn, prior, 0.6)
    return {
        "agcrn_shape": list(agcrn.shape),
        "agcrn_row_sum_max_error": float((agcrn.sum(-1) - 1.0).abs().max().detach()),
        "mtgnn_shape": list(mtgnn.shape),
        "mtgnn_row_sum_max_error": float((mtgnn.sum(-1) - 1.0).abs().max().detach()),
        "mtgnn_max_nonzero_per_row": int((mtgnn > 0).sum(-1).max().detach()),
        "blended_row_sum_max_error": float(
            (blended.sum(-1) - 1.0).abs().max().detach()
        ),
        "status": "structural_only_not_evaluated_on_vehicle_panel",
    }


def run_stage10b_literature_market(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Stage10BResult:
    """Compare Stage 10A with literature-informed short-series challengers."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    runs: dict[str, Any] = {}
    for target, column in (("ev", "EV_Sales"), ("total", "Total_Sales")):
        baseline = run_multiwindow_market_benchmark(
            source_csv,
            destination / f"01_{target}_stage10a_models",
            value_column=column,
            models=STAGE10A_MODEL_NAMES,
            overwrite=True,
        )
        expanded = run_multiwindow_market_benchmark(
            source_csv,
            destination / f"02_{target}_literature_models",
            value_column=column,
            models=STAGE10B_MODEL_NAMES,
            overwrite=True,
        )
        runs[target] = {
            "baseline": baseline.manifest,
            "expanded": expanded.manifest,
        }

    comparison = [
        *_comparison_rows("EV_Sales", runs["ev"]["baseline"], runs["ev"]["expanded"]),
        *_comparison_rows(
            "Total_Sales",
            runs["total"]["baseline"],
            runs["total"]["expanded"],
        ),
    ]
    _write_csv(destination / "stage10a_vs_stage10b.csv", comparison)
    model_test_metrics = [
        *_aggregate_test_models(
            destination / "02_ev_literature_models" / "predictions.csv",
            "EV_Sales",
        ),
        *_aggregate_test_models(
            destination / "02_total_literature_models" / "predictions.csv",
            "Total_Sales",
        ),
    ]
    _write_csv(destination / "literature_model_test_metrics.csv", model_test_metrics)

    months, ev_values, total_values = _read_market(source_csv)
    regime_rows = [
        *_regime_rows(months, ev_values, "EV_Sales"),
        *_regime_rows(months, total_values, "Total_Sales"),
    ]
    _write_csv(destination / "regime_diagnostics.csv", regime_rows)
    top_changes = sorted(
        regime_rows,
        key=lambda row: float(row["change_score"]),
        reverse=True,
    )[:12]

    improvements = [float(row["relative_improvement_percent"]) for row in comparison]
    improved_count = sum(value > 0.0 for value in improvements)
    severe_regressions = sum(value < -5.0 for value in improvements)
    selected_new_models = sorted(
        {
            str(row["selected_model"])
            for target in runs.values()
            for row in target["expanded"]["selected_fold_metrics"]
            if str(row["selected_model"]) not in STAGE10A_MODEL_NAMES
        }
    )
    promotion = (
        "promote_to_shadow_champion"
        if improved_count >= 4 and severe_regressions == 0
        else "retain_stage10a_champion_keep_literature_models_as_challengers"
    )
    graph_audit = _adaptive_graph_audit()
    identity = {
        "schema_version": "autogfm-stage10b-v1",
        "source_sha256": sha256_file(source_csv),
        "stage10b_models": STAGE10B_MODEL_NAMES,
        "baseline_ids": {
            target: values["baseline"]["benchmark_id"]
            for target, values in runs.items()
        },
        "expanded_ids": {
            target: values["expanded"]["benchmark_id"]
            for target, values in runs.items()
        },
    }
    run_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "run_id": run_id,
        "engineering_status": "pass",
        "evidence_scope": "public_real_aggregate_literature_informed",
        "adopted_ideas": {
            "nlinear": "last-value normalization before a linear readout",
            "dlinear": "moving-average trend/remainder decomposition",
            "change_point": "two-window robust level/slope/volatility discrepancy",
            "tft_tide": "known-calendar covariates and a lightweight expert gate",
            "agcrn_mtgnn": "adaptive soft and directed Top-K graph scaffolds",
        },
        "comparison": comparison,
        "literature_model_test_metrics": model_test_metrics,
        "improved_cells": improved_count,
        "severe_regressions": severe_regressions,
        "selected_new_models": selected_new_models,
        "promotion_decision": promotion,
        "top_regime_changes": top_changes,
        "adaptive_graph_audit": graph_audit,
        "vehicle_panel_status": "adapter_ready_large_raw_panel_not_local",
        "graph_research_claim": "no-claim",
        "timesfm_status": "external_challenger_deferred",
        "claim_boundary": (
            "Literature mechanisms are implemented and aggregate-tested. "
            "Adaptive graph code is structural only until the vehicle panel runs."
        ),
    }
    write_json(destination / "stage10b_manifest.json", manifest)
    _write_report(destination, manifest)
    return Stage10BResult(destination, run_id, manifest)


def _write_report(destination: Path, manifest: dict[str, Any]) -> None:
    rows = [
        "| Target | H | Stage 10A WAPE | Stage 10B WAPE | Relative change |",
        "|---|---:|---:|---:|---:|",
    ]
    rows.extend(
        "| {target} | {horizon} | {stage10a_wape_percent:.2f}% | "
        "{stage10b_wape_percent:.2f}% | {relative_improvement_percent:+.2f}% |".format(
            **row
        )
        for row in manifest["comparison"]
    )
    lines = [
        "# AutoGraphFM v0.15.0 Stage 10B Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        "- Engineering: **PASS**",
        f"- Promotion: **{manifest['promotion_decision']}**",
        f"- New models selected in at least one fold: "
        f"`{manifest['selected_new_models']}`",
        "- Vehicle-level graph claim: **NO-CLAIM**",
        "",
        "## Real aggregate comparison",
        "",
        *rows,
        "",
        "## Adopted literature mechanisms",
        "",
        "- NLinear-inspired last-value normalization.",
        "- DLinear-inspired moving-average trend/remainder decomposition.",
        "- Window change-point features for level, slope, and volatility shifts.",
        "- TFT/TiDE-inspired known-calendar features and lightweight expert gate.",
        "- AGCRN-style soft adjacency and MTGNN-style directed Top-K adjacency.",
        "",
        "## Boundary",
        "",
        manifest["claim_boundary"],
        "",
    ]
    (destination / "STAGE10B_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    payload = json.dumps(manifest, ensure_ascii=False)
    (destination / "STAGE10B_REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>AutoGraphFM Stage 10B</title>"
        "<style>body{font-family:Arial;max-width:1120px;margin:40px auto;"
        "line-height:1.55}.card{border:1px solid #bbb;padding:16px;margin:14px 0}"
        ".warn{background:#fff3cd}</style><h1>AutoGraphFM Stage 10B</h1>"
        f"<p>Run ID: <code>{manifest['run_id']}</code></p>"
        f"<div class='card warn'><b>Decision:</b> "
        f"{manifest['promotion_decision']}. Graph claim remains blocked.</div>"
        "<p>See <code>stage10a_vs_stage10b.csv</code>, "
        "<code>literature_model_test_metrics.csv</code>, and "
        "<code>regime_diagnostics.csv</code> for auditable evidence.</p>"
        f"<script type='application/json' id='manifest'>{payload}</script>",
        encoding="utf-8",
    )
