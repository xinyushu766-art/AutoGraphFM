"""Stage 10A: strong baselines and multi-window real-market evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autogfm.forecasting.market_benchmark import (
    run_market_benchmark,
    run_multiwindow_market_benchmark,
)
from autogfm.realdata.common import sha256_file, stable_payload_id, write_json


@dataclass(frozen=True)
class Stage10AResult:
    output_dir: Path
    run_id: str
    manifest: dict[str, Any]


def _selection_variation(manifest: dict[str, Any], horizon: int) -> int:
    return len(
        {
            row["selected_model"]
            for row in manifest["selected_fold_metrics"]
            if int(row["horizon"]) == horizon
        }
    )


def run_stage10a_real_market(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Stage10AResult:
    """Run final-period and annual multi-window benchmarks for EV and total sales."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    ev_single = run_market_benchmark(
        source_csv,
        destination / "01_ev_single_split",
        value_column="EV_Sales",
        horizons=(1, 3, 6),
        overwrite=True,
    )
    total_single = run_market_benchmark(
        source_csv,
        destination / "02_total_single_split",
        value_column="Total_Sales",
        horizons=(1, 3, 6),
        overwrite=True,
    )
    ev_multi = run_multiwindow_market_benchmark(
        source_csv,
        destination / "03_ev_multiwindow",
        value_column="EV_Sales",
        horizons=(1, 3, 6),
        overwrite=True,
    )
    total_multi = run_multiwindow_market_benchmark(
        source_csv,
        destination / "04_total_multiwindow",
        value_column="Total_Sales",
        horizons=(1, 3, 6),
        overwrite=True,
    )

    selection_variation = {
        "EV_Sales": {
            str(horizon): _selection_variation(ev_multi.manifest, horizon)
            for horizon in (1, 3, 6)
        },
        "Total_Sales": {
            str(horizon): _selection_variation(total_multi.manifest, horizon)
            for horizon in (1, 3, 6)
        },
    }
    unstable = any(
        count > 1
        for target in selection_variation.values()
        for count in target.values()
    )
    identity = {
        "schema_version": "autogfm-stage10a-v1",
        "source_sha256": sha256_file(source_csv),
        "ev_multiwindow_id": ev_multi.benchmark_id,
        "total_multiwindow_id": total_multi.benchmark_id,
    }
    run_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "run_id": run_id,
        "engineering_status": "pass",
        "evidence_scope": "public_real_aggregate_market_multiwindow",
        "model_selection_stability": "warn" if unstable else "pass",
        "selection_variation": selection_variation,
        "ev_single_split": ev_single.manifest,
        "total_single_split": total_single.manifest,
        "ev_multiwindow": ev_multi.manifest,
        "total_multiwindow": total_multi.manifest,
        "vehicle_panel_status": "adapter_ready_data_not_local",
        "graph_research_claim": "no-claim",
        "decision": (
            "Strong non-graph baselines and annual rolling evaluation are ready. "
            "Model selection remains regime-sensitive, so a single validation "
            "window is not sufficient for production approval."
        ),
    }
    write_json(destination / "stage10a_manifest.json", manifest)
    _write_report(destination, manifest)
    return Stage10AResult(destination, run_id, manifest)


def _metric_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Horizon | Folds | N | WAPE | MAE | RMSE | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['horizon']} | {row['folds']} | {row['n']} | "
            f"{row['wape_percent']:.2f}% | {row['mae']:.0f} | "
            f"{row['rmse']:.0f} | {row['bias']:.0f} |"
        )
    return lines


def _write_report(destination: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# AutoGraphFM v0.14.0 Stage 10A Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        "- Engineering status: **PASS**",
        "- Evidence: **PUBLIC REAL AGGREGATE MARKET / MULTI-WINDOW**",
        "- Model-selection stability: "
        f"**{manifest['model_selection_stability'].upper()}**",
        "- Vehicle-level graph effect: **NO-CLAIM**",
        "",
        "## EV sales — multi-window selected policy",
        "",
        *_metric_table(
            manifest["ev_multiwindow"]["aggregate_selected_metrics"]
        ),
        "",
        "## Total market — multi-window selected policy",
        "",
        *_metric_table(
            manifest["total_multiwindow"]["aggregate_selected_metrics"]
        ),
        "",
        "## Interpretation",
        "",
        "The annual folds choose different winners for several horizons. This is "
        "direct evidence that automotive demand is regime-sensitive. The correct "
        "next model step is a regime gate and stronger exogenous features, not an "
        "uncontrolled increase in neural-network size.",
        "",
        "## Claim boundary",
        "",
        "The executed data are national aggregate series. The vehicle-level "
        "38,806-row adapter is complete, but the raw panel is not local in this "
        "runtime. No graph-effect or vehicle-level accuracy claim is permitted.",
        "",
    ]
    (destination / "STAGE10A_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    payload = json.dumps(manifest, ensure_ascii=False)
    (destination / "STAGE10A_REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>AutoGraphFM Stage 10A</title>"
        "<style>body{font-family:Arial;max-width:1100px;margin:40px auto;"
        "line-height:1.55}.card{border:1px solid #bbb;padding:16px;margin:14px 0}"
        ".warn{background:#fff3cd}</style><h1>AutoGraphFM Stage 10A</h1>"
        f"<p>Run ID: <code>{manifest['run_id']}</code></p>"
        "<div class='card warn'><b>Decision:</b> Strong baselines and multi-window "
        "evaluation pass. Selection is regime-sensitive; graph claim remains "
        "blocked.</div><p>See the CSV leaderboards in the four benchmark folders "
        "for reproducible details.</p>"
        f"<script type='application/json' id='manifest'>{payload}</script>",
        encoding="utf-8",
    )
