"""Standalone Stage 4 audit report generation."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from autogfm.realdata.common import (
    read_csv_rows,
    sha256_file,
    stable_payload_id,
    write_json,
)


@dataclass(frozen=True)
class Stage4ReportSpec:
    """Presentation limits for the self-contained audit report."""

    leaderboard_rows: int = 18
    monitored_series_rows: int = 12
    forecast_rows: int = 12
    evidence_scope: str = "auto"

    def validate(self) -> None:
        for name in ("leaderboard_rows", "monitored_series_rows", "forecast_rows"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        allowed = {
            "auto",
            "demo_only",
            "synthetic_only",
            "real_unverified",
            "user_data",
            "official_real_data",
        }
        if self.evidence_scope not in allowed:
            raise ValueError(f"evidence_scope must be one of {sorted(allowed)}.")


@dataclass(frozen=True)
class Stage4ReportResult:
    output_dir: Path
    report_id: str
    decision: dict[str, Any]
    manifest: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return "NaN"
        return html.escape(f"{value:.4f}")
    return html.escape(str(value))


def _status_badge(status: str) -> str:
    normalized = status.lower()
    css = {
        "pass": "pass",
        "go": "pass",
        "warn": "warn",
        "block": "block",
        "no-claim": "block",
    }.get(normalized, "neutral")
    return f'<span class="badge {css}">{_escape(status.upper())}</span>'


def _table(rows: Iterable[Mapping[str, Any]], columns: list[tuple[str, str]]) -> str:
    materialized = list(rows)
    head = "".join(f"<th>{_escape(label)}</th>" for _, label in columns)
    if not materialized:
        body = f'<tr><td colspan="{len(columns)}">No rows</td></tr>'
    else:
        body = "".join(
            "<tr>"
            + "".join(f"<td>{_escape(row.get(key, ''))}</td>" for key, _ in columns)
            + "</tr>"
            for row in materialized
        )
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _all_true(values: Iterable[bool]) -> bool:
    materialized = list(values)
    return bool(materialized) and all(materialized)


def _graph_decision(
    backtest_manifest: Mapping[str, Any],
    neural_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rolling_checks = backtest_manifest.get("graph_claim_checks", {})
    rolling_pass = _all_true(
        check
        for horizon in rolling_checks.values()
        for check in horizon.values()
    )
    neural_gate = neural_manifest.get("claim_gate", {}) if neural_manifest else {}
    neural_pass = _all_true(neural_gate.values()) if neural_gate else False
    return {
        "status": "go" if rolling_pass and neural_pass else "no-claim",
        "rolling_pass": rolling_pass,
        "neural_pass": neural_pass,
        "rolling_checks": rolling_checks,
        "neural_gate": neural_gate or None,
    }


def _resolve_evidence_scope(source_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    normalized = source_name.lower()
    if "demo" in normalized:
        return "demo_only"
    if "synthetic" in normalized or "fixture" in normalized:
        return "synthetic_only"
    if "srni" in normalized:
        return "real_unverified"
    return "user_data"


def build_stage4_audit_report(
    run_dir: str | Path,
    output_dir: str | Path,
    spec: Stage4ReportSpec | None = None,
    *,
    overwrite: bool = False,
) -> Stage4ReportResult:
    """Build a self-contained HTML and Markdown audit report from a completed run."""
    spec = spec or Stage4ReportSpec()
    spec.validate()
    root = Path(run_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    stage3 = _load_json(root / "manifest.json")
    canonical = _load_json(root / "01_canonical" / "manifest.json")
    panel = _load_json(root / "02_panel" / "manifest.json")
    graph = _load_json(root / "03_graph" / "manifest.json")
    backtest = _load_json(root / "04_backtest" / "manifest.json")
    production = _load_json(root / "06_forecast" / "manifest.json")
    monitoring = _load_json(root / "07_monitoring" / "manifest.json")
    neural_path = root / "05_neural" / "manifest.json"
    neural = _load_json(neural_path) if neural_path.is_file() else None

    _, leaderboard, _ = read_csv_rows(root / "04_backtest" / "leaderboard.csv")
    _, selected_metrics, _ = read_csv_rows(
        root / "04_backtest" / "selected_model_test_metrics.csv"
    )
    _, series_monitoring, _ = read_csv_rows(
        root / "07_monitoring" / "series_monitoring.csv"
    )
    _, forecasts, _ = read_csv_rows(root / "06_forecast" / "production_forecast.csv")

    graph_decision = _graph_decision(backtest, neural)
    source_name = canonical.get("config", {}).get("source_name", "unknown")
    evidence_scope = _resolve_evidence_scope(source_name, spec.evidence_scope)
    deployable_scopes = {"user_data", "official_real_data"}
    decision = {
        "engineering": "pass",
        "deployment": monitoring["status"],
        "graph_research_claim": graph_decision["status"],
        "evidence_scope": evidence_scope,
        "release_ready": (
            monitoring["status"] != "block"
            and evidence_scope in deployable_scopes
        ),
    }

    selected_by_horizon = {
        str(row["horizon"]): row for row in selected_metrics
    }
    decision_cards = [
        ("Engineering pipeline", decision["engineering"]),
        ("Deployment gate", decision["deployment"]),
        ("Graph research claim", decision["graph_research_claim"]),
        ("Evidence scope", decision["evidence_scope"]),
    ]
    cards_html = "".join(
        "<article class=\"card\"><h3>"
        + _escape(label)
        + "</h3><div>"
        + _status_badge(status)
        + "</div></article>"
        for label, status in decision_cards
    )

    selected_rows = []
    for horizon, model in backtest["selected_models"].items():
        row = selected_by_horizon.get(str(horizon), {})
        selected_rows.append(
            {
                "horizon": horizon,
                "model": model,
                "validation_wape": row.get("validation_wape_percent", ""),
                "test_wape": row.get("test_wape_percent", ""),
                "test_mae": row.get("test_mae", ""),
                "delta": row.get("test_delta_wape_vs_last_value_pp", ""),
            }
        )

    graph_rows = []
    for horizon, checks in backtest["graph_claim_checks"].items():
        graph_rows.append(
            {
                "horizon": horizon,
                "pooled": checks["beats_pooled"],
                "identity": checks["beats_identity"],
                "permuted": checks["beats_permuted"],
                "decision": "PASS" if all(checks.values()) else "NO-CLAIM",
            }
        )

    top_series = series_monitoring[: spec.monitored_series_rows]
    forecast_sample = forecasts[: spec.forecast_rows]
    leaderboard_sample = leaderboard[: spec.leaderboard_rows]
    psi = monitoring["global_checks"]["global_psi"]
    missing = monitoring["global_checks"]["recent_missing_ratio"]

    style = """
:root{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:#18212b;background:#f3f5f7}
*{box-sizing:border-box}body{margin:0}.shell{max-width:1260px;margin:auto;padding:32px 28px 72px}
h1{font-size:34px;margin:0 0 6px}.subtitle{color:#5c6875;margin-bottom:28px}.grid{display:grid;gap:14px}
.cards{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}.card,.section{background:white;border:1px solid #dfe4e8;border-radius:12px;box-shadow:0 2px 8px rgba(20,30,40,.04)}
.card{padding:18px}.card h3{font-size:14px;color:#52606d;margin:0 0 12px}.section{padding:22px;margin-top:18px}.section h2{margin:0 0 14px;font-size:21px}.metrics{grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin:12px 0}.metric{padding:14px;background:#f7f9fa;border-radius:8px}.metric strong{display:block;font-size:21px;margin-top:5px}.metric span{font-size:12px;color:#65727e}
.badge{display:inline-block;padding:5px 9px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:.04em}.pass{background:#dff5e8;color:#17653a}.warn{background:#fff1c7;color:#7b5700}.block{background:#fde2e2;color:#8e2424}.neutral{background:#e7ebef;color:#48535f}
.table-wrap{overflow:auto;border:1px solid #e3e7ea;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:13px}th{background:#f4f6f8;text-align:left;color:#4a5662}th,td{padding:9px 10px;border-bottom:1px solid #e8ecef;white-space:nowrap}tr:last-child td{border-bottom:0}.note{padding:12px 14px;border-left:4px solid #788694;background:#f5f7f8;color:#46515c}.block-note{border-left-color:#b43b3b;background:#fff2f2}.footer{margin-top:24px;color:#6c7781;font-size:12px}
code{background:#eef1f3;padding:2px 5px;border-radius:4px}@media(max-width:600px){.shell{padding:20px 14px}h1{font-size:27px}}
"""
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoGraphFM Stage 4 Audit</title><style>{style}</style></head><body><main class="shell">
<h1>AutoGraphFM Stage 4 Audit</h1><div class="subtitle">Run <code>{_escape(stage3['run_id'])}</code> · source <code>{_escape(source_name)}</code></div>
<div class="grid cards">{cards_html}</div>
<section class="section"><h2>Executive decision</h2>
<div class="note {'block-note' if monitoring['status']=='block' else ''}">Engineering completion does not imply deployment approval or a graph-effect claim. This run is <strong>{_escape(decision['evidence_scope'])}</strong>; deployment is <strong>{_escape(decision['deployment'])}</strong>; graph evidence is <strong>{_escape(decision['graph_research_claim'])}</strong>.</div>
<div class="grid metrics">
<div class="metric"><span>Canonical series</span><strong>{canonical['quality']['canonical_series']}</strong></div>
<div class="metric"><span>Observed panel ratio</span><strong>{panel['quality']['observed_ratio']:.2%}</strong></div>
<div class="metric"><span>Graph largest component</span><strong>{graph['diagnostics']['largest_component_ratio']:.2%}</strong></div>
<div class="metric"><span>Global PSI</span><strong>{psi['value']:.3f}</strong><span>{_status_badge(psi['severity'])}</span></div>
<div class="metric"><span>Recent missingness</span><strong>{missing['value']:.2%}</strong><span>{_status_badge(missing['severity'])}</span></div>
<div class="metric"><span>Forecast alerts</span><strong>{monitoring['global_checks']['forecast_alerts']}</strong></div>
</div></section>
<section class="section"><h2>Data lineage and quality</h2>{_table([
{'metric':'Month range','value':f"{canonical['quality']['month_min']} → {canonical['quality']['month_max']}"},
{'metric':'Monthly rows','value':canonical['quality']['canonical_monthly_rows']},
{'metric':'Rejected sales rows','value':canonical['quality']['rejected_sales_rows']},
{'metric':'Aggregated duplicate keys','value':canonical['quality']['duplicate_monthly_keys_aggregated']},
{'metric':'Unmatched rows','value':canonical['quality']['unmatched_rows']},
{'metric':'Imputed panel cells','value':panel['quality']['imputed_cells']},
], [('metric','Metric'),('value','Value')])}</section>
<section class="section"><h2>Validation-selected models and held-out measurement</h2>{_table(selected_rows,[('horizon','Horizon'),('model','Selected model'),('validation_wape','Validation WAPE %'),('test_wape','Test WAPE %'),('test_mae','Test MAE'),('delta','Δ WAPE vs Last Value pp')])}</section>
<section class="section"><h2>Graph falsification controls</h2>{_table(graph_rows,[('horizon','Horizon'),('pooled','Beats pooled'),('identity','Beats identity'),('permuted','Beats permuted'),('decision','Decision')])}<p class="note">A graph claim requires the real graph to beat pooled temporal, identity-graph, and permuted-graph controls on held-out observations; the neural control gate is also required when neural experiments are enabled.</p></section>
<section class="section"><h2>Backtest leaderboard sample</h2>{_table(leaderboard_sample,[(key,key) for key in (leaderboard_sample[0].keys() if leaderboard_sample else [])])}</section>
<section class="section"><h2>Monitoring: highest-priority series</h2>{_table(top_series,[('series_key','Series'),('brand_name','Brand'),('series_name','Name'),('level_shift_z','Shift z'),('recent_missing_ratio','Missing ratio'),('severity','Severity')])}</section>
<section class="section"><h2>Production forecast sample</h2>{_table(forecast_sample,[('series_key','Series'),('forecast_month','Forecast month'),('horizon','Horizon'),('selected_model','Model'),('prediction','Prediction'),('lower_80','80% low'),('upper_80','80% high'),('lower_95','95% low'),('upper_95','95% high')])}</section>
<section class="section"><h2>Blocking and warning reasons</h2>{_table([{'level':'BLOCK','reason':r} for r in monitoring['block_reasons']]+[{'level':'WARN','reason':r} for r in monitoring['warn_reasons']],[('level','Level'),('reason','Reason')])}</section>
<div class="footer">Generated from content-addressed manifests. Report ID is written to the adjacent manifest.</div>
</main></body></html>"""
    html_path = destination / "AUDIT_REPORT.html"
    html_path.write_text(html_doc, encoding="utf-8")

    summary_lines = [
        "# AutoGraphFM Stage 4 Audit Summary",
        "",
        f"- Engineering: **{decision['engineering'].upper()}**",
        f"- Deployment gate: **{decision['deployment'].upper()}**",
        f"- Graph research claim: **{decision['graph_research_claim'].upper()}**",
        f"- Evidence scope: **{decision['evidence_scope'].upper()}**",
        f"- Release ready: **{decision['release_ready']}**",
        "",
        "## Core IDs",
        "",
        f"- Run: `{stage3['run_id']}`",
        f"- Dataset: `{canonical['dataset_id']}`",
        f"- Panel: `{panel['panel_id']}`",
        f"- Graph: `{graph['graph_id']}`",
        f"- Backtest: `{backtest['backtest_id']}`",
        f"- Forecast: `{production['forecast_id']}`",
        f"- Monitoring: `{monitoring['monitoring_id']}`",
        "",
        "## Interpretation",
        "",
        "The HTML report is a consolidated engineering and evidence audit. "
        "Demo, synthetic, or unverified rankings must not be cited as official "
        "real-world performance.",
        "",
    ]
    summary_path = destination / "AUDIT_SUMMARY.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    identity = {
        "schema_version": "autogfm-stage4-audit-report-v1",
        "run_id": stage3["run_id"],
        "monitoring_id": monitoring["monitoring_id"],
        "spec": asdict(spec),
        "decision": decision,
        "html_sha256": sha256_file(html_path),
        "summary_sha256": sha256_file(summary_path),
    }
    report_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "report_id": report_id,
        "graph_decision": graph_decision,
        "files": {
            "AUDIT_REPORT.html": {
                "path": "AUDIT_REPORT.html",
                "sha256": identity["html_sha256"],
            },
            "AUDIT_SUMMARY.md": {
                "path": "AUDIT_SUMMARY.md",
                "sha256": identity["summary_sha256"],
            },
        },
    }
    write_json(destination / "manifest.json", manifest)
    return Stage4ReportResult(destination, report_id, decision, manifest)
