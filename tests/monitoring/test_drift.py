from autogfm.monitoring import MonitoringSpec, run_monitoring
from autogfm.realdata.common import read_csv_rows


def test_monitoring_writes_explicit_operational_gate(
    stage3_artifacts: dict[str, object], tmp_path
) -> None:
    panel = stage3_artifacts["panel"]
    backtest = stage3_artifacts["backtest"]
    production = stage3_artifacts["production"]
    result = run_monitoring(
        panel.output_dir,
        backtest.output_dir,
        production.output_dir,
        tmp_path / "monitoring",
        MonitoringSpec(recent_months=4),
        overwrite=True,
    )
    assert result.status in {"pass", "warn", "block"}
    assert (result.output_dir / "MONITORING_REPORT.md").is_file()
    assert "MONITORING_REPORT.md" in result.manifest["files"]
    _, alerts, _ = read_csv_rows(result.output_dir / "forecast_alerts.csv")
    assert alerts == []


def test_monitoring_rejects_invalid_thresholds() -> None:
    spec = MonitoringSpec(psi_warn=0.5, psi_block=0.2)
    try:
        spec.validate()
    except Exception as exc:
        assert "psi" in str(exc)
    else:
        raise AssertionError("Invalid thresholds should fail validation.")
