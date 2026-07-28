from autogfm.reporting import Stage4ReportSpec


def test_stage4_report_spec_validation() -> None:
    Stage4ReportSpec().validate()
    try:
        Stage4ReportSpec(forecast_rows=0).validate()
    except ValueError as exc:
        assert "forecast_rows" in str(exc)
    else:
        raise AssertionError("Zero report rows should fail validation.")
