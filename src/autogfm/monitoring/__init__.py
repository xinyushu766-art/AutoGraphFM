"""Production monitoring, drift gates, and realized-outcome evaluation."""

from autogfm.monitoring.drift import MonitoringResult, MonitoringSpec, run_monitoring
from autogfm.monitoring.outcomes import (
    OutcomeResult,
    OutcomeSpec,
    evaluate_forecast_outcomes,
)

__all__ = [
    "MonitoringResult",
    "MonitoringSpec",
    "OutcomeResult",
    "OutcomeSpec",
    "evaluate_forecast_outcomes",
    "run_monitoring",
]
