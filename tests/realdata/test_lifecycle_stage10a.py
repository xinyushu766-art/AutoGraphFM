from __future__ import annotations

import numpy as np

from autogfm.realdata.lifecycle import LifecycleSpec, profile_series_lifecycle


def test_lifecycle_labels_use_train_cutoff_only() -> None:
    months = tuple(f"2020-{month:02d}-01" for month in range(1, 13)) + tuple(
        f"2021-{month:02d}-01" for month in range(1, 13)
    )
    values = np.full(24, np.nan)
    values[10:] = np.arange(14) + 100
    result = profile_series_lifecycle(
        values,
        months,
        train_end_index=11,
        test_end_index=23,
        spec=LifecycleSpec(recent_launch_months=6),
    )
    assert result["lifecycle_at_train_end"] == "new_launch_at_train_end"
    assert result["observed_train_months"] == 2
    assert result["observed_total_months"] == 14
    assert result["outcome_status_at_test_end"] == "active_at_test_end"


def test_future_launch_is_audit_only_not_train_feature() -> None:
    months = tuple(f"2020-{month:02d}-01" for month in range(1, 13)) + tuple(
        f"2021-{month:02d}-01" for month in range(1, 13)
    )
    values = np.full(24, np.nan)
    values[15:] = 50
    result = profile_series_lifecycle(
        values,
        months,
        train_end_index=11,
        test_end_index=23,
    )
    assert result["lifecycle_at_train_end"] == "not_observed_by_train"
    assert result["outcome_status_at_test_end"] == "launched_after_train"
    assert result["cold_start_evaluable"] is True
