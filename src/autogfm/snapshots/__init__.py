"""Versioned model-input snapshot builders."""

from autogfm.snapshots.monthly_sales import (
    SnapshotError,
    SnapshotResult,
    SnapshotSpec,
    build_monthly_sales_snapshot,
)

__all__ = [
    "SnapshotError",
    "SnapshotResult",
    "SnapshotSpec",
    "build_monthly_sales_snapshot",
]
