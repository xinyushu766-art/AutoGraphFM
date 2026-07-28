"""Forecasting models."""

from autogfm.models.adaptive_graph_residual_gru import AdaptiveGraphResidualGRU
from autogfm.models.last_value import LastValueForecaster
from autogfm.models.residual_static_graph_gru import ResidualStaticGraphGRU
from autogfm.models.residual_temporal_gru import ResidualTemporalGRU
from autogfm.models.safe_graph_residual_gru import SafeGraphResidualGRU
from autogfm.models.sparse_residual_graph_gru import SparseResidualGraphGRU
from autogfm.models.static_graph_gru import StaticGraphGRU
from autogfm.models.temporal_gru import TemporalOnlyGRU

# Backward-compatible name from Stage 0.1.
SimpleGraphTemporalForecaster = StaticGraphGRU

__all__ = [
    "AdaptiveGraphResidualGRU",
    "LastValueForecaster",
    "ResidualStaticGraphGRU",
    "ResidualTemporalGRU",
    "SafeGraphResidualGRU",
    "SparseResidualGraphGRU",
    "SimpleGraphTemporalForecaster",
    "StaticGraphGRU",
    "TemporalOnlyGRU",
]
