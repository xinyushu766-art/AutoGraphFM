"""Backward-compatible import for the Stage 0.1 model name."""

from autogfm.models.static_graph_gru import StaticGraphGRU

SimpleGraphTemporalForecaster = StaticGraphGRU

__all__ = ["SimpleGraphTemporalForecaster"]
