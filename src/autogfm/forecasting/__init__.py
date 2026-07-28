"""Real-panel baselines, evaluation, and production forecasting."""

from autogfm.forecasting.backtest import BacktestResult, BacktestSpec, run_rolling_backtest
from autogfm.forecasting.baselines import (
    PooledResidualRidge,
    damped_trend_forecast,
    regime_guarded_forecast,
    seasonal_growth_forecast,
)
from autogfm.forecasting.market_benchmark import (
    MarketBenchmarkResult,
    MarketFoldSpec,
    MultiWindowMarketBenchmarkResult,
    run_market_benchmark,
    run_multiwindow_market_benchmark,
)
from autogfm.forecasting.production import (
    ProductionForecastResult,
    ProductionForecastSpec,
    build_production_forecast,
)

__all__ = [
    "BacktestResult",
    "BacktestSpec",
    "MarketBenchmarkResult",
    "MarketFoldSpec",
    "MultiWindowMarketBenchmarkResult",
    "PooledResidualRidge",
    "ProductionForecastResult",
    "ProductionForecastSpec",
    "build_production_forecast",
    "damped_trend_forecast",
    "regime_guarded_forecast",
    "run_market_benchmark",
    "run_multiwindow_market_benchmark",
    "seasonal_growth_forecast",
    "run_rolling_backtest",
]
