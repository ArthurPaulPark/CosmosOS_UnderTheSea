"""Ocean Significant Wave Height Forecasting Module (Problem 3)."""

from __future__ import annotations

from .adapter import OceanForecastAdapter
from .cases import (
    ForecastCase,
    HistoricalForecastCaseBuilder,
    VALID_LEAD_HOURS,
    align_historical_grid,
)
from .features import WaveForecastFeaturePlugin
from .schema import validate_forecasting_schema
from .submission import generate_forecast_submission
from .validation import (
    build_primary_forecast_validation,
    build_rolling_forecast_backtest,
    calculate_forecast_metrics,
)

__all__ = [
    "OceanForecastAdapter",
    "HistoricalForecastCaseBuilder",
    "ForecastCase",
    "WaveForecastFeaturePlugin",
    "validate_forecasting_schema",
    "generate_forecast_submission",
    "build_primary_forecast_validation",
    "build_rolling_forecast_backtest",
    "calculate_forecast_metrics",
    "align_historical_grid",
    "VALID_LEAD_HOURS",
]
