"""Baseline forecasting strategies for comparative analysis.

Exposes the naive and moving-average baselines as standalone callables so
that experiment scripts can import them directly from this module without
depending on the full forecasting_models pipeline.
"""

from __future__ import annotations

import pandas as pd

from src.forecasting_models import (
    ForecastResult,
    fit_naive_forecast,
    fit_moving_average_forecast,
)


def naive_baseline(train: pd.Series, test: pd.Series) -> ForecastResult:
    """Last-value carry-forward baseline (paper Table IV: Naive)."""
    return fit_naive_forecast(train, test)


def moving_average_baseline(
    train: pd.Series,
    test: pd.Series,
    window: int = 3,
) -> ForecastResult:
    """Recursive moving-average baseline with configurable window (paper Table IV: MA-3)."""
    return fit_moving_average_forecast(train, test, window=window)
