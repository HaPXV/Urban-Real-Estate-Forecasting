"""Minimal forecasting model implementations for the ICCE paper artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.data_processing import get_main_price_series


@dataclass
class ForecastResult:
    """Container for forecasts and optional confidence intervals."""

    model: str
    y_true: pd.Series
    y_pred: pd.Series
    lower: Optional[pd.Series] = None
    upper: Optional[pd.Series] = None



def train_test_split_time_series(
    series: pd.Series,
    train_size: int = 37,
    test_size: int = 10,
) -> tuple[pd.Series, pd.Series]:
    """Split a monthly series into contiguous train and test windows."""
    if len(series) < train_size + test_size:
        raise ValueError(
            f"Series length {len(series)} is too short for requested split {train_size}/{test_size}."
        )

    train = series.iloc[:train_size].copy()
    test = series.iloc[train_size : train_size + test_size].copy()
    return train, test



def fit_naive_forecast(train: pd.Series, test: pd.Series) -> ForecastResult:
    """Forecast each test month using the last observed training value."""
    last_value = float(train.iloc[-1])
    preds = pd.Series(last_value, index=test.index, name="naive_pred")
    return ForecastResult(model="Naive", y_true=test, y_pred=preds)



def fit_moving_average_forecast(train: pd.Series, test: pd.Series, window: int = 3) -> ForecastResult:
    """Recursive moving-average forecast over the test horizon."""
    if window < 1:
        raise ValueError("window must be >= 1")
    history = list(train.astype(float).values)
    preds = []
    for _ in range(len(test)):
        recent = history[-window:] if len(history) >= window else history
        next_val = float(np.mean(recent))
        preds.append(next_val)
        history.append(next_val)
    pred_series = pd.Series(preds, index=test.index, name=f"ma_{window}_pred")
    return ForecastResult(model=f"Moving Average ({window})", y_true=test, y_pred=pred_series)



def fit_arima_forecast(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int] = (3, 1, 3),
) -> ForecastResult:
    """Fit ARIMA and forecast over the test horizon."""
    model = ARIMA(train.astype(float), order=order)
    fit = model.fit()
    forecast_res = fit.get_forecast(steps=len(test))
    preds = pd.Series(forecast_res.predicted_mean.values, index=test.index, name="arima_pred")
    conf_int = forecast_res.conf_int(alpha=0.05)
    lower = pd.Series(conf_int.iloc[:, 0].values, index=test.index, name="lower")
    upper = pd.Series(conf_int.iloc[:, 1].values, index=test.index, name="upper")
    return ForecastResult(model=f"ARIMA {order}", y_true=test, y_pred=preds, lower=lower, upper=upper)



def fit_prophet_forecast(train: pd.Series, test: pd.Series) -> ForecastResult:
    """Fit Prophet on monthly data and forecast test months."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise RuntimeError(
            "Prophet is not installed. Install with `pip install prophet` to run this model."
        ) from exc

    prophet_train = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_train)

    future = pd.DataFrame({"ds": test.index})
    fcst = model.predict(future)

    preds = pd.Series(fcst["yhat"].values, index=test.index, name="prophet_pred")
    lower = pd.Series(fcst["yhat_lower"].values, index=test.index, name="lower")
    upper = pd.Series(fcst["yhat_upper"].values, index=test.index, name="upper")
    return ForecastResult(model="Prophet", y_true=test, y_pred=preds, lower=lower, upper=upper)



def fit_lstm_forecast(
    train: pd.Series,
    test: pd.Series,
    lookback: int = 3,
    epochs: int = 200,
    batch_size: int = 8,
    seed: int = 42,
) -> ForecastResult:
    """Fit a compact single-feature LSTM and forecast recursively."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, LSTM
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for LSTM forecasting. Install with `pip install tensorflow`."
        ) from exc

    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_values = train.astype(float).values
    if len(train_values) <= lookback:
        raise ValueError("Training series is too short for chosen lookback.")

    min_v, max_v = float(train_values.min()), float(train_values.max())
    denom = max(max_v - min_v, 1e-8)

    def scale(values: np.ndarray) -> np.ndarray:
        return (values - min_v) / denom

    def inverse(values: np.ndarray) -> np.ndarray:
        return values * denom + min_v

    scaled = scale(train_values)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i])
        y.append(scaled[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)

    model = Sequential([LSTM(16, input_shape=(lookback, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    history = list(scaled)
    preds_scaled = []
    for _ in range(len(test)):
        x_input = np.array(history[-lookback:]).reshape(1, lookback, 1)
        next_scaled = float(model.predict(x_input, verbose=0)[0, 0])
        preds_scaled.append(next_scaled)
        history.append(next_scaled)

    preds = pd.Series(inverse(np.array(preds_scaled)), index=test.index, name="lstm_pred")
    return ForecastResult(model="LSTM", y_true=test, y_pred=preds)



def run_primary_split_models(series: Optional[pd.Series] = None) -> dict[str, ForecastResult]:
    """Run all paper models on the primary 37/10 split."""
    target = get_main_price_series() if series is None else series
    train, test = train_test_split_time_series(target, train_size=37, test_size=10)

    results: dict[str, ForecastResult] = {
        "Naive": fit_naive_forecast(train, test),
        "Moving Average (3)": fit_moving_average_forecast(train, test, window=3),
        "ARIMA (3,1,3)": fit_arima_forecast(train, test, order=(3, 1, 3)),
    }

    try:
        results["Prophet"] = fit_prophet_forecast(train, test)
    except RuntimeError:
        pass

    try:
        results["LSTM"] = fit_lstm_forecast(train, test)
    except RuntimeError:
        pass

    return results
