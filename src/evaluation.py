"""Evaluation helpers for paper-aligned forecast comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.forecasting_models import ForecastResult, train_test_split_time_series



def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE for aligned arrays."""
    true = pd.Series(y_true).astype(float)
    pred = pd.Series(y_pred).astype(float).reindex(true.index)

    if pred.isna().any():
        raise ValueError("y_pred contains missing values after aligning with y_true index.")

    errors = true - pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    non_zero = true != 0
    if not non_zero.any():
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((errors[non_zero] / true[non_zero]) * 100)))

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}



def build_model_comparison_table(
    model_results: dict[str, ForecastResult],
    output_path: Path | None = Path("results/tables/table2_model_performance.csv"),
) -> pd.DataFrame:
    """Build Table II style summary with Model/MAE/RMSE/MAPE columns."""
    rows = []
    for model_name, result in model_results.items():
        metrics = compute_metrics(result.y_true, result.y_pred)
        rows.append({"Model": model_name, **metrics})

    table = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "MAPE"])
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    return table



def build_robustness_table(
    series: pd.Series,
    model_functions: dict[str, Callable[[pd.Series, pd.Series], ForecastResult]],
    splits: list[tuple[int, int]],
    output_path: Path | None = Path(
        "results/tables/table3_robustness_evaluation_across_different_train_test_splits.csv"
    ),
) -> pd.DataFrame:
    """Build robustness table over multiple train/test split configurations."""
    rows = []
    for train_size, test_size in splits:
        train, test = train_test_split_time_series(series, train_size=train_size, test_size=test_size)
        split_label = f"{train_size}/{test_size}"

        for model_name, model_fn in model_functions.items():
            forecast = model_fn(train, test)
            metrics = compute_metrics(forecast.y_true, forecast.y_pred)
            rows.append({"Split": split_label, "Model": model_name, **metrics})

    table = pd.DataFrame(rows, columns=["Split", "Model", "MAE", "RMSE", "MAPE"])
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    return table



def build_baseline_comparison_table(
    baseline_results: dict[str, ForecastResult],
    output_path: Path | None = Path("results/tables/table4_comparison_with_simple_forecasting_baselines.csv"),
) -> pd.DataFrame:
    """Build baseline comparison table with paper schema."""
    rows = []
    for model_name, result in baseline_results.items():
        metrics = compute_metrics(result.y_true, result.y_pred)
        rows.append({"Model": model_name, **metrics})

    table = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "MAPE"])
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    return table
