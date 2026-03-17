"""Plotting helpers for paper figures and supporting experiment visuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



def plot_price_timeseries(
    price_series: pd.Series,
    output_path: Path = Path("results/figures/figure2_price_timeseries.png"),
) -> Path:
    """Recreate Figure 2 with Dataset A monthly average price series."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(price_series.index, price_series.values, color="#1f77b4", linewidth=2)
    ax.set_title("Dataset A Monthly Average Price per m²")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (million VND/m²)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path



def plot_actual_vs_predicted(
    forecast_df: pd.DataFrame,
    output_path: Path = Path("results/figures/figure3_actual_vs_predicted.png"),
) -> Path:
    """Recreate Figure 3 using actual/predicted Prophet test forecasts."""
    required = {"date", "actual_million", "predicted_million"}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for actual-vs-predicted plot: {sorted(missing)}")

    data = forecast_df.copy()
    data["date"] = pd.to_datetime(data["date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(data["date"], data["actual_million"], marker="o", label="Actual", linewidth=2)
    ax.plot(data["date"], data["predicted_million"], marker="o", label="Predicted", linewidth=2)

    if {"lower_million", "upper_million"}.issubset(data.columns):
        ax.fill_between(
            data["date"],
            data["lower_million"],
            data["upper_million"],
            alpha=0.2,
            label="95% interval",
        )

    ax.set_title("Figure 3: Actual vs Predicted (Prophet, 37/10 split)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (million VND/m²)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path



def plot_robustness_mape(
    robustness_table: pd.DataFrame,
    output_path: Path = Path("results/figures/fig_exp1_split_robustness_mape.png"),
) -> Path:
    """Plot MAPE across train/test split robustness experiments."""
    required = {"Split", "Model", "MAPE"}
    missing = required - set(robustness_table.columns)
    if missing:
        raise ValueError(f"Missing required columns for robustness plot: {sorted(missing)}")

    pivot = robustness_table.pivot(index="Split", columns="Model", values="MAPE")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Robustness Experiment: MAPE Across Splits")
    ax.set_xlabel("Train/Test Split")
    ax.set_ylabel("MAPE (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path



def plot_baseline_comparison(
    baseline_table: pd.DataFrame,
    output_path: Path = Path("results/figures/fig_exp2_baseline_comparison.png"),
) -> Path:
    """Plot baseline model MAE/RMSE/MAPE comparison."""
    required = {"Model", "MAE", "RMSE", "MAPE"}
    missing = required - set(baseline_table.columns)
    if missing:
        raise ValueError(f"Missing required columns for baseline comparison plot: {sorted(missing)}")

    metrics_df = baseline_table.set_index("Model")[["MAE", "RMSE", "MAPE"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics_df.plot(kind="bar", ax=ax)
    ax.set_title("Baseline Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
