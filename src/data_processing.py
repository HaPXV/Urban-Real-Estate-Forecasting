"""Data loading and preprocessing utilities for public ICCE artifact datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_PUBLIC_DATA_DIR = Path("data/public")
YEAR_MONTH_COL = "year_month"


class DataValidationError(ValueError):
    """Raised when a dataset is missing required columns or invalid values."""



def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise DataValidationError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )



def _prepare_monthly_frame(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    _validate_columns(df, required_columns, dataset_name)

    frame = df.copy()
    frame[YEAR_MONTH_COL] = pd.to_datetime(frame[YEAR_MONTH_COL], format="%Y-%m", errors="coerce")
    if frame[YEAR_MONTH_COL].isna().any():
        invalid_rows = frame[frame[YEAR_MONTH_COL].isna()].index.tolist()
        raise DataValidationError(
            f"{dataset_name} has invalid {YEAR_MONTH_COL} values at row indices: {invalid_rows}"
        )

    frame = frame.sort_values(YEAR_MONTH_COL).drop_duplicates(subset=[YEAR_MONTH_COL], keep="last")
    frame = frame.set_index(YEAR_MONTH_COL)
    inferred_freq = pd.infer_freq(frame.index)
    if inferred_freq:
        frame = frame.asfreq(inferred_freq)
    return frame



def _load_public_csv(filename: str, data_dir: Path = DEFAULT_PUBLIC_DATA_DIR) -> pd.DataFrame:
    csv_path = Path(data_dir) / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {csv_path}")
    return pd.read_csv(csv_path)



def load_dataset_a_public(data_dir: Path = DEFAULT_PUBLIC_DATA_DIR) -> pd.DataFrame:
    """Load Dataset A monthly public series indexed by month."""
    df = _load_public_csv("dataset_a_monthly_public.csv", data_dir=data_dir)
    required = [YEAR_MONTH_COL, "avg_price_m2_million", "transaction_count"]
    return _prepare_monthly_frame(df, "dataset_a_monthly_public", required)



def load_dataset_b_public(data_dir: Path = DEFAULT_PUBLIC_DATA_DIR) -> pd.DataFrame:
    """Load Dataset B monthly registration counts indexed by month."""
    df = _load_public_csv("dataset_b_monthly_public.csv", data_dir=data_dir)
    required = [YEAR_MONTH_COL, "registration_count"]
    return _prepare_monthly_frame(df, "dataset_b_monthly_public", required)



def load_dataset_c_public(data_dir: Path = DEFAULT_PUBLIC_DATA_DIR) -> pd.DataFrame:
    """Load Dataset C macro indicators indexed by month."""
    df = _load_public_csv("dataset_c_macro_public.csv", data_dir=data_dir)
    required = [YEAR_MONTH_COL, "interest_rate", "gold_price_sjc", "vn_index"]
    return _prepare_monthly_frame(df, "dataset_c_macro_public", required)



def load_unified_panel_public(data_dir: Path = DEFAULT_PUBLIC_DATA_DIR) -> pd.DataFrame:
    """Load the merged monthly panel indexed by month."""
    df = _load_public_csv("unified_monthly_panel_public.csv", data_dir=data_dir)
    required = [
        YEAR_MONTH_COL,
        "avg_price_m2_million",
        "transaction_count",
        "registration_count",
        "interest_rate",
        "gold_price_sjc",
        "vn_index",
    ]
    return _prepare_monthly_frame(df, "unified_monthly_panel_public", required)



def get_main_price_series(data_dir: Path = DEFAULT_PUBLIC_DATA_DIR) -> pd.Series:
    """Return the paper's main monthly target series from Dataset A."""
    dataset_a = load_dataset_a_public(data_dir=data_dir)
    series = dataset_a["avg_price_m2_million"].astype(float)
    series.name = "avg_price_m2_million"
    return series
