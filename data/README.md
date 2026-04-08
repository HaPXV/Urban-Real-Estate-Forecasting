# Data Directory — Availability and Withholding Policy

This document describes all datasets used in the paper and their public release status.

---

## Task 1 Data (Market Forecasting) — INCLUDED

The following monthly aggregate datasets are safe for public release and are included under `data/public/`:

### `dataset_a_monthly_public.csv`
- **Content:** Monthly aggregated real-estate transaction series (Dataset A).
- **Columns:** `year_month`, `avg_price_m2_million`, `transaction_count`
- **Why public:** Only month-level aggregate statistics — no row-level records, addresses, or identifiers.
- **Scope:** 47 months used in Task 1 forecasting experiments.

### `dataset_b_monthly_public.csv`
- **Content:** Monthly aggregated land-registration activity counts (Dataset B).
- **Columns:** `year_month`, `registration_count`
- **Why public:** Monthly counts only — no registration-level personal or administrative details.

### `dataset_c_macro_public.csv`
- **Content:** Monthly macroeconomic indicators (Dataset C).
- **Columns:** `year_month`, `interest_rate`, `gold_price_sjc`, `vn_index`, `cpi_index`
- **Why public:** Macro-level aggregates; no personal records.

### `unified_monthly_panel_public.csv`
- **Content:** Merged monthly analysis panel (Dataset A + B + C) for Task 1 forecasting workflows.
- **Why public:** Composed only of sanitized monthly aggregate variables.

---

## Task 2 Data (Property-Level Valuation) — WITHHELD

### `task2_modeling_trimmed.parquet` — **NOT RELEASED**
- **Content:** 5,005 individual property transaction records with structural, spatial, and temporal features used for Task 2 (XGBoost / H3 hedonic valuation).
- **Why withheld:** Contains transaction-level records sourced from non-public data providers. Individual record release would expose confidential property, price, and location data at a level inappropriate for public release.
- **Feature categories (not released):**
  - Structural: `area_m2`, `so_tang` (floors), `cap_cong_trinh` (quality grade), `frontage_m`, `is_mat_tien` (street-facing)
  - Location: `district`, `ward_key`
  - Temporal: `transaction_date`, `transaction_year`, `transaction_quarter`
  - Spatial: `dist_ben_thanh_km` (road-network distance to Ben Thanh Market)
  - Target: `price_m2` (VND per m²)

### Aggregate Task 2 results (INCLUDED in `results/tables/`)
Although the raw dataset is withheld, the following aggregate artifacts are included for reviewer verification:
- `results/tables/task2_p3_benchmark.csv` — MAPE, MAE, RMSE, R² for all Task 2 models on the P3 test set.
- `results/tables/bootstrap_xgb_vs_h3.json` — Paired bootstrap inference results (5,000 replicates), XGBoost vs H3.

---

## No PII in Any Public Artifact

No personally identifiable information (PII) is included in any publicly released file. All public files contain only statistical summaries at the monthly aggregate level.

---

## Data Access for Reviewers

Reviewers requiring access to the Task 2 dataset for deeper verification may contact the corresponding author. Access may be granted under a data-sharing agreement subject to source data licensing terms.
