# ICCE 2026 Paper Artifact Repository

## Paper
**A Multi-Source Framework for Forecasting Urban Real Estate Trends in Ho Chi Minh City**

This repository is a concise, reviewer-facing artifact package for ICCE 2026. It is designed for fast verification of paper-aligned datasets and outputs, not as a full thesis repository.

## 60-Second Reviewer Overview
- **Forecasting setup in paper:** 47 monthly observations with a primary **37/10 train-test split**.
- **Three data sources used in the framework:**
  - **Dataset A:** monthly aggregated real-estate transaction series.
  - **Dataset B:** monthly aggregated land-registration activity counts.
  - **Dataset C:** monthly macroeconomic indicators.
- **Privacy constraint:** raw Dataset A and Dataset B are not public because source-level records are sensitive/non-public.
- **Public sanitized data release:** all reviewer-accessible datasets are under `data/public/`.

## Public Sanitized Data Files
- `data/public/dataset_a_monthly_public.csv`
- `data/public/dataset_b_monthly_public.csv`
- `data/public/dataset_c_macro_public.csv`
- `data/public/unified_monthly_panel_public.csv`

No personally identifiable information (PII) is included in these public artifacts.

## Paper Artifacts in This Repository
- **Paper PDF:** `paper_assets/ICCE2026_PhamVinhHa_RealEstateForecasting.pdf`
- **Figure 1:** `results/figures/figure1_framework.png`
- **Figure 2:** `results/figures/figure2_price_timeseries.png`
- **Figure 3:** `results/figures/figure3_actual_vs_predicted.png`
- **Table II:** `results/tables/table2_model_performance.csv`
- **Table III:** `results/tables/table3_robustness_evaluation_across_different_train_test_splits.csv`
- **Table IV:** `results/tables/table4_comparison_with_simple_forecasting_baselines.csv`
- **Supporting Prophet artifacts:**
  - `results/tables/paper_prophet_summary.csv`
  - `results/tables/figure3_prophet_forecast_data.csv`
- **Supporting experiment figures:**
  - `results/figures/fig_exp1_split_robustness_mape.png`
  - `results/figures/fig_exp2_baseline_comparison.png`

## How to Verify This Repository
1. Confirm the ICCE 2026 paper PDF is present:
   ```bash
   test -f paper_assets/ICCE2026_PhamVinhHa_RealEstateForecasting.pdf
   ```
2. Confirm public sanitized datasets are present:
   ```bash
   test -f data/public/dataset_a_monthly_public.csv
   test -f data/public/dataset_b_monthly_public.csv
   test -f data/public/dataset_c_macro_public.csv
   test -f data/public/unified_monthly_panel_public.csv
   ```
3. Confirm main paper figure/table artifacts are present:
   ```bash
   test -f results/figures/figure1_framework.png
   test -f results/figures/figure2_price_timeseries.png
   test -f results/figures/figure3_actual_vs_predicted.png
   test -f results/tables/table2_model_performance.csv
   test -f results/tables/table3_robustness_evaluation_across_different_train_test_splits.csv
   test -f results/tables/table4_comparison_with_simple_forecasting_baselines.csv
   ```
4. Optionally inspect file headers for quick sanity checks:
   ```bash
   head -n 2 data/public/dataset_a_monthly_public.csv
   head -n 2 data/public/unified_monthly_panel_public.csv
   ```

## Scope Note
This artifact repository is intentionally minimal and paper-aligned for reviewer reproducibility checks.
