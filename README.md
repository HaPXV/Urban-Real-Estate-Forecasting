# ICCE 2026 Paper Artifact Repository

## Paper
**A Multi-Source Framework for Forecasting Urban Real Estate Trends in Ho Chi Minh City**

This repository is a **reviewer-facing ICCE 2026 artifact**. It is intentionally compact and focused on the paper outputs, not a full thesis codebase.

## 60-Second Reviewer Overview
- **Forecasting setup:** monthly forecasting with **47 observations** using a primary **37/10 train-test split**.
- **Data design:** three monthly sources used in the paper:
  - **Dataset A:** monthly aggregated real-estate transaction series.
  - **Dataset B:** monthly aggregated land-registration activity counts.
  - **Dataset C:** monthly macroeconomic indicators.
- **Privacy:** raw Dataset A and Dataset B records are **not public** due to sensitive source information.
- **Public sanitized files:** all reviewer-available data are under `data/public/`.
- **Paper artifact outputs:** figures and result tables are available under `results/` and the paper PDF is under `paper_assets/`.

## Public Data Files (Sanitized)
- `data/public/dataset_a_monthly_public.csv`
- `data/public/dataset_b_monthly_public.csv`
- `data/public/dataset_c_macro_public.csv`
- `data/public/unified_monthly_panel_public.csv`

No personally identifiable information (PII) is included in these public artifacts.

## Paper Artifact Outputs
- Paper PDF: `paper_assets/ICCE2026_PhamVinhHa_RealEstateForecasting.pdf`
- Figure 2 artifact: `results/figures/figure2_price_timeseries.png`
- Figure 3 artifact: `results/figures/figure3_actual_vs_predicted.png`
- Table 2 artifact: `results/tables/table2_model_performance.csv`
- Supporting table (Prophet summary): `results/tables/paper_prophet_summary.csv`
- Supporting data for Figure 3: `results/tables/figure3_prophet_forecast_data.csv`
- Multi-source layout template (not a finalized benchmark table): `results/tables/table3_multisource_template.csv`

## How to Verify This Repository
1. Confirm the paper file exists:
   ```bash
   test -f paper_assets/ICCE2026_PhamVinhHa_RealEstateForecasting.pdf
   ```
2. Confirm reproduced figures/tables exist:
   ```bash
   test -f results/figures/figure2_price_timeseries.png
   test -f results/figures/figure3_actual_vs_predicted.png
   test -f results/tables/table2_model_performance.csv
   ```
3. Confirm public sanitized datasets exist:
   ```bash
   test -f data/public/dataset_a_monthly_public.csv
   test -f data/public/dataset_b_monthly_public.csv
   test -f data/public/dataset_c_macro_public.csv
   test -f data/public/unified_monthly_panel_public.csv
   ```
4. Optionally inspect a few rows:
   ```bash
   head -n 5 data/public/unified_monthly_panel_public.csv
   ```

## Scope Note
This artifact is intentionally minimal for paper review and reproducibility checks. It does not claim to release sensitive raw data or full thesis-scale experimentation assets.
