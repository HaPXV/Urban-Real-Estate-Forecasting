# Paper-to-Repository Alignment (ICCE 2026)

## Scope and Contribution Mapping
This repository corresponds to the ICCE 2026 paper artifact for:

**“A Multi-Source Framework for Forecasting Urban Real Estate Trends in Ho Chi Minh City.”**

It provides a compact, reviewer-facing package of:
- the paper PDF,
- public sanitized monthly datasets,
- final reproduced figure/table artifacts,
- and supporting experiment outputs.

It is intentionally not a thesis-scale repository.

## Data Source Mapping (Paper -> Public Files)
The paper describes three monthly data sources. They map to the following public artifacts:

- **Dataset A (monthly aggregated real-estate transaction series)**
  - `data/public/dataset_a_monthly_public.csv`
- **Dataset B (monthly aggregated land-registration activity counts)**
  - `data/public/dataset_b_monthly_public.csv`
- **Dataset C (monthly macroeconomic indicators)**
  - `data/public/dataset_c_macro_public.csv`
- **Unified monthly panel for aligned modeling inputs**
  - `data/public/unified_monthly_panel_public.csv`

Raw source-level Dataset A and Dataset B records are withheld due to sensitivity/non-public constraints; only sanitized monthly aggregates are released.

## Forecasting Setup Mapping
The main forecasting setup reported in the paper is represented by this artifact package as:
- **47 monthly observations total**
- **primary 37/10 train-test split**

## Result Artifact Mapping

### A) Final reproduced result artifacts (main paper outputs)
- **Paper PDF** -> `paper_assets/ICCE2026_PhamVinhHa_RealEstateForecasting.pdf`
- **Figure 1 (framework/pipeline overview)** -> `results/figures/figure1_framework.png`
- **Figure 2** -> `results/figures/figure2_price_timeseries.png`
- **Figure 3** -> `results/figures/figure3_actual_vs_predicted.png`
- **Table II** -> `results/tables/table2_model_performance.csv`
- **Table III** -> `results/tables/table3_robustness_evaluation_across_different_train_test_splits.csv`
- **Table IV** -> `results/tables/table4_comparison_with_simple_forecasting_baselines.csv`

### B) Supporting forecast data (Prophet / Figure 3 support)
These are supporting artifacts for the Prophet experiment and Figure 3 context, not the main cross-model comparison tables:
- `results/tables/paper_prophet_summary.csv`
- `results/tables/figure3_prophet_forecast_data.csv`

### C) Supporting experiment figures (robustness/baseline support)
These are supporting experiment visuals, not the main numbered paper figures:
- `results/figures/fig_exp1_split_robustness_mape.png`
- `results/figures/fig_exp2_baseline_comparison.png`

## Notes for Reviewers
For rapid verification, check file existence and inspect CSV headers/rows directly in the paths above.
