# Experimental Protocol

## Time horizon and split policy
- The artifact reflects **47 monthly observations**.
- The primary setup is a **37/10 train-test split** (train/test months).

## Models and evaluation scope
- Core model outputs are represented in `results/tables/table2_model_performance.csv`.
- Split robustness outcomes are provided in `results/tables/table3_robustness_evaluation_across_different_train_test_splits.csv`.
- Baseline comparison outcomes are provided in `results/tables/table4_comparison_with_simple_forecasting_baselines.csv`.

## Figure/table traceability
- Figure 2 is a price time-series visualization (`results/figures/figure2_price_timeseries.png`).
- Figure 3 is the actual-vs-predicted visualization (`results/figures/figure3_actual_vs_predicted.png`).
- Prophet-specific support for Figure 3 is in `results/tables/figure3_prophet_forecast_data.csv`.
- `results/tables/paper_prophet_summary.csv` is a supporting summary artifact and does not replace Table II/III/IV.

## Positioning note
This protocol document is intentionally brief to match ICCE artifact-review needs.
