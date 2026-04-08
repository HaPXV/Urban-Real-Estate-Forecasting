# A Dual-Task Framework for Real Estate Intelligence: Market Forecasting and Property Valuation in Ho Chi Minh City

**Conference:** ICCE 2026
**Authors:** Pham Vinh Ha et al.

---

## Paper Summary

This paper introduces a dual-task framework for urban real estate analysis applied to Ho Chi Minh City:

- **Task 1 — Market Forecasting:** Predicts ward-level average price per m² as a monthly time series using classical and ML-based forecasting models (MA, ARIMA, Prophet, LSTM). Primary benchmark: 37-month training / 10-month hold-out.
- **Task 2 — Property-Level Valuation:** Predicts individual property price per m² using gradient boosting (XGBoost, LightGBM, CatBoost) against an interpretable OLS hedonic baseline (H3). Primary benchmark: **P3 temporal split** (train: transactions before 2025-01-01; test: 2025 transactions, N=1,658).

---

## Repository Contents

```
paper_assets/
  PAPER_final.pdf               ← Final submitted paper (ICCE 2026)
  ICCE_2026___Vinh_Ha.pdf       ← Earlier review version (same content)

results/
  figures/
    figure1_framework.png       ← Overall dual-task framework diagram
    figure2_price_timeseries.png
    figure3_actual_vs_predicted.png
    fig_road_network_accessibility.png  ← Road-network spatial accessibility map (Task 2)
    fig_exp1_split_robustness_mape.png
    fig_exp2_baseline_comparison.png
  tables/
    task2_p3_benchmark.csv      ← Task 2 P3 benchmark: all models (XGBoost, LightGBM, etc.)
    bootstrap_xgb_vs_h3.json    ← Paired bootstrap results: XGBoost vs H3 (5,000 replicates)
    table2_model_performance.csv
    table3_robustness_evaluation_across_different_train_test_splits.csv
    table4_comparison_with_simple_forecasting_baselines.csv
    paper_prophet_summary.csv
    figure3_prophet_forecast_data.csv

scripts/
  run_bootstrap.py              ← Reproduces XGBoost vs H3 paired bootstrap (Task 2)
  run_hedonic_h3.py             ← Reproduces H3 OLS hedonic model on P3 split
  run_task2_p3_benchmark.py     ← Reproduces full Task 2 P3 benchmark (all ML models)

src/                            ← Task 1 forecasting modules (data processing, models, evaluation)
configs/                        ← ARIMA/Prophet/LSTM config YAML files

data/
  public/                       ← Sanitized monthly aggregate data (safe to release)
    dataset_a_monthly_public.csv
    dataset_b_monthly_public.csv
    dataset_c_macro_public.csv
    unified_monthly_panel_public.csv
  README.md                     ← Data availability and withholding policy

docs/                           ← Methodology and reproducibility notes
```

---

## What Is Intentionally Excluded

| Item | Reason |
|---|---|
| Raw transaction-level Dataset A (parquet/csv) | Confidential — contains non-public property records |
| Raw Dataset B registration records | Non-public administrative data |
| Trained model binaries (.pkl, .joblib) | Not required for reviewer verification; models are fully reproducible from scripts |
| `.venv`, `env/`, conda environments | Standard exclusion |
| `__pycache__`, `.ipynb_checkpoints` | Junk |
| Draft manuscript files, old PDFs | Not reviewer-relevant |
| Large temporary outputs, logs | Not reviewer-relevant |
| Shapefile GIS data (GADM, OSM) | Large; publicly available from GADM/OSM directly |

---

## Finding the Paper PDF

```
paper_assets/PAPER_final.pdf
```

---

## Primary Benchmark Note (Task 2)

The **P3 temporal split** is the primary evaluation protocol for Task 2:
- Train: all transactions with `transaction_date < 2025-01-01` (N=3,347)
- Test: all transactions with `transaction_date >= 2025-01-01` (N=1,658)

XGBoost achieves **MAPE = 16.55%**, R² = 0.640 on the P3 test set.
H3 (OLS, HC3-robust, same P3 protocol): **MAPE = 18.27%**, R² = 0.522.
Paired bootstrap (B=5,000): 95% CI for MAPE gap = **[1.10, 2.45] pp** (one-sided p < 0.001).

See `results/tables/task2_p3_benchmark.csv` and `results/tables/bootstrap_xgb_vs_h3.json`.

---

## Reproducibility Note

**Task 2 (Property Valuation):**
1. Requires `task2_modeling_trimmed.parquet` (not publicly released — see `data/README.md`).
2. With the dataset, run `scripts/run_task2_p3_benchmark.py` to reproduce Table 2a metrics.
3. Run `scripts/run_bootstrap.py` to reproduce the paired bootstrap comparison.
4. Run `scripts/run_hedonic_h3.py` for the H3 OLS hedonic benchmark.

**Task 1 (Market Forecasting):**
1. Requires `data/public/unified_monthly_panel_public.csv` (included).
2. Source code is in `src/`. Config files are in `configs/`.

**Environment:** Python 3.11+. See `requirements.txt` for dependencies.

---

## Raw / Private Data Notice

Raw transaction-level data (Dataset A) is **not publicly released** due to confidentiality of source records. Only monthly aggregate statistics are provided under `data/public/`. See `data/README.md` for full disclosure.
