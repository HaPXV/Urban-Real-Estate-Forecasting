# Paper-to-Repository Alignment (ICCE 2026)

## Paper Scope in This Repository
This repository packages the reviewer-facing artifacts for:

**“A Multi-Source Framework for Forecasting Urban Real Estate Trends in Ho Chi Minh City.”**

The scope here is limited to concise, reproducible paper evidence: the paper PDF, public sanitized monthly datasets, and reproduced result artifacts (figures/tables).

## Dataset Alignment
The paper uses three monthly data sources. In this repository, they are represented by the following **public sanitized files**:

- **Dataset A (monthly aggregated real-estate transaction series):**
  - `data/public/dataset_a_monthly_public.csv`
- **Dataset B (monthly aggregated land-registration activity counts):**
  - `data/public/dataset_b_monthly_public.csv`
- **Dataset C (monthly macroeconomic indicators):**
  - `data/public/dataset_c_macro_public.csv`
- **Unified monthly analysis panel used for aligned modeling inputs:**
  - `data/public/unified_monthly_panel_public.csv`

Raw source-level records for Dataset A and Dataset B are not included in this artifact repository due to privacy and non-public data constraints.

## Forecasting Protocol Alignment
The primary forecasting setup documented in the paper is reflected as:
- **47 monthly observations total**
- **Primary split: 37 months train / 10 months test**

A compact summary matching this setup is provided in:
- `results/tables/paper_prophet_summary.csv`

## Result Artifact Alignment

### Final reproduced result artifacts
- **Paper PDF:** `paper_assets/ICCE2026_PhamVinhHa_RealEstateForecasting.pdf`
- **Figure 2:** `results/figures/figure2_price_timeseries.png`
- **Figure 3:** `results/figures/figure3_actual_vs_predicted.png`
- **Table 2 (model performance comparison):** `results/tables/table2_model_performance.csv`

### Supporting forecast data (used to generate figure-level evidence)
- **Figure 3 underlying forecast/interval data:**
  - `results/tables/figure3_prophet_forecast_data.csv`
- **Paper-specific Prophet metrics summary:**
  - `results/tables/paper_prophet_summary.csv`

### Templates / structure helpers (not finalized performance claims)
- **Multi-source result template:**
  - `results/tables/table3_multisource_template.csv`

`table3_multisource_template.csv` should be interpreted as a template/supporting artifact (layout scaffold) rather than a finalized benchmark table.

## Consistency and Limits
This repository is intentionally concise for artifact review. It should be read as a paper artifact package, not a full thesis repository or complete raw-data release.
