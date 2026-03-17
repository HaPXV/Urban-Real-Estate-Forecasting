# Urban Real Estate Forecasting

## 60-second overview

This repository reproduces the forecasting study on **monthly urban real-estate dynamics**, evaluating how well compact time-series models predict the next months of market movement from a short history window.

### Paper scope
- **Goal:** forecast monthly real-estate outcomes in an urban market and compare model performance under a fixed, reproducible setup.
- **Unit of analysis:** one monthly target series with aligned exogenous covariates.
- **Evaluation focus:** out-of-sample forecasting quality and reproducibility of the paper’s reported visuals/tables.

### Three datasets used in the paper
1. **Dataset A — Public market series (public):** monthly city-level real-estate target series used for forecasting.
2. **Dataset B — Public contextual covariates (public):** monthly macro/market covariates aligned to Dataset A timestamps.
3. **Dataset C — Proprietary micro-data (withheld):** privacy-sensitive, record-level data used only to derive aggregate features.

### What is public vs. withheld
- **Public in this repository:**
  - processed, reproducible monthly modeling tables derived from public sources;
  - code/configuration needed to regenerate forecasts, figures, and tables;
  - documentation of split logic and evaluation protocol.
- **Withheld for privacy:**
  - raw, record-level proprietary transactions/identifiers;
  - any data fields that could re-identify individuals, properties, or counterparties.

### Main forecasting setup
- **Total history length:** **47 monthly observations**.
- **Primary training/test split:** **37 months train / 10 months test**.
- **Forecasting task:** train on the first 37 observations, forecast the final 10, and report test-set errors.

### Where to find reproduced outputs
- **Figures:** `results/figures/`
- **Tables:** `results/tables/`

(These folders are the canonical locations for paper reproduction artifacts.)
