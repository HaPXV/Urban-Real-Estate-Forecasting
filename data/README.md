# Data Directory Notes

This directory documents the public, sanitized monthly datasets released with the ICCE 2026 paper artifact repository.

## Privacy and Release Policy
- Public data are provided only at monthly aggregate level.
- No personally identifiable information (PII) is included in any public artifact.
- Raw source-level records for Dataset A and Dataset B are not publicly released due to sensitive/non-public source information.

## Public Dataset Files

### 1) `data/public/dataset_a_monthly_public.csv`
- **Represents:** Dataset A monthly aggregated real-estate transaction series.
- **Why it is safe to publish:** only month-level aggregate statistics are provided; no row-level transaction records, addresses, or individual/entity identifiers are included.
- **High-level columns:**
  - `year_month`: monthly time key
  - `avg_price_m2_million`: monthly average unit price (million VND per m²)
  - `transaction_count`: monthly aggregated transaction volume

### 2) `data/public/dataset_b_monthly_public.csv`
- **Represents:** Dataset B monthly aggregated land-registration activity counts.
- **Why it is safe to publish:** only monthly counts are released; no registration-level personal or administrative record details are included.
- **High-level columns:**
  - `year_month`: monthly time key
  - `registration_count`: monthly aggregated land-registration activity count

### 3) `data/public/dataset_c_macro_public.csv`
- **Represents:** Dataset C monthly macroeconomic indicators used as contextual explanatory variables.
- **Why it is safe to publish:** indicators are macro-level aggregates and do not contain personal records.
- **High-level columns:**
  - `year_month`: monthly time key
  - `interest_rate`: monthly interest-rate indicator
  - `gold_price_sjc`: monthly SJC gold price indicator
  - `vn_index`: monthly VN-Index indicator

### 4) `data/public/unified_monthly_panel_public.csv`
- **Represents:** merged monthly analysis panel aligning Dataset A, Dataset B, and Dataset C fields for forecasting workflows.
- **Why it is safe to publish:** composed only of sanitized monthly aggregate variables; no person-level, household-level, or transaction-level raw records are included.
- **High-level columns:**
  - `year_month`: monthly time key
  - Dataset A aggregate fields (price and transaction count)
  - Dataset B aggregate field (registration count)
  - Dataset C macro fields (interest rate, gold price, VN-Index)
