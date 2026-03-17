# Data Directory Notes

This directory documents the **public sanitized datasets** released with the ICCE 2026 paper artifact repository.

## Privacy Statement
All public files are monthly aggregated and anonymized. No personally identifiable information (PII) is included in these public artifacts.

## Public Dataset Files

### 1) `data/public/dataset_a_monthly_public.csv`
- **Represents:** monthly aggregated real-estate transaction series used as a core market signal.
- **Why safe to publish:** values are aggregated to month-level summary statistics; no row-level transaction records or identities are included.
- **High-level columns:**
  - `year_month`: month key
  - `avg_price_m2_million`: average price per square meter (million VND scale)
  - `transaction_count`: monthly transaction count (aggregated)

### 2) `data/public/dataset_b_monthly_public.csv`
- **Represents:** monthly aggregated land-registration activity counts.
- **Why safe to publish:** only monthly count totals are released; no registrant-level or record-level details are included.
- **High-level columns:**
  - `year_month`: month key
  - `registration_count`: monthly registration activity count

### 3) `data/public/dataset_c_macro_public.csv`
- **Represents:** monthly macroeconomic indicators used as contextual explanatory inputs.
- **Why safe to publish:** macro indicators are published at aggregate market level and do not contain personal records.
- **High-level columns:**
  - `year_month`: month key
  - `interest_rate`: monthly policy/market interest-rate indicator
  - `gold_price_sjc`: monthly gold price indicator
  - `vn_index`: monthly VN-Index level

### 4) `data/public/unified_monthly_panel_public.csv`
- **Represents:** merged monthly panel combining Dataset A, Dataset B, and Dataset C fields for aligned analysis.
- **Why safe to publish:** constructed only from anonymized monthly aggregates; no personal or row-level source records are exposed.
- **High-level columns:**
  - `year_month`: month key
  - Dataset A aggregate fields (price and transaction count)
  - Dataset B aggregate field (registration count)
  - Dataset C macro fields (interest rate, gold price, VN-Index)

## Withheld Data
Raw source data for Dataset A and Dataset B are not publicly distributed in this repository because they may contain sensitive information in non-aggregated form.
