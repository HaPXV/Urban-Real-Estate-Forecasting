# Privacy and Anonymization

## Public release principle
Only **sanitized monthly aggregated data** are released in this repository.

## Dataset A and Dataset B treatment
- **Dataset A** (real-estate transactions) is published only as monthly aggregate fields in `data/public/dataset_a_monthly_public.csv`.
- **Dataset B** (land-registration activity) is published only as monthly aggregate counts in `data/public/dataset_b_monthly_public.csv`.
- Raw source-level records for Dataset A and Dataset B are not included in version control.

## Protected details not released
The public artifact does not include person-level, household-level, property-level, registration-level, or address-level raw records.

## Public files with anonymized/sanitized content
- `data/public/dataset_a_monthly_public.csv`
- `data/public/dataset_b_monthly_public.csv`
- `data/public/dataset_c_macro_public.csv`
- `data/public/unified_monthly_panel_public.csv`
