# ICCE 2026 Research Artifact

## Title
**A Multi-Source Framework for Forecasting Urban Real Estate Trends in Ho Chi Minh City**

This repository is a **concise, reviewer-friendly paper artifact** prepared for ICCE 2026 evaluation.
It is intentionally minimal and focused on reproducibility signals rather than full thesis-scale content.

## Scope
The artifact provides:
- Lightweight project structure for data, code, configs, and outputs.
- Placeholder modules for processing, forecasting, evaluation, baselines, and plotting.
- Starter documentation describing protocol, provenance, and limitations.

The artifact does **not** include a full end-to-end thesis repository.

## Data Availability and Privacy
Dataset A and Dataset B used in the study contain sensitive information and cannot be released in raw form.
Only **anonymized monthly aggregated data** is intended for public sharing under `data/public/`.
Schema and metadata references can be documented under `data/schema/` and `docs/`.

## Repository Layout
```text
.
├── AGENTS.md
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── configs/
├── data/
│   ├── public/
│   └── schema/
├── docs/
├── notebooks/
├── paper_assets/
├── results/
└── src/
```

## Quick Start
1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add public-ready aggregates to `data/public/`.
4. Configure model settings in `configs/`.
5. Implement and run pipelines from `src/`.

## Intended Review Use
Reviewers can inspect structure, documentation placeholders, and configuration templates to assess artifact readiness and reproducibility planning.
