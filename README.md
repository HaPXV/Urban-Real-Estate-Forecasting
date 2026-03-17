# ICCE 2026 Research Artifact

## Title
**A Multi-Source Framework for Forecasting Urban Real Estate Trends in Ho Chi Minh City**

This repository is a **concise, reviewer-friendly paper artifact** prepared for ICCE 2026 evaluation.
It is intentionally minimal and focused on reproducibility signals rather than full thesis-scale content.

## 60-Second Reviewer Overview
- **What this is:** a compact ICCE 2026 artifact for reproducing core forecasting outputs, not a full thesis repository.
- **Forecasting setup:** monthly prediction with **47 observations** and primary split of **37 train / 10 test**.
- **Data design:** three aligned monthly datasets (transactions, macro context, and online market sentiment/activity).
- **Privacy policy:** only anonymized monthly aggregates are public; sensitive raw records remain withheld.
- **Where reproduced outputs appear:** charts in `results/figures/` and tables in `results/tables/`.

## Scope
The artifact provides:
- Lightweight project structure for data, code, configs, and outputs.
- Placeholder modules for processing, forecasting, evaluation, baselines, and plotting.
- Starter documentation describing protocol, provenance, and limitations.

The artifact does **not** include a full end-to-end thesis repository.

## Datasets (Three-Source Design)
The study aligns three monthly data sources:

1. **Dataset A — Urban housing transactions (core target source).**
   - Purpose: supply historical housing market dynamics for forecasting.
   - Access: sensitive row-level records are **withheld for privacy**.

2. **Dataset B — Socioeconomic and macro indicators (context source).**
   - Purpose: capture broader drivers linked to real-estate movement.
   - Access: sensitive or licensed granular records are **withheld for privacy/licensing**.

3. **Dataset C — Online market activity/sentiment indicators (auxiliary source).**
   - Purpose: provide high-level behavioral signals that complement A/B.
   - Access: only non-identifying, monthly aggregated representations are exposed.

### Public vs. Withheld
- **Public:** anonymized monthly aggregated data placed under `data/public/`.
- **Withheld:** raw, row-level or identifying source data for Datasets A/B (and any sensitive intermediate joins).
- Supporting schema/provenance notes belong in `data/schema/` and `docs/`.

## Core Forecasting Protocol
- Time granularity: monthly.
- Total observations in the main setup: **47**.
- Primary evaluation split: **37 training / 10 testing**.
- Configs and model templates are in `configs/` and `src/` for reproducible execution.

## Reproduced Artifact Locations
- Figures: `results/figures/`
- Tables: `results/tables/`

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
