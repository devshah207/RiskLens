# RiskLens

**Explainable Financial Risk Modeling, Validation & AI Research Platform**

RiskLens is an **educational research prototype** motivated by applied financial machine learning and model-validation workflows. It is **not investment advice, a trading strategy, or a regulatory-compliant risk engine**.

## Phase 1 status

This repository currently contains **project scope, finance/data documentation, and an installable Python package skeleton only**. It does **not** yet fetch or process market data, train models, calculate risk metrics, create a dashboard, or provide RAG/AI capabilities. No model results exist yet.

## Problem definition

After NIFTY 50 price-index trading session `t` closes, estimate the probability that the following trading session closes **at least 1% below** that close.

For index closing levels `P_t` and `P_(t+1)`:

```text
r_(t+1) = P_(t+1) / P_t - 1
label_t = 1 if r_(t+1) <= -0.01 else 0
```

Features must only use data available **at or before** the prediction time (after close of session `t`). The next-session close is used **only to form historical labels**, never as an input feature.

The `-1%` threshold is an **initial, documented design choice**, subject to checking event frequency using development data *before* evaluating final models.

## Getting started

Requires Python >=3.11.

```bash
# From repository root; activate an isolated virtual environment first if desired
python -m pip install -e .
python -c "import risklens; print('RiskLens import successful')"
```

On some systems, replace `python` with `python3`. No data is required for the Phase 1 skeleton.

## Repository map

- `docs/project_charter.md` — goals, problem, boundaries, and future roadmap.
- `docs/finance_glossary.md` — basic financial terms.
- `docs/data_contract.md` — planned input columns, quality rules, provenance, leakage policy.
- `docs/decisions.md` — design choices, rationale, alternatives, and limitations.
- `src/risklens/` — future Python application code (currently an importable skeleton).
- `tests/` — future automated tests.
- `data/raw/` and `data/processed/` — private working data; intentionally excluded from Git.
- `data/fixtures/` — synthetic, redistributable testing fixtures may be added later.

## Planned phases

1. Project and finance foundations **(current)**
2. Financial data acquisition and quality checks
3. Exploratory analysis and leakage-safe feature engineering
4. Baseline and ML model development
5. Independent model validation and risk backtesting
6. Explainability, monitoring, documentation
7. Grounded financial-report retrieval and RAG
8. Interface, integration, testing
9. GitHub, evidence-based resume, demo
10. Technical interview defense

## Responsible use

- Do not treat predictions as financial recommendations.
- Do not claim meaningful performance before running actual experiments.
- Do not publish or redistribute market data unless permitted by the provider.
- Do not use the final holdout test set for iterative model selection.
