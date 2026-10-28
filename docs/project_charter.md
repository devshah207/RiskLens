# RiskLens — Project Charter

## Intended audience and motivation

An educational project showing how to build and validate financial machine-learning models and later ground financial-research answers in public documents. Relevant technical skills include Python, statistical modeling, model validation, time-series data, explainability, and retrieval evaluation.

## Initial scope

- **Instrument:** NIFTY 50 **price** index, not an individual stock and not the Total Returns Index.
- **Forecast origin:** Immediately after trading session `t` closes, assuming official close data is available and finalized.
- **Prediction horizon:** Close-to-close return over the **next trading session**, not the next calendar day.
- **Target:** `y_t = 1` when `Close_(t+1)/Close_t - 1 <= -0.01`, otherwise `0`.
- **Output:** Estimated probability of that downside event, not the size of a decline and not a direct trading signal.
- **Risk domain:** Market downside risk; not a credit-default probability.

## Why begin with an index?

A single index series simplifies the first version's data quality, chronology, feature definition, and evaluation. Future company-level extensions require additional attention to corporate actions, universe selection, and survivorship bias.

## Planned modeling and validation

- Begin with a constant event-frequency baseline based exclusively on the appropriate training period.
- Try logistic regression, then only justified tree-based models.
- Use chronological development and holdout periods. For tuning, use development-period validation or walk-forward splits.
- Evaluate discrimination (e.g. PR-AUC, ROC-AUC), probability quality (e.g. Brier score and calibration), and relevant failure modes.
- Check event prevalence before finalizing the modeling protocol. Avoid revising targets based on final holdout results.
- Keep Value at Risk (VaR) as a **separate** future risk measure with its own definition and backtesting.

## Research assistant (later phase)

Retrieve selected company reports, cite document passages, and evaluate answers for source support and retrieval quality. Do not imply an LLM explains the causal mechanism behind statistical risk forecasts.

## Non-goals

- No production brokerage integration, live trading, or financial recommendations.
- No claims of regulatory compliance or guaranteed predictive performance.
- No private client banking data or credit-risk scoring in the initial version.
- No model training, scraping, market-data ingestion, or LLM features during Phase 1 Task 1.

## Principal risks and limitations

Financial patterns can change over time; rare-event targets may have little data; source series can be revised; market data rights constrain sharing; inaccurate feature timestamps may create look-ahead leakage; a good backtest does not guarantee future performance.
