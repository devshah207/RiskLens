# Phase 5 — One-time locked 2025–2026 final holdout

**Implementation status:** Added to a dedicated code path with synthetic-only tests. Do **not** confuse this with a completed real-data independent evaluation. Real results can be produced only on your local, original market sources, after predeclared checks.

## Frozen specification (established before opening)

- **Champion candidate:** fixed 11-feature XGBoost, unchanged `FIXED_XGB_PARAMS`.
- **Challenger:** true Optuna LSTM trial #86 from saved SQLite study; 45-session history, 2 layers, 24 hidden, dropout 0.3413771488170494, learning rate 0.0027467789382135, weight decay 0.0006684358163984, batch 128, 24 epochs.
- Task: next-session NIFTY 50 price-index close decline of at least 1%, predicted after NIFTY trading day t closes.
- **Train only on pre-2025 observations**, purge the last training date if its next-day label crosses the 2025 boundary. No 2025+ labels or model-selection outcomes enter fitting.
- One-time out-of-time window: from 2025-01-01 through 2026-09-30 source dates. A prediction date needs a known next observed NIFTY close **within** the window: Sep 30, 2026 is not itself labeled when the file ends on that date.
- Max external source age: 7 calendar days, strictly earlier VIX and S&P 500 observation dates (never same-day), no future filling. LSTM histories must include all 45 complete external sessions or that prediction date is excluded for **both** candidates.
- Primary score: paired Brier; descriptive secondary metrics log loss, ROC-AUC, average precision, annual stability and fixed-bin reliability. Historical 250-session VaR 95% and 99% exception monitoring is **separate** from the ML classifiers.
- Maintain the preselected XGBoost candidate after this test. Do not retune or select a model using these observed outcomes. Record failures and limitations without changing the protocol.

## Required source files: collect before unlocking

The repository already has the full NIFTY price history through Sep 30, 2026, and the **pre-2025** integrated external feature file. But **it does not contain verified India VIX and S&P 500 2025–2026 sources**. The old pre-2025 external CSV is *insufficient* for this phase.

Prepare two local CSVs (not committed to Git):

1. **India VIX**: `date,open,high,low,close`, containing at least December 2024 context and 2025-01-01 through 2026-09-30 available observations. NSE annual files may be combined into one chronological CSV. Original structural OHLC checks are applied; only the predefined Open=High=Low=0 with positive close anomaly is quarantined with its count recorded. Do not add new discretionary cleaning rules after observing the holdout.
2. **S&P 500 index**: `date,open,high,low,close` with at least December 2024 context through Sep 30, 2026. The return at a US session is its close divided by the previous US close minus one, and only a **strictly earlier US session date** can be used to predict NIFTY t.

Use dates as ISO `YYYY-MM-DD` or the established accepted date formats. Keep raw data source provenance/URLs and license considerations. The holdout command refuses newer-than-Sep-2026 raw observations so the evaluated period is fixed.

## One-time command

**Do not run this until the external data are complete and verified.** The command is intentionally not included in `risklens demo` or other development automation.

```bash
git switch main
git pull --ff-only origin main
python -m pip install -e '.[dev,deep,tuning]'
python -m pytest -q
ruff check src tests scripts --select F,E4,E7,E9

# Replace these with your real sourced CSV paths:
risklens evaluate-holdout \
  --vix data/raw/india_vix_2024_2026.csv \
  --sp500 data/raw/sp500_2024_2026.csv \
  --device mps \
  --confirm-locked-protocol
```

The full report is written only once to `data/processed/phase5_final_holdout/`; a nonempty existing output directory is rejected. There is no `--overwrite` option, and the opt-in flag is mandatory. Never use a new output location to repeatedly inspect results and adjust model decisions.

Outputs:
- `final_holdout_predictions.csv`: aligned one-time probabilities, dates and labels.
- `final_holdout_reliability.csv`: fixed-bin reliability with bin-size warnings.
- `final_holdout_error_cases.csv`: retrospective low-probability realized events and high-probability non-events, for both frozen candidates.
- `final_holdout_report.json`: frozen parameters, study/source checksums, counts, yearly metrics, calibration, 95%/99% VaR coverage and independence checks.
- `final_holdout_verdict.md`: human-readable report with limitations.

## Gate for marking Phase 5 complete

1. GitHub Actions unit and lint checks pass.
2. Market data sources are complete, structurally validated and time-aligned.
3. The holdout is evaluated **once**, without reopening, retuning or model switching.
4. The full report includes both successes and failures, year-wise performance, calibration, error/failure cases, and VaR breach statistics.
5. Results and limitations are integrated into the final validation memo. No fabricated metrics.

The technical implementation is only one part of this phase; if inputs are missing, keep Phase 5 marked **incomplete** and do not pretend an independent result exists.
