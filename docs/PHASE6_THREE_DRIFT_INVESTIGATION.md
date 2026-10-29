# Phase 6 — Investigation of the 3 Flagged Feature Drifts

## Purpose

**Investigate first** before claiming the 3 PSI flags are market regime changes. Neither 'drift' nor 'PSI > 0.20' by itself provides causal explanation. This analysis is strictly retrospective on the **already-scored** 2025–Sep 2026 holdout dates, and does not retrain/reselect either model.

The three flagged features from the user's real local run are `ema_50_distance`, `india_vix_level`, and `rsi_14`.

## One command on the actual local data

```bash
git switch main
git pull --ff-only origin main
risklens investigate-drift \
  --vix data/processed/holdout_sources/india_vix_2024_2026.csv \
  --sp500 data/raw/sp500/sp500_daily.csv
```

**Use the existing files.** No extra Optuna runs, no model refit, no data restructuring, no rerun of the one-time holdout.

## Investigation design

1. **Formula integrity:** Recompute the entire EMA50 (recursive adjust=False) and Wilder RSI14 on original NIFTY OHLC closes *independently*; compare every reconstructed development and scored-holdout feature within tolerance; reject discrepancy or RSI outside 0–100.
2. **VIX source integrity:** Cross-compare 2024 original integrated development VIX source dates and values against the newly combined 2024–2026 VIX CSV, using the original strict prior-date / 7-day rule. Report overlap count, conflicting dates, level differences and quarantined zero-OHL anomalies. A failed overlap is a **data-source issue requiring review before making market interpretations.**
3. **VIX lag:** For each already-scored 2025/2026 NIFTY date, verify VIX source date is strictly earlier, not older than 7 calendar days; report actual lag distribution. This does not claim exchange publication timestamps are verified.
4. **Reference-year sensitivity:** Compare distribution against the original all-2016–2024 baseline, the actual 2020–2024 development window, and recent 2023–2024; also separate 2025 from 2026. Quantile-bin PSI uses the *reference only*.
5. **Sustained vs extreme shifts:** Report 2025–2026 monthly medians, how many lie outside 2020–2024 IQR, 10 dates with greatest/least values per flagged feature, tail shares beyond baseline 10th/90th percentiles.
6. **Specific finance diagnostics:** EMA50 distance >0 rate (price above EMA); India VIX level >30 rate; RSI below 30 and above 70 rates. These are descriptive context, not calibrated model thresholds.
7. **Retrospective outcomes:** Stratify each flagged feature into below-baseline-P10 / middle / above-P90 in 2025 and 2026, reporting n, events, event rates, mean frozen model probabilities and Brier by model; mark sparse samples. **Never change model selection, features or hyperparameters based on these slices.**

## Outputs

`data/processed/phase6_drift_investigation/`:

- `three_feature_investigation.md`: primary written evidence, cross-period table and integrity results
- `three_feature_investigation.json`: source hashes, full period summaries, formula/source checks, limitations
- `three_feature_periods.csv`: all baselines and both calendar years per feature, median/percentiles/PSI, tail fractions, indicator regime proportions
- `three_feature_months.csv`: monthly medians, tail proportions and persistence flags
- `three_feature_extremes.csv`: most extreme date/feature pairs and VIX observation-age details
- `three_feature_risk_slices.csv`: outcome-conditioned diagnostics, small-sample flags and both candidates' frozen scores

**Completion gate:** We must inspect actual output values and source overlap before declaring any of the three drifts explained. Do not pretend data are available on GitHub: these are local, ignored files.

Next: verify actual findings and write precise feature-by-feature governance conclusions before resuming unrelated phases.
