# Initial Data Contract (Task 2: Structural Validation Implemented)

## Data source

Preferred starting point: manually downloaded historical **NIFTY 50 price-index** series from NSE Indices. Record the source URL, retrieval timestamp (including time zone), data coverage, license/use restrictions, source format/version and file checksum.

Do not claim the raw CSV has been downloaded, tested or licensed at this phase.

## Expected canonical columns

- `date`: trading session date, ISO `YYYY-MM-DD` after parsing; unique and ordered.
- `open`, `high`, `low`, `close`: finite, positive numeric index levels.
- Additional provider columns may be present. They must be documented rather than silently used as features.

Current parser normalizes capitalization and surrounding whitespace of the five required headers. It tolerates extra columns but **ignores** them in validated output (and reports them through the CLI). Accepted dates: `YYYY-MM-DD`, `DD-Mon-YYYY`, `DD Mon YYYY`. Prices may include conventional properly grouped thousands separators (e.g. `25,123.45`). Further provider-specific mappings will be introduced only after inspecting the actual CSV.

## Structural validation requirements (implemented for Phase 1 Task 2)

1. Required columns exist after deliberate column mapping.
2. Dates parse without ambiguity and are unique.
3. Numeric columns parse as finite positive numbers.
4. `low <= min(open, close)` and `high >= max(open, close)` for each row; additionally `low <= high`.
5. Chronological order is enforced in the validated output, and any reorder is reported through the CLI.
6. Invalid records trigger actionable errors, not silent repairs.
7. Market holidays/weekends are not incorrectly interpreted as missing sessions.
8. Missing values and duplicate dates cause validation failure. Statistical outlier detection remains future work; no unapproved forward-fill or row deletion occurs.
9. Preserve raw downloaded files unchanged locally; outputs belong in `data/processed/`.

## Information availability and label policy

Prediction is made after close of session `t` using only information available by then. The future `close_(t+1)` forms the *historical training label*, but is not an input feature. Rolling features must not unintentionally include future rows.

## Data privacy and licensing

Do not commit or redistribute downloaded market data until terms permit. Tests use clearly labelled synthetic fixtures. Never mix synthetic test records with reported real-world model metrics.

## Splitting and experiment reproducibility

Use chronological partitions and record exact date ranges. Fit imputers/scalers and feature selection only within applicable training folds. Keep a final holdout isolated from iterative model selection.

## Important limitations

Validation does **not** authenticate that a CSV came from NSE, detect subtle stale/revised data, know the official trading-calendar schedule, correct corporate actions, or judge whether a statistically extreme but internally consistent price is authentic. Those require additional checks in Phase 2.
