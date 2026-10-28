# Design Decision Log

Record decisions before implementation; revisit only with stated evidence.

| ID | Decision | Reason | Alternatives/risks | Status |
| --- | --- | --- | --- | --- |
| D001 | Start with NIFTY 50 **price** index | One series is manageable for learning market-risk ML | Individual stock corporate actions and survivorship; total return index differs | Proposed |
| D002 | Binary event: next session closing return <= -1% | Clear probability target for classification | Threshold may lead to sparse labels; check development-period prevalence | Provisional |
| D003 | Forecast after session `t` close | Makes information availability explicit | Data publication delays; revise cutoff if source finalization is late | Proposed |
| D004 | Use chronological evaluation | Avoid future-to-past validation leakage | Nonstationarity; consider rolling/walk-forward tests | Proposed |
| D005 | Constant-rate baseline followed by logistic regression | Baseline and interpretable learned model | Tree ensembles may capture nonlinear patterns but overfit | Proposed |
| D006 | Keep VaR separate from classification probability | They represent different risk questions | Requires independent design and backtesting | Proposed |
| D007 | Download raw index CSV manually at first | Keeps early setup reliable and inspectable | Later source adapter can automate with permission | Proposed |
| D008 | Task 1 = documentation + importable package only | Prevents unreviewable feature scope and fabricated results | Task 2 extends the foundation with validation | Accepted |
| D009 | Reject malformed CSV observations rather than silently repair them | Prevents undocumented changes to market data | Fails early; requires explicit investigation and correction | Implemented |
| D010 | Use synthetic fixture for automated tests | Allows repeatable testing without licensing real market data | Synthetic prices do not establish market authenticity or predictive validity | Implemented |

## Open questions for later phases

- Confirm actual NSE CSV schema, available coverage and redistribution terms.
- Measure event frequency on development data before freezing evaluation protocol.
- Define chronological date boundaries for training, validation and holdout.
- Evaluate which additional features are supported by reliable timely data.
- Choose the financial reports and RAG evaluation set for StockGro relevance.
