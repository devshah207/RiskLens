# RiskLens — Frozen Model and Evidence Record (review-only)

**Freeze record prepared:** 30 October 2026  
**GitHub status:** Review-branch publication approved by owner; not merged into `main`. Final governance approval remains pending.  
**Baseline main commit:** `8f519527acc742ad38d84a13f64404b8f7b99451`  
**Scope:** Freeze **model selection, feature specification, hyperparameters, holdout result identity, and evidence hashes**. This is NOT a new training run, new final evaluation, deployment approval, or archive of fitted model weights.

## Frozen specification

- **Champion:** Fixed XGBoost (`fixed_xgboost_11`), original 11 features, 80 estimators, depth 2, learning rate 0.04, min child weight 10, gamma 1, subsample 0.8, column subsample 0.9, L2 10, L1 0, seed 42. See exact parameters in `PHASE6_MODEL_FREEZE_MANIFEST.json`.
- **Challenger:** Actual Optuna LSTM **Trial #86**, original 11 features, 45 trading-session sequences, hidden size 24, 2 layers, dropout 0.3413771488170494, learning rate 0.0027467789382135, weight decay 0.0006684358163984, batch size 128, 24 epochs; original holdout fit seed **4051**. This is NOT the early hand-built LSTM.
- **Train/validation boundary:** Training observations strictly before `2025-01-01`; year-boundary label purge preserved.
- **Original locked final holdout:** 433 original NIFTY prediction dates, 2025 to September 2026. No retuning, re-scoring, post-hoc feature choice, threshold adjustment or candidate switch.

| Original final holdout measure | XGBoost champion | Optuna LSTM #86 challenger |
|---|---:|---:|
| ROC-AUC | 0.672756 | 0.684874 |
| Average precision | 0.186103 | 0.218973 |
| Brier score | 0.084773 | 0.084299 |
| Log loss | 0.303295 | 0.300362 |

The difference in original holdout scores is observational; it does **not** authorize a post-hoc champion switch. Both classifiers produced no positive alerts at 0.5. Historical 95% VaR failed the independence and conditional-coverage research tests under their assumptions; VaR is a separate method, not a third classifier.

## Research disposition

- Three-feature audits: EMA-50 and RSI-14 formulas independently agree; India VIX 2024 source-overlap values agree. Drifts are investigated, not proven harmless.
- Ablations and regime-context variants: exploratory. No reliable feature-deletion benefit.
- Seven-seed Trial #86 robustness: regime-context benefit did not persist consistently; original 11-feature Trial #86 remains the challenger.
- LSTM monitoring score PSI 0.4702: explanation remains limited by comparisons across different annual-fitted development models versus one holdout fitted model. Additional **saved-prediction-only** diagnostic may be appropriate; no re-scoring.
- Phase 6 risk-register/documentation approval still pending; no live trading/banking approval.

## Reproducibility and what is actually frozen

The JSON manifest preserves **exact original frozen parameters**, SHA-256 fingerprints from the original one-time final-holdout report and the monitoring report, and the original Git reference. The read-only verifier (`scripts/verify_phase6_freeze.py`) checks actual bytes from your local project against these fingerprints and rechecks the original holdout report's model specifications. It does **not** fit, train, change, or score any models.

**Important limitation:** The original final-holdout pipeline saved prediction and report artifacts, not a serialized fitted XGBoost estimator or LSTM state dictionary. Therefore we can freeze the specification and historical evidence now, but cannot truthfully claim the exact original trained weight objects have been archived. Rebuilding those weights later would constitute a separate refit, not recovery of the identical original instance, particularly on MPS.

## File placement

- `docs/RiskLens_Phase6_Review_Draft.md` — owner-review scientific report, still a draft.
- `data/processed/phase6_governance/RiskLens_VaR_Independent_Reconstruction.csv` — research-only source evidence; this folder is ignored by Git.
- `docs/PHASE6_MODEL_FREEZE_RECORD.md` — this governance decision record.
- `docs/PHASE6_MODEL_FREEZE_MANIFEST.json` — frozen configuration and SHA-256 evidence.
- `scripts/verify_phase6_freeze.py` — read-only verification, standard library only.

From the RiskLens root run:

```bash
python scripts/verify_phase6_freeze.py
```

The command reports PASS only if the original files are present and byte-identical to those used in the studies; otherwise it reports missing/mismatched files without modifying anything. If your local data lives at different paths, move/copy those *original* files to their established locations rather than changing hashes. See the JSON for all exact paths.

## Publication gate

**Owner approved publication to a separate review branch only.** No merge into `main` is authorized. Obtain separate explicit owner approval for any merge or later GitHub push.
