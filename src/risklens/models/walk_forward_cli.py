"""Run fixed-model expanding-window experiments on pre-2025 development data."""

import argparse

from risklens.models.walk_forward import WalkForwardError, save_walk_forward_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="RiskLens: yearly expanding-window development validation")
    parser.add_argument("csv", help="Combined OHLC history, e.g., data/processed/multiyear/nifty50_ohlc.csv")
    parser.add_argument("--output-dir", default="data/processed/walk_forward")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_walk_forward_validation(args.csv, args.output_dir, overwrite=args.overwrite)
    except (WalkForwardError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: Fixed baseline and Logistic Regression across 2020-2024 validation folds")
    for fold in report["folds"]:
        scores = fold["validation_scores"]
        auc = scores["logistic_regression"]["roc_auc"]
        auc_text = f"{auc:.4f}" if auc is not None else "undefined (only one class)"
        print(f"{fold['validation_year']}: train={fold['training_range']['rows']} "
              f"validation={fold['validation_range']['rows']} "
              f"events={fold['validation_range']['events']} LR ROC-AUC={auc_text}")
    for name, score in report["pooled_probability_scores"].items():
        print(f"{name}: pooled Brier={score['brier_score']:.5f}, log loss={score['log_loss']:.5f}")
    print(f"Outputs written under: {args.output_dir}")
    print("NOTE: 2025+ held out; folds are development research, not untouched final tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
