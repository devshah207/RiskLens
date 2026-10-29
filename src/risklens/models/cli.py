"""Evaluate a fixed baseline and logistic classifier without touching final holdout."""

import argparse

from risklens.models.baselines import ModelInputError, save_comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RiskLens training-only baseline and logistic validation"
    )
    parser.add_argument(
        "--train-csv", default="data/processed/features/train_features.csv"
    )
    parser.add_argument(
        "--validation-csv", default="data/processed/features/validation_features.csv"
    )
    parser.add_argument("--output-dir", default="data/processed/models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_comparison(
            args.train_csv,
            args.validation_csv,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (ModelInputError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    tr, val = report["training"], report["validation"]
    print(f"PASS: trained on {tr['rows']} rows ({tr['events']} downside events)")
    print(f"Validation: {val['rows']} rows ({val['events']} downside events)")
    for model_name, scores in report["validation_scores"].items():
        auc = "undefined" if scores["roc_auc"] is None else f"{scores['roc_auc']:.4f}"
        ap = (
            "undefined"
            if scores["average_precision"] is None
            else f"{scores['average_precision']:.4f}"
        )
        print(
            f"{model_name}: ROC-AUC={auc}, AP={ap}, "
            f"Brier={scores['brier_score']:.5f}, LogLoss={scores['log_loss']:.5f}"
        )
    print(f"Files saved under: {args.output_dir}")
    print(
        "NOTE: Results describe a single historical validation period, not the final holdout or trading skill."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
