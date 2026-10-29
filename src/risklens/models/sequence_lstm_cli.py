"""Run the leakage-audited PyTorch LSTM experiment and offline risk dashboard."""

import argparse
from risklens.models.sequence_lstm import save_sequence_benchmark, SequenceModelError


def main():
    p = argparse.ArgumentParser(
        description="RiskLens LSTM vs XGBoost; 2020-2024 development ONLY"
    )
    p.add_argument("nifty_csv")
    p.add_argument("integrated_csv")
    p.add_argument("xgboost_predictions_csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--var-report",
        default=None,
        help="Optional historical_var_report.json to embed in offline dashboard",
    )
    args = p.parse_args()
    try:
        r = save_sequence_benchmark(
            args.nifty_csv,
            args.integrated_csv,
            args.xgboost_predictions_csv,
            args.output_dir,
            overwrite=args.overwrite,
            var_report_path=args.var_report,
        )
    except (SequenceModelError, ValueError, OSError) as exc:
        p.exit(2, f"FAILED: {exc}\n")
    print(
        f"PASS: {r['n_validation_predictions']} development predictions, {r['n_events']} events; 2025+ sealed"
    )
    for key, val in r["pooled_scores"].items():
        print(
            f"{key}: Brier={val['brier_score']:.6f}, logloss={val['log_loss']:.6f}, ROC-AUC={val['roc_auc']:.3f}, AP={val['average_precision']:.3f}"
        )
    print("Open dashboard:", args.output_dir + "/risk_model_validation_dashboard.html")


if __name__ == "__main__":
    main()
