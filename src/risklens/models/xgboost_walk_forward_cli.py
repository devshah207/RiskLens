"""Terminal entry point for the fixed XGBoost walk-forward experiment."""
import argparse

from risklens.models.xgboost_walk_forward import XGBComparisonError, save_xgboost_walk_forward


def main() -> int:
    parser = argparse.ArgumentParser(description="RiskLens: fixed-parameter XGBoost vs baselines (development only)")
    parser.add_argument("csv", help="Combined NIFTY 50 OHLC CSV")
    parser.add_argument("--output-dir", default="data/processed/xgboost_comparison")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_xgboost_walk_forward(args.csv, args.output_dir, overwrite=args.overwrite)
    except (XGBComparisonError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {report['total_validation_rows']} development validation rows, {report['total_validation_events']} events")
    print("Pooled Brier score (lower is better):")
    for label, values in report["pooled_probability_scores"].items():
        print(f"  {label}: {values['brier_score']:.5f}")
    print(f"Files saved to: {args.output_dir}")
    print("NOTE: 2025+ final holdout not evaluated; no parameter tuning or model selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
