"""Run the controlled 3-feature vs 6-feature walk-forward comparison."""
import argparse

from risklens.models.ohlc_comparison import OHLCComparisonError, save_ohlc_feature_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare NIFTY 50 3-feature and 6-feature Logistic Regression")
    parser.add_argument("csv", help="Combined, validated NIFTY 50 OHLC CSV")
    parser.add_argument("--output-dir", default="data/processed/ohlc_comparison")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_ohlc_feature_comparison(args.csv, args.output_dir, overwrite=args.overwrite)
    except (OHLCComparisonError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {len(report['folds'])} walk-forward years, {report['total_validation_rows']} validation observations")
    print("Pooled Brier score (lower is better):")
    for key, scores in report["pooled_probability_scores"].items():
        print(f"  {key}: {scores['brier_score']:.5f}")
    print(f"Results saved under: {args.output_dir}")
    print("NOTE: Development-only results; 2025+ final holdout remains sealed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
