"""Run fixed 6-vs-9 XGBoost comparison without final-holdout analysis."""
import argparse
from risklens.models.technical_comparison import TechnicalComparisonError, save_technical_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="RiskLens Phase 3.8 technical indicators comparison")
    parser.add_argument("csv", help="Existing validated NIFTY 50 combined OHLC CSV")
    parser.add_argument("--output-dir", default="data/processed/technical_comparison")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_technical_comparison(args.csv, args.output_dir, overwrite=args.overwrite)
    except (TechnicalComparisonError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {report['total_validation_rows']} development rows, {report['total_validation_events']} downside events")
    for name, scores in report["pooled_probability_scores"].items():
        print(f"{name}: Brier={scores['brier_score']:.5f}, Log loss={scores['log_loss']:.5f}")
    print(f"Files written under: {args.output_dir}")
    print("NOTE: 2025+ holdout not evaluated. Results are exploratory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
