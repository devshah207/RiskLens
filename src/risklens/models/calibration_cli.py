"""Run the fixed development-only probability calibration experiment."""
import argparse

from risklens.models.calibration import CalibrationError, save_calibration_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="RiskLens XGBoost time-separated sigmoid calibration")
    parser.add_argument("csv", help="Validated combined NIFTY 50 OHLC CSV")
    parser.add_argument("--output-dir", default="data/processed/calibration")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_calibration_comparison(args.csv, args.output_dir, overwrite=args.overwrite)
    except (CalibrationError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {len(report['validation_years'])} validation years; "
          f"{report['total_validation_rows']} observations; "
          f"{report['total_validation_events']} downside events")
    for label, metrics in report["pooled_probability_scores"].items():
        print(f"{label}: Brier={metrics['brier_score']:.5f}; log loss={metrics['log_loss']:.5f}")
    print(f"Files saved to: {args.output_dir}")
    print("NOTE: 2025+ holdout sealed. This is repeated development research, not a final test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
