"""CLI: controlled nine-feature versus eleven-feature XGBoost model comparison."""
import argparse

from risklens.models.multifactor_comparison import (
    MultiFactorComparisonError, save_multifactor_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="RiskLens Phase 3.10 multifactor comparison; development only")
    parser.add_argument("nifty_csv", help="Original NIFTY combined OHLC CSV")
    parser.add_argument("external_csv", help="Step 3.9 nifty50_external_development.csv")
    parser.add_argument("--output-dir", default="data/processed/multifactor_comparison")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_multifactor_comparison(
            args.nifty_csv, args.external_csv, args.output_dir, overwrite=args.overwrite,
        )
    except (MultiFactorComparisonError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {report['total_validation_rows']} validation rows, "
          f"{report['total_validation_events']} downside events; 2025+ sealed")
    for model, score in report["pooled_probability_scores"].items():
        print(f"{model}: Brier={score['brier_score']:.5f}, Log loss={score['log_loss']:.5f}")
    print(f"Saved: {args.output_dir}")
    print("NOTE: repeated development analysis; not an independent future performance estimate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
