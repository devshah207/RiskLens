"""CLI for conservative, development-only India VIX and S&P 500 integration."""

import argparse
from risklens.data.external_integration import (
    ExternalDataError,
    save_external_development,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join earlier-date India VIX and S&P 500 to development-only NIFTY 50"
    )
    parser.add_argument("nifty_csv", help="Combined validated NIFTY OHLC CSV")
    parser.add_argument("vix_dir", help="Directory of annual India VIX CSV files")
    parser.add_argument("sp500_csv", help="Historical S&P 500 OHLC CSV")
    parser.add_argument("--output-dir", default="data/processed/external")
    parser.add_argument("--max-age-days", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_external_development(
            args.nifty_csv,
            args.vix_dir,
            args.sp500_csv,
            args.output_dir,
            max_age_days=args.max_age_days,
            overwrite=args.overwrite,
        )
    except (ExternalDataError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {report['nifty_development_sessions']} pre-2025 NIFTY sessions")
    print(f"India VIX matches: {report['nifty_rows_with_vix']}")
    print(f"Prior US return matches: {report['nifty_rows_with_sp500_return']}")
    print(
        f"Rows with both external features: {report['nifty_rows_with_both_external_features']}"
    )
    print(
        f"Quarantined VIX zero-OHL records: {report['vix_source_audit']['zero_ohl_count']}"
    )
    print(f"Output directory: {args.output_dir}")
    print(
        "CAUTION: Strictly earlier source dates only; no same-date external observations, no holdout analysis, no model trained."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
