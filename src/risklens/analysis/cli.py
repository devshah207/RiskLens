"""Run development-only NIFTY 50 exploratory analysis from a terminal."""

import argparse

from risklens.analysis.eda import EDAError, save_development_eda


def main() -> int:
    parser = argparse.ArgumentParser(description="RiskLens development-only financial EDA")
    parser.add_argument("csv", help="Combined, validated NIFTY 50 OHLC CSV")
    parser.add_argument("--output-dir", default="data/processed/eda")
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_development_eda(
            args.csv, args.output_dir,
            holdout_start=args.holdout_start,
            rolling_window=args.rolling_window,
            overwrite=args.overwrite,
        )
    except (EDAError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {report['development_price_rows']} development price rows, "
          f"{report['valid_daily_returns']} daily returns")
    print(f"Development coverage: {report['development_first_session']} "
          f"to {report['development_last_session']}")
    print(f"Annualized volatility: {report['annualized_volatility']:.2%}")
    print(f"Development downside close-date events: "
          f"{report['downside_event_close_dates']}/{report['valid_daily_returns']} "
          f"({report['downside_event_fraction']:.2%})")
    print(f"Outputs written under: {args.output_dir}")
    print("NOTE: Holdout outcomes (2025 onward with default cutoff) remain unexamined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
