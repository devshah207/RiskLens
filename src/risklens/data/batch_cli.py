"""CLI for importing multiple annual NIFTY 50 CSVs without external APIs."""

import argparse

from risklens.data.batch_ingestion import import_nifty50_batch
from risklens.data.ingestion import ImportErrorData


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine, validate and audit annual NIFTY 50 CSVs")
    parser.add_argument("input_dir", help="Directory holding annual NIFTY 50 .csv downloads")
    parser.add_argument("--output-dir", default="data/processed/multiyear")
    parser.add_argument("--source-page", help="Declared web source, recorded but not independently verified")
    parser.add_argument("--assume-nifty50", action="store_true", help="Confirm manually if CSVs omit Index Name")
    parser.add_argument("--holdout-start", default="2025-01-01", help="Predeclared final holdout cutoff, YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly overwrite earlier batch outputs")
    args = parser.parse_args()
    try:
        report, events = import_nifty50_batch(
            args.input_dir, args.output_dir,
            source_page=args.source_page,
            assume_nifty50=args.assume_nifty50,
            holdout_start=args.holdout_start,
            overwrite=args.overwrite,
        )
    except ImportErrorData as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"PASS: {report['number_of_files']} files, {report['final_unique_trading_sessions']} unique trading sessions")
    print(f"Coverage: {report['first_session']} to {report['last_session']}")
    print(f"Development events (before {events['holdout_starts_on']}): "
          f"{events['downside_event_days']}/{events['eligible_prediction_days']} "
          f"({events['event_rate']:.2%})")
    print(f"Outputs written under: {args.output_dir}")
    for warning in report["warnings"]:
        print(f"NOTE: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
