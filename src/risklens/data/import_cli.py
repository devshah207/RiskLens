"""Command-line interface for importing an existing NIFTY 50 CSV file."""

import argparse

from risklens.data.ingestion import ImportErrorData, import_nifty50_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a manually downloaded NIFTY 50 CSV. No network requests."
    )
    parser.add_argument("input_csv", help="Path to the original CSV file")
    parser.add_argument(
        "--output-dir", default="data/processed", help="Directory for validated CSV and report"
    )
    parser.add_argument(
        "--source-page", help="Page where you obtained the data (recorded, not verified)"
    )
    parser.add_argument(
        "--assume-nifty50", action="store_true",
        help="Only if the CSV lacks an index column and you verified its identity manually",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly allow overwriting earlier outputs"
    )
    args = parser.parse_args()

    try:
        report = import_nifty50_csv(
            args.input_csv,
            args.output_dir,
            source_page=args.source_page,
            assume_nifty50=args.assume_nifty50,
            overwrite=args.overwrite,
        )
    except ImportErrorData as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"PASS: Imported {report['rows_imported']} NIFTY 50 trading sessions")
    print(f"Coverage: {report['first_session']} to {report['last_session']}")
    print(f"Saved files under: {args.output_dir}")
    print("CAUTION: This result verifies data structure, not market-data authenticity.")
    for item in report["warnings"]:
        print(f"NOTE: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
