"""Small command-line interface for validating a historical OHLC CSV."""

import argparse

from risklens.data.validation import OHLCValidationError, load_ohlc_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate daily OHLC data (no market data is downloaded)."
    )
    parser.add_argument("csv_path", help="Path to a local OHLC CSV file")
    args = parser.parse_args()

    try:
        validated = load_ohlc_csv(args.csv_path)
    except OHLCValidationError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"PASS: {len(validated)} valid trading-day rows")
    print(
        "Coverage: "
        f"{validated['date'].min().date()} to {validated['date'].max().date()}"
    )
    if validated.attrs["reordered"]:
        print("NOTE: Source rows were out of order and were sorted chronologically.")
    if validated.attrs["ignored_columns"]:
        print(f"NOTE: Extra columns not used: {validated.attrs['ignored_columns']}")
    print("No real-world accuracy or source authenticity is implied by PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
