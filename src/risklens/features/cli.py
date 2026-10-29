"""Command-line interface for development-only feature/target preparation."""

import argparse

from risklens.features.dataset import FeatureDatasetError, save_development_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build leakage-aware RiskLens development features"
    )
    parser.add_argument("csv", help="Validated combined NIFTY 50 OHLC file")
    parser.add_argument("--output-dir", default="data/processed/features")
    parser.add_argument("--validation-start", default="2023-01-01")
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_development_features(
            args.csv,
            args.output_dir,
            validation_start=args.validation_start,
            holdout_start=args.holdout_start,
            overwrite=args.overwrite,
        )
    except (FeatureDatasetError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    tr, va = report["train"], report["validation"]
    print(f"PASS: train={tr['rows']} rows, validation={va['rows']} rows")
    print(
        f"Training events: {tr['downside_events']}/{tr['rows']} ({tr['event_frequency']:.2%})"
    )
    print(
        f"Validation events: {va['downside_events']}/{va['rows']} ({va['event_frequency']:.2%})"
    )
    print(
        f"Training boundary labels purged: {report['training_boundary_labels_purged']}"
    )
    print(f"Files saved to: {args.output_dir}")
    print("NOTE: No model was fitted; final holdout outcomes were not examined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
