"""CLI to audit existing (not refitted) Step 3.5 development predictions."""

import argparse

from risklens.models.diagnostics import DiagnosticsError, save_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit existing RiskLens walk-forward probabilities (2020–2024 only)")
    parser.add_argument("predictions_csv", help="Step 3.5 xgboost_walk_forward_predictions.csv")
    parser.add_argument("--output-dir", default="data/processed/model_diagnostics")
    parser.add_argument("--example-count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = save_diagnostics(args.predictions_csv, args.output_dir,
                                  overwrite=args.overwrite, example_count=args.example_count)
    except (DiagnosticsError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    baseline = report["pooled_scores"]["constant_baseline"]
    xgb3 = report["pooled_scores"]["xgboost_3_features"]
    print(f"PASS: {report['rows']} development predictions ({report['observed_events']} downside events)")
    print(f"Years: {report['years'][0]}–{report['years'][-1]} (2025+ holdout refused)")
    print(f"Pooled Brier score: baseline={baseline['brier_score']:.5f}, XGBoost-3={xgb3['brier_score']:.5f}")
    print(f"Outputs written under: {args.output_dir}")
    print("NOTE: Retrospective diagnostics only; no model fitted or parameters changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
