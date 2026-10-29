#!/usr/bin/env python3
"""Read-only verification of RiskLens Phase 6 frozen config and evidence.

No model training, feature generation, predictions, score calculations, or file mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="RiskLens repository root (default: parent of scripts/)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "docs/PHASE6_MODEL_FREEZE_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"FAIL: Cannot read freeze manifest: {exc}")
        return 1

    failures = []
    checked = 0
    for relative, expected in manifest["protected_file_sha256"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"MISSING: {relative}")
            continue
        hasher = hashlib.sha256()
        with path.open("rb") as inp:
            for chunk in iter(lambda: inp.read(1048576), b""):
                hasher.update(chunk)
        found = hasher.hexdigest()
        if found != expected:
            failures.append(
                f"SHA256 MISMATCH: {relative} expected {expected}, got {found}"
            )
        else:
            checked += 1

    report_path = root / "data/processed/phase5_final_holdout/final_holdout_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            contracts = {
                "selection_remains_frozen": report["selection_remains_frozen"] is True,
                "champion": (
                    report["selected_before_opening"] == manifest["champion"]["name"]
                ),
                "features": (
                    report["features"]
                    == manifest["champion"]["features"]
                    == manifest["challenger"]["features"]
                ),
                "xgboost_parameters": (
                    report["frozen_xgb_params"] == manifest["champion"]["parameters"]
                ),
                "lstm_parameters": (
                    report["frozen_lstm_params"] == manifest["challenger"]["parameters"]
                ),
                "lstm_seed": report["lstm_seed"] == manifest["challenger"]["seed"],
                "lstm_trial": (
                    report["study_record"]["trial_number"]
                    == manifest["challenger"]["trial_number"]
                ),
                "holdout_rows": (
                    report["eligible_holdout_rows"] == manifest["holdout"]["rows"]
                ),
                "training_cutoff": (
                    report["training_cutoff_exclusive"]
                    == manifest["development_training_cutoff_exclusive"]
                ),
                "champion_metrics": (
                    report["model_scores"]["xgboost_fixed"]
                    == manifest["champion"]["holdout_metrics"]
                ),
                "challenger_metrics": (
                    report["model_scores"]["lstm_optuna_trial86"]
                    == manifest["challenger"]["holdout_metrics"]
                ),
            }
            failed_contracts = [
                name for name, passed in contracts.items() if not passed
            ]
            if failed_contracts:
                failures.append(
                    "MODEL FREEZE CONTRACT MISMATCH: " + ", ".join(failed_contracts)
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(
                f"MODEL FREEZE CONTRACT MISMATCH: {type(exc).__name__}: {exc}"
            )

    if failures:
        print(
            "FAIL: Freeze evidence verification did not pass; no files were modified."
        )
        for problem in failures:
            print(" -", problem)
        print(
            f"{checked}/{len(manifest['protected_file_sha256'])} file hashes matched"
        )
        return 1

    print(
        f"PASS: {checked} freeze-evidence hashes match, champion/challenger "
        "specifications and original holdout results match."
    )
    print(
        "This verifies existing files and model selection; it does not archive "
        "fitted model weights or authorize deployment."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
