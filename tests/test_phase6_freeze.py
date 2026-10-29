"""Synthetic-only regression tests for the read-only Phase 6 freeze verification."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_phase6_freeze.py"
OPTUNA = "data/processed/optuna_search/optuna_studies.sqlite3"
HOLDOUT = "data/processed/phase5_final_holdout/final_holdout_report.json"


def _make_fixture(tmp_path):
    db_hash = sha256(b"synthetic-optuna-study").hexdigest()
    champion = {"name": "fixed_xgboost_11", "features": ["feature_a"],
                "parameters": {"n_estimators": 80}, "holdout_metrics": {"roc_auc": 0.63}}
    challenger = {"name": "lstm_optuna_trial86", "features": ["feature_a"],
                  "parameters": {"lookback": 45}, "holdout_metrics": {"roc_auc": 0.64},
                  "trial_db_sha256": db_hash, "trial_number": 86, "seed": 4051}
    report = {
        "selection_remains_frozen": True,
        "selected_before_opening": champion["name"],
        "features": champion["features"],
        "frozen_xgb_params": champion["parameters"],
        "frozen_lstm_params": challenger["parameters"],
        "lstm_seed": challenger["seed"],
        "study_record": {"trial_number": 86, "database_sha256": db_hash},
        "eligible_holdout_rows": 2,
        "training_cutoff_exclusive": "2025-01-01",
        "model_scores": {"xgboost_fixed": champion["holdout_metrics"],
                         "lstm_optuna_trial86": challenger["holdout_metrics"]},
    }
    db = tmp_path / OPTUNA
    db.parent.mkdir(parents=True)
    db.write_bytes(b"synthetic-optuna-study")
    report_file = tmp_path / HOLDOUT
    report_file.parent.mkdir(parents=True)
    report_file.write_text(json.dumps(report), encoding="utf-8")
    manifest = {
        "protected_file_sha256": {OPTUNA: db_hash, HOLDOUT: sha256(report_file.read_bytes()).hexdigest()},
        "champion": champion, "challenger": challenger,
        "holdout": {"rows": 2},
        "development_training_cutoff_exclusive": "2025-01-01",
    }
    manifest_path = tmp_path / "docs" / "PHASE6_MODEL_FREEZE_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True)
    _write(manifest_path, manifest)
    return manifest_path, manifest, report_file, report


def _write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _verify(root):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root)],
                          capture_output=True, text=True, check=False)


def test_freeze_accepts_consistent_synthetic_evidence(tmp_path):
    _make_fixture(tmp_path)
    result = _verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 2 freeze-evidence hashes match" in result.stdout


def test_freeze_rejects_conflicting_optuna_manifest_fingerprints(tmp_path):
    manifest_path, manifest, _, _ = _make_fixture(tmp_path)
    manifest["protected_file_sha256"][OPTUNA] = "0" * 64
    _write(manifest_path, manifest)
    result = _verify(tmp_path)
    assert result.returncode != 0
    assert "MANIFEST INCONSISTENCY" in result.stdout


def test_freeze_rejects_missing_holdout_report(tmp_path):
    _manifest_path, _manifest, report_file, _report = _make_fixture(tmp_path)
    report_file.unlink()
    result = _verify(tmp_path)
    assert result.returncode != 0
    assert "MISSING: data/processed/phase5_final_holdout/final_holdout_report.json" in result.stdout


def test_freeze_rejects_disagreeing_holdout_optuna_fingerprint(tmp_path):
    manifest_path, manifest, report_file, report = _make_fixture(tmp_path)
    report["study_record"]["database_sha256"] = "0" * 64
    _write(report_file, report)
    manifest["protected_file_sha256"][HOLDOUT] = sha256(report_file.read_bytes()).hexdigest()
    _write(manifest_path, manifest)
    result = _verify(tmp_path)
    assert result.returncode != 0
    assert "optuna_study_sha256" in result.stdout
