"""Safety checks for the optional Pass 5 deletion helper."""

from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "cleanup_pass5",
    Path(__file__).resolve().parents[1] / "scripts" / "cleanup_pass5.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_removal_targets_are_only_legacy_cli_wrappers():
    assert len(module.LEGACY) == 5
    assert all(
        p.startswith("src/risklens/models/") and p.endswith("_cli.py")
        for p in module.LEGACY
    )
    assert all(
        "sequence_lstm" not in p and "market_risk_monitor" not in p
        for p in module.LEGACY
    )


def test_detects_static_references(tmp_path):
    f = tmp_path / "src" / "risklens" / "models"
    f.mkdir(parents=True)
    (f / "user.py").write_text("from risklens.models.diagnostics_cli import main\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    assert module.check_references(tmp_path)


def test_allows_no_references(tmp_path):
    f = tmp_path / "src" / "risklens"
    f.mkdir(parents=True)
    (f / "other.py").write_text(
        "from risklens.models.diagnostics import save_diagnostics\n"
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    assert not module.check_references(tmp_path)
