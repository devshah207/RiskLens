"""Offline no-Streamlit-installed smoke via a tiny inert UI fake.

This checks Python execution paths, NOT actual visual rendering or installed
Streamlit compatibility; the Mac must run real Streamlit separately.
"""
from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path
from types import ModuleType


class UIStub(ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __getattr__(self, name):
        if name == "sidebar":
            return self
        if name == "cache_data":
            return lambda **kwargs: lambda fn: fn
        if name == "columns":
            return lambda n, **kwargs: [self] * (n if isinstance(n, int) else len(n))
        if name == "tabs":
            return lambda options, **kwargs: [self] * len(options)
        if name in {"expander", "container", "form", "spinner"}:
            return lambda *args, **kwargs: self
        if name == "radio":
            return lambda label, options, index=0, **kwargs: options[index]
        if name == "selectbox":
            return lambda label, options, index=0, **kwargs: list(options)[index]
        if name == "text_input":
            return lambda label, value="", **kwargs: value
        if name == "text_area":
            return lambda label, value="", **kwargs: value
        if name in {"button", "checkbox", "form_submit_button"}:
            return lambda *args, **kwargs: False
        if name == "stop":
            return lambda: (_ for _ in ()).throw(AssertionError("Unexpected Streamlit stop"))
        return lambda *args, **kwargs: None


def test_all_three_pages_execute_without_api_or_saved_files(monkeypatch):
    fake = UIStub()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    root = Path(__file__).resolve().parents[1]
    for path in [root / "app/Home.py", root / "app/pages/1_Market_Risk.py",
                 root / "app/pages/2_Financial_Research.py"]:
        runpy.run_path(str(path), run_name="__main__")


def test_market_page_never_imports_holdout_evaluator_or_training_code():
    root = Path(__file__).resolve().parents[1]
    for file in (root / "app/pages/1_Market_Risk.py",
                 root / "app/services/saved_risk_evidence.py"):
        source = file.read_text(encoding="utf-8")
        assert "import risklens" not in source
        assert "from risklens" not in source
        assert "evaluate_holdout(" not in source
        assert "model.fit(" not in source


def test_market_page_still_renders_when_original_private_outputs_are_absent(monkeypatch, tmp_path):
    """Clean-checkout demo must not crash or fabricate saved market predictions."""
    fake = UIStub()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    source = Path(__file__).resolve().parents[1] / "app"
    (tmp_path / "app/pages").mkdir(parents=True)
    shutil.copy(source / "pages/1_Market_Risk.py", tmp_path / "app/pages/1_Market_Risk.py")
    runpy.run_path(str(tmp_path / "app/pages/1_Market_Risk.py"), run_name="__main__")
