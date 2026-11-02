"""Prevent regression of the Streamlit v2 asset_dir registration failure.

These tests exercise the actual delivered component wrapper without npm, a
browser, a research corpus, an API key, or model training.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import risklens_story


class FakeV2:
    def __init__(self):
        self.registration = None
        self.mount = None

    def component(self, name, *, html, js, css, isolate_styles):
        self.registration = {
            "name": name, "html": html, "js": js,
            "css": css, "isolate_styles": isolate_styles,
        }

        def mount(*, key, data):
            self.mount = {"key": key, "data": data}

        return mount


def test_uses_inline_js_and_css_not_file_backed_asset_paths(tmp_path, monkeypatch):
    """Mimics a built Vite bundle and tests the real registration call."""
    (tmp_path / "index-abc123.js").write_text(
        "export default function(component) { return () => {}; }",
        encoding="utf-8",
    )
    (tmp_path / "index.css").write_text(
        ".react-flow { background: #0d1628; }", encoding="utf-8",
    )
    monkeypatch.setattr(risklens_story, "_ASSETS", tmp_path)
    risklens_story._read_trusted_bundle.cache_clear()
    v2 = FakeV2()
    st = SimpleNamespace(components=SimpleNamespace(v2=v2))
    data = {"market": {"nodes": [{"id": "xgb"}]}, "research": {"nodes": []}}

    assert risklens_story.frontend_is_built()
    risklens_story.render_explorer(st, data)
    assert "export default function" in v2.registration["js"]
    assert ".react-flow { background" in v2.registration["css"]
    assert v2.registration["isolate_styles"] is True
    assert v2.mount["data"] is data
    assert v2.mount["key"] == "risklens-architecture-v1"
    assert v2.registration["js"] != "index-*.js"
    assert v2.registration["css"] != "index.css"


def test_rebuild_updates_inline_assets(tmp_path, monkeypatch):
    bundle = tmp_path / "index-xyz.js"
    css = tmp_path / "index.css"
    bundle.write_text("export default function first() {}", encoding="utf-8")
    css.write_text(".first { color: red; }", encoding="utf-8")
    monkeypatch.setattr(risklens_story, "_ASSETS", tmp_path)
    risklens_story._read_trusted_bundle.cache_clear()
    v2 = FakeV2()
    st = SimpleNamespace(components=SimpleNamespace(v2=v2))
    risklens_story.render_explorer(st, {})
    assert ".first" in v2.registration["css"]

    css.write_text(".updated { color: cyan; }", encoding="utf-8")
    # Ensure a distinct mtime on filesystems with coarse clock resolution.
    current = css.stat()
    import os
    os.utime(css, ns=(current.st_atime_ns, current.st_mtime_ns + 3_000_000_000))
    risklens_story.render_explorer(st, {})
    assert ".updated" in v2.registration["css"]


def test_unbuilt_frontend_reports_specific_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(risklens_story, "_ASSETS", tmp_path)
    st = SimpleNamespace(components=SimpleNamespace(v2=FakeV2()))
    assert not risklens_story.frontend_is_built()
    with pytest.raises(FileNotFoundError, match="npm run build"):
        risklens_story.render_explorer(st, {})


def test_invalid_install_missing_streamlit_v2(tmp_path, monkeypatch):
    monkeypatch.setattr(risklens_story, "_ASSETS", tmp_path)
    with pytest.raises(RuntimeError, match="1.51"):
        risklens_story.render_explorer(SimpleNamespace(), {})
