"""Regression for a real Streamlit 1.66.0 / 1.66.1 filename heuristic.

Official upstream:
streamlit/components/v2/component_path_utils.py
ComponentPathUtils.looks_like_inline_content().
The actual Mac StreamlitAPIException occurred because Vite minifies CSS/JS into
one line with `/`, so path detection classified the contents as file-backed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


import risklens_story


def streamlit_166_thinks_inline(value: str) -> bool:
    """Equivalent to upstream streamlit 1.66 ComponentPathUtils implementation.

    This independently checks that BOTH inline strings pass the actual rule;
    testing only whether 'index.css' differs from the arg is insufficient.
    """
    s = value.strip()
    if "\n" in s or "\r" in s:
        return True
    if any(char in s for char in ("*", "?", "[", "]")):
        return False
    if s.startswith(("./", "/", "\\")):
        return False
    if "/" in s or "\\" in s:
        return False
    return not s.lower().endswith((".js", ".css", ".mjs", ".cjs"))


class StrictV2:
    def __init__(self):
        self.registration = None
        self.mounted = None

    def component(self, name, *, html, js, css, isolate_styles):
        # Emulates exactly the asset_root=None branch in Streamlit's resolver.
        for label, source in (("css", css), ("js", js)):
            if not streamlit_166_thinks_inline(source):
                raise ValueError(f"must be declared in pyproject.toml with asset_dir to use file-backed {label}")
        self.registration = (name, html, js, css, isolate_styles)

        def mount(*, key, data):
            self.mounted = (key, data)

        return mount


def test_original_minified_vite_css_was_mistaken_for_a_file():
    css = ".react-flow{background:url(/assets/diagram.png)}"
    js = "export default function run(){return '/example'};"
    assert not streamlit_166_thinks_inline(css)
    assert not streamlit_166_thinks_inline(js)
    # A bare leading newline is stripped and cannot fix this.
    assert not streamlit_166_thinks_inline("\n" + css)


def test_compiled_single_line_bundles_mount_without_asset_dir(tmp_path, monkeypatch):
    minified_js = 'export default function run(){return "/assets/flow"};'
    minified_css = '.react-flow{background:url(/assets/background.svg)}'
    (tmp_path / "index-abc123.js").write_text(minified_js, encoding="utf-8")
    (tmp_path / "index.css").write_text(minified_css, encoding="utf-8")
    monkeypatch.setattr(risklens_story, "_ASSETS", tmp_path)
    risklens_story._read_trusted_bundle.cache_clear()
    strict_v2 = StrictV2()
    stories = {"market": {"nodes": [{"id": "xgb"}]}, "research": {"nodes": []}}
    st = SimpleNamespace(components=SimpleNamespace(v2=strict_v2))
    risklens_story.render_explorer(st, stories)
    assert strict_v2.registration is not None
    name, html, js, css, isolate_styles = strict_v2.registration
    assert name == "risklens-educational.story_explorer"
    assert html == '<div id="risklens-reactflow-root"></div>'
    assert minified_js in js
    assert minified_css in css
    assert streamlit_166_thinks_inline(js)
    assert streamlit_166_thinks_inline(css)
    assert strict_v2.mounted == ("risklens-architecture-v1", stories)
    assert isolate_styles


def test_inline_headers_must_survive_strip():
    assert streamlit_166_thinks_inline("/* trusted CSS */\n.react-flow{a:url(/x)}")
    assert streamlit_166_thinks_inline("/* trusted JS */\nexport default () => '/x'")


def test_no_private_original_sources_or_network_in_wrapper():
    source = Path(risklens_story.__file__).read_text(encoding="utf-8")
    for forbidden in ("GROQ_API_KEY", "run_query_xbrl_first", "model.fit", "fetch("):
        assert forbidden not in source
