"""Regression: Streamlit's browser must not require a Node.js `process` global.

Uses a representative Vite library-mode dependency reference because the user's
real compiled JS bundle is local to their Mac, not in the repository ZIP.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import risklens_story


class _ComponentRecorder:
    def __init__(self):
        self.arguments = None
        self.mount_arguments = None

    def component(self, name, **kwargs):
        self.arguments = (name, kwargs)

        def mount(**props):
            self.mount_arguments = props

        return mount


def _mounted_component(monkeypatch, tmp_path):
    js = tmp_path / "index-example.js"
    css = tmp_path / "index.css"
    js.write_text('globalThis.storyTestOk=process.env.NODE_ENV==="production";', encoding="utf-8")
    css.write_text('body{background:url(data:image/svg+xml;base64,AAAA)}', encoding="utf-8")
    monkeypatch.setattr(risklens_story, "_ASSETS", tmp_path)
    risklens_story._read_trusted_bundle.cache_clear()
    recorder = _ComponentRecorder()
    st = SimpleNamespace(components=SimpleNamespace(v2=recorder))
    stories = {"market": {"nodes": []}, "research": {"nodes": []}}
    risklens_story.render_explorer(st, stories)
    assert recorder.mount_arguments == {"key": "risklens-architecture-v1", "data": stories}
    return recorder.arguments[1]


def test_previous_browser_process_error_is_avoided(monkeypatch, tmp_path):
    component = _mounted_component(monkeypatch, tmp_path)
    js = component["js"]
    assert js.index("var process =") < js.index("process.env.NODE_ENV")
    assert "\n" in js.strip()
    assert "\n" in component["css"].strip()  # Streamlit 1.66 inline heuristic
    assert component["isolate_styles"] is True
    assert component["html"] == '<div id="risklens-reactflow-root"></div>'

    if shutil.which("node") is None:
        pytest.skip("Node.js not present to execute browser-like JS sandbox")
    runner = (
        'const vm=require("node:vm");'
        'const code=process.argv[1];'
        'const sandbox={};'
        'vm.runInNewContext(code,sandbox);'
        'if(sandbox.storyTestOk!==true) process.exit(7);'
    )
    # The sandbox intentionally has NO process: this resembles Chrome.
    verified = subprocess.run(
        ["node", "-e", runner, js], capture_output=True, text=True, check=False
    )
    assert verified.returncode == 0, verified.stderr


def test_vite_compiles_out_node_process_env():
    config = (
        Path(__file__).resolve().parents[1]
        / "src/risklens_story/frontend/vite.config.ts"
    ).read_text(encoding="utf-8")
    assert "'process.env.NODE_ENV': JSON.stringify('production')" in config
    assert "formats: ['es']" in config
    assert "inlineDynamicImports: true" in config


def test_original_browser_failure_is_reproducible_without_shim():
    if shutil.which("node") is None:
        pytest.skip("Node.js not present")
    runner = (
        'const vm=require("node:vm");'
        'try {vm.runInNewContext("globalThis.x=process.env.NODE_ENV;",{});'
        'process.exit(3)} catch (e) {if(e.name!=="ReferenceError")process.exit(2)}'
    )
    result = subprocess.run(["node", "-e", runner], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
