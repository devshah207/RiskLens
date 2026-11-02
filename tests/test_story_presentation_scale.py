"""Regression guards for the readable, presentation-scale React Flow explorer.

These are source/integration checks. They do NOT replace an actual npm build
or a Mac browser rendering test of the installed React Flow component.
"""
from __future__ import annotations

from pathlib import Path
import re

from app.services.story_blueprints import build_stories

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "src/risklens_story/frontend/src"
EXPLORER = (FRONT / "Explorer.tsx").read_text(encoding="utf-8")
CSS = (FRONT / "styles.css").read_text(encoding="utf-8")
PAGE = (ROOT / "app/pages/3_Explore_RiskLens.py").read_text(encoding="utf-8")


def test_focused_camera_magnifies_selected_node_after_measurement():
    assert "useNodesInitialized" in EXPLORER
    assert "const GUIDED_ZOOM = 1.13;" in EXPLORER
    assert "const NODE_WIDTH = 298;" in EXPLORER
    assert "const NODE_HEIGHT = 154;" in EXPLORER
    assert "flow.setCenter" in EXPLORER
    assert "onInit={() => setCameraSignal((v) => v + 1)}" in EXPLORER
    assert "signal={cameraSignal}" in EXPLORER


def test_toolbar_controls_are_functional_not_decorative():
    for item in (
        "View whole map", "Focus stage", "Hide details", "Show details",
        "Presentation mode", "Exit presentation", "onClick={() => setCameraSignal",
        "onClick={() => { setDetailsOpen", "onClick={() => void togglePresentation()}",
        "aria-expanded={detailsOpen}", "aria-pressed={focusMode}",
    ):
        assert item in EXPLORER
    assert "{detailsOpen && <aside" in EXPLORER
    assert "flow-layout${!detailsOpen ? ' details-collapsed' : ''}" in EXPLORER


def test_real_presentation_uses_fullscreen_on_click_and_keeps_escape():
    assert "shellRef.current.requestFullscreen()" in EXPLORER
    assert "document.exitFullscreen()" in EXPLORER
    assert "fullscreenchange" in EXPLORER
    assert "getRootNode()" in EXPLORER  # supports Streamlit shadow DOM
    assert "event.key === 'Escape'" in EXPLORER
    assert ".explorer-shell.is-presentation" in CSS
    assert ".explorer-shell:fullscreen" in CSS
    assert "height: 100dvh" in CSS
    assert "prefers-reduced-motion: reduce" in CSS


def test_large_map_and_responsive_collapse():
    assert "minmax(270px, 300px)" in CSS
    assert ".flow-layout.details-collapsed" in CSS
    assert "height: min(82vh, 890px)" in CSS
    assert "width: 298px" in CSS
    assert "min-height: 154px" in CSS
    assert "font-size: 19px" in CSS
    assert "@media (max-width: 910px)" in CSS
    assert "@media (max-width: 650px)" in CSS


def test_expanded_coordinates_still_leave_room_for_nodes():
    graphs = build_stories(ROOT)
    for name in ("market", "research"):
        nodes = {node["id"]: node for node in graphs[name]["nodes"]}
        for a in nodes.values():
            for b in nodes.values():
                if a is b:
                    continue
                x_gap = abs(a["position"]["x"] - b["position"]["x"]) * 1.40
                y_gap = abs(a["position"]["y"] - b["position"]["y"]) * 1.48
                assert x_gap >= 298 or y_gap >= 154, (name, a["id"], b["id"])


def test_page_width_scope_and_research_boundary():
    assert 'layout=\'wide\'' in PAGE
    assert '[data-testid="stMainBlockContainer"]' in PAGE
    assert "rl-explorer-intro" in PAGE
    assert "render_explorer(st, stories)" in PAGE
    assert "run_query_xbrl_first(" not in PAGE
    assert "model.fit(" not in PAGE
    assert "fetch(" not in EXPLORER
    assert not re.search(r"https?://(?!(?:github\.com/))", EXPLORER)
