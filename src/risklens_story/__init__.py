"""Mount the original React Flow explorer through Streamlit Components v2.

The frontend is the owner's locally built, trusted Vite output. Passing the
compiled JS/CSS source inline avoids a fragile package-asset registration step
in some Streamlit / editable-install combinations. No model or RAG code runs
here; the Python application passes only the local architecture blueprint.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "frontend" / "build"


def frontend_is_built() -> bool:
    """True only if exactly one Vite JS bundle and its CSS have been built."""
    return (len(list(_ASSETS.glob("index-*.js"))) == 1
            and (_ASSETS / "index.css").is_file())


@lru_cache(maxsize=4)
def _read_trusted_bundle(js_path: Path, js_mtime: int, css_mtime: int) -> tuple[str, str]:
    """Read only first-party Vite output; mtime keys refresh on rebuild.

    Returning *source text*, not paths, is essential: file-backed CSS/JS
    requires a discovered Streamlit asset_dir registration, which can be
    unavailable even if the files and component-level manifest exist.
    """
    del js_mtime, css_mtime
    return (js_path.read_text(encoding="utf-8"),
            (_ASSETS / "index.css").read_text(encoding="utf-8"))


def render_explorer(st, stories: dict) -> None:
    """Mount the actual compiled React Flow UI in the existing Streamlit app."""
    if not hasattr(st, "components") or not hasattr(st.components, "v2"):
        raise RuntimeError("Streamlit >=1.51 is required for Components v2")
    bundles = list(_ASSETS.glob("index-*.js"))
    css_path = _ASSETS / "index.css"
    if len(bundles) != 1 or not css_path.is_file():
        raise FileNotFoundError(
            "React Flow frontend is not built: run `npm run build` inside "
            "`src/risklens_story/frontend`, then restart Streamlit."
        )

    bundle = bundles[0]
    javascript, stylesheet = _read_trusted_bundle(
        bundle, bundle.stat().st_mtime_ns, css_path.stat().st_mtime_ns,
    )
    if not javascript.strip() or not stylesheet.strip():
        raise RuntimeError("Compiled React Flow JS or CSS is empty. Rebuild the frontend.")

    # Streamlit 1.66's actual ComponentPathUtils looks_like_inline_content()
    # first STRIPS surrounding whitespace, then checks for INTERNAL newlines.
    # A minified one-line Vite CSS/JS bundle containing '/' can otherwise be
    # MISCLASSIFIED as an asset path, even when the full source text is passed.
    # A trusted CSS/JS block comment followed by an INTERNAL newline makes
    # each bundle unambiguously inline, without relying on manifest discovery.
    # Do not replace these headers with a bare leading newline: strip() would
    # remove it and reproduce the exact user's StreamlitAPIException.
    # Vite's ES-library build deliberately leaves process.env.NODE_ENV in some
    # third-party React code unless it is explicitly replaced. Chrome has no
    # Node.js `process` global. Prefer the Vite compile-time `define` (included
    # in this repair); keep this SMALL, module-local compatibility binding so
    # the owner's already-built JS also works without an npm reinstall.
    #
    # Since the trusted bundle is registered as an ES-module entry by Streamlit,
    # `var process` is in its module scope, NOT a window/globalThis mutation.
    # This does not provide arbitrary Node process APIs to browser code.
    javascript = (
        '/* RiskLens trusted locally built JavaScript */\n'
        'var process = { env: { NODE_ENV: "production" } };\n'
        + javascript
    )
    stylesheet = "/* RiskLens trusted locally built CSS */\n" + stylesheet
    component = st.components.v2.component(
        "risklens-educational.story_explorer",
        html='<div id="risklens-reactflow-root"></div>',
        js=javascript,
        css=stylesheet,
        isolate_styles=True,
    )
    component(key="risklens-architecture-v1", data=stories)
