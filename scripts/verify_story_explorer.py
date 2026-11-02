"""Read-only readiness diagnostics for the custom React Flow + Streamlit v2 UI."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.services.story_blueprints import build_stories  # noqa: E402
from risklens_story import frontend_is_built  # noqa: E402


def check_node() -> tuple[bool, str]:
    executable = shutil.which('node')
    if not executable:
        return False, 'Node.js is unavailable (install Node.js 24 LTS)'
    try:
        done = subprocess.run([executable, '--version'], capture_output=True,
                              text=True, timeout=5, check=True)
        major = int(done.stdout.strip().lstrip('v').split('.')[0])
        return major >= 20, f'Node.js {done.stdout.strip()} installed'
    except (ValueError, OSError, subprocess.SubprocessError):
        return False, 'Unable to determine Node.js version'


def main() -> int:
    print('RiskLens interactive story explorer — READ-ONLY readiness check')
    data = build_stories(ROOT)
    print(f'Blueprints: Market {len(data["market"]["nodes"])} nodes; '
          f'Research {len(data["research"]["nodes"])} nodes')
    print('Saved holdout note from original JSON:',
          'read and validated' if data['context']['savedHoldoutVerified'] else 'not independently validated')
    node_ok, detail = check_node()
    print('Node:', detail)
    try:
        streamlit_version = importlib.metadata.version('streamlit')
        major, minor = (int(x) for x in streamlit_version.split('.')[:2])
        streamlit_ok = (major, minor) >= (1, 51)
        print('Streamlit:', streamlit_version, 'supports v2' if streamlit_ok else 'too old')
    except (ValueError, importlib.metadata.PackageNotFoundError):
        print('Streamlit: not installed')
        streamlit_ok = False
    build_ok = frontend_is_built()
    print('React Flow bundle:', 'built' if build_ok else 'NOT BUILT')
    if not build_ok:
        print('FIX: cd src/risklens_story/frontend && npm install && npm run build')
    if not streamlit_ok:
        print("FIX: python -m pip install -e '.[ui,dev]'")
    if not node_ok:
        print('NOTE: Node.js is needed for building, not for running a previously built bundle.')
    if build_ok and streamlit_ok:
        print('READY FOR LOCAL BROWSER REVIEW. This is not an actual visual/browser test.')
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
