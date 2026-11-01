"""Load optional project-root credentials without leaking them to logs or Git.

An activated venv does not turn pyvenv.cfg entries into environment variables.
The project-root .env is local and gitignored; process environment takes precedence.
"""
from __future__ import annotations

from pathlib import Path


def load_project_env(path: str | Path | None = None) -> Path:
    dotenv_path = Path(path) if path is not None else Path(__file__).resolve().parents[3] / ".env"
    if dotenv_path.is_file():
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError("Install python-dotenv to read the project-root .env") from exc
        load_dotenv(dotenv_path=dotenv_path, override=False)
    return dotenv_path
