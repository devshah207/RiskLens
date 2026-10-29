"""Safely retire five obsolete research-only CLI wrappers after unified CLI migration.

Dry run by default; --apply removes explicitly listed wrappers. Never deletes model
libraries, training code, raw/processed data, or experiment evidence.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys

LEGACY = (
    "src/risklens/models/calibration_cli.py",
    "src/risklens/models/diagnostics_cli.py",
    "src/risklens/models/ohlc_comparison_cli.py",
    "src/risklens/models/walk_forward_cli.py",
    "src/risklens/models/xgboost_walk_forward_cli.py",
)


def check_references(root: Path, legacy: tuple[str, ...] = LEGACY) -> list[str]:
    """Abort removal when project Python code still imports a legacy wrapper."""
    names = {"risklens.models." + Path(s).stem for s in legacy}
    issues = []
    for file in sorted(
        list((root / "src").rglob("*.py"))
        + list((root / "scripts").rglob("*.py"))
        + list((root / "tests").rglob("*.py"))
    ):
        if file.relative_to(root).as_posix() in legacy:
            continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeError) as exc:
            issues.append(f"{file.relative_to(root)}: cannot inspect AST: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == n or alias.name.startswith(n + ".") for n in names
                    ):
                        issues.append(
                            f"{file.relative_to(root)}:{node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module in names:
                    issues.append(
                        f"{file.relative_to(root)}:{node.lineno}: from {node.module}"
                    )
                elif node.module == "risklens.models":
                    for alias in node.names:
                        if "risklens.models." + alias.name in names:
                            issues.append(
                                f"{file.relative_to(root)}:{node.lineno}: imported {alias.name}"
                            )
        # Dynamic import strings are unlikely but fail closed if explicit wrappers appear.
        source = file.read_text(encoding="utf-8")
        if any(n in source for n in names) and not any(
            str(file.relative_to(root)) in issue for issue in issues
        ):
            issues.append(
                f"{file.relative_to(root)}: mentions legacy module dynamically"
            )
    return issues


def run(root: Path, apply: bool = False) -> int:
    proc = subprocess.run(
        ["git", "branch", "--show-current"], cwd=root, text=True, capture_output=True
    )
    if proc.returncode or proc.stdout.strip() != "refactor/codebase-cleanup":
        print("REFUSED: checkout refactor/codebase-cleanup first.", file=sys.stderr)
        return 2
    if not (root / "src/risklens/cli.py").is_file():
        print(
            "REFUSED: unified CLI from Pass 4 is missing; apply Pass 4 first.",
            file=sys.stderr,
        )
        return 2
    if not (root / "tests/test_unified_cli.py").is_file():
        print("REFUSED: unified CLI regression tests missing.", file=sys.stderr)
        return 2
    issues = check_references(root)
    if issues:
        print(
            "REFUSED: wrapper references detected:\n" + "\n".join(issues),
            file=sys.stderr,
        )
        return 2
    for path in LEGACY:
        file = root / path
        print(
            ("REMOVE" if apply else "WOULD REMOVE")
            + ": "
            + path
            + ("" if file.exists() else " (already absent)")
        )
        if apply and file.is_file():
            file.unlink()
    print(
        "Complete. Run pytest and Citi demo before committing."
        if apply
        else "Dry run only. Use --apply to delete."
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    return run(args.root.resolve(), args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
