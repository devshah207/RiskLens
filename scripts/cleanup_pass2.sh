#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--apply" ]]; then
  echo "Dry run only; execute bash scripts/cleanup_pass2.sh --apply to delete the listed obsolete files"
fi
branch=$(git branch --show-current)
if [[ "$branch" != "refactor/codebase-cleanup" ]]; then
  echo "Refusing cleanup on branch: $branch" >&2; exit 2
fi
files=(
  "docs/phase2_step1_learning_guide.md"
  "docs/phase2_step2_learning_guide.md"
  "docs/phase2_step3_learning_guide.md"
  "docs/phase3_step10_learning_guide.md"
  "docs/phase3_step1_learning_guide.md"
  "docs/phase3_step2_learning_guide.md"
  "docs/phase3_step3_learning_guide.md"
  "docs/phase3_step4_learning_guide.md"
  "docs/phase3_step5_learning_guide.md"
  "docs/phase3_step6_learning_guide.md"
  "docs/phase3_step7_learning_guide.md"
  "docs/phase3_step8_learning_guide.md"
  "docs/phase3_step9_learning_guide.md"
  "docs/task2_learning_guide.md"
  "src/risklens/models/technical_comparison_cli.py"
)
for f in "${files[@]}"; do
  if [[ -f "$f" ]]; then
    if [[ "${1:-}" == "--apply" ]]; then
      rm -- "$f"
      echo "Removed $f"
    else
      echo "Would remove $f"
    fi
  fi
done
