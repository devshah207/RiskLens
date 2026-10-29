# Cleanup Pass 2 — Remove unused presentation artifacts

This is a deletion-only pass plus removal of an unused optional dependency. It deliberately **does not delete model implementations or training outputs**. Git history preserves removed guides and the former `technical_comparison_cli.py` entry point; the `technical_comparison.py` research implementation and its tests remain.

## Why these files

- 14 step-by-step tutorial guides are superseded by README, data contract, architecture, model validation and Optuna guides. These are historical teaching material, not runtime resources.
- `technical_comparison_cli.py` is unused by active demo and tests. The underlying `technical_comparison.py` remains callable for reproducibility.
- `imbalanced-learn` is removed from the optional dependency groups because SMOTE is not part of the retained pipeline.

## Apply on `refactor/codebase-cleanup`

```bash
bash scripts/cleanup_pass2.sh          # Preview the paths
bash scripts/cleanup_pass2.sh --apply  # Delete only listed paths
python -m pip install -e ".[dev,deep,tuning]"
python -m pytest -q
bash scripts/run_citi_demo.sh
```

Check `git status` and `git diff --stat` before committing. Preserve the local `data/processed/optuna_search` and its separate backup. **Do not delete any other modules in this pass**: later architectural consolidation is needed to break existing imports safely.

The full test suite should pass on your Mac; focused suites were checked in the build environment, but the full suite exceeded its runtime limit.
