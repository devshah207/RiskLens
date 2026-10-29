# RiskLens Optuna on Apple Silicon M3

## Correct hardware support

- **PyTorch LSTM**: Apple GPU via **MPS** (Metal Performance Shaders). `--device mps` forces MPS and raises a useful error if unavailable; `--device auto` chooses MPS when available, otherwise CPU.
- **XGBoost**: XGBoost's GPU path is based on **CUDA**, which Apple M3 does not support. Keep `tree_method=hist` on the Mac CPU. `--device mps` only applies to PyTorch LSTM; it does not make XGBoost GPU accelerated.
- Bigger neural networks are not automatically faster on MPS, especially for tiny datasets; benchmark with a few trials. MPS and CPU floating-point behavior can vary slightly.

## Setup (from RiskLens root, native arm64 Python environment)

```bash
uname -m      # should print arm64
source .venv/bin/activate
python -m pip install -e '.[dev,deep,tuning]'
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('MPS built:', torch.backends.mps.is_built())
print('MPS available:', torch.backends.mps.is_available())
PY
```

If `MPS available` is False, confirm you are using a native arm64 Python on macOS 12.3+ with a current compatible PyTorch build. Do not use `cuda` on Apple Silicon.

## Continue the existing persistent Optuna study

```bash
python -m risklens.models.optuna_research \
  data/processed/multiyear/nifty50_ohlc.csv \
  data/processed/external/nifty50_external_development.csv \
  --output-dir data/processed/optuna_search \
  --xgb-trials 0 --lstm-trials 200 --device mps
```

The SQLite Optuna storage in `--output-dir` is reused. Trials are **additional**. For XGBoost on the Mac CPU, run separately:

```bash
python -m risklens.models.optuna_research \
  data/processed/multiyear/nifty50_ohlc.csv \
  data/processed/external/nifty50_external_development.csv \
  --output-dir data/processed/optuna_search \
  --xgb-trials 500 --lstm-trials 0
```

If MPS throws an unsupported-operation error, prefer `--device cpu` for the problematic run rather than silently mixing devices. Avoid `PYTORCH_ENABLE_MPS_FALLBACK=1` for a controlled benchmark unless the fallback is explicitly reported.

**Methodology:** These trials only tune on the already examined 2020–2022 development folds; later 2023–2024 diagnostics are also not independent due to previous experiments. 2025+ remains unopened. Running more trials cannot guarantee a better out-of-sample model.
