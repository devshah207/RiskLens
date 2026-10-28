# Synthetic OHLC fixture

`sample_ohlc.csv` contains **invented values for testing software only**. It does **not** contain historical NIFTY 50 data, and its dates/prices must never be reported as real-world observations or used as evidence of model performance.

`invalid_ohlc.csv` is deliberately **invalid synthetic data**: the recorded high is less than the closing price. It is only for demonstrating a FAIL result from the validator.
