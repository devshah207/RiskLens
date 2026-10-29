"""Historical-simulation 95%/99% one-session VaR breach monitor for Citi-style risk validation.

Value-at-Risk (VaR) is a loss quantile, NOT a guarantee or a capital calculation.
Each t forecast uses the previous 250 returns including session t. The
following t+1 return is used ONLY after prediction to score an exception.
Entire computation is restricted to the pre-2025 development window.
"""
from pathlib import Path
import json
from datetime import datetime, timezone
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from risklens.data.validation import load_ohlc_csv
from risklens.models.sequence_lstm import _check_reference, SequenceModelError
from risklens.evaluation.splits import FINAL_HOLDOUT_START

WINDOW=250


def backtest_historical_var(nifty_csv: str|Path, reference_csv: str|Path,*,window:int=WINDOW):
    """Report a baseline market-risk coverage assessment on exact ML validation dates."""
    if window<30:raise SequenceModelError('Historical VaR window must be >=30 returns')
    ref=_check_reference(reference_csv)
    prices=load_ohlc_csv(nifty_csv)
    d=prices.loc[prices.date<pd.Timestamp(FINAL_HOLDOUT_START),['date','close']].copy().reset_index(drop=True)
    d['historical_return_1d']=d.close.pct_change(fill_method=None)
    # Include session t return, no t+1 future outcome in the rolling quantile.
    for conf,q in [('95',0.05),('99',0.01)]:
        d[f'var_{conf}_loss_fraction']=-d.historical_return_1d.rolling(window,min_periods=window).quantile(q)
    d['next_session_return']=d.close.shift(-1)/d.close-1
    # Merge only requested development predictions; no last-of-year leaked label.
    m=ref[['date','validation_year']].merge(d[['date','next_session_return','var_95_loss_fraction','var_99_loss_fraction']],on='date',how='left',validate='one_to_one')
    if m[['next_session_return','var_95_loss_fraction','var_99_loss_fraction']].isna().any().any():
        raise SequenceModelError('Insufficient historical VaR window or missing evaluation outcome')
    y=m.next_session_return.to_numpy(dtype=float)
    if not np.isfinite(y).all():raise SequenceModelError('Nonfinite realized returns')
    scores={}
    for confidence,p in [('95',0.05),('99',0.01)]:
        losses=m[f'var_{confidence}_loss_fraction'].to_numpy(dtype=float)
        exceptions=(-y>losses)
        if not np.isfinite(losses).all():raise SequenceModelError('Nonfinite historical VaR')
        m[f'var_{confidence}_exception']=exceptions.astype(int)
        k=int(exceptions.sum());n=len(exceptions)
        scores[f'var_{confidence}']={'confidence_level':int(confidence),'expected_exception_rate':p,'observed_exception_rate':round(k/n,8),'exception_count':k,'n_observations':n,'expected_exceptions':round(n*p,2),'mean_loss_fraction_threshold':round(float(np.mean(losses)),8), 'exact_binomial_coverage_p_value':round(float(binomtest(k,n,p).pvalue),8),'per_year':{str(year):{'observations':int(len(sub)),'exceptions':int(sub[f'var_{confidence}_exception'].sum())} for year,sub in m.groupby('validation_year')}}
    report={'title':'RiskLens | Historical VaR Exception Monitoring','generated_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'scope':'2020-2024 development backtest; 2025+ untouched','historical_return_window':window,'method':'Trailing empirical lower-tail quantile of the past 250 observed daily returns, inclusive of forecast date t','exception_definition':'Realized next-session loss exceeds the positive VaR loss threshold; strict >','results':scores,'limitations':['Historical simulation assumes recent realized return distribution reflects the next session; extreme tail has few observations.','P-values are descriptive under a simple independent Bernoulli exception model; financial exceptions may cluster in time.','This is not a bank regulatory VaR capital model, FRTB calculation or portfolio exposure model.','This 2020-2024 sample was previously observed during model development; no final holdout evaluation.']}
    return m,report


def save_var_backtest(nifty_csv,reference_csv,out_dir,*,overwrite=False):
    out=Path(out_dir)
    paths=[out/'historical_var_backtest.csv',out/'historical_var_report.json']
    if any(p.exists() for p in paths) and not overwrite:raise SequenceModelError('VaR output exists; use --overwrite')
    f,r=backtest_historical_var(nifty_csv,reference_csv)
    out.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        z=Path(td);f.date=f.date.dt.strftime('%Y-%m-%d')
        f.to_csv(z/paths[0].name,index=False,float_format='%.12g')
        (z/paths[1].name).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        for p in paths:(z/p.name).replace(p)
    return r
