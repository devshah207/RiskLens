import numpy as np
import pandas as pd
from risklens.models.optuna_research import brier, xgb_space, lstm_space, TUNING_YEARS, DIAGNOSTIC_YEARS

class Trial:
    def suggest_int(self,n,a,b):return a
    def suggest_float(self,n,a,b,log=False):return a
    def suggest_categorical(self,n,choices):return choices[0]

def test_tuning_chronology():
    assert max(TUNING_YEARS)<min(DIAGNOSTIC_YEARS)<2025

def test_xgboost_space_safe():
    d=xgb_space(Trial());assert d['n_estimators']>=1 and d['max_depth']>=1
    assert d['scale_pos_weight']==1.0 and d['n_jobs']==1

def test_lstm_space_safe():
    d=lstm_space(Trial());assert d['lookback']>=2 and d['epochs']>=1

def test_brier_exact():
    assert brier([0,1],[.2,.8])==np.mean((np.array([0,1])-np.array([.2,.8]))**2)

def test_device_selection_cpu_and_invalid():
    import pytest
    from risklens.models.optuna_research import resolve_torch_device
    assert resolve_torch_device('cpu').type == 'cpu'
    with pytest.raises(ValueError, match='device must be'):
        resolve_torch_device('cuda')


def test_lstm_one_cpu_smoke():
    from risklens.models.optuna_research import _lstm_one
    rng=np.random.default_rng(1)
    x=rng.normal(size=(12, 4, 3)).astype(np.float32)
    y=np.array([0,1]*6,dtype=int)
    va=rng.normal(size=(4,4,3)).astype(np.float32)
    params={'hidden':8,'layers':1,'dropout':0.0,'lr':0.001,'weight_decay':0.001,'batch':4,'epochs':1}
    p=_lstm_one(x,y,va,params,seed=7,device='cpu')
    assert p.shape == (4,)
    assert np.isfinite(p).all()
    assert ((p>=0)&(p<=1)).all()
