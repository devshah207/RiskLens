"""Tests of development-only descriptive analysis; entirely synthetic prices."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklens.analysis.eda import EDAError, analyze_development, save_development_eda


def make_file(tmp_path: Path, dates=None, closes=None, name='prices.csv'):
    if dates is None:
        dates = ['2024-12-26','2024-12-27','2024-12-30','2024-12-31','2025-01-01','2025-01-02']
    if closes is None:
        closes = [100.,102.,99.,99.,100.,101.]
    # OHLC exactly equal to close values: valid, simple controlled test data.
    data = pd.DataFrame({'date':dates,'open':closes,'high':closes,'low':closes,'close':closes})
    file = tmp_path/name
    data.to_csv(file,index=False)
    return file


def test_return_formula_and_missing_first_day(tmp_path):
    path=make_file(tmp_path)
    dev, report = analyze_development(path, rolling_window=2)
    assert len(dev)==4
    assert pd.isna(dev.loc[0,'daily_return'])
    assert dev.loc[1,'daily_return']==pytest.approx(.02)
    assert dev.loc[2,'daily_return']==pytest.approx(99/102-1)
    assert dev.loc[3,'daily_return']==pytest.approx(0)
    assert report['valid_daily_returns']==3
    assert report['development_last_session']=='2024-12-31'


def test_log_return_formula(tmp_path):
    data, _ = analyze_development(make_file(tmp_path), rolling_window=2)
    assert data.loc[1,'log_return']==pytest.approx(np.log(1.02))
    assert data.loc[2,'log_return']==pytest.approx(np.log(99/102))


def test_rolling_volatility_past_and_current_only(tmp_path):
    data, report=analyze_development(make_file(tmp_path),rolling_window=2)
    assert pd.isna(data.loc[1,'rolling_vol_2d_annualized'])
    expected=np.std([.02,99/102-1],ddof=1)*np.sqrt(252)
    assert data.loc[2,'rolling_vol_2d_annualized']==pytest.approx(expected)
    assert report['rolling_volatility_nonnull_rows']==2


def test_holdout_prices_do_not_change_development_statistics(tmp_path):
    first=make_file(tmp_path,name='original.csv')
    changed=make_file(tmp_path,closes=[100,102,99,99,1000,1100],name='changed.csv')
    a,aa=analyze_development(first,rolling_window=2)
    b,bb=analyze_development(changed,rolling_window=2)
    pd.testing.assert_frame_equal(a,b)
    keys={'source_file','source_sha256'}
    assert {k:v for k,v in aa.items() if k not in keys} == {k:v for k,v in bb.items() if k not in keys}


def test_only_return_close_date_counted(tmp_path):
    _, report=analyze_development(make_file(tmp_path),rolling_window=2)
    assert report['downside_event_close_dates']==1  # Dec30 -2.94%
    assert report['downside_event_fraction']==pytest.approx(1/3)


def test_invalid_cutoff_rejected(tmp_path):
    path=make_file(tmp_path)
    for bad in ['2025/01/01','2025-02-30','25-01-01','2025-1-1']:
        with pytest.raises(EDAError,match='YYYY-MM-DD'):
            analyze_development(path,holdout_start=bad)


def test_cutoff_leaves_too_few_observations(tmp_path):
    with pytest.raises(EDAError,match='at least two'):
        analyze_development(make_file(tmp_path),holdout_start='2024-12-27')


def test_parameters_validated(tmp_path):
    path=make_file(tmp_path)
    for window in (0,1,1.5,True):
        with pytest.raises(EDAError,match='rolling_window'):
            analyze_development(path,rolling_window=window)
    for freq in (0,-1,False):
        with pytest.raises(EDAError,match='sessions_per_year'):
            analyze_development(path,sessions_per_year=freq)
    for threshold in (0,-.01,1,2,float('nan')):
        with pytest.raises(EDAError,match='downside_threshold'):
            analyze_development(path,downside_threshold=threshold)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(EDAError,match='not found'):
        analyze_development(tmp_path/'missing.csv')


def test_bad_ohlc_rejected(tmp_path):
    path=make_file(tmp_path)
    frame=pd.read_csv(path);frame.loc[1,'high']=1;frame.to_csv(path,index=False)
    with pytest.raises(EDAError,match='Inconsistent OHLC'):
        analyze_development(path)


def test_writes_only_development_period(tmp_path):
    path=make_file(tmp_path)
    dest=tmp_path/'out'
    report=save_development_eda(path,dest,rolling_window=2)
    frame=pd.read_csv(dest/'nifty50_eda_development.csv')
    metadata=json.loads((dest/'nifty50_eda_development.json').read_text())
    assert len(frame)==4
    assert frame['date'].max()=='2024-12-31'
    assert (frame['date']<'2025-01-01').all()
    assert report==metadata
    assert metadata['source_sha256']


def test_refuses_overwrite(tmp_path):
    path=make_file(tmp_path)
    dest=tmp_path/'out'
    save_development_eda(path,dest,rolling_window=2)
    with pytest.raises(EDAError,match='already exists'):
        save_development_eda(path,dest,rolling_window=2)
    save_development_eda(path,dest,rolling_window=2,overwrite=True)


def test_annual_group_uses_close_date(tmp_path):
    path=make_file(tmp_path,dates=['2023-12-29','2024-01-01','2024-01-02','2025-01-01'],closes=[100,99,98,70])
    _,report=analyze_development(path,rolling_window=2)
    assert report['annual_statistics'][0]['return_observations']==0
    assert report['annual_statistics'][1]['return_observations']==2
    assert report['downside_event_close_dates']==2


def test_short_valid_history_has_no_rolling_vol(tmp_path):
    path=make_file(tmp_path,dates=['2024-12-30','2024-12-31','2025-01-01'],closes=[100,99,110])
    data,report=analyze_development(path,rolling_window=20)
    assert len(data)==2
    assert report['rolling_volatility_nonnull_rows']==0
    assert report['rolling_volatility_max'] is None
    assert report['annualized_volatility'] is None
    assert report['sample_skewness'] is None
