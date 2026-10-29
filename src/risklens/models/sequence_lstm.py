"""Citi-oriented sequence-model benchmark with strict point-in-time development folds.

A 30-session multi-factor input sequence predicts the following NIFTY session's
>=1% downside event. No dates >=2025-01-01 are turned into features or labels.
Torch is an optional dependency: pip install -e '.[deep,dev]'.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import html
import json
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from risklens.data.validation import load_ohlc_csv
from risklens.features.dataset import (
    FULL_FEATURE_COLUMNS, TARGET_COLUMN, _wilder_rsi_14, build_development_features,
)
from risklens.models.multifactor_comparison import _validate_integrated
from risklens.models.ohlc_comparison import _pooled_probability_scores
from risklens.models.baselines import _metrics
from risklens.models.walk_forward import FOLD_YEARS, FINAL_HOLDOUT_START
from risklens.models.xgboost_walk_forward import EXTERNAL_FEATURE_COLUMNS, MULTIFACTOR_FEATURE_COLUMNS

LOOKBACK = 30
HIDDEN_SIZE = 24
MAX_EPOCHS = 28
PATIENCE = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 64
SEED = 73
PREDICTIONS_FILENAME = 'sequence_lstm_predictions.csv'
REPORT_FILENAME = 'sequence_lstm_validation_report.json'
DASHBOARD_FILENAME = 'risk_model_validation_dashboard.html'


class SequenceModelError(ValueError):
    """Unsafe or incompatible historical sequence comparison inputs."""


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise SequenceModelError("PyTorch not installed: python -m pip install -e '.[deep,dev]'") from e
    return torch, nn


def build_causal_history(nifty_csv: str | Path, external_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Create full, unlabeled development feature history for sequence lookbacks.

    Existing feature-builder training/validation values are independently checked
    for every fold before model fitting. No forward fill, smoothing across dates,
    or input labels. External joins have already passed exact provenance checks.
    """
    external, audit = _validate_integrated(nifty_csv, external_csv)
    prices = load_ohlc_csv(nifty_csv)
    d = prices.loc[prices.date < pd.Timestamp(FINAL_HOLDOUT_START), ['date','open','high','low','close']].copy().reset_index(drop=True)
    d['return_1d'] = d.close.pct_change(fill_method=None)
    d['return_5d'] = d.close.pct_change(periods=5, fill_method=None)
    d['volatility_20d'] = d.return_1d.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    d['intraday_range'] = (d.high-d.low)/d.open
    d['intraday_return'] = d.close/d.open - 1
    d['opening_gap'] = d.open/d.close.shift(1) - 1
    d['rsi_14'] = _wilder_rsi_14(d.close)
    d['ema_50_distance'] = d.close/d.close.ewm(span=50,adjust=False,min_periods=50).mean() - 1
    d['volatility_10d'] = d.return_1d.rolling(10,min_periods=10).std(ddof=1) * np.sqrt(252)
    d = d[['date', *FULL_FEATURE_COLUMNS]].merge(external[['date', *EXTERNAL_FEATURE_COLUMNS]],on='date',how='left',validate='one_to_one')
    if not d.date.is_monotonic_increasing or d.date.duplicated().any():
        raise SequenceModelError('Feature history dates not unique and sorted')
    return d, audit


def make_sequences(history: pd.DataFrame, labeled: pd.DataFrame, features: tuple[str,...], lookback: int=LOOKBACK):
    """Labels belong only to the END date. Earlier window dates may be unlabeled.

    No future features: window is [t-lookback+1, ..., t] and does not include t+1.
    Missing external rows are excluded rather than imputed from future sources.
    """
    if lookback < 2 or not set(features).issubset(history.columns):
        raise SequenceModelError('Invalid feature sequence specification')
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise SequenceModelError('History dates must be unique/increasing')
    if labeled.date.duplicated().any():
        raise SequenceModelError('Duplicate label dates')
    if (history.date >= pd.Timestamp(FINAL_HOLDOUT_START)).any() or (labeled.date >= pd.Timestamp(FINAL_HOLDOUT_START)).any():
        raise SequenceModelError('2025+ holdout access forbidden')
    positions = pd.Series(np.arange(len(history)),index=history.date)
    if not labeled.date.isin(positions.index).all():
        raise SequenceModelError('One or more label dates missing from feature history')
    vals = history.loc[:,list(features)].to_numpy(dtype=np.float64)
    ys = labeled[TARGET_COLUMN].to_numpy(dtype=int)
    if not np.isin(ys,[0,1]).all():
        raise SequenceModelError('Expected binary labels')
    xs, outputs, times = [], [], []
    for date,y in zip(labeled.date,ys):
        i=int(positions.loc[date])
        if i+1 < lookback:
            continue
        window=vals[i-lookback+1:i+1]
        if not np.isfinite(window).all():
            continue
        xs.append(window)
        outputs.append(y)
        times.append(date)
    if not xs:
        raise SequenceModelError('No complete lookback sequences')
    return np.asarray(xs,dtype=np.float32),np.asarray(outputs,dtype=np.int64),pd.DatetimeIndex(times)


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, *, seed: int, max_epochs: int=MAX_EPOCHS):
    """Only pre-validation-year labels select epochs; final model refits all earlier data.

    Stage A: final 15% of earlier years used for *internal* early stopping, with
    the training/inner-validation boundary's last label purged. Stage B: learn
    from ALL earlier-year labeled samples using selected epoch count. Scalers
    are fitted on stage-specific training history only. The reported validation
    year is never used by optimizer, scaler, or epoch selection.
    """
    torch, nn = _torch()
    if train_x.ndim!=3 or valid_x.ndim!=3 or train_x.shape[1:]!=valid_x.shape[1:]:
        raise SequenceModelError('Invalid sequence tensor shapes')
    if len(train_x)<80 or len(valid_x)<1 or len(np.unique(train_y))!=2:
        raise SequenceModelError('Not enough labeled history for chronological fit')
    if not np.isfinite(train_x).all() or not np.isfinite(valid_x).all():
        raise SequenceModelError('Nonfinite features')
    torch.set_num_threads(1)  # predictable laptop-compatible CPU runtime
    torch.use_deterministic_algorithms(True)

    class LSTMNet(nn.Module):
        def __init__(self, inputs):
            super().__init__()
            self.lstm=nn.LSTM(input_size=inputs,hidden_size=HIDDEN_SIZE,num_layers=2,dropout=0.15,batch_first=True)
            self.head=nn.Sequential(nn.LayerNorm(HIDDEN_SIZE),nn.Linear(HIDDEN_SIZE,12),nn.ReLU(),nn.Linear(12,1))
        def forward(self,x):
            h,_=self.lstm(x)
            return self.head(h[:,-1,:]).squeeze(-1)

    def scale_fit(fit_x, *others):
        s=StandardScaler().fit(fit_x.reshape(-1,fit_x.shape[-1]))
        def transform(x):
            z=s.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape)
            return torch.as_tensor(np.clip(z,-8,8),dtype=torch.float32)
        return [transform(fit_x),*(transform(x) for x in others)]

    def train_model(x,y, epochs, this_seed):
        torch.manual_seed(this_seed)
        model=LSTMNet(x.shape[-1])
        opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=0.01)
        lossfn=nn.BCEWithLogitsLoss()
        yy=torch.as_tensor(y,dtype=torch.float32)
        generator=torch.Generator().manual_seed(this_seed)
        for _ in range(epochs):
            model.train()
            for ix in torch.randperm(len(x),generator=generator).split(BATCH_SIZE):
                opt.zero_grad()
                loss=lossfn(model(x[ix]),yy[ix])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
        model.eval()
        return model

    cut=int(len(train_x)*0.85)
    # purge the last inner-fit label because its outcome uses the next session
    inner_x,inner_y=train_x[:cut-1],train_y[:cut-1]
    stop_x,stop_y=train_x[cut:],train_y[cut:]
    a,b=scale_fit(inner_x,stop_x)
    torch.manual_seed(seed)
    model=LSTMNet(a.shape[-1]); opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=0.01)
    yy=torch.as_tensor(inner_y,dtype=torch.float32); vy=torch.as_tensor(stop_y,dtype=torch.float32)
    criterion=nn.BCEWithLogitsLoss()
    gen=torch.Generator().manual_seed(seed)
    best_loss=float('inf');best_epoch=1;stall=0;loss_history=[]
    for epoch in range(1,max_epochs+1):
        model.train()
        for ix in torch.randperm(len(a),generator=gen).split(BATCH_SIZE):
            opt.zero_grad(); loss=criterion(model(a[ix]),yy[ix]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        model.eval()
        with torch.no_grad():
            stop_loss=float(criterion(model(b),vy).item())
        loss_history.append(round(stop_loss,6))
        if stop_loss < best_loss-1e-5:
            best_loss=stop_loss;best_epoch=epoch;stall=0
        else:
            stall+=1
            if stall>=PATIENCE:break

    # IMPORTANT: refit using ALL previous-year labels, never the evaluated year.
    xx,vv=scale_fit(train_x,valid_x)
    final=train_model(xx,train_y,best_epoch,seed+1000)
    with torch.no_grad():
        prob=torch.sigmoid(final(vv)).numpy().astype(float)
    if not np.isfinite(prob).all() or not ((prob>=0)&(prob<=1)).all():
        raise SequenceModelError('Invalid model probabilities')
    audit={'early_stop_best_epoch':int(best_epoch),'early_stop_best_log_loss':round(best_loss,8),
           'early_stop_inner_train_rows':len(inner_x),'early_stop_inner_validation_rows':len(stop_x),
           'final_fit_rows':len(train_x),'final_fit_epochs':int(best_epoch),
           'fit_test_separation':'internal stopping strictly before evaluation year',
           'fit_device':'cpu','random_seed':int(seed)}
    return prob,audit


def _check_reference(reference_csv: str | Path) -> pd.DataFrame:
    ref=pd.read_csv(reference_csv)
    required=['validation_year','date','actual_downside_event','baseline_probability','xgboost_9_features_probability','xgboost_11_features_probability']
    if list(ref.columns)!=required:raise SequenceModelError('Step 3.10 reference columns inconsistent')
    ref.date=pd.to_datetime(ref.date,format='%Y-%m-%d',errors='raise')
    if ref.date.duplicated().any() or not ref.date.is_monotonic_increasing or (ref.date>=pd.Timestamp(FINAL_HOLDOUT_START)).any():
        raise SequenceModelError('Reference has duplicate/out-of-order/holdout dates')
    if not (ref.validation_year.to_numpy()==ref.date.dt.year.to_numpy()).all() or not set(ref.validation_year)==set(FOLD_YEARS):
        raise SequenceModelError('Reference fold years invalid')
    if not np.isin(ref.actual_downside_event,[0,1]).all():
        raise SequenceModelError('Reference outcome values not binary')
    for col in required[3:]:
        arr=ref[col].to_numpy(dtype=float)
        if not np.isfinite(arr).all() or ((arr<0)|(arr>1)).any():
            raise SequenceModelError(f'Invalid probabilities: {col}')
    return ref


def run_sequence_benchmark(nifty_csv: str|Path, external_csv: str|Path, reference_csv: str|Path, *,max_epochs:int=MAX_EPOCHS,lookback:int=LOOKBACK):
    """Five expanding walk-forward experiments; fixed architecture and two ablations."""
    if not 2<=lookback<=120 or not 1<=max_epochs<=100:
        raise SequenceModelError('Unsupported lookback or epoch budget')
    history,input_audit=build_causal_history(nifty_csv,external_csv)
    reference=_check_reference(reference_csv)
    out=[];audits=[]
    for year in FOLD_YEARS:
        train,valid,split_audit=build_development_features(nifty_csv,validation_start=f'{year}-01-01',holdout_start=f'{year+1}-01-01',include_ohlc_features=True,include_technical_features=True)
        external_dates=history[['date',*EXTERNAL_FEATURE_COLUMNS]]
        for frame in (train,valid):
            # Independent reconstruction must numerically match original features.
            original=frame.merge(history[['date',*FULL_FEATURE_COLUMNS]],on='date',validate='one_to_one',suffixes=('_prior','_audit'))
            for feature in FULL_FEATURE_COLUMNS:
                if not np.allclose(original[f'{feature}_prior'],original[f'{feature}_audit'],rtol=1e-10,atol=1e-10):
                    raise SequenceModelError(f'Independent historical {feature} does not match original builder')
        foldref=reference.loc[reference.validation_year==year].reset_index(drop=True)
        actual=valid.merge(external_dates,on='date',how='left',validate='one_to_one')
        actual=actual.loc[actual[list(EXTERNAL_FEATURE_COLUMNS)].notna().all(axis=1)].reset_index(drop=True)
        if not actual.date.equals(foldref.date) or not np.array_equal(actual[TARGET_COLUMN].to_numpy(dtype=int),foldref.actual_downside_event.to_numpy(dtype=int)):
            raise SequenceModelError(f'{year}: reference dates/labels do not match primary source')
        train_x,train_y,train_dates=make_sequences(history,train,MULTIFACTOR_FEATURE_COLUMNS,lookback)
        # Require same target dates as reference and complete 11-feature histories.
        val_x,val_y,val_dates=make_sequences(history,actual,MULTIFACTOR_FEATURE_COLUMNS,lookback)
        if not val_dates.equals(pd.DatetimeIndex(foldref.date)) or not np.array_equal(val_y,foldref.actual_downside_event.to_numpy(dtype=int)):
            raise SequenceModelError(f'{year}: incomplete external sequence or validation-label mismatch')
        if train_dates.max() >= val_dates.min() or (train_dates>=pd.Timestamp(f'{year}-01-01')).any():
            raise SequenceModelError(f'{year}: training chronology crossed validation boundary')
        p9,a9=_fit_predict(train_x[:,:,:len(FULL_FEATURE_COLUMNS)],train_y,val_x[:,:,:len(FULL_FEATURE_COLUMNS)],seed=SEED+year,max_epochs=max_epochs)
        p11,a11=_fit_predict(train_x,train_y,val_x,seed=SEED+year+100,max_epochs=max_epochs)
        f=foldref.copy();f['lstm_9_features_probability']=p9;f['lstm_11_features_probability']=p11
        out.append(f)
        audits.append({'year':year,'train_sequences':len(train_x),'train_events':int(train_y.sum()),'validation_sequences':len(val_x),'validation_events':int(val_y.sum()),'first_validation_date':str(val_dates.min().date()),'last_training_date':str(train_dates.max().date()),'training_boundary_labels_purged':split_audit['training_boundary_labels_purged'],'lstm_9':a9,'lstm_11':a11,
            'year_scores':{name:_metrics(val_y,p) for name,p in {'lstm_9':p9,'lstm_11':p11,'xgb_11':foldref.xgboost_11_features_probability.to_numpy(dtype=float)}.items()}})
    result=pd.concat(out,ignore_index=True)
    if result.date.duplicated().any() or (result.date>=pd.Timestamp(FINAL_HOLDOUT_START)).any():
        raise SequenceModelError('Fold output overlaps or contains final holdout')
    cols={'constant_baseline':'baseline_probability','xgboost_9':'xgboost_9_features_probability','xgboost_11':'xgboost_11_features_probability','lstm_9':'lstm_9_features_probability','lstm_11':'lstm_11_features_probability'}
    pooled={key:{**_pooled_probability_scores(result,col), 'roc_auc':round(float(roc_auc_score(result.actual_downside_event,result[col])),8),'average_precision':round(float(average_precision_score(result.actual_downside_event,result[col])),8)} for key,col in cols.items()}
    report={'title':'RiskLens | Independent Model Development and Validation Research','generated_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'scope':'Repeated exploratory development validation only; 2025+ final holdout fully sealed','target':'Next trading session NIFTY close-to-close return <= -1%','prediction_time':'After NIFTY trading session t closes','architecture':{'network':'2-layer PyTorch LSTM + LayerNorm + MLP binary logit head','sequence_length':lookback,'hidden_size':HIDDEN_SIZE,'dropout':0.15,'max_internal_early_stopping_epochs':max_epochs,'patience':PATIENCE,'optimizer':'AdamW','learning_rate':LEARNING_RATE,'gradient_clipping_norm':1.0,'class_weights':'none (retain probability interpretation)','seed':SEED,'device':'CPU'},'validation_years':list(FOLD_YEARS),'n_validation_predictions':len(result),'n_events':int(result.actual_downside_event.sum()),'data_integrity':input_audit,'folds':audits,'pooled_scores':pooled,'source_sha256':{'nifty':sha256(Path(nifty_csv).read_bytes()).hexdigest(),'integrated':sha256(Path(external_csv).read_bytes()).hexdigest(),'xgboost_reference':sha256(Path(reference_csv).read_bytes()).hexdigest()},'limitations':['Small single-index sample and relatively rare downside events; deep networks may overfit.','Development validation years have been consulted for prior model decisions; results are NOT a clean independent estimate.','Earlier XGBoost probabilities are supplied as verified Step 3.10 reference; this stage does not retrain or retune XGBoost.','These external features intentionally use strictly prior source dates, which is conservative for India VIX.','Repeated overlapping 30-session windows reduce effective statistical independence.','Using observed next-session returns as labels is for research, not a deployable trading or regulatory capital model.','This tests neither credit scoring nor regulatory VaR approval; no real-world investment or bank risk claims.','2025-2026 holdout remains sealed; do not choose architectures or tune on those outcomes.']}
    return result,report


def render_dashboard(report:dict, var_report:dict|None=None) -> str:
    """Self-contained, offline HTML model validation card; no external JS or network."""
    scores=report['pooled_scores']; folds=report['folds']
    def row(k,name):
        d=scores[k]
        return f'<tr><td>{html.escape(name)}</td><td>{d["brier_score"]:.5f}</td><td>{d["log_loss"]:.5f}</td><td>{d["roc_auc"]:.3f}</td><td>{d["average_precision"]:.3f}</td></tr>'
    table=''.join([row('constant_baseline','Historical event-rate baseline'),row('xgboost_9','XGBoost — 9 features'),row('xgboost_11','XGBoost — 11 features'),row('lstm_9','LSTM — 9 features'),row('lstm_11','LSTM — 11 features')])
    yr=''.join('<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.4f}</td><td>{:.4f}</td></tr>'.format(x['year'],x['validation_sequences'],x['validation_events'],x['year_scores']['lstm_11']['brier_score'],x['year_scores']['lstm_11']['roc_auc']) for x in folds)
    var_section=''
    if var_report is not None:
        if not isinstance(var_report,dict) or 'results' not in var_report:
            raise SequenceModelError('Invalid VaR audit report')
        v=var_report['results']
        cells=''.join(f'<tr><td>{int(v[k]["confidence_level"])}%</td><td>{v[k]["expected_exception_rate"]:.2%}</td><td>{v[k]["observed_exception_rate"]:.2%}</td><td>{int(v[k]["exception_count"])}</td><td>{v[k]["exact_binomial_coverage_p_value"]:.4f}</td></tr>' for k in ('var_95','var_99'))
        var_section='<section><h2>Historical VaR exception monitoring</h2><table><thead><tr><th>VaR confidence</th><th>Expected breaches</th><th>Observed breaches</th><th>Exceptions</th><th>Binomial p-value*</th></tr></thead><tbody>'+cells+'</tbody></table><p class="small">*Simple binomial test assumes independent exceptions and does not capture breach clustering.</p></section>'
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RiskLens | Risk Model Validation</title><style>body{margin:0;background:#0c1424;color:#eaf1fb;font:15px -apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif}main{max-width:1050px;margin:auto;padding:38px 25px}.eyebrow{color:#59d7c8;text-transform:uppercase;letter-spacing:.19em;font-weight:700;font-size:12px}h1{font-size:41px;margin:16px 0}p{line-height:1.7;color:#b8c5d9}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:15px;margin:27px 0}.card,section{background:#152238;border:1px solid #30415b;border-radius:13px;padding:22px}.num{font-size:29px;font-weight:800;margin:6px 0}.small{color:#9eafc9;font-size:12px}table{width:100%;border-collapse:collapse;text-align:left;min-width:540px}th,td{padding:13px;border-bottom:1px solid #30415b}th{color:#59d7c8;font-size:12px;text-transform:uppercase;letter-spacing:.04em}section{margin:20px 0;overflow:auto}h2{margin:0 0 15px;font-size:20px}.notice{border-left:4px solid #59d7c8;padding:13px 19px;background:#172a3c}.foot{font-size:12px;color:#8ba0b9;margin-top:30px}</style></head><body><main><div class="eyebrow">RISK ANALYTICS / MODEL VALIDATION RESEARCH</div><h1>RiskLens <span style="color:#59d7c8">/</span> Model Risk Monitor</h1><p>Multi-factor NIFTY 50 next-session downside probability. Chronological, leakage-audited research benchmarking tabular XGBoost against sequence-based LSTM.</p><div class="cards"><div class="card"><div class="small">Development evaluation</div><div class="num">2020–24</div></div><div class="card"><div class="small">Out-of-training predictions</div><div class="num">{n}</div></div><div class="card"><div class="small">Observed downside events</div><div class="num">{e}</div></div><div class="card"><div class="small">Final holdout</div><div class="num">SEALED</div><div class="small">2025+</div></div></div><section><h2>Cross-model probability and ranking scores</h2><table><thead><tr><th>Model</th><th>Brier ↓</th><th>Log loss ↓</th><th>ROC-AUC ↑</th><th>Avg precision ↑</th></tr></thead><tbody>{scores}</tbody></table><p class="small">These are repeated development results, not an unbiased final holdout estimate. Selecting models based on them adds selection bias.</p></section><section><h2>Yearly LSTM-11 stability / failure analysis</h2><table><thead><tr><th>Validation year</th><th>Days</th><th>Events</th><th>Brier ↓</th><th>ROC-AUC ↑</th></tr></thead><tbody>{years}</tbody></table></section><div class="notice"><b>Risk control:</b> Data joined using strictly earlier external-market observation dates. All feature windows end on prediction date t. Standardization and early stopping use only training-era data. Last cross-year training label is purged. No 2025+ features, labels, predictions, or performance were accessed.</div><p class="foot">Educational model risk research. No investment advice, credit-risk score, production deployment, or regulatory claim. Source/authenticity verification is separate from structural data checks.</p></main></body></html>'''.replace('{n}',str(report['n_validation_predictions'])).replace('{e}',str(report['n_events'])).replace('{scores}',table).replace('{years}',yr).replace('<div class="notice">',var_section+'<div class="notice">')


def save_sequence_benchmark(nifty_csv,external_csv,reference_csv,output_dir,*,overwrite=False,max_epochs=MAX_EPOCHS,lookback=LOOKBACK,var_report_path=None):
    out=Path(output_dir); paths=[out/PREDICTIONS_FILENAME,out/REPORT_FILENAME,out/DASHBOARD_FILENAME]
    if any(p.exists() for p in paths) and not overwrite:
        raise SequenceModelError('Output exists; use another directory or --overwrite')
    predictions, report=run_sequence_benchmark(nifty_csv,external_csv,reference_csv,max_epochs=max_epochs,lookback=lookback)
    out.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        staged=Path(tmp)
        export=predictions.copy();export['date']=export.date.dt.strftime('%Y-%m-%d')
        export.to_csv(staged/PREDICTIONS_FILENAME,index=False,float_format='%.12g')
        (staged/REPORT_FILENAME).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        var_report=json.loads(Path(var_report_path).read_text(encoding='utf-8')) if var_report_path else None
        (staged/DASHBOARD_FILENAME).write_text(render_dashboard(report,var_report),encoding='utf-8')
        for p in paths:(staged/p.name).replace(p)
    return report
