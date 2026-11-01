"""Phase 8 presentation of existing Phase 4/5/6 saved evidence; no fitting."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from app.components.charts import (  # noqa: E402
    BLUE, MINT, PURPLE, calibration_chart, feature_chart, timeline_chart,
    var_chart, yearly_auc_chart,
)
from app.components.ui import (  # noqa: E402
    hero, inject_theme, mini_stats, note, section, sidebar_brand,
)
from app.services.demo_data import synthetic_timeline  # noqa: E402
from app.services.saved_risk_evidence import (  # noqa: E402
    EvidenceError, MODEL_NAMES, PROBABILITY_COLUMNS, load_evidence,
    model_comparison, recorded_comparison,
)

st.set_page_config(page_title='RiskLens | Market Risk', page_icon='📈', layout='wide',
                   initial_sidebar_state='expanded')
inject_theme(st)
sidebar_brand(st, 'Market Risk Analytics')
hero(
    st, '01 / Market Risk Analytics', 'Understand the downside.',
    'Saved NIFTY 50 probability estimates, frozen model evidence and independent market-risk monitoring — with no new fitting, holdout rescoring or trading signals.',
)

@st.cache_data(ttl=60, show_spinner=False)
def get_saved(root: str, fingerprint: tuple):
    # Cache key includes actual file modification timestamps and sizes.
    return load_evidence(Path(root))

fingerprint = tuple(
    (str(p), p.stat().st_mtime_ns, p.stat().st_size)
    for p in sorted((ROOT / 'data/processed').glob('phase*/*'))
    if p.is_file() and p.suffix in {'.json', '.csv'}
)
try:
    e = get_saved(str(ROOT), fingerprint)
except EvidenceError as exc:
    st.error(f'Saved evidence validation failed: {exc}. This app will not make up replacement results.')
    st.stop()

comparison = model_comparison(e.report) if e.report else recorded_comparison()
if not e.report:
    st.warning('The original holdout report is unavailable here. Comparison values are recorded handover figures, NOT a new local verification.')

mini_stats(st, [
    ('Evaluation window', '433', '2025–September 2026 sessions'),
    ('Downside events', '42', 'Recorded original holdout cases'),
    ('Best recorded ROC-AUC', '0.685', 'LSTM #86, not promoted as champion'),
    ('Recall at 0.50', '0%', 'Both models missed all 42 events'),
])
note(st, 'Historical event: after NIFTY 50 closes, will the next completed trading session close at least 1% lower? ROC-AUC measures ranking; zero recall at 0.50 means these fitted models were not useful as 0.50-threshold alerts.', warning=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Overview', 'Probability explorer', 'Historical VaR', 'Explanations & drift', 'Governance',
])

with tab1:
    section(st, '01 / Final model comparison')
    st.subheader('Champion vs. challenger')
    st.caption('All figures below are taken from the original saved final report if available. Higher AUC/AP is better; lower Brier/log loss is better.')
    st.dataframe(
        comparison.style.format({c: '{:.6f}' for c in comparison.columns if c != 'Model'}),
        hide_index=True, use_container_width=True,
    )
    note(st, 'XGBoost remains the preselected champion. The LSTM trial #86 performed slightly better on the single final holdout, but the governance agreement forbids retroactively changing the champion.')
    if e.report:
        year_chart = yearly_auc_chart(e.report)
        if year_chart:
            st.markdown('#### Year-by-year discrimination')
            st.plotly_chart(year_chart, use_container_width=True, config={'displaylogo': False})
            st.caption('The original saved 2025 and 2026 scores are shown separately. Both candidates weakened in 2026.')
        with st.expander('Detailed original year-by-year metrics'):
            for year, group in sorted((e.report.get('yearly_scores') or {}).items()):
                st.markdown(f'**{year} · {group.get("observations", "?")} sessions / {group.get("events", "?")} events**')
                st.dataframe(group.get('models', {}), use_container_width=True)
    else:
        st.info('The full saved year-by-year breakdown is unavailable in this checkout.')

with tab2:
    section(st, '02 / Probability history')
    st.subheader('What did the models estimate over time?')
    if e.predictions is None:
        st.warning('Original saved probability CSV is missing. The preview below is deliberately fictional and must not be presented as historical research evidence.')
        fict = synthetic_timeline()
        st.line_chart(fict.set_index('fictional_session'), height=300)
        st.caption('SYNTHETIC DISPLAY FIXTURE. No real dates, prices, probabilities or events.')
    else:
        selected = st.selectbox('Frozen candidate', list(MODEL_NAMES), format_func=lambda m: MODEL_NAMES[m])
        st.plotly_chart(timeline_chart(e.predictions, selected), use_container_width=True,
                        config={'displaylogo': False})
        st.caption('The colored line is the saved predicted probability. Coral diamonds mark actual downside-event prediction dates at their original predicted values; they are NOT probability-1.0 labels.')
        with st.expander('All 42 actual event dates and corresponding original probabilities'):
            prob = PROBABILITY_COLUMNS[selected]
            hits = e.predictions.loc[e.predictions['actual_downside_event'] == 1, ['date', prob]].copy()
            hits['date'] = hits['date'].dt.strftime('%d %b %Y')
            st.dataframe(hits.rename(columns={prob: 'Saved probability (0–1)'}),
                         hide_index=True, use_container_width=True)
    section(st, '03 / Calibration')
    st.subheader('Predicted risk versus observed frequency')
    st.caption('A well-calibrated 10% forecast means about 10 realized events per 100 comparable dates. These are the saved historical bins — the UI does not create new bins or optimize thresholds.')
    if e.calibration is not None:
        st.plotly_chart(calibration_chart(e.calibration), use_container_width=True,
                        config={'displaylogo': False})
        st.caption('Circle size represents the number of observations in an original bin. Tiny bins are uncertain; a dotted line indicates ideal alignment.')
        with st.expander('Read the original reliability-bin table'):
            st.dataframe(e.calibration, hide_index=True, use_container_width=True)
    else:
        st.info('Saved calibration bins are unavailable; no new holdout calibration is calculated.')

with tab3:
    section(st, '04 / Historical Value at Risk')
    st.subheader('How often was the historical loss threshold exceeded?')
    note(st, 'Value at Risk (VaR) is separate from the XGBoost and LSTM probabilities. The original study estimated loss thresholds from earlier market returns, not from these classifiers.')
    var = (e.report or {}).get('historical_var', {})
    c95, c99 = st.columns(2, gap='large')
    for conf, observed, col in [('95', 30, c95), ('99', 7, c99)]:
        result = var.get(conf, {}) if isinstance(var, dict) else {}
        n = result.get('observations', 433)
        count = result.get('exceptions', observed)
        pct = result.get('observed_rate', observed / 433)
        with col:
            st.metric(f'{conf}% historical VaR breaches', f'{count}/{n}')
            st.caption(f'Observed: {pct:.2%} · Expected: {1-int(conf)/100:.0%}')
    if isinstance(var, dict) and all(isinstance(var.get(k), dict) for k in ('95', '99')):
        st.plotly_chart(var_chart(var), use_container_width=True,
                        config={'displaylogo': False})
        with st.expander('Original independence and conditional-coverage tests'):
            for level in ('95', '99'):
                item = var[level]
                st.markdown(f'**{level}% VaR**')
                st.write({'independence': item.get('independence'),
                          'conditional_coverage': item.get('conditional_coverage'),
                          'annual_exceptions': item.get('annual_exceptions')})
    else:
        st.caption('Counts on this tab are the recorded historical values from the handover; the original VaR report was not mounted.')
    note(st, '95% VaR exceptions clustered in 2026, including March. Historical backtests flagged independence and conditional-coverage concerns; they did not identify a proven macroeconomic cause.', warning=True)

with tab4:
    section(st, '05 / Model explanations')
    st.subheader('Which recorded inputs mattered to the studied models?')
    left, right = st.columns(2, gap='large')
    with left:
        st.markdown('#### Fixed XGBoost · TreeSHAP')
        if e.shap is not None:
            st.plotly_chart(feature_chart(e.shap, value='mean_absolute_shap_log_odds',
                                         label='Mean absolute SHAP (log-odds)', color=MINT),
                            use_container_width=True, config={'displaylogo': False})
        else:
            st.info('Original TreeSHAP summary CSV missing.')
        st.caption('Calculated in the earlier pre-2025 explanatory study. These are descriptive log-odds contributions, not proof of financial causation.')
    with right:
        st.markdown('#### LSTM #86 · Temporal sensitivity')
        if e.lstm_sensitivity is not None:
            st.plotly_chart(feature_chart(e.lstm_sensitivity, value='mean_brier_increase',
                                         label='Brier change from permutation', color=PURPLE),
                            use_container_width=True, config={'displaylogo': False})
        else:
            st.info('Original LSTM sensitivity CSV missing.')
        st.caption('A 2024 development study shuffled complete historical feature sequences. This is NOT SHAP and its scale cannot be compared directly with TreeSHAP.')
    section(st, '06 / Drift watch')
    st.subheader('How did input feature distributions shift?')
    if e.feature_drift is not None:
        st.plotly_chart(feature_chart(e.feature_drift, value='population_stability_index',
                                     label='Saved population stability index', color=BLUE,
                                     threshold=.20), use_container_width=True,
                        config={'displaylogo': False})
        with st.expander('Original feature drift table and investigation details'):
            st.dataframe(e.feature_drift, hide_index=True, use_container_width=True)
            if e.drift_investigation:
                st.json(e.drift_investigation, expanded=False)
    else:
        st.info('The saved Phase 6 feature PSI table is missing. No replacement values were computed.')
    st.caption('PSI = Population Stability Index. A value above the research 0.20 investigation heuristic is a warning, not proof of what caused a market move.')

with tab5:
    section(st, '07 / Governance and traceability')
    st.subheader('Read-only, reproducible and candid')
    if e.manifest:
        st.success('Original Phase 6 model-freeze manifest found. Presence does not itself verify all SHA-256 evidence hashes.')
        st.caption(str(e.manifest.get('purpose', 'Historical model freeze')))
    else:
        st.warning('Freeze manifest missing from this checkout.')
    note(st, 'Nothing in this app fits a model, changes a frozen threshold, promotes the LSTM challenger, reruns the sealed 433-row holdout, or makes post-2026 live predictions.')
    st.markdown('#### Saved monitoring and risk register')
    if e.monitoring is not None:
        st.caption(str(e.monitoring.get('selection_status', 'Selection remains frozen')))
        register = e.monitoring.get('risk_register') or []
        if register:
            st.dataframe(register, hide_index=True, use_container_width=True)
        with st.expander('Raw monitoring statistics / saved annual frequencies'):
            st.json({'holdout_annual_event_frequencies':
                     e.monitoring.get('holdout_annual_event_frequencies', {})}, expanded=False)
    else:
        st.info('Saved monitoring summary not mounted.')
    if e.monitoring_periods is not None or e.monitoring_bins is not None:
        with st.expander('Original saved score-monitoring tables'):
            if e.monitoring_periods is not None:
                st.markdown('**Development vs holdout periods**')
                st.dataframe(e.monitoring_periods, hide_index=True, use_container_width=True)
            if e.monitoring_bins is not None:
                st.markdown('**Saved score distribution bins**')
                st.dataframe(e.monitoring_bins, hide_index=True, use_container_width=True)
    with st.expander('Original local files that are not present'):
        if e.missing:
            for category, path in e.missing.items():
                st.text(f'{category}: {path}')
        else:
            st.write('All documented Phase 8 market-risk evidence files are present.')
