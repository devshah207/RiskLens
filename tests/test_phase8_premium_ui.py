"""UI regression tests: answer visibility, untrusted HTML, saved-source plots."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import pytest

from app.components import ui
from app.components import charts


class CaptureMarkdown:
    def __init__(self):
        self.calls = []

    def markdown(self, message, **kwargs):
        self.calls.append(message)


def test_answer_card_always_displays_actual_xbrl_amount_and_full_answer():
    st = CaptureMarkdown()
    original = 'INFY consolidated RevenueFromOperations (FY2025): ₹162,990 crore'
    result = {
        'status': 'answered', 'answer': original,
        'route': 'xbrl_exact_fact', 'provider': 'nse_xbrl_xml_local',
        'verification_status': 'not_human_certified',
    }
    ui.render_answer_card(st, result)
    rendered = '\n'.join(st.calls)
    assert '₹162,990 crore' in rendered
    assert original in rendered
    assert 'Source-backed answer' not in rendered  # status is a separate element
    assert 'Answer from source evidence' in rendered


def test_pdf_and_llm_answer_html_escaped_and_not_executable():
    st = CaptureMarkdown()
    hostile = '<script>alert("XSS")</script><img src=x onerror=alert(1)>'
    ui.render_answer_card(st, {
        'status': 'needs_review', 'answer': hostile,
        'provider': 'groq', 'route': 'llm_numeric_unverified',
        'verification_status': 'needs_review',
    })
    rendered = ''.join(st.calls)
    assert '<script>' not in rendered and '<img src=' not in rendered
    assert html.escape(hostile, quote=True) in rendered
    assert 'Provisional answer' in rendered


def test_status_does_not_replace_or_hide_the_answer():
    st = CaptureMarkdown()
    result = {'status': 'abstained', 'answer': 'No eligible indexed report.', 'route': 'empty_index'}
    ui.show_status(st, result)
    ui.render_answer_card(st, result)
    assert 'No eligible indexed report.' in ''.join(st.calls)
    assert len(st.calls) == 2


def test_theme_does_not_load_external_fonts():
    st = CaptureMarkdown()
    ui.inject_theme(st)
    assert '@import url' not in st.calls[0]
    assert 'http://' not in st.calls[0] and 'https://' not in st.calls[0]


def test_plotly_chart_uses_saved_samples_without_mutating_rows():
    pytest.importorskip('plotly')
    sample = pd.DataFrame({
        'date': pd.to_datetime(['2025-01-02', '2025-01-03', '2025-01-06']),
        'actual_downside_event': [0, 1, 0],
        'xgboost_11_fixed_probability': [.10, .13, .08],
        'lstm_optuna_86_probability': [.11, .15, .09],
    })
    original = sample.copy(deep=True)
    fig = charts.timeline_chart(sample, 'xgboost_fixed')
    assert list(fig.data[0].y) == [10.0, 13.0, 8.0]
    assert len(fig.data[1].x) == 1
    pd.testing.assert_frame_equal(original, sample)


def test_saved_calibration_bins_not_recomputed_and_psi_threshold_read_only():
    pytest.importorskip('plotly')
    bins = pd.DataFrame([
        {'model': 'xgboost_fixed', 'mean_probability': .08, 'event_frequency': .06, 'observations': 50},
        {'model': 'lstm_optuna_trial86', 'mean_probability': .10, 'event_frequency': .12, 'observations': 43},
    ])
    before = bins.copy(deep=True)
    plot = charts.calibration_chart(bins)
    assert len(plot.data) == 3  # two original models + ideal diagonal
    pd.testing.assert_frame_equal(before, bins)
    shifts = pd.DataFrame({'feature': ['rsi_14', 'return_1d'], 'population_stability_index': [.40, .03]})
    shifts_before = shifts.copy(deep=True)
    fig = charts.feature_chart(shifts, value='population_stability_index',
                               label='saved PSI', threshold=.20)
    assert len(fig.data[0].x) == 2
    pd.testing.assert_frame_equal(shifts, shifts_before)


def test_no_model_training_or_holdout_eval_in_any_ui_file():
    root = Path(__file__).resolve().parents[1] / 'app'
    for source_file in [root / 'Home.py', *list((root / 'pages').glob('*.py')),
                        *list((root / 'components').glob('*.py'))]:
        contents = source_file.read_text(encoding='utf-8')
        assert 'evaluate_holdout(' not in contents
        assert 'model.fit(' not in contents
        assert 'run_tuning(' not in contents
