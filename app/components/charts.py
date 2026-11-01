"""Read-only Plotly visualizations of saved RiskLens evidence.

No model training, rescore, calibration-bin recomputation, tuning or data writes.
Every plotted number comes from an existing saved CSV/JSON; calculated
percentages are formatting of those stored numbers only.
"""
from __future__ import annotations

import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:  # charts are only required for the optional UI installation
    go = None

from app.services.saved_risk_evidence import MODEL_NAMES, PROBABILITY_COLUMNS

BG = "#0d1828"
TXT = "#e7f2fb"
MUTED = "#9cb1c7"
GRID = "rgba(150,177,198,0.15)"
MINT = "#5de2be"
BLUE = "#6baaf9"
PURPLE = "#a79aff"
CORAL = "#ff928b"
GOLD = "#efc47f"


def _require_plotly() -> None:
    if go is None:
        raise RuntimeError("Plotly is required for charts: install the project's optional [ui] extra")


def _style(fig: go.Figure, *, height: int = 370, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"color": TXT, "family": "Arial, sans-serif", "size": 12},
        margin={"l": 20, "r": 15, "t": 30, "b": 40},
        hoverlabel={"bgcolor": "#22374d", "font_color": "#edf6ff"},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"color": MUTED}},
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    return fig


def timeline_chart(frame: pd.DataFrame, model: str) -> go.Figure:
    """Display SAVED probabilities; dot markers identify actual event dates."""
    _require_plotly()
    prob = PROBABILITY_COLUMNS[model]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame['date'], y=frame[prob] * 100,
        mode='lines', name='Saved probability',
        line={"color": MINT if model == 'xgboost_fixed' else BLUE, "width": 2.7},
        fill='tozeroy', fillcolor='rgba(93,226,190,0.07)',
        hovertemplate='%{x|%d %b %Y}<br>Saved probability: %{y:.2f}%<extra></extra>',
    ))
    events = frame.loc[frame['actual_downside_event'] == 1]
    fig.add_trace(go.Scatter(
        x=events['date'], y=events[prob] * 100,
        mode='markers', name='Actual -1% event (at its prediction date)',
        marker={"color": CORAL, "size": 7.5, "symbol": "diamond", "line": {"color": BG, "width": 1}},
        hovertemplate='%{x|%d %b %Y}<br>Actual event: yes<br>Saved probability: %{y:.2f}%<extra></extra>',
    ))
    _style(fig, height=390)
    fig.update_yaxes(title='Saved probability (%)', rangemode='tozero', ticksuffix='%')
    fig.update_xaxes(title=None)
    return fig


def yearly_auc_chart(report: dict) -> go.Figure | None:
    _require_plotly()
    yearly = report.get('yearly_scores') or {}
    if not yearly:
        return None
    fig = go.Figure()
    for model, color in [('xgboost_fixed', MINT), ('lstm_optuna_trial86', PURPLE)]:
        years, aucs = [], []
        for year, group in sorted(yearly.items()):
            metric = group.get('models', {}).get(model, {}).get('roc_auc')
            if metric is not None:
                years.append(str(year))
                aucs.append(float(metric))
        if years:
            fig.add_trace(go.Scatter(
                x=years, y=aucs, mode='lines+markers',
                name=MODEL_NAMES[model], line={"color": color, "width": 3},
                marker={"size": 10}, hovertemplate='Year %{x}<br>ROC-AUC %{y:.4f}<extra></extra>',
            ))
    _style(fig, height=290)
    fig.update_yaxes(range=[0.5, 0.9], title='ROC-AUC (holdout by year)')
    return fig


def calibration_chart(saved_bins: pd.DataFrame) -> go.Figure:
    """Use original precomputed reliability bins; do not rebucket the holdout."""
    _require_plotly()
    data = saved_bins.dropna(subset=['mean_probability', 'event_frequency']).copy()
    if 'observations' in data:
        data = data.loc[data['observations'] > 0]
    fig = go.Figure()
    for model, color in [('xgboost_fixed', MINT), ('lstm_optuna_trial86', PURPLE)]:
        chunk = data.loc[data['model'] == model]
        if chunk.empty:
            continue
        counts = chunk['observations'].astype(float) if 'observations' in chunk else pd.Series(35, index=chunk.index)
        fig.add_trace(go.Scatter(
            x=chunk['mean_probability'] * 100,
            y=chunk['event_frequency'] * 100,
            mode='markers+lines', name=MODEL_NAMES[model],
            line={"color": color, "width": 1.5},
            marker={"color": color, "size": counts.clip(10, 250).pow(.5).mul(1.22).clip(7, 22)},
            customdata=counts.to_numpy().reshape(-1, 1),
            hovertemplate='Predicted mean: %{x:.2f}%<br>Observed rate: %{y:.2f}%<br>Sessions: %{customdata[0]:.0f}<extra></extra>',
        ))
    max_percent = max([40.0, *list(data['mean_probability'].mul(100)), *list(data['event_frequency'].mul(100))])
    limit = min(100, max_percent * 1.07)
    fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit],
                             name='Ideal alignment', mode='lines',
                             line={"color": '#7b8da7', "dash": 'dash', "width": 1.5}))
    _style(fig, height=380)
    fig.update_xaxes(title='Mean predicted probability (%)', range=[0, limit], ticksuffix='%')
    fig.update_yaxes(title='Observed event frequency (%)', range=[0, limit], ticksuffix='%')
    return fig


def var_chart(var: dict) -> go.Figure:
    """Only chart the saved observed and expected exception frequencies."""
    _require_plotly()
    labels, observed, expected = [], [], []
    for level in ('95', '99'):
        entry = var.get(level)
        if isinstance(entry, dict):
            labels.append(level + '% VaR')
            observed.append(float(entry['observed_rate']) * 100)
            expected.append(float(entry['expected_rate']) * 100)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=expected, name='Expected exception rate', marker_color='#4b7090',
                         text=[f'{v:.1f}%' for v in expected], textposition='outside'))
    fig.add_trace(go.Bar(x=labels, y=observed, name='Observed historical exception rate', marker_color=GOLD,
                         text=[f'{v:.2f}%' for v in observed], textposition='outside'))
    _style(fig, height=300)
    fig.update_layout(barmode='group')
    fig.update_yaxes(title='Exception rate (%)', rangemode='tozero', ticksuffix='%')
    return fig


def feature_chart(frame: pd.DataFrame, *, value: str, label: str,
                  color: str = MINT, threshold: float | None = None) -> go.Figure:
    _require_plotly()
    data = frame[['feature', value]].sort_values(value, ascending=True).copy()
    bar_colors = [CORAL if threshold is not None and v >= threshold else color for v in data[value]]
    fig = go.Figure(go.Bar(
        x=data[value], y=data['feature'], orientation='h', marker_color=bar_colors,
        customdata=data['feature'],
        hovertemplate='%{customdata}: %{x:.5f}<extra></extra>',
    ))
    _style(fig, height=max(340, 31 * len(data) + 75), legend=False)
    fig.update_xaxes(title=label, zeroline=True, zerolinecolor=GRID)
    fig.update_yaxes(automargin=True, title=None)
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash='dash', line_color=GOLD,
                      annotation_text='Investigate', annotation_position='top right')
    return fig
