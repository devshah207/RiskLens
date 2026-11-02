"""A read-only, source-anchored explanation of existing RiskLens architecture.

This is narrative metadata, NEVER prediction logic or an answer generator.
The explainer never invokes ML inference, Groq, document extraction, or evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

_PUBLIC_ROOT = 'https://github.com/devshah207/RiskLens/blob/main/'


def _node(code: str, title: str, subtitle: str, x: int, y: int,
          plain: str, technical: str, evidence: str, source: str,
          kind: str = 'process') -> dict:
    return {
        'id': code, 'title': title, 'subtitle': subtitle, 'position': {'x': x, 'y': y},
        'plain': plain, 'technical': technical, 'evidence': evidence,
        'sourcePath': source, 'kind': kind,
    }


def _edge(source: str, target: str, label: str = '') -> dict:
    return {'id': f'{source}--{target}', 'source': source, 'target': target, 'label': label}


def _story(story_id: str, title: str, subtitle: str, nodes: list[dict],
           edges: list[dict], journey: list[str], footer: str) -> dict:
    ids = {node['id'] for node in nodes}
    if len(ids) != len(nodes) or any(s not in ids for s in journey):
        raise ValueError('Invalid project story blueprint')
    if any(e['source'] not in ids or e['target'] not in ids for e in edges):
        raise ValueError('Invalid project story edge')
    return {'id': story_id, 'title': title, 'subtitle': subtitle, 'nodes': nodes,
            'edges': edges, 'journey': journey, 'footer': footer}


def _frozen_holdout_summary(root: Path) -> dict:
    """Display only a *saved* summary; never calculate fresh holdout scores."""
    path = root / 'data/processed/phase5_final_holdout/final_holdout_report.json'
    if not path.is_file():
        return {'verified': False, 'detail': 'Original holdout JSON is not mounted in this checkout; see docs/PHASE5_HOLDOUT_FINDINGS.md.'}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        scores = data['model_scores']
        frozen = data['selected_before_opening']
        xgb = scores['xgboost_fixed']
        lstm = scores['lstm_optuna_trial86']
        n = data['eligible_holdout_rows']
        confusion = xgb['confusion_matrix']
        if (n != 433 or frozen != 'fixed_xgboost_11'
                or abs(float(xgb['roc_auc']) - 0.67275606) > 1e-6
                or abs(float(lstm['roc_auc']) - 0.68487395) > 1e-6
                or confusion['tp'] + confusion['fn'] != 42
                or lstm['confusion_matrix']['tp'] + lstm['confusion_matrix']['fn'] != 42):
            raise ValueError('Noncanonical frozen report')
        return {
            'verified': True, 'detail': (
                f'Loaded saved report: {n} holdout sessions, 42 actual events. '
                f'XGBoost ROC-AUC {xgb["roc_auc"]:.3f}; '
                f'LSTM #86 ROC-AUC {lstm["roc_auc"]:.3f}. '
                f'Both models had 0% recall at the original 0.50 threshold.'
            ),
        }
    except (ValueError, KeyError, TypeError, OSError):
        return {'verified': False, 'detail': 'Original report could not be validated for this walkthrough; consult frozen source evidence directly.'}


def build_stories(root: Path) -> dict:
    """Return a static narrative + an optional read-only frozen-report note."""
    root = Path(root)
    score = _frozen_holdout_summary(root)
    market_nodes = [
        _node('market-data', 'Historical market data', 'NIFTY 50 · OHLC', 10, 100,
              'We start with genuine historical daily prices and checks that reject impossible price records.',
              'OHLC means open, high, low and close. Validated 2016–September 2026 data; the research target is based on the next completed session.',
              'Original market data stays local and is not included in this animated walkthrough.',
              'src/risklens/data/validation.py', 'source'),
        _node('market-time', 'Time alignment', 'No future information', 300, 100,
              'Every input must have been known when the market closed. Future prices are never allowed into the model inputs.',
              'Chronological folds and boundary label purges; India VIX and S&P 500 are aligned to earlier completed observations.',
              'A next-session closing price is used only to label historical outcomes, not as an input.',
              'src/risklens/features/dataset.py', 'guard'),
        _node('market-features', 'Feature engineering', '11 frozen inputs', 590, 100,
              'The program transforms past prices into signals describing momentum, trend and volatility.',
              'Frozen features include returns/ranges, RSI-14, EMA-50 distance, rolling volatility and external market context. Features remain unchanged.',
              'No features were added after examining the sealed final holdout.',
              'src/risklens/evaluation/features.py'),
        _node('market-xgb', 'XGBoost champion', 'Fixed selection', 900, -20,
              'An ensemble of decision trees produces a downside-event probability.',
              'The fixed 11-feature XGBoost was selected before the final holdout was opened and remains the research champion.',
              'Champion was not switched after the challenger showed stronger holdout ROC-AUC.',
              'src/risklens/evaluation/final_holdout.py', 'model'),
        _node('market-lstm', 'LSTM challenger', 'Optuna trial #86', 900, 220,
              'A sequential neural network uses recent history to estimate the same downside-event probability.',
              'The frozen Optuna trial #86 LSTM consumes a 45-session sequence. It remains the challenger and is not retuned.',
              'Historical comparison only; not a new training run or ensemble.',
              'src/risklens/evaluation/final_holdout.py', 'model'),
        _node('market-holdout', 'One-time validation', '433 sessions · 42 events', 1230, 100,
              'We compare already-saved predictions with what actually happened, rather than claim the model is always right.',
              'The 2025–September 2026 final holdout was opened once. At threshold 0.50, both classifiers had zero recall. ROC-AUC describes ranking, not working alerts.',
              score['detail'], 'src/risklens/evaluation/final_holdout.py', 'evidence'),
        _node('market-govern', 'Explain & govern', 'VaR · SHAP · PSI', 1530, 100,
              'We inspect how the models behave, where inputs shifted and what the saved risk checks warn about.',
              'Historical VaR is independent of the classifier. XGBoost TreeSHAP, LSTM permutation sensitivity, feature PSI, drift investigation and freeze governance are separate diagnostics.',
              'Original saved findings include clustered 95% VaR breaches, monitored distribution shifts and documented model limitations.',
              'src/risklens/evaluation/monitoring.py', 'governance'),
    ]
    market_edges = [
        _edge('market-data','market-time'), _edge('market-time','market-features'),
        _edge('market-features','market-xgb'), _edge('market-features','market-lstm'),
        _edge('market-xgb','market-holdout'), _edge('market-lstm','market-holdout'),
        _edge('market-holdout','market-govern'),
    ]
    research_nodes = [
        _node('research-question','Financial question','Company · year · scope',20,100,
              'A person asks a financial question about an actual company filing.',
              'The question must identify the issuer, financial year, reporting basis (standalone/consolidated) and unit when needed.',
              'The walkthrough never submits a query or sends anything to an external provider.',
              'app/pages/2_Financial_Research.py','source'),
        _node('research-guard','Interpret request','No silent assumptions',300,100,
              'The system checks that the requested company and accounting context are consistent.',
              'Phase 8 adapter validates conflicts before delegating the question; the backend enforces accounting concept, time period, unit and scope for exact facts.',
              'A question about an ambiguous year or reporting scope must not be silently converted into another question.',
              'app/services/filing_research.py','guard'),
        _node('research-xbrl','Exact XBRL lookup','8 NSE XML filings',590,100,
              'It first looks for an exact tagged financial value in the original NSE filings.',
              'The XBRL parser reads eight company/scope XML instances and checks reporting context, concept, year, dates, units and issuer.',
              'A valid exact match can produce a fact and original XML citation WITHOUT calling Groq.',
              'src/risklens/research/xbrl_rag_v1.py','model'),
        _node('research-answer','Direct XBRL answer','No Groq needed',920,-55,
              'If the exact financial fact is found, the system can return the figure with its original source.',
              'The exact-fact route returns source context, units, accounting concept and provenance as supplied by the original parser.',
              'An answered status does not equal an independently human-certified accounting answer.',
              'src/risklens/research/xbrl_rag_v1.py','evidence'),
        _node('research-pdf','PDF retrieval fallback','BM25 + MiniLM',920,210,
              'If there is no exact XBRL fact, the system searches indexed annual-report pages for useful passages.',
              'The original indexed PDF retrieval combines lexical BM25 with MiniLM semantic embeddings and identifies candidate source pages.',
              'Retrieved candidate pages are NOT automatically verified citations or generated answers.',
              'src/risklens/research/llm_guided_v7.py','process'),
        _node('research-groq','Optional Groq reasoning','Explicit consent only',1240,210,
              'With consent, Groq may read retrieved passages and draft a source-grounded response.',
              'Live Groq requests are only permitted after explicit user authorization; the existing Phase 7 verification policy can trigger multiple API calls.',
              'Offline mode finds evidence but does not pretend a generated answer exists.',
              'app/services/filing_research.py','model'),
        _node('research-review','Evidence & abstention','Answer · review · refuse',1540,100,
              'The final page shows an answer, a needs-review warning, or a refusal—and where its evidence came from.',
              'Original statuses and citations are preserved. No answer is promoted to human-certified; quoted passages and untrusted content are handled as data.',
              'Historical exposed benchmark: 18 source-supported, 6 correct refusals, 4 incorrect refusals, 6 untested. Not an independent blind test.',
              'src/risklens/research/llm_guided_v7.py','governance'),
    ]
    research_edges = [
        _edge('research-question','research-guard'), _edge('research-guard','research-xbrl'),
        _edge('research-xbrl','research-answer','Exact fact'),
        _edge('research-xbrl','research-pdf','No exact fact'),
        _edge('research-answer','research-review'), _edge('research-pdf','research-groq'),
        _edge('research-groq','research-review'),
    ]
    result = {
        'market': _story('market', 'Market Risk Analytics',
                         'Saved model research, explained step by step', market_nodes, market_edges,
                         ['market-data','market-time','market-features','market-xgb','market-lstm','market-holdout','market-govern'],
                         'Frozen, historical evidence only · Not live trading or forward testing.'),
        'research': _story('research', 'Financial Research Assistant',
                           'Source-constrained financial answers through the original Phase 7 pipeline',
                           research_nodes, research_edges,
                           ['research-question','research-guard','research-xbrl','research-answer','research-pdf','research-groq','research-review'],
                           'Two alternative routes: exact XBRL match OR PDF/Groq fallback. Groq never runs automatically.'),
        'context': {'savedHoldoutVerified': score['verified'],
                    'publicRepository': 'https://github.com/devshah207/RiskLens',
                    'notes': 'Architecture walkthrough only; no ML/RAG calls are performed.'},
    }
    # Do not expose local filesystem paths or file contents to client.
    return deepcopy(result)


def source_link(path: str) -> str:
    """Only audited local source paths become repository links."""
    if not isinstance(path, str) or not path.startswith(('src/risklens/', 'app/')) or '..' in path.split('/'):
        raise ValueError('Unapproved source path')
    return _PUBLIC_ROOT + path
