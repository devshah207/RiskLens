#!/usr/bin/env python3
"""Offline Phase 7 XBRL-first evaluation; never imports gold into answering path.

With --db, run the *existing* PDF RAG on unmatched questions through run_query.
Without --db, count those questions as pending PDF RAG (not wrong answers).
No live Groq calls.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from risklens.research.xbrl_rag_v1 import lookup_xbrl, run_query_xbrl_first


def expected_matches(result: dict, expected: str, unit: str) -> str:
    """The benchmark-only grader. This function is never called by the answerer."""
    cite = result['citations'][0]
    raw = Decimal(cite['raw_reported_value'])
    goal = Decimal(expected.replace(',', ''))
    if unit == 'crore':
        goal = goal * Decimal(10_000_000)
    elif unit == 'thousand_rupees':
        goal = goal * Decimal(1000)
    elif unit == 'rupees':
        if cite['unit_measures'] != ['iso4217:INR', 'xbrli:shares']:
            return 'wrong_quantity_unit'
    elif unit == 'percent':
        return 'not_comparable'
    else:
        return 'not_comparable'
    delta = abs(raw - goal)
    if delta == 0:
        return 'exact_match'
    dec = cite.get('decimals')
    if dec and dec.upper() != 'INF':
        try:
            tol = Decimal(10) ** (-int(dec)) / Decimal(2)
        except (InvalidOperation, ValueError):
            return 'unknown_rounding'
        if delta <= tol:
            return 'within_source_declared_rounding'
    return 'mismatch'


def run(questions: Path, archive: Path, output: Path, db: Path | None = None,
        bank_dir: Path | None = None, pdf_dir: Path | None = None) -> dict:
    cases = list(csv.DictReader(questions.open(encoding='utf-8', newline='')))
    if len(cases) != 34 or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('Expected the intact original 34-question benchmark')
    # PDF RAG resources are loaded only when a real index is supplied.
    store = None
    embedder = None
    if db is not None:
        if not db.is_file():
            raise FileNotFoundError(f'Original Phase 7 index missing: {db}')
        from risklens.research.index import ResearchStore
        from risklens.research.embeddings import SentenceEmbedder, create_embedder
        class LazySentence:
            name = SentenceEmbedder.name
            dimension = SentenceEmbedder.dimension
            def __init__(self): self._model = None
            def encode(self, values):
                if self._model is None:
                    self._model = create_embedder('sentence')
                return self._model.encode(values)
        store = ResearchStore(db)
        embedder = LazySentence()
    rows = []
    try:
        for case in cases:
            question = case['question']
            if store:
                extra = {}
                if bank_dir is not None: extra['bank_dir'] = bank_dir
                if pdf_dir is not None: extra['pdf_dir'] = pdf_dir
                result = run_query_xbrl_first(store, embedder, question, provider='offline',
                                   ticker=case['ticker'], xbrl_archive=archive, **extra)
            else:
                result = lookup_xbrl(question, archive=archive, ticker=case['ticker'])
                if result is None:
                    result = {'route': 'pdf_rag_pending_not_run', 'status': 'not_evaluated',
                              'answer': '', 'citations': []}
            is_xbrl = result.get('route') == 'xbrl_exact_fact'
            assessment = (expected_matches(result, case['expected_value'], case['expected_unit'])
                          if is_xbrl and case['kind'] == 'numeric' and case['expected_value'] else
                          'not_graded')
            cite = result['citations'][0] if is_xbrl else {}
            rows.append({
                'id': case['id'], 'kind': case['kind'], 'ticker': case['ticker'],
                'question': question, 'route': result.get('route'),
                'status': result['status'], 'answer': result.get('answer',''),
                'source_concept': cite.get('concept_qname',''),
                'source_file': cite.get('source_file',''),
                'period_kind': cite.get('period_kind',''),
                'period_start': cite.get('period_start',''),
                'period_end': cite.get('period_end',''),
                'declared_decimals': cite.get('decimals',''),
                'reference_check': assessment,
                'reference_value': case['expected_value'] if is_xbrl else '',
                'reference_unit': case['expected_unit'] if is_xbrl else '',
            })
    finally:
        if store is not None:
            # ResearchStore exposes a context manager and conn, not close().
            # This is the same SQLite cleanup performed by ResearchStore.__exit__.
            store.conn.close()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={
        'benchmark':'existing_34_questions_exposed_not_blind',
        'provider':'offline', 'mode':'xbrl_then_existing_pdf_rag' if db else 'xbrl_only_pdf_rag_not_run',
        'total':len(rows), 'numeric':sum(x['kind']=='numeric' for x in rows),
        'explanations':sum(x['kind']=='explanation' for x in rows),
        'unsupported_cases':sum(x['kind']=='unsupported' for x in rows),
        'xbrl_answers':sum(x['route']=='xbrl_exact_fact' for x in rows),
        'xbrl_exact_with_reference':sum(x['reference_check']=='exact_match' for x in rows),
        'xbrl_within_declared_rounding':sum(x['reference_check']=='within_source_declared_rounding' for x in rows),
        'xbrl_mismatched_with_reference':sum(x['reference_check']=='mismatch' for x in rows),
        'pdf_rag_not_evaluated':sum(x['route']=='pdf_rag_pending_not_run' for x in rows),
        'route_counts':dict(Counter(x['route'] for x in rows)),
        'note':'No Groq calls. No untested PDF fallback is counted as a wrong answer. Earlier questions are exposed development cases, not a blind accuracy evaluation.',
    }
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--xbrl-archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--db',type=Path,default=None,help='Optional existing Phase 7 SQLite index')
    p.add_argument('--bank-dir',type=Path,default=None)
    p.add_argument('--pdf-dir',type=Path,default=None)
    args=p.parse_args()
    print(json.dumps(run(args.questions,args.xbrl_archive,args.out,args.db,
                         args.bank_dir,args.pdf_dir),indent=2))
