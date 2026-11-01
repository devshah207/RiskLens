#!/usr/bin/env python3
"""Read-only, checkpointed regression on previously exposed 44 questions.

This is NOT a new blind benchmark. Gold values are not passed to the answerer.
The offline run makes zero Groq requests; --live requires explicit approval.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import sleep

from risklens.research.index import ResearchStore
from risklens.research.embeddings import SentenceEmbedder, create_embedder
from risklens.research.llm_guided_v7 import BANKS, run_query


class LazySentence:
    name = SentenceEmbedder.name
    dimension = SentenceEmbedder.dimension

    def __init__(self):
        self.actual = None

    def encode(self, texts):
        if self.actual is None:
            self.actual = create_embedder('sentence')
        return self.actual.encode(texts)


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions', type=Path,
                   default=Path('docs/phase7_v7_exposed_acceptance_questions.json'))
    p.add_argument('--db', type=Path,
                   default=Path('data/processed/research/real_acceptance/real_index.sqlite3'))
    p.add_argument('--bank-dir', type=Path, default=BANKS)
    p.add_argument('--pdf-dir', type=Path, default=Path('data/raw/research_reports'))
    p.add_argument('--output', type=Path,
                   default=BANKS/'phase7_v7_exposed_acceptance_checkpoint.json')
    p.add_argument('--live', action='store_true')
    p.add_argument('--max-live', type=int, default=3)
    p.add_argument('--pause-seconds', type=float, default=65.0)
    p.add_argument('--max-questions', type=int, default=44)
    args = p.parse_args()
    if args.max_live < 1 or args.max_live > 5:
        p.error('max-live must be between 1 and 5')
    if args.max_questions < 1 or args.max_questions > 44:
        p.error('max-questions must be between 1 and 44')
    if args.pause_seconds < 0 or args.pause_seconds > 600:
        p.error('pause-seconds must be between 0 and 600')
    if not args.db.is_file():
        p.error('Missing Phase 7 index. No new database will be created')
    raw = json.loads(args.questions.read_text(encoding='utf-8'))
    cases = raw['questions'] if isinstance(raw, dict) else raw
    if not isinstance(cases, list) or len({x['id'] for x in cases}) != len(cases):
        raise ValueError('Invalid or duplicate benchmark questions')
    output = {'schema_version': 7, 'benchmark': 'EXPOSED_44_NOT_BLIND',
              'cases': {}, 'note': 'No automatic sign-off; source audit required.'}
    if args.output.is_file():
        output = json.loads(args.output.read_text(encoding='utf-8'))
        if output.get('benchmark') != 'EXPOSED_44_NOT_BLIND':
            raise RuntimeError('Checkpoint is from a different evaluation')
    embedder = LazySentence()
    calls = 0
    with ResearchStore(args.db) as store:
        if not store.list_documents():
            raise RuntimeError('Empty Phase 7 index; refusing to benchmark')
        for case in cases[:args.max_questions]:
            cid = case['id']
            prior = output['cases'].get(cid, {}).get('response', {})
            if cid in output['cases'] and not (args.live and
                 prior.get('status') in {'not_evaluated', 'provider_unavailable'}):
                continue
            q = case['question']
            # Zero-key offline source attribution first. No gold answer used.
            offline = run_query(store, embedder, q, provider='offline',
                                ticker=case.get('ticker'),
                                bank_dir=args.bank_dir, pdf_dir=args.pdf_dir)
            called_groq = False
            if not args.live or offline['status'] in {'answered', 'abstained', 'needs_review'}:
                result = offline
            else:
                if calls >= args.max_live:
                    break
                result = run_query(store, embedder, q, provider='groq',
                                   ticker=case.get('ticker'),
                                   bank_dir=args.bank_dir, pdf_dir=args.pdf_dir)
                calls += 1
                called_groq = True
            output['cases'][cid] = {'question': q, 'response': result}
            _atomic_json(args.output, output)
            print(cid, result['status'], result.get('route'), flush=True)
            if result['status'] == 'provider_unavailable':
                print('Groq unavailable. Stopping to protect API capacity.', flush=True)
                break
            if args.live and called_groq and calls < args.max_live:
                sleep(args.pause_seconds)
    print('CHECKPOINT:', args.output, 'recorded:', len(output['cases']), flush=True)


if __name__ == '__main__':
    main()
