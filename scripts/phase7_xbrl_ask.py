#!/usr/bin/env python3
"""Run original RiskLens PDF RAG with exact NSE XBRL answers as the first route.

Works on an XBRL match even without a local PDF index. If a question needs
PDF evidence, requires the pre-existing Phase 7 SQLite index. Never builds
or modifies the index. --provider groq must be explicitly selected.
"""
import argparse
import json
from pathlib import Path
from risklens.research.xbrl_rag_v1 import lookup_xbrl, run_query_xbrl_first


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('question')
    p.add_argument('--xbrl-archive', type=Path, required=True)
    p.add_argument('--db', type=Path, default=Path('data/processed/research/real_acceptance/real_index.sqlite3'))
    p.add_argument('--bank-dir', type=Path, default=Path('data/processed/research/real_acceptance/phase7_completion'))
    p.add_argument('--pdf-dir', type=Path, default=Path('data/raw/research_reports'))
    p.add_argument('--ticker', default=None)
    p.add_argument('--fiscal-year', default=None)
    p.add_argument('--as-of', default=None)
    p.add_argument('--provider', choices=['offline','groq'], default='offline')
    p.add_argument('--top-k', type=int, default=5)
    args=p.parse_args()
    if not 1 <= args.top_k <= 20:
        p.error('--top-k must be 1..20')
    try:
        direct=lookup_xbrl(args.question, archive=args.xbrl_archive,
                           ticker=args.ticker, fiscal_year=args.fiscal_year,
                           as_of=args.as_of)
    except (ValueError, FileNotFoundError) as exc:
        p.error(f'Invalid XBRL source: {exc}')
    if direct is not None:
        print(json.dumps(direct, ensure_ascii=False, indent=2))
        return
    if not args.db.is_file():
        print(json.dumps({
            'question':args.question, 'answer':'No exact XBRL match; the existing PDF RAG index is missing.',
            'status':'not_evaluated','route':'pdf_rag_requires_existing_index','citations':[]
        },ensure_ascii=False,indent=2))
        return
    from risklens.research.index import ResearchStore
    from risklens.research.embeddings import SentenceEmbedder, create_embedder
    class LazySentence:
        name=SentenceEmbedder.name
        dimension=SentenceEmbedder.dimension
        def __init__(self):self._model=None
        def encode(self,txt):
            if self._model is None:self._model=create_embedder('sentence')
            return self._model.encode(txt)
    with ResearchStore(args.db) as store:
        if not store.list_documents():
            p.error('The existing index has no documents')
        result=run_query_xbrl_first(
            store,LazySentence(),args.question,xbrl_archive=args.xbrl_archive,
            provider=args.provider,ticker=args.ticker,fiscal_year=args.fiscal_year,
            as_of=args.as_of,top_k=args.top_k,bank_dir=args.bank_dir,pdf_dir=args.pdf_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':main()
