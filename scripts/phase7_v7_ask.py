#!/usr/bin/env python3
"""Opt-in Phase 7 LLM-guided report QA; no changes to existing research-ask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from risklens.research.index import ResearchStore
from risklens.research.embeddings import SentenceEmbedder, create_embedder
from risklens.research.llm_guided_v7 import BANKS, MODEL, run_query


class LazySentence:
    name = SentenceEmbedder.name
    dimension = SentenceEmbedder.dimension

    def __init__(self):
        self._model = None

    def encode(self, texts):
        if self._model is None:
            self._model = create_embedder('sentence')
        return self._model.encode(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--provider', choices=['groq', 'offline'], default='offline')
    parser.add_argument('--db', type=Path,
                        default=Path('data/processed/research/real_acceptance/real_index.sqlite3'))
    parser.add_argument('--pdf-dir', type=Path, default=Path('data/raw/research_reports'))
    parser.add_argument('--bank-dir', type=Path, default=BANKS)
    parser.add_argument('--ticker', default=None)
    parser.add_argument('--fiscal-year', default=None)
    parser.add_argument('--as-of', default=None)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--no-second-review', action='store_true',
                        help='No extra Groq page review for unverified numeric answers')
    parser.add_argument('--vision-review', action='store_true',
                        help='Opt in to rendered-PDF Groq Vision verification')
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 20:
        parser.error('--top-k must be 1..20')
    if not args.db.is_file():
        parser.error('The existing Phase 7 document index is missing; refusing to make an empty new DB')
    with ResearchStore(args.db) as store:
        if not store.list_documents():
            parser.error('The selected Phase 7 index contains no source documents')
        answer = run_query(store, LazySentence(), args.question,
                           provider=args.provider, ticker=args.ticker,
                           fiscal_year=args.fiscal_year, as_of=args.as_of,
                           top_k=args.top_k, model=args.model,
                           bank_dir=args.bank_dir, pdf_dir=args.pdf_dir,
                           verify_financial=not args.no_second_review,
                           verify_vision=args.vision_review)
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
