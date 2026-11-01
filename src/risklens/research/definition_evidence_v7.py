"""Recover explicit financial definitions from ORIGINAL hash-checked PDF pages.

This module never writes to the index or the financial fact banks. It is not a
semantic claim checker: only an explicit same-page, same-metric definition can
be returned directly. Other matching pages merely supplement Groq evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

from .contracts import Chunk, Document, Hit, stable_id


@dataclass(frozen=True)
class DefinitionEvidence:
    hit: Hit
    explicit_definition: str | None


def strict_definition_request(question: str) -> bool:
    """A literal request for a definition or formula, not a 'why/how changed'."""
    q = ' '.join(question.casefold().split())
    return bool(re.search(
        r'\b(?:define|defined|definition|meaning|formula|means|mean)\b', q))


def _phrase_pattern(label: str) -> re.Pattern[str] | None:
    words = re.findall(r'[a-z0-9]+', label.casefold())
    if len(words) < 2 or len(words) > 12:
        return None
    return re.compile(r'(?<!\w)' + r'\s+'.join(map(re.escape, words)) + r'(?!\w)', re.I)


def _direct_definition(flat_text: str, metric: re.Pattern[str]) -> str | None:
    """Return the exact source sentence only when its verb defines the term."""
    # Non-explanatory sentences such as 'deposits are growing' must not pass.
    defining_predicate = re.compile(
        r'\s+(?:is|are)\s+(?:the|a|an)\s+'
        r'(?:(?:daily|annual|weighted|net|gross)\s+){0,2}'
        r'(?:ratio|percentage|average|measure|amount|sum|difference|total|'
        r'proportion|rate|calculation|number|value)\b|'
        r'\s+(?:is|are)\s+(?:defined|calculated|computed|measured)\s+as\b|'
        r'\s+(?:means|refers\s+to|equals)\b|\s*=\s*',
        re.I,
    )
    for match in metric.finditer(flat_text):
        tail = flat_text[match.end():match.end() + 560]
        predicate = defining_predicate.match(tail)
        if predicate is None:
            continue
        end = re.search(r'(?<=[.!?])(?:\s|$)', tail)
        if end is None or end.end() > 510:
            continue
        sentence = flat_text[match.start():match.end() + end.start()].strip()
        if 20 <= len(sentence) <= 530:
            return sentence
    return None


def find_definition_evidence(
    intent: Any, documents: list[Document], pdf_dir: Path,
    *, limit: int = 3,
) -> list[DefinitionEvidence]:
    """Find exact metric pages in eligible indexed originals, without Groq.

    Requires an explicit ticker to avoid mixing banks. Rechecks full PDF hashes
    before reading, and keeps physical PDF page numbers in each evidence hit.
    Fail closed on an altered/missing PDF; never scan other local files.
    """
    if intent.kind not in {'definition', 'explanation'} or not intent.ticker:
        return []
    metric = _phrase_pattern(intent.metric_label)
    if metric is None:
        return []
    try:
        import pymupdf
    except ImportError:
        return []
    matched: list[DefinitionEvidence] = []
    for doc in documents:
        if doc.spec.ticker != intent.ticker:
            continue
        path = pdf_dir / Path(doc.spec.local_path).name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != doc.sha256:
            continue
        with pymupdf.open(path) as pdf:
            if len(pdf) != doc.pages:
                continue
            for i, page in enumerate(pdf):
                original = page.get_text(sort=False)
                # Whitespace normalization preserves the label/definition
                # wording while connecting lines split by PDF extraction.
                flat = ' '.join(original.split())
                if not metric.search(flat):
                    continue
                definition = _direct_definition(flat, metric)
                if definition is None:
                    # A metric label on a page is only supporting evidence,
                    # never proof that a definition was found.
                    pos = metric.search(flat)
                    assert pos is not None
                    start = max(0, pos.start() - 140)
                    passage = flat[start:min(len(flat), pos.end() + 850)]
                else:
                    passage = definition
                chunk = Chunk(
                    chunk_id=stable_id('explicit-definition-search', doc.sha256,
                                       i + 1, intent.metric_label),
                    doc_id=doc.doc_id, page_no=i + 1,
                    start=0, end=len(passage), text=passage,
                    section='PDF label definition lexical recovery',
                )
                matched.append(DefinitionEvidence(
                    Hit(chunk, doc, 1.0 if definition else 0.5,
                        ('original_pdf_lexical',)), definition))
    # Prefer a real definition to a financial table that only mentions the term.
    matched.sort(key=lambda item: (item.explicit_definition is None,
                                   item.hit.document.spec.ticker,
                                   item.hit.chunk.page_no))
    # Never truncate away a conflicting explicit source definition.
    explicit = [item for item in matched if item.explicit_definition]
    supporting = [item for item in matched if not item.explicit_definition]
    return [*explicit, *supporting[:limit]]


def verified_definition_answer(
    question: str, evidence: list[DefinitionEvidence],
) -> dict[str, Any] | None:
    """Answer verbatim only if all matching explicit definitions agree.

    Different definitions in the eligible source set require review; neither
    the first PDF nor a model's preference may silently win.
    """
    valid = [e for e in evidence if e.explicit_definition]
    if not valid:
        return None
    claims = {' '.join(e.explicit_definition.casefold().split()) for e in valid}
    if len(claims) != 1:
        return {
            'question': question,
            'answer': 'Different explicit definitions were found in the indexed reports.',
            'status': 'needs_review',
            'route': 'conflicting_original_pdf_definitions',
            'verification_status': 'conflicting_source_definitions',
            'citations': [_citation(e) for e in valid[:3]],
        }
    chosen = valid[0]
    return {
        'question': question,
        'answer': chosen.explicit_definition,
        'status': 'answered',
        'provider': 'original_pdf',
        'route': 'explicit_original_pdf_definition',
        'verification_status': 'original_pdf_hash_and_definition_phrase_checked',
        'citations': [_citation(chosen)],
    }


def _citation(item: DefinitionEvidence) -> dict[str, Any]:
    hit = item.hit
    return {
        'chunk_id': hit.chunk.chunk_id,
        'file_sha256': hit.document.sha256,
        'ticker': hit.document.spec.ticker,
        'pdf_page': hit.chunk.page_no,
        'source_url': hit.document.spec.source_url,
        'source_file': Path(hit.document.spec.local_path).name,
        'source_quote': item.explicit_definition or hit.chunk.text,
        'verification': 'original_pdf_file_hash_and_exact_phrase_checked',
    }
