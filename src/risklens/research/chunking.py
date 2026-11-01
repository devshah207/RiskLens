"""Page-preserving, deterministic span-based chunker with section/table context."""
from __future__ import annotations

from hashlib import sha256
import re

from .contracts import Chunk, Page, ResearchError, stable_id

_WORD = re.compile(r"\S+")
_HEADING = re.compile(r"^(?:#{1,5}\s+|[A-Z][A-Z /&\-]{8,75}$)")


def _section_at(text: str, start: int) -> str:
    section = ""
    for line in text[:start].splitlines():
        line = line.strip()
        if _HEADING.match(line) and len(line) <= 120:
            section = line.lstrip("# ")
    return section


def make_chunks(page: Page, *, max_tokens: int = 450, overlap: int = 50) -> list[Chunk]:
    """Exact character spans; never mix two pages or silently remove table lines.

    Token budget is a whitespace proxy, not the transformer tokenizer's length.
    Oversized table rows are split with overlap and always retain page provenance.
    """
    if not (64 <= max_tokens <= 2000) or not (0 <= overlap < max_tokens // 2):
        raise ResearchError("Chunk parameters must satisfy 64<=max_tokens<=2000 and 0<=overlap<max_tokens/2")
    text = page.text
    spans = list(_WORD.finditer(text))
    if not spans:
        return []
    output: list[Chunk] = []
    ix = 0
    while ix < len(spans):
        stop = min(ix + max_tokens, len(spans))
        if stop < len(spans):
            # Seek a nearby line break to avoid breaking table rows or sentences.
            for j in range(stop - 1, max(ix + max_tokens * 3 // 4, ix + 1), -1):
                if "\n" in text[spans[j - 1].end():spans[j].start()]:
                    stop = j
                    break
        start_char = spans[ix].start()
        end_char = spans[stop - 1].end()
        part = text[start_char:end_char]
        if part.strip():
            output.append(Chunk(
                chunk_id=stable_id(page.doc_id, page.page_no, start_char, end_char,
                                   sha256(part.encode("utf-8")).hexdigest()),
                doc_id=page.doc_id,
                page_no=page.page_no,
                start=start_char,
                end=end_char,
                text=part,
                section=_section_at(text, start_char),
                review_required=page.review_required,
            ))
        if stop == len(spans):
            break
        ix = max(ix + 1, stop - overlap)
    return output
