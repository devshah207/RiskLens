"""Local transformer embeddings; explicit deterministic hash fixture for offline QA only."""
from __future__ import annotations

from hashlib import sha256
import math
import re

from .contracts import ResearchError


class HashFixtureEmbedder:
    """Not semantic embeddings. Only for synthetic tests, demos and no-download smoke."""

    name = "fixture-hash-64-not-semantic"
    dimension = 64

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dimension
            for token in re.findall(r"[a-z0-9]+", t.lower()):
                h = int.from_bytes(sha256(token.encode()).digest()[:8], "big")
                v[h % self.dimension] += 1
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class SentenceEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384

    def __init__(self, *, local_files_only: bool = False) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ResearchError("Install optional research-embeddings dependencies") from exc
        try:
            self.model = SentenceTransformer(self.name, local_files_only=local_files_only)
        except Exception as exc:
            raise ResearchError(f"Could not initialize local embedding model: {exc}") from exc
        if self.model.get_sentence_embedding_dimension() != self.dimension:
            raise ResearchError("Unexpected embedding dimension; refusing to mix vector spaces")

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]


def create_embedder(mode: str):
    if mode == "sentence":
        return SentenceEmbedder()
    if mode == "hash-fixture":
        return HashFixtureEmbedder()
    raise ResearchError(f"Unknown embedding mode: {mode}")
