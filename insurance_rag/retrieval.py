"""Hybrid retrieval: pgvector cosine search + Postgres full-text search, merged with Reciprocal Rank Fusion (RRF).

Why both: embeddings find paraphrases ("what do I pay to see a specialist" -> "Specialist visit: $60 copay"), keyword
search finds exact tokens that embeddings blur (CPT codes, "$350", drug names, "PET scan"). RRF needs no score
calibration between the two systems: each list contributes 1/(60 + rank).
"""
from __future__ import annotations

from dataclasses import dataclass

from insurance_rag.embeddings import Embedder
from insurance_rag.store import ChunkStore

RRF_K = 60


@dataclass
class Hit:
    id: int
    doc_id: str
    plan: str
    source: str
    page: int | None
    headings: list[str]
    text: str
    score: float
    cosine: float | None = None          # set when the chunk appeared in the vector results
    vector_rank: int | None = None
    keyword_rank: int | None = None


def rrf_fuse(vector_rows: list[dict], keyword_rows: list[dict], k: int = 10, rrf_k: int = RRF_K) -> list[Hit]:
    by_id: dict[int, Hit] = {}
    for name, rows in (("vector", vector_rows), ("keyword", keyword_rows)):
        for rank, r in enumerate(rows, start=1):
            h = by_id.get(r["id"])
            if h is None:
                h = by_id[r["id"]] = Hit(**{f: r[f] for f in ("id", "doc_id", "plan", "source", "page", "headings", "text")},
                                         score=0.0)
            h.score += 1.0 / (rrf_k + rank)
            if name == "vector":
                h.vector_rank, h.cosine = rank, float(r["score"])
            else:
                h.keyword_rank = rank
    return sorted(by_id.values(), key=lambda h: (-h.score, h.id))[:k]


class Retriever:
    def __init__(self, store: ChunkStore, embedder: Embedder, candidates: int = 20):
        self.store, self.embedder, self.candidates = store, embedder, candidates

    def search(self, query: str, k: int = 5, plan: str | None = None, mode: str = "hybrid") -> list[Hit]:
        if mode not in ("hybrid", "vector", "keyword"):
            raise ValueError("mode must be hybrid, vector or keyword")
        n = max(self.candidates, k)
        vec = self.store.vector_search(self.embedder.embed_queries([query])[0], n, plan) if mode != "keyword" else []
        kw = self.store.keyword_search(query, n, plan) if mode != "vector" else []
        return rrf_fuse(vec, kw, k)
