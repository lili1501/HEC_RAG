"""Text embedders. `SentenceTransformerEmbedder` is the real one; `HashingEmbedder` is a deterministic, dependency-free
stand-in that captures only word overlap, used to test the retrieval plumbing (it says nothing about semantic quality)."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


_STOP = set("a an and are as at be by can do does for from has have how i if in is it me my of on or the to what when "
            "which who will with you your".split())


class HashingEmbedder:
    name = "hashing-standin"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = [t for t in re.findall(r"[a-z0-9$%]+", text.lower()) if t not in _STOP]
        feats = [(t, 1.0) for t in toks] + [(a + "_" + b, 0.5) for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for tok, w in feats:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += w * (1.0 if (h >> 63) & 1 else -1.0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    embed_queries = embed_documents


class SentenceTransformerEmbedder:
    """Default model: intfloat/e5-base-v2 (MIT, Microsoft), which expects 'query: ' / 'passage: ' prefixes.
    UNTESTED here (model weights could not be downloaded in the build sandbox)."""

    def __init__(self, model_id: str = "intfloat/e5-base-v2", query_prefix: str = "query: ",
                 doc_prefix: str = "passage: ", device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id, device=device)
        self.name, self.qp, self.dp, self.bs = model_id, query_prefix, doc_prefix, batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def _enc(self, texts, prefix):
        return self.model.encode([prefix + t for t in texts], batch_size=self.bs, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)

    def embed_documents(self, texts):
        return self._enc(texts, self.dp)

    def embed_queries(self, texts):
        return self._enc(texts, self.qp)


def make_embedder(kind: str, model_id: str | None = None) -> Embedder:
    if kind == "hash":
        return HashingEmbedder()
    if kind == "st":
        return SentenceTransformerEmbedder(model_id) if model_id else SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder '{kind}' (use 'st' or 'hash')")
