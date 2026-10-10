"""Parse PDFs, embed the chunks, and load them into pgvector.

  python -m insurance_rag.ingest --pdf "data/sample/cedar_silver_3000.pdf::Cedar Silver 3000" --parser pypdf --embedder hash
  python -m insurance_rag.ingest --pdf "plans/aetna_sbc.pdf::Aetna Silver 2025" --parser docling --embedder st
"""
from __future__ import annotations

import argparse
import json
import time

from insurance_rag.embeddings import make_embedder
from insurance_rag.parse import parse
from insurance_rag.store import DEFAULT_DSN, ChunkStore


def ingest(pdfs: list[tuple[str, str]], parser: str, embedder, store: ChunkStore, batch: int = 64) -> dict:
    t0, n_chunks = time.time(), 0
    for path, plan in pdfs:
        chunks = parse(path, plan, parser)
        if not chunks:
            raise ValueError(f"{path}: parsed to zero chunks")
        import numpy as np
        vecs = np.concatenate([embedder.embed_documents([c.embed_text() for c in chunks[i:i + batch]])
                               for i in range(0, len(chunks), batch)])
        n_chunks += store.add(chunks, vecs)
    return {"documents": len(pdfs), "chunks": n_chunks, "seconds": round(time.time() - t0, 1), "total_in_store": store.count()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="append", required=True, help='"path/to.pdf::Plan Name" (repeatable)')
    ap.add_argument("--parser", default="docling", choices=["docling", "pypdf"])
    ap.add_argument("--embedder", default="st", choices=["st", "hash"])
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--table", default="chunks")
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the table (needed when changing embedder)")
    a = ap.parse_args()
    pairs = []
    for spec in a.pdf:
        if "::" not in spec:
            raise SystemExit(f"--pdf needs 'path::Plan Name', got {spec!r}")
        p, plan = spec.split("::", 1)
        pairs.append((p, plan.strip()))
    emb = make_embedder(a.embedder, a.embed_model)
    store = ChunkStore(emb.dim, a.dsn, a.table, a.recreate)
    print(json.dumps(ingest(pairs, a.parser, emb, store), indent=2))


if __name__ == "__main__":
    main()
