"""FastAPI service.   POST /ask  {"question": "...", "plan": "Cedar Silver 3000", "k": 5}   GET /plans   GET /health

Run:  EMBEDDER=st LLM_URL=http://localhost:8000 uvicorn insurance_rag.api:create_app --factory --port 8080
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from insurance_rag.answer import answer_question
from insurance_rag.embeddings import make_embedder
from insurance_rag.llm import LLMError, VLLMClient
from insurance_rag.retrieval import Retriever
from insurance_rag.store import DEFAULT_DSN, ChunkStore


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    plan: str | None = None
    k: int = Field(5, ge=1, le=10)


def create_app(retriever: Retriever | None = None, llm=None, min_cosine: float | None = None) -> FastAPI:
    if retriever is None:
        emb = make_embedder(os.environ.get("EMBEDDER", "st"), os.environ.get("EMBED_MODEL"))
        retriever = Retriever(ChunkStore(emb.dim, os.environ.get("DATABASE_URL", DEFAULT_DSN),
                                         os.environ.get("TABLE", "chunks")), emb)
    if llm is None:
        llm = VLLMClient(os.environ.get("LLM_URL", "http://localhost:8000"),
                         os.environ.get("LLM_MODEL", "microsoft/Phi-3.5-mini-instruct"))
    app = FastAPI(title="Health insurance coverage assistant")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": retriever.store.count(), "embedder": retriever.embedder.name}

    @app.get("/plans")
    def plans():
        return {"plans": retriever.store.plans()}

    @app.post("/ask")
    def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(422, "empty question")
        if req.plan and req.plan not in retriever.store.plans():
            raise HTTPException(404, f"unknown plan '{req.plan}'")
        try:
            a = answer_question(req.question, retriever, llm, req.k, req.plan, min_cosine)
        except LLMError as e:
            raise HTTPException(502, str(e))
        return {
            "answer": a.text, "refused": a.refused, "grounded": a.grounded, "citations": a.cited,
            "invalid_citations": a.invalid_citations, "uncited": a.uncited, "short_circuited": a.short_circuited,
            "sources": [{"n": i, "plan": h.plan, "source": h.source, "page": h.page, "section": " > ".join(h.headings),
                         "text": h.text} for i, h in enumerate(a.sources, start=1)],
            "timings_ms": {"retrieval": round(a.retrieval_ms, 1), "llm": round(a.llm_ms, 1)},
            "disclaimer": "Informational only; not a coverage determination. Confirm with your insurer.",
        }

    return app
