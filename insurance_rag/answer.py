"""Retrieve -> prompt -> generate -> check citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from insurance_rag.prompt import NOT_FOUND, build_messages
from insurance_rag.retrieval import Hit, Retriever

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    cited: list[int]                     # valid excerpt numbers (1-based) the answer cites
    invalid_citations: list[int]         # cited numbers with no matching excerpt (a hallucinated citation)
    uncited: bool                        # a substantive answer with no citation at all
    sources: list[Hit] = field(default_factory=list)
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    short_circuited: bool = False        # refused without calling the LLM because retrieval looked irrelevant

    @property
    def grounded(self) -> bool:
        return self.refused or (bool(self.cited) and not self.invalid_citations)


def check_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    nums = [int(x) for x in _CITE.findall(text)]
    return sorted({n for n in nums if 1 <= n <= n_sources}), sorted({n for n in nums if not 1 <= n <= n_sources})


def answer_question(question: str, retriever: Retriever, llm, k: int = 5, plan: str | None = None,
                    min_cosine: float | None = None) -> Answer:
    """min_cosine: if set and no retrieved chunk reaches it, answer NOT_FOUND without calling the LLM.
    The right value depends on the embedding model: calibrate it on questions you know are unanswerable."""
    q = question.strip()
    if not q:
        raise ValueError("empty question")
    t0 = time.perf_counter()
    hits = retriever.search(q, k=k, plan=plan)
    r_ms = (time.perf_counter() - t0) * 1000

    if not hits or (min_cosine is not None and max((h.cosine or 0.0) for h in hits) < min_cosine):
        return Answer(q, NOT_FOUND, True, [], [], False, hits, r_ms, 0.0, short_circuited=True)

    t1 = time.perf_counter()
    text = llm.chat(build_messages(q, hits))
    l_ms = (time.perf_counter() - t1) * 1000
    refused = NOT_FOUND.rstrip(".").lower() in text.lower()
    cited, invalid = check_citations(text, len(hits))
    return Answer(q, text, refused, cited, invalid, uncited=(not refused and not cited and not invalid),
                  sources=hits, retrieval_ms=r_ms, llm_ms=l_ms)
