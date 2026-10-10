"""A fake 'LLM' for tests and offline demos: returns the best-overlapping sentence from the excerpts with a citation,
or the refusal. It is NOT a language model; metrics produced with it only exercise the harness, never quote them."""
from __future__ import annotations

import re

from insurance_rag.embeddings import _STOP
from insurance_rag.prompt import NOT_FOUND


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$%]+", s.lower()) if w not in _STOP}


class ExtractiveStubLLM:
    def __init__(self, min_overlap: int = 2):
        self.min_overlap = min_overlap
        self.calls = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls += 1
        user = messages[-1]["content"]
        question = user.rsplit("Question:", 1)[1].split("Answer:")[0]
        qw = _words(question)
        best, best_n, best_score = None, 0, 0
        for m in re.finditer(r"\[(\d+)\] \([^\n]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", user, re.S):
            for sent in re.split(r"(?<=[.!?])\s+|\n", m.group(2)):
                sc = len(qw & _words(sent))
                if sc > best_score:
                    best, best_n, best_score = sent.strip(), int(m.group(1)), sc
        if best is None or best_score < self.min_overlap:
            return NOT_FOUND
        return f"{best} [{best_n}]"
