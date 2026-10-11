"""Evaluation harness: retrieval quality and answer quality against an eval set built from known facts.

Eval item: {"question", "plan", "facts": ["$3,000", ...], "answerable": true|false}
A chunk / answer "contains the facts" if every fact string appears in it (case- and whitespace-insensitive).

Retrieval: Recall@k = share of answerable questions with an answer-bearing chunk (from the right plan) in the top k;
           MRR = mean reciprocal rank of the first such chunk.
Answers (needs an LLM): fact_match (answer states all facts), valid_citations (cites >=1 excerpt, none invalid),
           cited_supported (a cited excerpt from the right plan contains the facts), false_refusal (refuses an answerable
           question), correct_refusal (refuses an unanswerable one), grounded_correct (the headline number: the answer states
           the facts AND cites an excerpt of the right plan that contains them).
These are string-match proxies, cheaper and stricter than an LLM judge, but they can miss correct paraphrases
(e.g. "thirty dollars"); report them as such.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from insurance_rag.answer import answer_question
from insurance_rag.retrieval import Retriever


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def has_facts(text: str, facts: list[str]) -> bool:
    t = norm(text)
    return all(norm(f) in t for f in facts)


_DECLINE = re.compile(r"could not find|couldn'?t find|no (?:specific |relevant )?information|"
                      r"(?:do|does) not (?:specify|provide|contain|mention|include|state|say|list)|not (?:specified|provided|mentioned)", re.I)


def declines(text: str) -> bool:
    """Heuristic: the answer says the documents do not contain the information, in ANY wording. Evaluation only.
    The strict flag (`refused`) needs the exact refusal sentence, which a model often paraphrases."""
    return bool(_DECLINE.search(text))


def load_eval(path: str) -> list[dict]:
    with open(path) as f:
        items = [json.loads(line) for line in f if line.strip()]
    for it in items:
        if it["answerable"] and not it["facts"]:
            raise ValueError(f"answerable item without facts: {it['question']}")
    return items


def _pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def retrieval_eval(retriever: Retriever, items: list[dict], ks=(1, 3, 5), plan_filter: bool = True,
                   mode: str = "hybrid") -> dict:
    ans = [it for it in items if it["answerable"]]
    kmax = max(ks)
    ranks = []
    for it in ans:
        hits = retriever.search(it["question"], k=kmax, plan=it["plan"] if plan_filter else None, mode=mode)
        r = next((i for i, h in enumerate(hits, 1) if h.plan == it["plan"] and has_facts(h.text, it["facts"])), None)
        ranks.append(r)
    out = {f"recall@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in ks}
    out["mrr"] = float(np.mean([0.0 if r is None else 1.0 / r for r in ranks]))
    out.update({"n_answerable": len(ans), "mode": mode, "plan_filter": plan_filter})
    return out


def generation_eval(retriever: Retriever, llm, items: list[dict], k: int = 5, plan_filter: bool = True,
                    min_cosine: float | None = None) -> dict:
    rows, r_ms, l_ms = [], [], []
    for it in items:
        a = answer_question(it["question"], retriever, llm, k, it["plan"] if plan_filter else None, min_cosine)
        r_ms.append(a.retrieval_ms)
        if not a.short_circuited:
            l_ms.append(a.llm_ms)
        row = {"question": it["question"], "plan": it["plan"], "answerable": it["answerable"], "refused": a.refused,
               "declined": declines(a.text), "answer": a.text}
        if it["answerable"]:
            # `answer_facts` = the value(s) a correct answer must state; `facts` (below) = strings that locate the source chunk
            row["fact_match"] = has_facts(a.text, it.get("answer_facts") or it["facts"]) and not a.refused
            row["valid_citations"] = bool(a.cited) and not a.invalid_citations
            row["cited_supported"] = any(a.sources[n - 1].plan == it["plan"] and has_facts(a.sources[n - 1].text, it["facts"])
                                         for n in a.cited)
            row["grounded_correct"] = bool(row["fact_match"] and row["cited_supported"])
        rows.append(row)
    ans = [r for r in rows if r["answerable"]]; un = [r for r in rows if not r["answerable"]]
    m = lambda rs, key: float(np.mean([r[key] for r in rs])) if rs else float("nan")
    return {
        "n_answerable": len(ans), "n_unanswerable": len(un),
        "fact_match": m(ans, "fact_match"), "valid_citations": m(ans, "valid_citations"),
        "cited_supported": m(ans, "cited_supported"), "grounded_correct": m(ans, "grounded_correct"),
        "false_refusal": m(ans, "refused"),
        "correct_refusal": m(un, "refused"),
        # same two measures with the looser paraphrase-tolerant detector (see `declines`)
        "false_refusal_loose": m(ans, "declined"),
        "correct_refusal_loose": m(un, "declined"),
        "retrieval_ms_p50": _pct(r_ms, 50), "retrieval_ms_p95": _pct(r_ms, 95),
        "llm_ms_p50": _pct(l_ms, 50), "llm_ms_p95": _pct(l_ms, 95),
        "rows": rows,
    }


def main():
    from insurance_rag.embeddings import make_embedder
    from insurance_rag.llm import VLLMClient
    from insurance_rag.store import DEFAULT_DSN, ChunkStore
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--embedder", default="st", choices=["st", "hash"])
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--llm", default="vllm", choices=["vllm", "none"], help="'none' = retrieval metrics only")
    ap.add_argument("--llm-url", default="http://localhost:8000")
    ap.add_argument("--llm-model", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--table", default="chunks")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="reports/eval.json")
    a = ap.parse_args()
    emb = make_embedder(a.embedder, a.embed_model)
    retriever = Retriever(ChunkStore(emb.dim, a.dsn, a.table), emb)
    items = load_eval(a.eval)
    rep = {"retrieval": {m: retrieval_eval(retriever, items, mode=m) for m in ("vector", "keyword", "hybrid")},
           "retrieval_hybrid_no_plan_filter": retrieval_eval(retriever, items, plan_filter=False)}
    if a.llm == "vllm":
        rep["generation"] = generation_eval(retriever, VLLMClient(a.llm_url, a.llm_model), items, a.k)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps({k: (v if k != "generation" else {x: y for x, y in v.items() if x != "rows"}) for k, v in rep.items()}, indent=2))


if __name__ == "__main__":
    main()
