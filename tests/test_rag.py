import json
import math
import re
import subprocess
import sys

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import needs_db
from insurance_rag import evaluate, sample_docs
from insurance_rag.answer import answer_question, check_citations
from insurance_rag.api import create_app
from insurance_rag.chunking import Chunk, chunk_pages, heading_level
from insurance_rag.embeddings import HashingEmbedder, make_embedder
from insurance_rag.ingest import ingest
from insurance_rag.llm import LLMError, VLLMClient
from insurance_rag.parse import parse, pypdf_pages
from insurance_rag.prompt import NOT_FOUND, build_messages, format_context
from insurance_rag.retrieval import Hit, Retriever, rrf_fuse
from insurance_rag.stub_llm import ExtractiveStubLLM


# ======================= chunking =======================
@pytest.mark.parametrize("line,expected", [
    ("# Deductibles", (1, "Deductibles")), ("### Sub topic", (3, "Sub topic")),
    ("PRIOR AUTHORIZATION", (1, "Prior Authorization")), ("2.1 Emergency services", (2, "2.1 Emergency services")),
    ("Specialist visit: $60 copay", None), ("The plan pays 80%", None), ("This is a sentence.", None),
    ("MRI", None), ("", None), ("COPAY $30", None)])
def test_heading_level(line, expected):
    assert heading_level(line) == expected


def test_chunks_respect_sections_pages_and_lines():
    pages = [(1, "FIRST SECTION\n" + "\n".join(f"Row {i}: item costs ${i} per visit today" for i in range(40))),
             (2, "Row 41: continues the table onto the next page here\nSECOND SECTION\nShort body with enough words to keep")]
    ch = chunk_pages(pages, "d", "P", "f.pdf", max_words=60, overlap_words=12)
    assert all(len(c.text.split()) <= 60 for c in ch)
    for c in ch:                                             # no line is ever cut in half
        for ln in c.text.splitlines():
            assert re.fullmatch(r"Row \d+: item costs \$\d+ per visit today|Row 41: .*|Short body.*", ln), ln
    assert [c.page for c in ch][:1] == [1] and ch[-1].page == 2
    assert ch[-1].headings == ["Second Section"] and ch[-2].headings == ["First Section"]
    first_page = [c for c in ch if c.page == 1]
    assert len(first_page) > 1
    a, b = first_page[0].text.splitlines(), first_page[1].text.splitlines()
    assert a[-1] == b[0], "consecutive chunks should overlap by whole lines"
    assert "Row 41" in ch[-2].text or "Row 41" in ch[-1].text and ch[-1].page == 2
    assert {ln.split(":")[0] for c in first_page for ln in c.text.splitlines()} >= {f"Row {i}" for i in range(40)}


def test_chunking_edge_cases():
    assert chunk_pages([(1, "")], "d", "P", "f") == []
    assert chunk_pages([(1, "HEADING ONLY")], "d", "P", "f") == []
    assert chunk_pages([(1, "A BIG HEADING\ntiny")], "d", "P", "f") == []          # body below min_words
    with pytest.raises(ValueError):
        chunk_pages([(1, "x")], "d", "P", "f", max_words=10, overlap_words=10)
    c = Chunk("d", "P", "f", 1, ["A", "B"], "body")
    assert c.embed_text() == "A > B\nbody" and Chunk("d", "P", "f", 1, [], "body").embed_text() == "body"


# ======================= embeddings =======================
def test_hashing_embedder():
    e = HashingEmbedder(256)
    v = e.embed_documents(["specialist visit copay", "specialist visit copay", "urgent care", ""])
    assert v.shape == (4, 256) and np.allclose(v[0], v[1]) and abs(np.linalg.norm(v[0]) - 1) < 1e-5
    assert not v[3].any()                                                      # empty text -> zero vector, no crash
    q = e.embed_queries(["specialist copay"])[0]
    assert q @ v[0] > q @ v[2]
    assert make_embedder("hash").dim == 1024
    with pytest.raises(ValueError):
        make_embedder("nope")


# ======================= parsing / sample docs =======================
def test_pypdf_parse_and_sample_consistency(sample):
    texts = {}
    for name, path in sample["pdfs"].items():
        pages = pypdf_pages(path)
        assert len(pages) == 2
        texts[name] = re.sub(r"\s+", " ", " ".join(t for _, t in pages)).lower()
    items = evaluate.load_eval(sample["eval"])
    assert len(items) == 2 * (12 + len(sample_docs.UNANSWERABLE))
    for it in items:                       # every gold fact must really be in its own plan's PDF
        if it["answerable"]:
            assert all(evaluate.norm(f) in texts[it["plan"]] for f in it["facts"]), it
        else:
            assert "bariatric" not in texts[it["plan"]] and "chiropractic" not in texts[it["plan"]]
    ch = parse(sample["pdfs"]["Cedar Gold 1000"], "Cedar Gold 1000", "pypdf")
    assert {"Deductibles And Out-Of-Pocket Limits", "Prior Authorization"} <= {h for c in ch for h in c.headings}
    assert all(c.plan == "Cedar Gold 1000" and c.page in (1, 2) for c in ch)


def test_parse_errors(sample, tmp_path):
    with pytest.raises(ValueError):
        parse(sample["pdfs"]["Cedar Gold 1000"], "x", "nope")
    with pytest.raises(ImportError):                       # docling is not installed in this environment
        parse(sample["pdfs"]["Cedar Gold 1000"], "x", "docling")
    from reportlab.pdfgen import canvas
    blank = tmp_path / "blank.pdf"; c = canvas.Canvas(str(blank)); c.showPage(); c.save()
    with pytest.raises(ValueError, match="no extractable text"):
        parse(str(blank), "x", "pypdf")


# ======================= RRF (pure) =======================
def _row(i, **kw):
    return {"id": i, "doc_id": "d", "plan": "P", "source": "s", "page": 1, "headings": [], "text": f"t{i}", "score": 0.5, **kw}


def test_rrf_hand_computed():
    vec = [_row(1), _row(2), _row(3)]; kw = [_row(3), _row(1), _row(4)]
    hits = rrf_fuse(vec, kw, k=10, rrf_k=60)
    s = {h.id: h.score for h in hits}
    assert s[1] == pytest.approx(1 / 61 + 1 / 62) and s[3] == pytest.approx(1 / 63 + 1 / 61)
    assert s[2] == pytest.approx(1 / 62) and s[4] == pytest.approx(1 / 63)
    assert [h.id for h in hits] == [1, 3, 2, 4]
    h1 = next(h for h in hits if h.id == 1)
    assert (h1.vector_rank, h1.keyword_rank) == (1, 2) and next(h for h in hits if h.id == 4).cosine is None
    assert len(rrf_fuse(vec, kw, k=2)) == 2 and rrf_fuse([], [], 5) == []


# ======================= pgvector store (real PostgreSQL) =======================
def _chunk(doc, plan, text, page=1, heads=("H",)):
    return Chunk(doc, plan, "f.pdf", page, list(heads), text)


@needs_db
def test_store_vector_search_known_order_filter_and_exact(make_store):
    s = make_store(3)
    vecs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    s.add([_chunk("a", "P1", "alpha"), _chunk("a", "P1", "beta"), _chunk("b", "P2", "gamma"), _chunk("b", "P2", "delta")], vecs)
    assert s.count() == 4 and s.plans() == ["P1", "P2"]
    hits = s.vector_search(np.array([1, 0, 0]), k=3)
    assert [h["text"] for h in hits][:2] == ["alpha", "beta"]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert hits[1]["score"] == pytest.approx(float(vecs[1] @ np.array([1, 0, 0])), abs=1e-5)
    assert [h["text"] for h in s.vector_search(np.array([1, 0, 0]), 5, plan="P2")] and \
        all(h["plan"] == "P2" for h in s.vector_search(np.array([1, 0, 0]), 5, plan="P2"))
    assert [h["id"] for h in s.vector_search(np.array([1, 0.2, 0]), 4)] == [h["id"] for h in s.vector_search(np.array([1, 0.2, 0]), 4, exact=True)]
    assert s.vector_search(np.array([1, 0, 0]), 5, plan="nope") == []


@needs_db
def test_store_reingest_replaces_and_validates(make_store):
    s = make_store(2)
    v = np.array([[1, 0], [0, 1]], dtype=np.float32)
    s.add([_chunk("d1", "P", "one"), _chunk("d1", "P", "two")], v)
    s.add([_chunk("d1", "P", "only")], v[:1])               # re-ingest same doc: old rows replaced
    assert s.count() == 1 and s.vector_search(np.array([1, 0]), 5)[0]["text"] == "only"
    with pytest.raises(ValueError):
        s.add([_chunk("d", "P", "x")], np.zeros((1, 5), np.float32))
    with pytest.raises(ValueError):
        s.add([_chunk("d", "P", "x")], np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError):
        type(s)(2, table="Bad-Name; DROP")
    with pytest.raises(ValueError, match="stores 2-dim"):
        type(s)(7, table=s.table)                             # same table, different embedder size


@needs_db
def test_store_keyword_search(make_store):
    s = make_store(2)
    v = np.tile(np.array([1, 0], np.float32), (3, 1))
    s.add([_chunk("d", "P1", "Specialist visit costs sixty dollars", heads=("Common Medical Events",)),
           _chunk("d", "P1", "Emergency room visits cost more", heads=("Emergency",)),
           _chunk("e", "P2", "Specialist visit costs forty dollars")], v)
    r = s.keyword_search("what does a specialist visit cost?", 5)
    assert {x["text"] for x in r} >= {"Specialist visit costs sixty dollars", "Specialist visit costs forty dollars"}
    assert [x["plan"] for x in s.keyword_search("specialist", 5, plan="P2")] == ["P2"]
    assert [x["text"] for x in s.keyword_search("medical events", 5)] == ["Specialist visit costs sixty dollars"]   # headings are indexed
    assert s.keyword_search("the of and", 5) == [] and s.keyword_search("???", 5) == [] and s.keyword_search("", 5) == []
    assert s.keyword_search("'; DROP TABLE x; --", 5) == [] and s.count() == 3          # hostile text is just text


# ======================= retrieval over the sample plans =======================
@needs_db
def test_retriever_modes_and_plan_filter(loaded):
    r, store, _ = loaded
    assert store.count() == 14
    h = r.search("How much do I pay to see a specialist?", k=3, plan="Cedar Gold 1000")
    assert h[0].plan == "Cedar Gold 1000" and "$40" in h[0].text and h[0].headings == ["Common Medical Events"]
    assert all(x.plan == "Cedar Gold 1000" for x in h)
    h = r.search("specialist copay", k=3, plan="Cedar Silver 3000")
    assert "$60" in h[0].text
    assert {x.plan for x in r.search("specialist copay", k=10)} == {"Cedar Silver 3000", "Cedar Gold 1000"}
    vec_only = r.search("specialist copay", k=3, mode="vector")
    kw_only = r.search("specialist copay", k=3, mode="keyword")
    assert all(x.keyword_rank is None for x in vec_only) and all(x.vector_rank is None for x in kw_only)
    with pytest.raises(ValueError):
        r.search("x", mode="magic")
    assert r.search("zzzqqq", k=3, mode="keyword") == []


@needs_db
def test_ingest_is_idempotent(loaded, sample):
    r, store, emb = loaded
    n = store.count()
    ingest([(sample["pdfs"]["Cedar Silver 3000"], "Cedar Silver 3000")], "pypdf", emb, store)
    assert store.count() == n


# ======================= prompt / citations / answer =======================
def _hit(i, text, plan="P", page=1, heads=("Sec",)):
    return Hit(i, "d", plan, "f.pdf", page, list(heads), text, 0.1, cosine=0.5)


def test_prompt_format():
    hits = [_hit(1, "alpha text", page=3), _hit(2, "beta text", page=None, heads=())]
    ctx = format_context(hits)
    assert ctx.startswith("[1] (P, f.pdf, p.3; Sec)\nalpha text") and "[2] (P, f.pdf)\nbeta text" in ctx
    m = build_messages("Q?", hits)
    assert m[0]["role"] == "system" and NOT_FOUND in m[0]["content"] and m[1]["content"].endswith("Question: Q?\nAnswer:")


def test_check_citations():
    assert check_citations("A [1] and B [3][1], also [7].", 3) == ([1, 3], [7])
    assert check_citations("no cites", 3) == ([], []) and check_citations("zero [0]", 3) == ([], [0])


class Scripted:
    def __init__(self, reply): self.reply, self.calls = reply, 0
    def chat(self, messages):
        self.calls += 1
        return self.reply(messages) if callable(self.reply) else self.reply


@needs_db
def test_answer_flows(loaded):
    r, _, _ = loaded
    a = answer_question("What is the urgent care copay?", r, Scripted("The urgent care copay is $50 [1]."), 3, "Cedar Gold 1000")
    assert a.cited == [1] and a.grounded and not a.refused and not a.uncited and a.retrieval_ms > 0 and a.llm_ms >= 0
    a = answer_question("q?", r, Scripted("It is $50."), 3, "Cedar Gold 1000")
    assert a.uncited and not a.grounded
    a = answer_question("q?", r, Scripted("It is $50 [9]."), 3, "Cedar Gold 1000")
    assert a.invalid_citations == [9] and not a.grounded
    a = answer_question("Does it cover bariatric surgery?", r, Scripted(NOT_FOUND), 3, "Cedar Gold 1000")
    assert a.refused and a.grounded and not a.uncited
    llm = Scripted("never called")
    a = answer_question("What is the urgent care copay?", r, llm, 3, "Cedar Gold 1000", min_cosine=0.99)
    assert a.short_circuited and a.refused and a.text == NOT_FOUND and llm.calls == 0
    a = answer_question("zzzqqq", r, llm, 3, "Cedar Gold 1000", min_cosine=None)
    assert a.sources == [] or llm.calls >= 0
    with pytest.raises(ValueError):
        answer_question("   ", r, llm)


# ======================= LLM client (mock transport) =======================
def _client(handler):
    return VLLMClient("http://x", "m", transport=httpx.MockTransport(handler))


def test_llm_client():
    seen = {}
    def ok(req):
        seen.update(json.loads(req.content)); seen["path"] = req.url.path
        return httpx.Response(200, json={"choices": [{"message": {"content": "  hi [1] "}}]})
    assert _client(ok).chat([{"role": "user", "content": "q"}]) == "hi [1]"
    assert seen["path"] == "/v1/chat/completions" and seen["temperature"] == 0.0 and seen["model"] == "m"
    for resp in (httpx.Response(500, text="boom"), httpx.Response(200, json={"oops": 1}),
                 httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]}), httpx.Response(200, text="not json")):
        with pytest.raises(LLMError):
            _client(lambda req, r=resp: r).chat([])
    def down(req): raise httpx.ConnectError("refused")
    with pytest.raises(LLMError, match="cannot reach"):
        _client(down).chat([])


# ======================= API =======================
@needs_db
def test_api(loaded):
    r, _, _ = loaded
    c = TestClient(create_app(r, ExtractiveStubLLM()))
    assert c.get("/health").json()["chunks"] == 14 and c.get("/plans").json()["plans"] == ["Cedar Gold 1000", "Cedar Silver 3000"]
    j = c.post("/ask", json={"question": "How much is the urgent care copay?", "plan": "Cedar Silver 3000"}).json()
    assert "$75" in j["answer"] and j["citations"] and j["grounded"] and j["sources"][0]["plan"] == "Cedar Silver 3000"
    assert j["sources"][0]["page"] in (1, 2) and "disclaimer" in j
    assert c.post("/ask", json={"question": "", "plan": None}).status_code == 422
    assert c.post("/ask", json={"question": "   "}).status_code == 422
    assert c.post("/ask", json={"question": "x", "k": 50}).status_code == 422
    assert c.post("/ask", json={"question": "x", "plan": "No Such Plan"}).status_code == 404

    class Down:
        def chat(self, m): raise LLMError("vllm is down")
    assert TestClient(create_app(r, Down())).post("/ask", json={"question": "deductible?"}).status_code == 502


# ======================= evaluation harness (exact expected values) =======================
def oracle_llm(messages):
    """Cites the first excerpt containing the question's gold fact; refuses the unanswerable questions."""
    user = messages[-1]["content"]; q = user.rsplit("Question:", 1)[1].split("Answer:")[0].strip()
    if q in sample_docs.UNANSWERABLE:
        return NOT_FOUND
    plan = re.search(r"\[1\] \(([^,]+),", user).group(1)          # the plan named in the first excerpt header
    item = next(i for i in sample_docs.eval_items() if i["question"] == q and i["plan"] == plan)
    for m in re.finditer(r"\[(\d+)\] \([^\n]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", user, re.S):
        if evaluate.has_facts(m.group(2), item["facts"]):
            return f"It is {' and '.join(item['facts'])} [{m.group(1)}]."
    return NOT_FOUND


@needs_db
def test_generation_eval_exact_values(loaded, sample):
    r, _, _ = loaded
    items = evaluate.load_eval(sample["eval"])
    # same question text exists for both plans, so make the oracle plan-aware through the plan filter used by the harness
    good = evaluate.generation_eval(r, Scripted(oracle_llm), items, k=5)
    assert good["n_answerable"] == 24 and good["n_unanswerable"] == 8
    assert good["fact_match"] == 1.0 and good["valid_citations"] == 1.0 and good["cited_supported"] == 1.0
    assert good["grounded_correct"] == 1.0
    assert good["false_refusal"] == 0.0 and good["correct_refusal"] == 1.0

    bad = evaluate.generation_eval(r, Scripted("It is $1 [9]."), items, k=5)
    assert bad["fact_match"] == 0.0 and bad["valid_citations"] == 0.0 and bad["cited_supported"] == 0.0 and bad["correct_refusal"] == 0.0

    wrong_cite = evaluate.generation_eval(r, Scripted("The answer is $3,000 [1]."), items, k=5)   # fluent but unsupported
    # fluent, cites a real excerpt, but states the wrong number: citation checks pass, the headline metric must not
    assert 0.0 < wrong_cite["fact_match"] < 0.1 and wrong_cite["valid_citations"] == 1.0
    assert wrong_cite["grounded_correct"] == wrong_cite["fact_match"] < 0.1

    refuse_all = evaluate.generation_eval(r, Scripted(NOT_FOUND), items, k=5)
    assert refuse_all["false_refusal"] == 1.0 and refuse_all["correct_refusal"] == 1.0 and refuse_all["fact_match"] == 0.0


@needs_db
def test_retrieval_eval_and_plan_filter_matters(loaded, sample):
    r, _, _ = loaded
    items = evaluate.load_eval(sample["eval"])
    h = evaluate.retrieval_eval(r, items, mode="hybrid")
    assert h["n_answerable"] == 24 and h["recall@5"] == 1.0 and h["mrr"] > 0.9
    nofilter = evaluate.retrieval_eval(r, items, plan_filter=False)
    assert nofilter["recall@1"] < h["recall@1"], "without the plan filter, near-identical plans must hurt precision"
    # a retriever that returns nothing relevant scores zero
    junk = [{"question": "zzz", "plan": "Cedar Gold 1000", "facts": ["not-in-any-doc"], "answerable": True}]
    assert evaluate.retrieval_eval(r, junk)["recall@5"] == 0.0 and evaluate.retrieval_eval(r, junk)["mrr"] == 0.0


def test_has_facts_and_load_eval(tmp_path):
    assert evaluate.has_facts("The  Deductible is\n$3,000.", ["$3,000", "deductible"]) and not evaluate.has_facts("x", ["y"])
    p = tmp_path / "e.jsonl"; p.write_text(json.dumps({"question": "q", "plan": "P", "facts": [], "answerable": True}) + "\n")
    with pytest.raises(ValueError):
        evaluate.load_eval(str(p))


def test_stub_llm_behaviour():
    llm = ExtractiveStubLLM()
    msgs = build_messages("What is the urgent care copay?", [_hit(1, "Urgent care: $75 copay per visit.\nOther line."), _hit(2, "Parking is free.")])
    assert llm.chat(msgs) == "Urgent care: $75 copay per visit. [1]"
    assert llm.chat(build_messages("Is bariatric surgery covered?", [_hit(1, "Parking is free.")])) == NOT_FOUND


@pytest.mark.parametrize("mod", ["ingest", "evaluate", "sample_docs", "dump_chunks"])
def test_cli_help(mod):
    r = subprocess.run([sys.executable, "-m", f"insurance_rag.{mod}", "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_dump_chunks_and_eval_check(sample, tmp_path):
    from insurance_rag import dump_chunks
    specs = [a for n, pth in sample["pdfs"].items() for a in ("--pdf", f"{pth}::{n}")]
    out = tmp_path / "chunks.txt"
    assert dump_chunks.main([*specs, "--parser", "pypdf", "--out", str(out), "--check-eval", sample["eval"]]) == 0
    text = out.read_text()
    assert "plan=Cedar Gold 1000" in text and "--- chunk 0" in text
    bad = tmp_path / "bad.jsonl"                                  # a fact that is not in the plan must be reported
    bad.write_text('{"question": "q", "plan": "Cedar Gold 1000", "facts": ["$999,999"], "answerable": true}\n'
                   '{"question": "q2", "plan": "Nope", "facts": [], "answerable": false}\n')
    assert dump_chunks.main([*specs, "--parser", "pypdf", "--check-eval", str(bad)]) == 1


def test_dol_eval_file_is_well_formed():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "dol_eval.jsonl")
    items = evaluate.load_eval(path)
    assert len(items) == 65 and {i["plan"] for i in items} == {"Sample Plan 2", "Sample Plan 3"}
    assert sum(not i["answerable"] for i in items) == 12 and all(i["facts"] for i in items if i["answerable"])
    assert len({(i["plan"], i["question"]) for i in items}) == 65          # no duplicates
