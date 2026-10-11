"""Parse PDFs and write the chunks as plain text, WITHOUT a database or an embedding model.

Use it to (1) see what the parser extracted from your plan PDFs, (2) write an eval set whose answers you can read in the
text, and (3) check that every gold fact in an eval file really appears in the parsed text of its own plan.

  python -m insurance_rag.dump_chunks --pdf "plans/sbc1.pdf::Plan A" --parser docling --out out/chunks.txt
  python -m insurance_rag.dump_chunks --pdf "plans/sbc1.pdf::Plan A" --parser docling --check-eval data/my_eval.jsonl
"""
from __future__ import annotations

import argparse
import sys

from insurance_rag.evaluate import has_facts, load_eval
from insurance_rag.parse import parse


def render(chunks) -> str:
    out = []
    for i, c in enumerate(chunks):
        out.append(f"--- chunk {i} | plan={c.plan} | page={c.page} | headings={' > '.join(c.headings)}\n{c.text}\n")
    return "\n".join(out)


def check_eval(items: list[dict], chunks) -> list[str]:
    """Problems found: an answerable item whose facts are not all inside one chunk of its own plan, or an unknown plan."""
    plans = {c.plan for c in chunks}
    problems = []
    for it in items:
        if it["plan"] not in plans:
            problems.append(f"unknown plan {it['plan']!r}: {it['question']}")
        elif it["answerable"] and not any(c.plan == it["plan"] and has_facts(c.text, it["facts"]) for c in chunks):
            problems.append(f"facts {it['facts']} not found together in one {it['plan']} chunk: {it['question']}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="append", required=True, help='"path/to.pdf::Plan Name" (repeatable)')
    ap.add_argument("--parser", default="docling", choices=["docling", "pypdf"])
    ap.add_argument("--out", default=None, help="write the chunks to this text file")
    ap.add_argument("--check-eval", default=None, help="eval .jsonl to verify against the parsed text")
    a = ap.parse_args(argv)
    chunks = []
    for spec in a.pdf:
        if "::" not in spec:
            raise SystemExit(f"--pdf needs 'path::Plan Name', got {spec!r}")
        p, plan = spec.split("::", 1)
        got = parse(p, plan.strip(), a.parser)
        print(f"{p}: {len(got)} chunks")
        chunks += got
    if a.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(render(chunks))
        print("wrote", a.out)
    if a.check_eval:
        items = load_eval(a.check_eval)
        problems = check_eval(items, chunks)
        for p in problems:
            print("PROBLEM:", p)
        print(f"{len(items)} eval items checked, {len(problems)} problem(s)")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
