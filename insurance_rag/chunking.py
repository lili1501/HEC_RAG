"""Heading-aware chunking of policy text.

Policy documents are organised by section ("DEDUCTIBLES", "COMMON MEDICAL EVENTS", ...), and a copay only makes sense
together with its section and plan. So a chunk (a) never spans two sections, (b) never splits a line, which keeps
table rows such as "Specialist visit: $60 copay" intact, (c) never spans pages (so citations carry an exact page),
and (d) records its heading path, which is prepended when the chunk is embedded.

This is the fallback chunker used with the pypdf text parser. With Docling, `parse.docling_chunks` uses Docling's own
layout-aware HybridChunker instead (untested here); both produce the same `Chunk` objects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    doc_id: str
    plan: str
    source: str
    page: int | None
    headings: list[str]
    text: str
    extra: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        return (" > ".join(self.headings) + "\n" + self.text) if self.headings else self.text


_MD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z][A-Za-z0-9 ,&/'\-]{2,70})$")
_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9 ,&/'\-]{3,70}$")


def heading_level(line: str) -> tuple[int, str] | None:
    """(level, title) if the line looks like a section heading, else None. Heuristic, see README limitations."""
    s = line.strip()
    if not s or "$" in s or "%" in s or s.endswith((".", ":", ",")):
        return None
    m = _MD.match(s)
    if m:
        return len(m.group(1)), m.group(2)
    m = _NUMBERED.match(s)
    if m and len(s.split()) <= 10:
        return m.group(1).count(".") + 1, s
    if _CAPS.match(s) and re.search(r"[A-Z]{3}", s) and len(s.split()) <= 10:
        return 1, s.title()
    return None


def _words(s: str) -> int:
    return len(s.split())


def chunk_pages(pages: list[tuple[int, str]], doc_id: str, plan: str, source: str,
                max_words: int = 180, overlap_words: int = 30, min_words: int = 6) -> list[Chunk]:
    """pages: [(page_number, text)]. Returns chunks in reading order."""
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    chunks: list[Chunk] = []
    stack: list[str] = []                       # current heading path, carried across pages

    for page_no, text in pages:
        buf: list[str] = []
        head: list[str] = list(stack)

        def flush(keep_overlap: bool):
            nonlocal buf
            body = "\n".join(buf).strip()
            if _words(body) >= min_words:
                chunks.append(Chunk(doc_id, plan, source, page_no, list(head), body))
            if keep_overlap:                    # carry the last lines forward so a fact split across chunks stays findable
                tail, n = [], 0
                for ln in reversed(buf):
                    if n + _words(ln) > overlap_words:
                        break
                    tail.insert(0, ln); n += _words(ln)
                buf = tail
            else:
                buf = []

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            h = heading_level(line)
            if h:
                flush(False)
                level, title = h
                stack = stack[:level - 1] + [""] * max(0, level - 1 - len(stack)) + [title]
                head = [x for x in stack if x]
                continue
            if buf and _words("\n".join(buf)) + _words(line) > max_words:
                flush(True)
            buf.append(line)
        flush(False)
    return chunks
