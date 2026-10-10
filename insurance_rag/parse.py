"""PDF -> chunks. Two parsers behind one interface:

  docling : IBM Docling layout analysis + table structure recovery + its HybridChunker (tokenizer-aware, keeps headings
            and page numbers in chunk metadata). This is the intended parser for real SBC / policy PDFs.
            UNTESTED here: Docling downloads layout models from Hugging Face, which the build sandbox cannot reach.
  pypdf   : plain text extraction + our heading-aware chunker. Tested. No table structure, no OCR, so scanned PDFs fail.
"""
from __future__ import annotations

import os
import re

from pypdf import PdfReader

from insurance_rag.chunking import Chunk, chunk_pages


def doc_id_for(path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", os.path.splitext(os.path.basename(path))[0].lower()).strip("-")


def pypdf_pages(path: str) -> list[tuple[int, str]]:
    pages = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(PdfReader(path).pages)]
    if not any(t.strip() for _, t in pages):
        raise ValueError(f"{path}: no extractable text (scanned PDF? use the docling parser with OCR)")
    return pages


def parse_pypdf(path: str, plan: str, max_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    return chunk_pages(pypdf_pages(path), doc_id_for(path), plan, os.path.basename(path), max_words, overlap_words)


def parse_docling(path: str, plan: str, max_tokens: int = 400) -> list[Chunk]:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(path).document
    chunker = HybridChunker(max_tokens=max_tokens)
    out = []
    for ch in chunker.chunk(dl_doc=doc):
        pages = sorted({p.page_no for item in ch.meta.doc_items for p in item.prov})
        out.append(Chunk(doc_id_for(path), plan, os.path.basename(path), pages[0] if pages else None,
                         list(ch.meta.headings or []), ch.text))
    return out


def parse(path: str, plan: str, parser: str = "docling", **kw) -> list[Chunk]:
    if parser == "docling":
        return parse_docling(path, plan, **kw)
    if parser == "pypdf":
        return parse_pypdf(path, plan, **kw)
    raise ValueError(f"unknown parser '{parser}' (use 'docling' or 'pypdf')")
