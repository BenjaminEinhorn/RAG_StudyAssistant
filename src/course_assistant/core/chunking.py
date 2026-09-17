"""Chunking: split page/slide text into retrievable units.

Chunks stay bound to their source page (slide) and inherit the page's rendered
image path, so a chunk retrieved by RAG can always show the supporting slide
*image* as visual evidence.

Method A (default, ``chunk_method="recursive"``): LangChain
RecursiveCharacterTextSplitter, paragraph-aware, with configurable size/overlap.
Method B (``chunk_method="fixed"``): fixed-size character splitter with overlap.
These two are compared in the README (see the answer-quality evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config.settings import Settings
from .parsing import ParsedDoc


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    page: int
    text: str
    image_path: str | None
    chunk_index: int


def _fallback_splitter():
    # import lazily so module loads even before langchain installed
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter


def build_splitter(method: str, size: int, overlap: int):
    """Return a callable ``split(text) -> list[str]``."""
    if method == "recursive":
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        sp = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return sp.split_text
    # fixed-size with right-overlap
    def fixed(text: str) -> list[str]:
        if len(text) <= size:
            return [text] if text.strip() else []
        out, i = [], 0
        while i < len(text):
            out.append(text[i:i + size])
            i += size - overlap
        return out
    return fixed


def chunk_document(parsed: ParsedDoc, settings: Settings,
                   method: str = "recursive") -> list[Chunk]:
    split_fn = build_splitter(method, settings.chunk_size, settings.chunk_overlap)
    doc_id = parsed.sha256[:16]
    chunks: list[Chunk] = []
    for page in parsed.pages:
        if not page.text.strip():
            continue
        pieces = split_fn(page.text)
        for idx, piece in enumerate(pieces):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_name=parsed.name,
                    page=page.page_num,
                    text=piece.strip(),
                    image_path=page.image_path,
                    chunk_index=idx,
                )
            )
    return chunks
