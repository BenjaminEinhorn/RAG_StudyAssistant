"""Answer generation: RAG retrieval -> evidence -> grounded answer.

Returns structured data with separate ``answer`` and ``sources`` fields,
and validates that every source is a real retrieved item (never hallucinated).
When nothing relevant is found the assistant says so explicitly instead of
inventing an answer or citation.

The ``answer`` method is a thin, testable wrapper: given a query and the doc
selection, it (1) retrieves evidence via :class:`Catalog`, (2) builds a prompt
that instructs the model to answer ONLY from the evidence and to refuse
missing-info questions, (3) calls the chat provider (optionally passing the
matching slide images for visual questions), and (4) returns validated output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config.settings import Settings
from ..services.chat import ChatProvider
from .index import Catalog

_NO_EVIDENCE = "<<no-evidence>>"


@dataclass
class Source:
    doc_name: str
    page: int
    text: str
    image_path: str | None = None

    def label(self) -> str:
        return f"{self.doc_name} — slide/page {self.page}"


@dataclass
class AnswerResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    used_images: list[str] = field(default_factory=list)
    found_evidence: bool = True


def _assistant_system() -> str:
    return (
        "You are a helpful course assistant for the MBAX 6418 course "
        "(large language models for business). Answer the student's question "
        "using ONLY the provided course-material excerpts and any slide images "
        "shown to you. Quote or paraphrase the material precisely. "
        "Rules:\n"
        "1. Cite the source(s) — document name and slide/page number — inside "
        "the answer using [Source: <doc>, slide N] markers.\n"
        "2. If the excerpts and images do NOT contain enough information to "
        "answer, or the question is off-topic for the loaded materials, say so "
        "explicitly: 'I couldn't find enough supporting material in the loaded "
        "documents to answer this.' Do NOT invent answers, facts, or citations.\n"
        "3. When the question is about a picture/diagram/chart on a slide, "
        "describe what the shown image contains and explain it.\n"
        "Answer in plain, concise prose.\n"
    )


class Assistant:
    def __init__(self, catalog: Catalog, chat: ChatProvider,
                 settings: Settings):
        self.catalog = catalog
        self.chat = chat
        self.settings = settings

    def retrieve(self, query: str, doc_names: list[str] | None = None,
                 k: int = 5, include_visual: bool = True):
        """Return evidence (text chunks + matching slide images) for a query."""
        chunks = self.catalog.search(query, k=k, doc_names=doc_names)
        slides = self.catalog.visual_search(query, k=3) if include_visual else []
        return chunks, slides

    def answer(self, query: str, doc_names: list[str] | None = None,
               k: int = 5, include_images: bool = True) -> AnswerResult:
        chunks, slides = self.retrieve(query, doc_names, k)
        srcs = [
            Source(c.doc_name, c.page, c.text, c.image_path)
            for c in chunks
        ]
        # relevant slide images: prefer the ones matching visual search, then
        # images attached to the top chunks.
        used_images: list[str] = []
        if include_images:
            seen: set[str] = set()
            # always show the primary source's slide image
            if srcs and srcs[0].image_path:
                used_images.append(srcs[0].image_path)
                seen.add(srcs[0].image_path)
            for s in slides:
                if s.image_path and s.image_path not in seen:
                    used_images.append(s.image_path)
                    seen.add(s.image_path)
            for s in srcs[:3]:
                if s.image_path and s.image_path not in seen:
                    used_images.append(s.image_path)
                    seen.add(s.image_path)

        found = len(srcs) > 0
        if not found:
            prompt = (
                f"{_NO_EVIDENCE}\n\nStudent question: {query}\n\n"
                "No supporting course material was retrieved for this question."
            )
        else:
            context_block = "\n\n".join(
                f"[{i}] ({s.label()})\n{s.text}"
                for i, s in enumerate(srcs, 1)
            )
            prompt = (
                "Course material excerpts (retrieved by search):\n\n"
                + context_block
                + "\n\nStudent question: " + query
                + "\n\nAnswer the question using ONLY the excerpts and images. "
                "Cite sources with [Source: <doc>, slide N]. If the material "
                "does not contain the answer, say so explicitly and do not "
                "invent anything."
            )
        answer_text = self.chat.complete(
            _assistant_system(), prompt,
            images=used_images if include_images else None,
        ).strip()
        return AnswerResult(
            answer=answer_text,
            sources=srcs,
            used_images=used_images,
            found_evidence=found,
        )
