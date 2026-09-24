"""Answer generation: RAG retrieval -> evidence -> grounded answer.

Returns structured data with separate ``answer`` and ``sources`` fields.
The model must reply with schema-constrained JSON::

    {"found": bool, "answer": str,
     "coverage": "full" | "partial" | "none", "beyond_slides": str,
     "citations": [{"source": <evidence number>, "excerpt": <verbatim quote>}]}

and every citation is then checked in code, never trusted:

- the cited number must be an item that retrieval actually returned
  (``in_evidence``), otherwise the citation is dropped and flagged;
- the quoted excerpt must occur in that item's text (``excerpt_found``).
  A citation with no excerpt counts as visual support only if the model was
  shown that slide's image.

When nothing relevant is found the assistant says so explicitly instead of
inventing an answer or citation. When a full answer would need knowledge from
outside the slides, the result says so (``outside_slides``) and names what the
slides do not cover; the code also raises that flag when the model claims an
answer but none of its citations checks out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config.settings import Settings
from ..services.chat import ChatProvider
from .index import Catalog

_NO_EVIDENCE = "<<no-evidence>>"
NOT_FOUND = ("I couldn't find enough supporting material in the loaded "
             "documents to answer this.")
MAX_IMAGES = 4
COVERAGE = ("full", "partial", "none")

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    # coverage is decided first (JSON is generated in property order), so a
    # partly covered question is answered for its covered part, not refused
    "required": ["coverage", "beyond_slides", "found", "answer", "citations"],
    "properties": {
        "coverage": {"type": "string", "enum": list(COVERAGE)},
        "beyond_slides": {"type": "string"},
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source", "excerpt"],
                "properties": {
                    "source": {"type": "integer"},
                    "excerpt": {"type": "string"},
                },
            },
        },
    },
}


@dataclass
class Source:
    doc_name: str
    page: int
    text: str
    image_path: str | None = None

    def label(self) -> str:
        return f"{self.doc_name} — slide/page {self.page}"


@dataclass
class Citation:
    number: int                 # evidence number the model cited (1-based)
    excerpt: str
    source: Source | None       # None when the number is not in the evidence
    in_evidence: bool
    excerpt_found: bool         # excerpt occurs verbatim in the cited text
    image_shown: bool           # the model saw this slide's image

    @property
    def supported(self) -> bool:
        if not self.in_evidence:
            return False
        return self.excerpt_found if self.excerpt.strip() else self.image_shown


@dataclass
class AnswerResult:
    answer: str
    sources: list[Source] = field(default_factory=list)      # cited + verified
    used_images: list[str] = field(default_factory=list)
    found_evidence: bool = True
    evidence: list[Source] = field(default_factory=list)     # everything retrieved
    citations: list[Citation] = field(default_factory=list)  # as the model cited
    coverage: str = "full"        # how much of the question the slides cover
    beyond_slides: str = ""       # what a full answer needs from outside them

    @property
    def outside_slides(self) -> bool:
        """True when answering fully would require going outside the slides."""
        return self.coverage != "full"

    @property
    def verified(self) -> bool:
        """True when the answer cites something and every citation checks out."""
        return bool(self.citations) and all(c.supported for c in self.citations)


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", re.sub(r"[^\w'\" ]+", " ", text)).strip()


def excerpt_in_text(excerpt: str, text: str) -> bool:
    """Whitespace/punctuation-insensitive substring check."""
    e = _norm(excerpt)
    return bool(e) and e in _norm(text)


def _assistant_system() -> str:
    return (
        "You are a helpful course assistant for the MBAX 6418 course "
        "(large language models for business). Answer the student's question "
        "using ONLY the numbered course-material evidence and any slide images "
        "shown to you.\n"
        "Rules:\n"
        "1. Reply with JSON: found (bool), answer (string), citations (list of "
        "{source, excerpt}). `source` is the evidence number you relied on. "
        "`excerpt` is a short phrase copied VERBATIM from that evidence's text; "
        "use an empty excerpt only when the support is what the slide image "
        "shows rather than its text.\n"
        "2. In the answer, mention document name and slide number for the "
        "evidence you use.\n"
        "3. If the evidence answers only part of the question, set found=true, "
        "answer just that part, and set coverage='partial'. If the evidence and "
        "images answer NONE of it, or the question is off-topic for the loaded "
        "materials, set found=false, "
        "citations=[], and answer: '" + NOT_FOUND + "' Do NOT invent answers, "
        "facts, or citations.\n"
        "4. When the question is about a picture/diagram/chart/meme on a "
        "slide, describe what the shown image contains and explain it.\n"
        "5. Set coverage: 'full' if the evidence fully answers the question; "
        "'partial' if it answers only part and a complete answer would need "
        "knowledge from outside the slides; 'none' if the slides do not answer "
        "it at all. For 'partial' or 'none', set beyond_slides to one sentence "
        "naming what the slides do not cover; otherwise leave it empty. Never "
        "fill the gap with outside knowledge in the answer.\n"
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
        if doc_names:
            slides = [s for s in slides if s.doc_name in set(doc_names)]
        return chunks, slides

    def answer(self, query: str, doc_names: list[str] | None = None,
               k: int = 5, include_images: bool = True) -> AnswerResult:
        chunks, slides = self.retrieve(query, doc_names, k)
        evidence = [Source(c.doc_name, c.page, c.text, c.image_path) for c in chunks]
        # slides found only by the visual index join the evidence list too
        seen_pages = {(s.doc_name, s.page) for s in evidence}
        for sl in slides:
            if (sl.doc_name, sl.page) not in seen_pages:
                evidence.append(Source(sl.doc_name, sl.page, sl.text, sl.image_path))
                seen_pages.add((sl.doc_name, sl.page))

        # images shown to the model: the best visual matches first, then the
        # slides behind the top text evidence
        used_images: list[str] = []
        if include_images:
            ranked = [sl.image_path for sl in slides] + [s.image_path for s in evidence[:3]]
            for img in ranked:
                if img and img not in used_images and len(used_images) < MAX_IMAGES:
                    used_images.append(img)

        found = len(evidence) > 0
        if not found:
            prompt = (f"{_NO_EVIDENCE}\n\nStudent question: {query}\n\n"
                      "No supporting course material was retrieved for this question.")
        else:
            by_image = {s.image_path: i for i, s in enumerate(evidence, 1) if s.image_path}
            context_block = "\n\n".join(
                f"[{i}] ({s.label()})\n{s.text.strip() or '(no text on this slide)'}"
                for i, s in enumerate(evidence, 1))
            image_block = "\n".join(
                f"Image {n} shows evidence [{by_image[img]}] "
                f"({evidence[by_image[img] - 1].label()})."
                for n, img in enumerate(used_images, 1))
            prompt = (
                "Course material evidence (retrieved by search):\n\n" + context_block
                + ("\n\nAttached slide images:\n" + image_block if image_block else "")
                + "\n\nStudent question: " + query
                + "\n\nAnswer using ONLY this evidence. Cite evidence numbers with "
                "verbatim excerpts. Answer whatever part it covers; set "
                "found=false only if it covers none of the question.")
        data = self.chat.complete_json(
            _assistant_system(), prompt, ANSWER_SCHEMA,
            images=used_images if include_images else None)

        answer_text = str(data.get("answer", "")).strip()
        model_found = bool(data.get("found", False))
        citations: list[Citation] = []
        for c in data.get("citations") or []:
            try:
                n = int(c.get("source"))
            except (TypeError, ValueError):
                n = 0
            excerpt = str(c.get("excerpt") or "")
            src = evidence[n - 1] if 1 <= n <= len(evidence) else None
            citations.append(Citation(
                number=n, excerpt=excerpt, source=src, in_evidence=src is not None,
                excerpt_found=bool(src) and excerpt_in_text(excerpt, src.text),
                image_shown=bool(src and src.image_path in used_images)))
        beyond = str(data.get("beyond_slides") or "").strip()
        coverage = data.get("coverage")
        if coverage not in COVERAGE:
            coverage = "full" if model_found else "none"
        if not model_found:
            citations = []
            answer_text = answer_text or NOT_FOUND
            coverage = "none"
        sources: list[Source] = []
        for c in citations:
            if c.supported and c.source not in sources:
                sources.append(c.source)
        if model_found and not sources and coverage == "full":
            # the model claims a full answer, but nothing it cited checks out
            coverage = "partial"
            beyond = beyond or ("None of the cited slide text could be verified, "
                                "so this answer may rely on knowledge from "
                                "outside the slides.")
        if coverage == "none" and not beyond:
            beyond = "The loaded slides do not cover this question."
        return AnswerResult(
            answer=answer_text,
            sources=sources,
            used_images=used_images,
            found_evidence=found and model_found,
            evidence=evidence,
            citations=citations,
            coverage=coverage,
            beyond_slides=beyond if coverage != "full" else "",
        )
