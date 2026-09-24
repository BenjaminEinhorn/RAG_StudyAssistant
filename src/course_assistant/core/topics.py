"""General topic categories per document, for the quiz topic dropdown.

When a document is added, the class LLM reads the opening line of every slide
or page (usually its title) and names the 3-8 broad categories the document
covers. The result is cached in ``data/topics.json`` keyed by document name, so
each document costs one LLM call. Documents loaded before this cache existed
(e.g. by ``seed_data.py``) get their topics the first time they are needed.
"""
from __future__ import annotations

import json
from collections import Counter

from ..config.settings import Settings
from ..services.chat import ChatProvider, LocalEchoChat
from .index import Catalog

MIN_TOPICS, MAX_TOPICS = 3, 8
_MAX_OUTLINE_CHARS = 12000

TOPICS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["topics"],
    "properties": {"topics": {"type": "array", "items": {"type": "string"}}},
}


def _outline(catalog: Catalog, doc_name: str) -> list[tuple[int, str]]:
    """(page, first line) for every page of a document, in page order."""
    first: dict[int, str] = {}
    for c in sorted(catalog._chunks.values(), key=lambda c: (c.page, c.chunk_index)):
        if c.doc_name != doc_name or c.page in first:
            continue
        line = next((ln.strip() for ln in c.text.splitlines() if ln.strip()), "")
        if line:
            first[c.page] = line[:120]
    return sorted(first.items())


def _clean(topics) -> list[str]:
    out, seen = [], set()
    for t in topics or []:
        t = " ".join(str(t).split()).strip(" .")
        if t and t.lower() not in seen and len(t) <= 60:
            seen.add(t.lower())
            out.append(t)
    return out[:MAX_TOPICS]


def extract_topics(catalog: Catalog, chat: ChatProvider, doc_name: str) -> list[str]:
    outline = _outline(catalog, doc_name)
    if not outline:
        return []
    if isinstance(chat, LocalEchoChat):
        # offline stand-in (tests only): the most repeated slide titles
        counts = Counter(line for _, line in outline)
        return _clean([t for t, _ in counts.most_common(MAX_TOPICS)])
    lines = "\n".join(f"slide/page {p}: {line}" for p, line in outline)
    data = chat.complete_json(
        "You organise course material for a study app. Reply with JSON only.",
        f"Below is the opening line (usually the title) of every slide or page "
        f"of the course document '{doc_name}'.\n\n{lines[:_MAX_OUTLINE_CHARS]}\n\n"
        f"List the {MIN_TOPICS} to {MAX_TOPICS} general topic categories this "
        "document covers, suitable as quiz topics. Each is a short noun phrase "
        "(2-5 words), broad enough to span several slides, and named in the "
        "material's own terms. Skip housekeeping such as agenda, logistics, "
        "title or thank-you slides. Order them as they appear.",
        TOPICS_SCHEMA)
    return _clean(data.get("topics"))


class TopicStore:
    """Cached topic categories per document name (``data/topics.json``)."""

    def __init__(self, settings: Settings):
        self.path = settings.data_dir / "topics.json"
        try:
            self._topics: dict[str, list[str]] = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            self._topics = {}

    def _save(self):
        self.path.write_text(json.dumps(self._topics, indent=1))

    def add(self, catalog: Catalog, chat: ChatProvider, doc_name: str) -> list[str]:
        self._topics[doc_name] = extract_topics(catalog, chat, doc_name)
        self._save()
        return self._topics[doc_name]

    def drop(self, doc_name: str):
        if self._topics.pop(doc_name, None) is not None:
            self._save()

    def ensure(self, catalog: Catalog, chat: ChatProvider):
        """Fill in topics for loaded documents that have none cached, and
        forget documents that are no longer loaded."""
        loaded = {d["doc_name"] for d in catalog.list_documents()}
        for name in sorted(loaded - set(self._topics)):
            self.add(catalog, chat, name)
        for name in set(self._topics) - loaded:
            self.drop(name)

    def topics_for(self, doc_names: list[str] | None) -> list[str]:
        """Union of the topics of the given documents (all when None/empty),
        in document order, without duplicates."""
        out, seen = [], set()
        for name in doc_names or sorted(self._topics):
            for t in self._topics.get(name, []):
                if t.lower() not in seen:
                    seen.add(t.lower())
                    out.append(t)
        return out
