"""Composition root: build the service provider stacks and app glue.

Kept free of Gradio imports so unit tests can construct a catalog + assistant
without the UI.
"""
from __future__ import annotations

from .config.settings import Settings
from .core.assistant import Assistant
from .core.index import Catalog
from .core.quiz import Quiz
from .core.topics import TopicStore
from .services.chat import build_chat
from .services.embeddings import build_text_embedder, build_visual_embedder
from .services.reranker import build_reranker


class AppState:
    """Holds the live catalog/assistant plus transient UI state."""

    def __init__(self, catalog: Catalog, assistant: Assistant,
                 settings: Settings):
        self.catalog = catalog
        self.assistant = assistant
        self.settings = settings
        self.quiz: Quiz | None = None
        self.topics = TopicStore(settings)

    def doc_choices(self) -> list[str]:
        return [d["doc_name"] for d in self.catalog.list_documents()]

    def sync_topics(self) -> str | None:
        """Bring the topic cache in line with the loaded documents. Returns an
        error message instead of raising, so a service outage never blocks
        adding, removing or quizzing."""
        try:
            self.topics.ensure(self.catalog, self.assistant.chat)
        except Exception as e:  # noqa: BLE001 — shown in the UI, key redacted
            from .config.settings import redact
            return redact(str(e))
        return None

    def topic_choices(self, doc_names: list[str] | None = None) -> list[str]:
        """Quiz topics for the selected documents (all loaded when none)."""
        self.sync_topics()
        return self.topics.topics_for(doc_names)


def build_app_state(settings: Settings,
                    chunk_method: str = "recursive",
                    rerank_enabled: bool = True) -> AppState:
    """Build the RAG stack for the given (injected) settings.

    ``chunk_method`` and ``rerank_enabled`` are controllable for the README
    answer-quality comparison (rerank off, different chunkers).
    """
    text_embed = build_text_embedder(settings)
    visual_embed = build_visual_embedder(settings)
    # rerank_enabled=False keeps the reciprocal-rank-fused keyword+vector order
    reranker = build_reranker(settings) if rerank_enabled else None
    catalog = Catalog(settings, text_embed, visual_embed, reranker,
                      chunk_method=chunk_method)
    chat = build_chat(settings)
    assistant = Assistant(catalog, chat, settings)
    return AppState(catalog, assistant, settings)
