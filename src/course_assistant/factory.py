"""Composition root: build the service provider stacks and app glue.

Kept free of Gradio imports so unit tests can construct a catalog + assistant
without the UI.
"""
from __future__ import annotations

from .config.settings import Settings
from .core.assistant import Assistant
from .core.courses import CourseRegistry
from .core.index import Catalog
from .core.quiz import Quiz
from .core.topics import TopicStore
from .services.chat import build_chat
from .services.embeddings import build_text_embedder, build_visual_embedder
from .services.reranker import build_reranker


class AppState:
    """Holds the live catalog/assistant of the current course plus transient
    UI state. Switching course swaps the catalog, assistant and topics in
    place, so handlers holding this object always see the current course."""

    def __init__(self, settings: Settings, chunk_method: str = "recursive",
                 rerank_enabled: bool = True):
        self.root_settings = settings
        self.chunk_method = chunk_method
        self.rerank_enabled = rerank_enabled
        self.courses = CourseRegistry(settings.root_data_dir)
        self.switch_course(self.courses.names()[0])

    def switch_course(self, name: str) -> None:
        settings = self.root_settings.for_data_dir(self.courses.dir_of(name))
        text_embed = build_text_embedder(settings)
        visual_embed = build_visual_embedder(settings)
        # rerank_enabled=False keeps the reciprocal-rank-fused keyword+vector order
        reranker = build_reranker(settings) if self.rerank_enabled else None
        self.catalog = Catalog(settings, text_embed, visual_embed, reranker,
                               chunk_method=self.chunk_method)
        self.assistant = Assistant(self.catalog, build_chat(settings), settings,
                                   course_name=name)
        self.topics = TopicStore(settings)
        self.settings = settings
        self.course_name = name
        self.quiz: Quiz | None = None

    def create_course(self, name: str) -> str:
        """Add an empty course and switch to it."""
        name = self.courses.create(name)
        self.switch_course(name)
        return name

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
    """Build the RAG stack for the given (injected) settings, opened on the
    first course.

    ``chunk_method`` and ``rerank_enabled`` are controllable for the README
    answer-quality comparison (rerank off, different chunkers).
    """
    return AppState(settings, chunk_method, rerank_enabled)
