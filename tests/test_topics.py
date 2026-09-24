"""Quiz topic categories: extracted per document, cached, and offered in the
Quiz tab's topic dropdown for the selected documents."""
import json

import pytest

from course_assistant.core.topics import TopicStore, extract_topics


class _TopicChat:
    """Stands in for the live LLM; records the prompt it was given."""

    def __init__(self, topics):
        self.topics = topics
        self.prompts = []

    def complete_json(self, system, user, schema, images=None):
        self.prompts.append(user)
        return {"topics": self.topics}


def _two_docs(tmp_path, app_state):
    a = tmp_path / "rag.txt"
    a.write_text("Hybrid RAG\nKeyword search and vector search are combined.")
    b = tmp_path / "prompts.txt"
    b.write_text("Prompt templates\nFew-shot examples guide the model.")
    for p in (a, b):
        app_state.catalog.add_file(str(p))
    return "rag.txt", "prompts.txt"


def test_extract_topics_sends_slide_titles_and_cleans_reply(app_state, tmp_path):
    rag, _ = _two_docs(tmp_path, app_state)
    chat = _TopicChat(["Hybrid RAG", " hybrid rag ", "Vector search.", "", "x" * 80])
    topics = extract_topics(app_state.catalog, chat, rag)
    assert topics == ["Hybrid RAG", "Vector search"]  # deduped, trimmed, bounded
    assert "slide/page 1: Hybrid RAG" in chat.prompts[0]
    assert "Few-shot" not in chat.prompts[0]           # only this document


def test_store_caches_filters_by_document_and_drops_removed(app_state, tmp_path):
    rag, prompts = _two_docs(tmp_path, app_state)
    store = TopicStore(app_state.settings)
    store.ensure(app_state.catalog, _TopicChat(["Shared topic", "Own topic"]))
    store._topics[prompts] = ["Prompt design", "shared TOPIC"]

    assert store.topics_for([prompts]) == ["Prompt design", "shared TOPIC"]
    assert store.topics_for(None) == ["Prompt design", "shared TOPIC", "Own topic"]

    chat = _TopicChat(["unused"])
    store.ensure(app_state.catalog, chat)
    assert chat.prompts == []          # cached: no second LLM call

    app_state.catalog.remove_document(rag)
    store.ensure(app_state.catalog, chat)
    assert rag not in json.loads(store.path.read_text())


def test_topic_failure_is_reported_not_raised(app_state, small_doc, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("class LLM call failed")

    monkeypatch.setattr(app_state.assistant.chat, "complete_json", boom)
    monkeypatch.setattr("course_assistant.core.topics.LocalEchoChat", type(None))
    app_state.catalog.add_file(str(small_doc))
    assert "class LLM call failed" in app_state.sync_topics()
    assert app_state.topic_choices() == []


def test_quiz_topic_is_a_dropdown_of_material_topics(app_state, tmp_path):
    pytest.importorskip("gradio")
    from course_assistant.ui.app import build_ui, topic_dropdown_update

    rag, prompts = _two_docs(tmp_path, app_state)
    app_state.topics.ensure(app_state.catalog, _TopicChat(["Hybrid RAG"]))
    app_state.topics._topics[prompts] = ["Prompt design"]

    demo = build_ui(app_state)
    dd = [b for b in demo.blocks.values()
          if type(b).__name__ == "Dropdown" and b.label == "Topic"]
    assert len(dd) == 1
    assert [c[0] for c in dd[0].choices] == ["Prompt design", "Hybrid RAG"]

    upd = topic_dropdown_update(app_state, [rag], "Prompt design")
    assert upd["choices"] == ["Hybrid RAG"] and upd["value"] == "Hybrid RAG"
    upd = topic_dropdown_update(app_state, [], "Hybrid RAG")
    assert upd["value"] == "Hybrid RAG"  # still offered, so the pick is kept
