""""Explore further" suggestions: general-knowledge subjects beyond the course
material, clearly labelled, never cited, and only for course-related questions."""
import pytest

from course_assistant.core.further import (clean_further, further_markdown,
                                           suggest_further_topics)


class _Chat:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete_json(self, system, user, schema, images=None):
        self.prompts.append((system + user, schema))
        return self.reply


def test_clean_further_dedupes_trims_and_caps():
    items = clean_further([{"topic": " BM25 ", "why": "keyword  search"},
                           {"topic": "bm25", "why": "dup"}, {"topic": "", "why": "x"},
                           "not a dict", {"topic": "x" * 90, "why": "too long"}]
                          + [{"topic": f"T{i}", "why": ""} for i in range(9)])
    assert items[0] == {"topic": "BM25", "why": "keyword search"}
    assert len(items) == 5 and len({i["topic"].lower() for i in items}) == 5


def test_markdown_is_labelled_as_outside_the_documents():
    md = further_markdown([{"topic": "BM25", "why": "keyword ranking"}])
    assert "beyond the course material" in md and "not from your documents" in md
    assert "**BM25** — keyword ranking" in md
    assert further_markdown([]) == ""


def test_answer_carries_suggestions_only_for_course_questions(app_state, small_doc):
    from course_assistant.core.assistant import ANSWER_SCHEMA, Assistant

    assert ANSWER_SCHEMA["required"][-2:] == ["question_is_about_course", "further_topics"]
    app_state.catalog.add_file(str(small_doc))
    base = {"coverage": "full", "beyond_slides": "", "found": True,
            "answer": "Hybrid RAG combines two searches.",
            "citations": [{"source": 1, "excerpt": "keyword search and vector search"}],
            "further_topics": [{"topic": "Cross-encoders", "why": "reranking"}]}

    def ask(**over):
        chat = _Chat({**base, **over})
        return Assistant(app_state.catalog, chat, app_state.settings).answer("hybrid RAG?")

    res = ask(question_is_about_course=True)
    assert [f["topic"] for f in res.further_topics] == ["Cross-encoders"]
    assert "Cross-encoders" not in res.answer            # never mixed into the answer
    assert ask(question_is_about_course=False).further_topics == []


def test_ask_handler_shows_block_after_sources(app_state, monkeypatch):
    pytest.importorskip("gradio")
    from course_assistant.core.assistant import AnswerResult
    from course_assistant.ui.app import ask_handler

    monkeypatch.setattr(app_state.assistant, "answer", lambda *a, **k: AnswerResult(
        answer="x", further_topics=[{"topic": "Vector databases", "why": "storage"}]))
    md, _ = ask_handler(app_state, "q", [], True)
    assert md.index("**Sources**") < md.index("Explore further")
    assert "Vector databases" in md


def test_quiz_suggestions_skip_covered_topics_and_survive_failures(app_state, monkeypatch):
    pytest.importorskip("gradio")
    from course_assistant.ui.app import quiz_further_md

    chat = _Chat({"further_topics": [{"topic": "RAG evaluation", "why": "quality"}]})
    items = suggest_further_topics(chat, "Hybrid RAG", covered=["Hybrid RAG Pipeline"])
    assert items == [{"topic": "RAG evaluation", "why": "quality"}]
    assert "Hybrid RAG Pipeline" in chat.prompts[0][0]   # told what is covered
    assert suggest_further_topics(chat, "  ") == []

    def boom(*a, **k):
        raise RuntimeError("class LLM call failed")

    monkeypatch.setattr("course_assistant.ui.app.suggest_further_topics", boom)
    assert "No further-learning suggestions" in quiz_further_md(app_state, "RAG", None)


def test_offline_mode_suggests_nothing(app_state):
    assert suggest_further_topics(app_state.assistant.chat, "Hybrid RAG") == []
