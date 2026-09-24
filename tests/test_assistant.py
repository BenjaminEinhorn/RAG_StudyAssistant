"""Unit tests for answer generation (structured answer/sources, missing-info)."""
from conftest import needs_soffice, REAL_DECK_W2, REAL_DECK_W5, needs_decks
from course_assistant.core.assistant import AnswerResult, Assistant


def test_answer_returns_structured_output(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = app_state.assistant.answer("What is hybrid RAG?")
    assert isinstance(res, AnswerResult)
    assert isinstance(res.answer, str) and res.answer
    assert res.sources and res.sources[0].doc_name == "notes.txt"
    assert res.sources[0].page == 1


@needs_decks
@needs_soffice
def test_answer_includes_image_when_available(app_state):
    r = app_state.catalog.add_file(str(REAL_DECK_W5))
    assert r["ok"] is True
    res = app_state.assistant.answer("Why use RAG?", include_images=True)
    # every source chunk is backed by a rendered slide image
    assert res.sources, "should find slides"
    assert any(s.image_path for s in res.sources)
    assert res.used_images, "should include at least one slide image"


def test_missing_information_is_acknowledged(app_state, small_doc):
    from course_assistant.services.chat import LocalEchoChat
    # The acknowledged-missing-info mechanism: when retrieval finds nothing,
    # the pipeline inserts a no-evidence marker and the fallback (and the LLM
    # prompt) must refuse rather than invent facts.
    out = LocalEchoChat().complete("sys", "<<no-evidence>>\nWhatever question",
                                   images=["some.png"])
    assert "could not find" in out.lower() and "won't guess" in out.lower()
    # And the assistant never fabricates a citation: any returned source must
    # be a real loaded document.
    app_state.catalog.add_file(str(small_doc))
    res = app_state.assistant.answer("hybrid eembedding vector", k=1)
    assert all(s.doc_name == "notes.txt" for s in res.sources)


def test_sources_do_not_invent_documents(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = app_state.assistant.answer("hybrid RAG")
    doc_names = {s.doc_name for s in res.sources}
    assert doc_names == {"notes.txt"}


@needs_decks
def test_visual_question_retains_slide(app_state):
    r = app_state.catalog.add_file(str(REAL_DECK_W2))
    assert r["ok"] is True
    res = app_state.assistant.answer(
        "Show me the Vibe Coding on Prod slide and explain it.")
    # the exact meme slide should be among the retrieved sources
    assert any(s.page == 33 and "Week 2" in s.doc_name for s in res.sources)


class _ScriptedChat:
    """Returns a fixed structured reply, to test citation verification."""

    def __init__(self, reply):
        self.reply = reply

    def complete_json(self, system, user, schema, images=None):
        return self.reply


def _assistant_with(app_state, reply):
    from course_assistant.core.assistant import Assistant as A

    return A(app_state.catalog, _ScriptedChat(reply), app_state.settings)


def test_verified_citation_becomes_a_source(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = _assistant_with(app_state, {
        "found": True, "answer": "Hybrid RAG combines two searches.",
        "citations": [{"source": 1, "excerpt": "keyword search and vector search"}],
    }).answer("What is hybrid RAG?")
    assert res.verified
    assert [s.doc_name for s in res.sources] == ["notes.txt"]


def test_fabricated_excerpt_and_missing_source_are_flagged(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = _assistant_with(app_state, {
        "found": True, "answer": "Made up.",
        "citations": [{"source": 1, "excerpt": "RAG was invented in 1850"},
                      {"source": 99, "excerpt": "anything"}],
    }).answer("What is hybrid RAG?")
    assert not res.verified
    assert res.sources == []                     # nothing unverified is cited
    assert res.citations[0].in_evidence and not res.citations[0].excerpt_found
    assert not res.citations[1].in_evidence


def test_not_found_reply_drops_citations(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = _assistant_with(app_state, {
        "found": False, "answer": "", "citations": [{"source": 1, "excerpt": "x"}],
    }).answer("Who won the 2024 Super Bowl?")
    assert not res.found_evidence and res.sources == [] and res.citations == []
    assert "couldn't find" in res.answer.lower()


def test_excerpt_matching_ignores_case_whitespace_and_quotes():
    from course_assistant.core.assistant import excerpt_in_text

    assert excerpt_in_text('Vibe Coding on "Prod"', "Title: Vibe  Coding on “Prod”\n")
    assert not excerpt_in_text("", "anything")
    assert not excerpt_in_text("not there", "something else")


# ------------------------------------------- answers that go beyond the slides

def test_full_coverage_is_not_flagged(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = _assistant_with(app_state, {
        "found": True, "answer": "Hybrid RAG combines two searches.",
        "coverage": "full", "beyond_slides": "",
        "citations": [{"source": 1, "excerpt": "keyword search and vector search"}],
    }).answer("What is hybrid RAG?")
    assert not res.outside_slides and res.beyond_slides == ""


def test_partial_coverage_says_what_is_outside_the_slides(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = _assistant_with(app_state, {
        "found": True, "answer": "Hybrid RAG combines two searches.",
        "coverage": "partial",
        "beyond_slides": "The slides do not say which vector database to use.",
        "citations": [{"source": 1, "excerpt": "keyword search and vector search"}],
    }).answer("What is hybrid RAG and which vector database should I use?")
    assert res.outside_slides and res.coverage == "partial"
    assert "vector database" in res.beyond_slides
    assert res.found_evidence  # the covered part is still answered


def test_unverified_full_answer_is_flagged_as_outside(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = _assistant_with(app_state, {
        "found": True, "answer": "RAG was invented in 1850.", "coverage": "full",
        "beyond_slides": "",
        "citations": [{"source": 1, "excerpt": "RAG was invented in 1850"}],
    }).answer("When was RAG invented?")
    assert res.outside_slides and "outside the slides" in res.beyond_slides


def test_not_found_is_marked_as_not_covered(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    res = _assistant_with(app_state, {
        "found": False, "answer": "", "coverage": "full", "beyond_slides": "",
        "citations": [],
    }).answer("Who won the 2024 Super Bowl?")
    assert res.coverage == "none" and res.beyond_slides


def test_ui_shows_outside_slides_notice(app_state, small_doc, monkeypatch):
    import pytest

    pytest.importorskip("gradio")
    from course_assistant.ui.app import ask_handler

    monkeypatch.setattr(app_state.assistant, "answer", lambda *a, **k: AnswerResult(
        answer="Partly answered.", coverage="partial",
        beyond_slides="The slides do not name a vector database."))
    md, _ = ask_handler(app_state, "q", [], True)
    assert "Partly outside the slides" in md and "vector database" in md
