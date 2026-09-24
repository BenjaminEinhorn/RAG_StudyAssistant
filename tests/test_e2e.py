"""End-to-end tests: exercise the real Gradio handler functions end to end
(add->ask->dedup->remove->quiz), in local (offline) mode.
"""
import pytest

pytest.importorskip("gradio")  # skip, not abort, on a bare clone without gradio

from conftest import needs_soffice, REAL_DECK_W2, needs_decks  # noqa: E402
from course_assistant.ui.app import (add_file_handler, ask_handler,  # noqa: E402
                                     quiz_feedback, remove_handler)


def _id_of(app_state, name):
    """The removal UI selects documents by id (the checkbox list values)."""
    return next(d["doc_id"] for d in app_state.catalog.list_documents()
                if d["doc_name"] == name)


@needs_decks
@needs_soffice
def test_full_add_ask_dedup_remove_flow(app_state):
    msg1, *_ = add_file_handler(app_state, str(REAL_DECK_W2), [])
    assert "Added" in msg1
    assert any("Week 2" in n for n in app_state.doc_choices())

    # ASK: the required meme — answer UI must surface sources incl. slide 33 image
    answer_md, gallery = ask_handler(
        app_state, "What does the Vibe Coding on Prod meme mean? Show the slide.",
        None, True)
    assert answer_md and "Sources" in answer_md
    assert "33" in answer_md, "should cite slide 33"
    assert gallery, "gallery should show a supporting slide image"

    # DEDUP: loading the same file twice must not duplicate
    msg_dup, *_ = add_file_handler(app_state, str(REAL_DECK_W2), [])
    assert "already loaded" in msg_dup
    assert len(app_state.catalog.list_documents()) == 1

    # REMOVE: content must be unreachable afterwards
    msg_rm, *_ = remove_handler(app_state, [_id_of(app_state, REAL_DECK_W2.name)])
    assert "Removed" in msg_rm
    assert app_state.catalog.list_documents() == []
    assert app_state.catalog.search("Vibe Coding") == []


@needs_decks
@needs_soffice
def test_meme_slide_image_is_displayed(app_state):
    add_file_handler(app_state, str(REAL_DECK_W2), [])
    answer_md, gallery = ask_handler(
        app_state, "Show me the Vibe Coding on Prod meme slide.", None, True)
    # the *actual* Week 2 slide 33 PNG must be in the evidence gallery
    assert any(p and "p033.png" in p for p, _ in gallery), f"got {gallery}"


def test_removed_doc_does_not_leak_into_answer(app_state, small_doc):
    from course_assistant.ui.app import add_file_handler, remove_handler
    add_file_handler(app_state, str(small_doc), [])
    remove_handler(app_state, [_id_of(app_state, "notes.txt")])
    res = app_state.assistant.answer("hybrid RAG")
    # no sources from the removed doc
    assert not any(s.doc_name == "notes.txt" for s in res.sources)


def test_quiz_feedback_scores_against_key_and_cites_sources(app_state, small_doc):
    from course_assistant.core.quiz import generate_quiz

    add_file_handler(app_state, str(small_doc), [])
    quiz = generate_quiz(app_state.catalog, app_state.assistant.chat, "RAG",
                         num_questions=2)
    key = {i: q.answer_index for i, q in enumerate(quiz.questions)}
    md = quiz_feedback(quiz, key)
    assert f"Score: {quiz.total()} / {quiz.total()}" in md
    assert "Source: notes.txt" in md
    wrong = {0: (quiz.questions[0].answer_index + 1) % len(quiz.questions[0].choices)}
    assert f"Score: 0 / {quiz.total()}" in quiz_feedback(quiz, wrong)
    revealed = quiz_feedback(quiz, {}, reveal_all=True)
    assert "Answer key" in revealed and "correct answer" in revealed
