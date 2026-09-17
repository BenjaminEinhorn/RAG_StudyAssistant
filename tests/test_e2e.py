"""End-to-end tests: exercise the real Gradio handler functions end to end
(add->ask->dedup->remove->quiz), in local (offline) mode.
"""
from conftest import REAL_DECK_W2, needs_decks
from course_assistant.ui.app import add_file_handler, ask_handler, remove_handler


@needs_decks
def test_full_add_ask_dedup_remove_flow(app_state):
    msg1, d1, d2 = add_file_handler(app_state, str(REAL_DECK_W2), [])
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
    msg_dup, _, _ = add_file_handler(app_state, str(REAL_DECK_W2), [])
    assert "already loaded" in msg_dup
    assert len(app_state.catalog.list_documents()) == 1

    # REMOVE: content must be unreachable afterwards
    msg_rm, _, _ = remove_handler(app_state, REAL_DECK_W2.name)
    assert "Removed" in msg_rm
    assert app_state.catalog.list_documents() == []
    assert app_state.catalog.search("Vibe Coding") == []


@needs_decks
def test_meme_slide_image_is_displayed(app_state):
    add_file_handler(app_state, str(REAL_DECK_W2), [])
    answer_md, gallery = ask_handler(
        app_state, "Show me the Vibe Coding on Prod meme slide.", None, True)
    # the *actual* Week 2 slide 33 PNG must be in the evidence gallery
    assert any(p and "p033.png" in p for p, _ in gallery), f"got {gallery}"


def test_removed_doc_does_not_leak_into_answer(app_state, small_doc):
    from course_assistant.ui.app import add_file_handler, remove_handler
    add_file_handler(app_state, str(small_doc), [])
    remove_handler(app_state, "notes.txt")
    res = app_state.assistant.answer("hybrid RAG")
    # no sources from the removed doc
    assert not any(s.doc_name == "notes.txt" for s in res.sources)
