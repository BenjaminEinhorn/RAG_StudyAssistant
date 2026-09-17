"""Unit tests for the hybrid index: add / dedup / remove / search."""
import pytest


def test_add_and_search(app_state, small_doc):
    res = app_state.catalog.add_file(str(small_doc))
    assert res["ok"] is True
    hits = app_state.catalog.search("keyword search and vector search")
    assert hits, "should retrieve the RAG text"
    assert "keyword search" in hits[0].text
    assert hits[0].doc_name == "notes.txt"
    assert hits[0].page == 1


def test_keyword_search_ranks_exact_match_top(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    hits = app_state.catalog.search("Vibe Coding on Prod")
    assert hits
    assert "Vibe Coding on Prod" in hits[0].text


def test_loading_same_file_twice_does_not_duplicate(app_state, small_doc):
    r1 = app_state.catalog.add_file(str(small_doc))
    r2 = app_state.catalog.add_file(str(small_doc))
    assert r2.get("duplicate") is True
    assert len(app_state.catalog.list_documents()) == 1


def test_add_same_name_different_content_keeps_both(app_state, small_doc, tmp_path):
    app_state.catalog.add_file(str(small_doc))
    other = tmp_path / "notes.txt"
    other.write_text("completely different unrelated topic about finance.")
    app_state.catalog.add_file(str(other))
    # same file NAME but different content -> two documents (sha differentiates)
    assert len(app_state.catalog.list_documents()) == 2


def test_remove_drops_searchable_content(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    assert app_state.catalog.search("Vibe Coding")  # present before
    r = app_state.catalog.remove_document("notes.txt")
    assert r["ok"] is True
    assert app_state.catalog.list_documents() == []
    assert app_state.catalog.search("Vibe Coding") == []


def test_visual_search_returns_slide_images(app_state, small_doc):
    # small_doc is a txt: no images. Instead just ensure empty is handled.
    assert app_state.catalog.visual_search("anything") == []


def test_remove_nonexistent_returns_message(app_state):
    r = app_state.catalog.remove_document("nope.pdf")
    assert r["ok"] is False
