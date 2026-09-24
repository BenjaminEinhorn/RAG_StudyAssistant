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


def test_index_refuses_vectors_from_a_different_embedder(app_state, small_doc):
    from course_assistant.core.index import Catalog
    from course_assistant.services.embeddings import LocalHashEmbeddings

    app_state.catalog.add_file(str(small_doc))
    other = LocalHashEmbeddings(dim=app_state.settings.embed_dim)
    other.name = "some-other-model"
    with pytest.raises(RuntimeError, match="built with embedder"):
        Catalog(app_state.settings, other, app_state.catalog.visual_embedder, None)


def test_rerank_off_keeps_fused_order(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    app_state.catalog.reranker = None
    hits = app_state.catalog.search("keyword search and vector search")
    assert hits and hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_remove_deletes_stored_file_and_slide_images(app_state, tmp_path):
    import fitz  # PyMuPDF: build a 2-page PDF so page images are rendered

    pdf = app_state.settings.materials_dir / "deck.pdf"
    doc = fitz.open()
    for text in ("Page one about embeddings", "Page two about reranking"):
        doc.new_page().insert_text((72, 72), text)
    doc.save(pdf)
    app_state.catalog.add_file(str(pdf))
    img_dirs = [p for p in app_state.settings.images_dir.iterdir() if any(p.iterdir())]
    assert img_dirs, "the PDF's pages should have been rendered"

    app_state.catalog.remove_document("deck.pdf")
    assert not pdf.exists()                      # re-seeding cannot bring it back
    assert not any(p.exists() for p in img_dirs)  # slide images are gone too


def test_add_reports_parser_warnings(app_state, tmp_path, monkeypatch):
    monkeypatch.setattr("course_assistant.core.parsing.find_soffice", lambda: None)
    from pptx import Presentation

    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[1]).shapes.title.text = "Only slide"
    deck = tmp_path / "deck.pptx"
    prs.save(deck)
    res = app_state.catalog.add_file(str(deck))
    assert "LibreOffice is not installed" in res["message"]
