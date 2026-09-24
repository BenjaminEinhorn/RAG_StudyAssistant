"""Tests for the Gradio UI handlers and event wiring (src/course_assistant/ui/app.py).

These run without launching a server. They pin the DOCUMENTS-tab contract:
- adding (drop), removing and refreshing all update the documents table AND the
  "Remove a document" dropdown, and re-publish the doc choices to both doc
  pickers (issues #11, #15 part 1);
- dropping a file is the single ingestion trigger, and failures are surfaced in
  the interface instead of only in the terminal (issue #12);
- each handler returns exactly the output arity the event wiring declares.
"""
import pytest

from course_assistant.ui.app import (
    add_file_handler,
    build_ui,
    refresh_docs,
    remove_handler,
)

REMOVE_LABEL = "Remove documents"
FILE_LABEL = "Add course material (pdf, pptx, docx, txt, md) — drop one or more files"


def _names(rows):
    return [r[0] for r in rows if r]


# ---------------------------------------------------------------- handlers

def test_refresh_docs_includes_doc_table_and_remove_dd(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    updates = refresh_docs(app_state)
    assert len(updates) == 4
    assert "notes.txt" in _names(updates[0]["value"])          # doc_table
    assert "notes.txt" in updates[1]["choices"]                  # remove_dd


def test_add_file_handler_updates_table_and_choices(app_state, small_doc):
    out = add_file_handler(app_state, str(small_doc), [])
    assert len(out) == 5  # message + doc_table + remove_dd + ask_docs + q_docs
    first, doc_table, remove_dd, ask_docs, q_docs = out
    assert "already loaded" not in first.lower()
    assert "notes.txt" in _names(doc_table["value"])
    assert "notes.txt" in remove_dd["choices"]
    assert "notes.txt" in ask_docs["choices"]
    assert "notes.txt" in q_docs["choices"]
    assert any(d["doc_name"] == "notes.txt"
               for d in app_state.catalog.list_documents())


def test_add_file_handler_single_ingestion(app_state, small_doc):
    first = add_file_handler(app_state, str(small_doc), [])
    assert "already loaded" not in first[0].lower()
    second = add_file_handler(app_state, str(small_doc), [])
    assert "already loaded" in second[0].lower()
    matching = [d for d in app_state.catalog.list_documents()
                if d["doc_name"] == "notes.txt"]
    assert len(matching) == 1  # re-running the handler never double-indexes


def test_add_file_handler_surfaces_failure(app_state, small_doc, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("libreoffice conversion failed")

    monkeypatch.setattr(app_state.catalog, "add_file", boom)
    out = add_file_handler(app_state, str(small_doc), [])
    assert len(out) == 5
    assert out[0].startswith("⚠️ Could not add 'notes.txt'")
    assert "libreoffice conversion failed" in out[0]


def test_add_file_handler_adds_multiple_files(app_state, small_doc):
    second = small_doc.parent / "more_notes.txt"
    second.write_text("Chunking splits documents into pieces before indexing.")
    out = add_file_handler(app_state, [str(small_doc), str(second)], [])
    assert len(out) == 5
    lines = out[0].split("  \n")
    assert len(lines) == 2  # one status line per file
    for name in ("notes.txt", "more_notes.txt"):
        assert name in _names(out[1]["value"])
        assert name in out[2]["choices"]


def test_add_file_handler_one_failure_does_not_block_others(app_state, small_doc,
                                                            monkeypatch):
    bad = small_doc.parent / "broken.pptx"
    bad.write_bytes(b"not a real deck")
    real_add = app_state.catalog.add_file

    def add_or_fail(path, *args, **kwargs):
        if path.endswith("broken.pptx"):
            raise RuntimeError("libreoffice conversion failed")
        return real_add(path, *args, **kwargs)

    monkeypatch.setattr(app_state.catalog, "add_file", add_or_fail)
    out = add_file_handler(app_state, [str(bad), str(small_doc)], [])
    first_line, second_line = out[0].split("  \n")
    assert first_line.startswith("⚠️ Could not add 'broken.pptx'")
    assert "already loaded" not in second_line.lower()
    assert "notes.txt" in _names(out[1]["value"])  # the good file still went in


def test_file_upload_accepts_multiple_files(app_state):
    demo = build_ui(app_state)
    file_ups = [b for b in demo.blocks.values()
                if getattr(b, "label", None) == FILE_LABEL]
    assert len(file_ups) == 1
    assert file_ups[0].file_count == "multiple"


def test_add_file_handler_no_selection(app_state):
    for empty in ("", None, []):
        out = add_file_handler(app_state, empty, [])
        assert out[0] == "No file selected."
        assert len(out) == 5


def test_remove_handler_updates_table_and_choices(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    out = remove_handler(app_state, "notes.txt")
    assert len(out) == 5
    assert "Removed" in out[0]
    assert "notes.txt" not in _names(out[1]["value"])          # doc_table
    assert "notes.txt" not in out[2]["choices"]                 # remove_dd
    assert app_state.catalog.list_documents() == []


def test_remove_handler_no_selection(app_state):
    out = remove_handler(app_state, "")
    for empty in ("", None, []):
        out = remove_handler(app_state, empty)
        assert out[0] == "Select at least one document to remove."
        assert len(out) == 5


# ------------------------------------------------------------- event wiring

def _comps(demo):
    return {c["id"]: (c.get("type"), (c.get("props") or {}).get("label"))
            for c in demo.config["components"]}


def test_event_wiring_drop_ingests_once_and_no_button_ingests(app_state):
    """Issue #12: file_up.change is the only ingestion path; add/remove/refresh
    all keep the table and remove-dropdown in sync (issues #11/#15)."""
    demo = build_ui(app_state)
    comps = _comps(demo)

    def find(pred):
        ids = [i for i, tv in comps.items() if pred(tv)]
        assert ids, "no matching component"
        return ids

    file_up_id = find(lambda tv: tv[0] == "file" and tv[1] == FILE_LABEL)[0]
    remove_dd_id = find(lambda tv: tv[0] == "dropdown" and tv[1] == REMOVE_LABEL)[0]
    doc_table_id = find(lambda tv: tv[0] == "dataframe")[0]

    deps = demo.config["dependencies"]

    def targeted(event):
        return [d for d in deps
                if any(ev == event for _, ev in (d.get("targets") or []))]

    # change events that touch the documents table (the quiz deck picker's
    # change event only refreshes the topic dropdown)
    change_deps = [d for d in targeted("change")
                   if doc_table_id in (d.get("outputs") or [])]
    click_deps = targeted("click")

    # dropping a file is the ONLY ingestion path
    assert len(change_deps) == 1
    ingestion = change_deps[0]
    assert (file_up_id, "change") in ingestion["targets"]
    # its status message is the markdown among its outputs
    add_msg_id = [o for o in ingestion["outputs"] if comps[o][0] == "markdown"]
    assert len(add_msg_id) == 1
    add_msg_id = add_msg_id[0]

    # issues #11/#15: ingestion updates the table and remove-dropdown too
    assert doc_table_id in ingestion["outputs"]
    assert remove_dd_id in ingestion["outputs"]

    # issue #12: no button click performs ingestion (drop fires once per file)
    for d in click_deps:
        assert add_msg_id not in (d.get("outputs") or [])

    # issues #11/#15: the remove/refresh buttons refresh table + dropdown
    doc_click_deps = [d for d in click_deps
                      if doc_table_id in d.get("outputs", [])
                      and remove_dd_id in d.get("outputs", [])]
    assert len(doc_click_deps) == 2  # remove_btn + refresh_btn
    for d in doc_click_deps:
        assert add_msg_id not in d.get("outputs", [])


# ------------------------------------------------ removing several documents

def _three_docs(app_state, tmp_path):
    names = []
    for n in ("a.txt", "b.txt", "c.txt"):
        p = tmp_path / n
        p.write_text(f"Document {n} talks about topic {n[0]} in detail.")
        app_state.catalog.add_file(str(p))
        names.append(n)
    return names


def test_remove_handler_removes_several_at_once(app_state, tmp_path):
    a, b, c = _three_docs(app_state, tmp_path)
    out = remove_handler(app_state, [a, b])
    assert len(out[0].split("  \n")) == 2
    assert app_state.doc_choices() == [c]
    assert _names(out[1]["value"]) == [c]
    assert out[2]["choices"] == [c]


def test_remove_clears_selections_so_the_next_event_does_not_crash(app_state, tmp_path):
    """The crash: after a removal the remove dropdown (and the Ask/Quiz deck
    pickers) kept the removed name selected, and Gradio rejected the next
    click because that value was no longer among the choices."""
    a, b, _ = _three_docs(app_state, tmp_path)
    _, _, remove_dd, ask_docs, q_docs = remove_handler(app_state, [a])
    for upd in (remove_dd, ask_docs, q_docs):
        assert upd["value"] == []
        assert a not in upd["choices"]


def test_add_keeps_ask_and_quiz_selections(app_state, small_doc):
    _, _, remove_dd, ask_docs, q_docs = add_file_handler(app_state, str(small_doc), [])
    assert remove_dd["value"] == []
    assert "value" not in ask_docs and "value" not in q_docs


def test_remove_dropdown_is_multiselect(app_state):
    demo = build_ui(app_state)
    dd = [b for b in demo.blocks.values() if getattr(b, "label", None) == REMOVE_LABEL]
    assert len(dd) == 1 and dd[0].multiselect
