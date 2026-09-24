"""Tests for the Gradio UI handlers and event wiring (src/course_assistant/ui/app.py).

These run without launching a server. They pin the DOCUMENTS-tab contract:
- the "Documents in this course" checkbox list (one checkbox per document,
  valued by document id) is the ONLY selection used for removal;
- adding (drop), removing and refreshing all update that list and re-publish
  the doc choices to both doc pickers (issues #11, #15 part 1);
- the Remove button is disabled until a document is ticked;
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

DOC_LIST_LABEL = "Documents in this course"
FILE_LABEL = "Add course material (pdf, pptx, docx, txt, md) — drop one or more files"


def _listed(update):
    """Document names in a doc_list update ((label, doc_id) choices)."""
    return [label.split("  ·  ")[0] for label, _ in update["choices"]]


def _id_of(app_state, name):
    return next(d["doc_id"] for d in app_state.catalog.list_documents()
                if d["doc_name"] == name)


# ---------------------------------------------------------------- handlers

def test_refresh_docs_lists_documents_by_id_and_disables_remove(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    doc_list, remove_btn, ask_docs, q_docs = refresh_docs(app_state)
    assert _listed(doc_list) == ["notes.txt"]
    assert doc_list["choices"][0][1] == _id_of(app_state, "notes.txt")
    assert doc_list["value"] == []                    # selection cleared
    assert remove_btn["interactive"] is False         # nothing ticked yet
    assert "notes.txt" in ask_docs["choices"] and "notes.txt" in q_docs["choices"]


def test_add_file_handler_updates_list_and_choices(app_state, small_doc):
    out = add_file_handler(app_state, str(small_doc), [])
    assert len(out) == 5  # message + doc_list + remove_btn + ask_docs + q_docs
    first, doc_list, _, ask_docs, q_docs = out
    assert "already loaded" not in first.lower()
    assert _listed(doc_list) == ["notes.txt"]
    assert "notes.txt" in ask_docs["choices"]
    assert "notes.txt" in q_docs["choices"]


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
    assert len(out[0].split("  \n")) == 2  # one status line per file
    assert sorted(_listed(out[1])) == ["more_notes.txt", "notes.txt"]


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
    assert _listed(out[1]) == ["notes.txt"]  # the good file still went in


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


def test_add_keeps_ask_and_quiz_selections(app_state, small_doc):
    _, doc_list, _, ask_docs, q_docs = add_file_handler(app_state, str(small_doc), [])
    assert doc_list["value"] == []
    assert "value" not in ask_docs and "value" not in q_docs


# ------------------------------------------------------------------ removing

def _three_docs(app_state, tmp_path):
    for n in ("a.txt", "b.txt", "c.txt"):
        p = tmp_path / n
        p.write_text(f"Document {n} talks about topic {n[0]} in detail.")
        app_state.catalog.add_file(str(p))
    return [_id_of(app_state, n) for n in ("a.txt", "b.txt", "c.txt")]


def test_remove_handler_removes_ticked_ids(app_state, tmp_path):
    a, b, _ = _three_docs(app_state, tmp_path)
    out = remove_handler(app_state, [a, b])
    assert len(out) == 5
    assert len(out[0].split("  \n")) == 2 and "Removed 'a.txt'" in out[0]
    assert app_state.doc_choices() == ["c.txt"]
    assert _listed(out[1]) == ["c.txt"]
    assert out[3]["choices"] == ["c.txt"]


def test_remove_handler_no_selection(app_state):
    for empty in ("", None, []):
        out = remove_handler(app_state, empty)
        assert out[0] == "Select at least one document to remove."
        assert len(out) == 5


def test_remove_refuses_ids_not_in_this_course(app_state, tmp_path):
    _three_docs(app_state, tmp_path)
    out = remove_handler(app_state, ["0123456789abcdef"])
    assert "No document with id" in out[0]
    assert len(app_state.doc_choices()) == 3


def test_remove_clears_selections_so_the_next_event_does_not_crash(app_state, tmp_path):
    """After a removal every selection that could still hold the removed
    document is cleared: Gradio rejects the next event if a component holds
    a value that is no longer among its choices."""
    a, _, _ = _three_docs(app_state, tmp_path)
    _, doc_list, remove_btn, ask_docs, q_docs = remove_handler(app_state, [a])
    for upd in (doc_list, ask_docs, q_docs):
        assert upd["value"] == []
    assert a not in [v for _, v in doc_list["choices"]]
    assert "a.txt" not in ask_docs["choices"]
    assert remove_btn["interactive"] is False


# ------------------------------------------------------------- event wiring

def _comps(demo):
    return {c["id"]: (c.get("type"), (c.get("props") or {}).get("label"))
            for c in demo.config["components"]}


def test_event_wiring_selection_ingestion_and_removal(app_state):
    """The checkbox list is the only removal selection; the drop is the only
    ingestion path (issue #12); add/remove/refresh keep the list in sync
    (issues #11/#15)."""
    demo = build_ui(app_state)
    comps = _comps(demo)

    def find(pred):
        ids = [i for i, tv in comps.items() if pred(tv)]
        assert ids, "no matching component"
        return ids

    file_up_id = find(lambda tv: tv[0] == "file" and tv[1] == FILE_LABEL)[0]
    doc_list_id = find(lambda tv: tv[1] == DOC_LIST_LABEL)[0]
    course_dd_id = find(lambda tv: tv[0] == "dropdown" and tv[1] == "Course")[0]
    # no dataframe or remove dropdown left: one source of truth for selection
    assert not find_all(comps, lambda tv: tv[0] == "dataframe")
    assert not find_all(comps, lambda tv: tv[1] and "Remove" in str(tv[1])
                        and tv[0] == "dropdown")

    deps = demo.config["dependencies"]

    def targeted(event):
        return [d for d in deps
                if any(ev == event for _, ev in (d.get("targets") or []))]

    # dropping a file is the ONLY ingestion path, and it empties the upload box
    ingest = [d for d in deps if doc_list_id in (d.get("outputs") or [])
              and file_up_id in (d.get("outputs") or [])
              and course_dd_id not in (d.get("outputs") or [])]
    assert len(ingest) == 1
    assert ingest[0]["targets"] == [(file_up_id, "upload")]

    # the Remove button reads exactly the checkbox list
    remove = [d for d in targeted("click") if d.get("inputs") == [doc_list_id]]
    assert len(remove) == 1
    assert doc_list_id in remove[0]["outputs"]
    remove_btn_id = remove[0]["targets"][0][0]
    assert comps[remove_btn_id][0] == "button"

    # ticking/unticking enables/disables the Remove button
    toggles = [d for d in targeted("change") if d.get("inputs") == [doc_list_id]
               and d.get("outputs") == [remove_btn_id]]
    assert len(toggles) == 1


def find_all(comps, pred):
    return [i for i, tv in comps.items() if pred(tv)]


def test_remove_button_starts_disabled_and_toggles(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    demo = build_ui(app_state)
    btn = [b for b in demo.blocks.values()
           if type(b).__name__ == "Button" and b.value == "Remove selected"]
    assert len(btn) == 1 and btn[0].interactive is False
    toggle = [f for f in demo.fns.values()
              if any(ev == "change" for _, ev in f.targets)
              and [getattr(i, "label", None) for i in f.inputs] == [DOC_LIST_LABEL]]
    assert len(toggle) == 1
    assert toggle[0].fn(["some-id"])["interactive"] is True
    assert toggle[0].fn([])["interactive"] is False
