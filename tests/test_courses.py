"""Courses: one install serves several classes, each with its own documents,
index and quiz topics."""
import pytest

from course_assistant.core.courses import CourseRegistry, slugify


def test_registry_starts_with_default_course_in_data_root(tmp_path):
    reg = CourseRegistry(tmp_path)
    assert len(reg.names()) == 1
    assert reg.dir_of(reg.names()[0]) == tmp_path.resolve()  # legacy data kept
    assert CourseRegistry(tmp_path).names() == reg.names()     # persisted


def test_create_course_rejects_duplicates_and_blank_names(tmp_path):
    reg = CourseRegistry(tmp_path)
    assert reg.create("  FIN   6100 ") == "FIN 6100"
    assert reg.dir_of("FIN 6100") == (tmp_path / "courses" / "fin-6100").resolve()
    for bad in ("fin 6100", "FIN-6100", "   ", "!!!"):
        with pytest.raises(ValueError):
            reg.create(bad)
    assert slugify("MBAX 6418: LLMs") == "mbax-6418-llms"


def test_courses_keep_documents_separate(app_state, small_doc):
    first = app_state.course_name
    app_state.catalog.add_file(str(small_doc))
    assert app_state.doc_choices() == ["notes.txt"]

    app_state.create_course("FIN 6100")
    assert app_state.course_name == "FIN 6100"
    assert app_state.doc_choices() == []                # nothing leaks across
    assert app_state.assistant.catalog is app_state.catalog
    assert "FIN 6100" in app_state.assistant.course_name
    assert app_state.settings.data_dir.name == "fin-6100"

    app_state.switch_course(first)
    assert app_state.doc_choices() == ["notes.txt"]     # still there


def test_switch_course_handler_updates_every_list(app_state, small_doc):
    pytest.importorskip("gradio")
    from course_assistant.ui.app import create_course_handler, switch_course_handler

    first = app_state.course_name
    app_state.catalog.add_file(str(small_doc))
    out = create_course_handler(app_state, "FIN 6100")
    course_dd, msg, hero, doc_table, remove_dd, ask_docs, q_docs, q_topic = out
    assert course_dd["value"] == "FIN 6100" and "FIN 6100" in course_dd["choices"]
    assert "FIN 6100" in hero and "no documents yet" in msg
    assert ask_docs["choices"] == [] and ask_docs["value"] == []
    assert q_topic["choices"] == []

    out = switch_course_handler(app_state, first)
    assert out[5]["choices"] == ["notes.txt"]

    bad = create_course_handler(app_state, "FIN 6100")
    assert bad[1].startswith("⚠️") and app_state.course_name == first


# --------------------------------------------- acceptance: no cross-course leaks

def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_acceptance_courses_never_share_documents_or_retrieval(app_state, tmp_path):
    """Course A gets A1/A2, course B gets B1/B2. Each course lists and
    retrieves only its own documents, even when the other course's text is a
    far better match for the query."""
    pytest.importorskip("gradio")
    from course_assistant.ui.app import add_file_handler, create_course_handler

    a1 = _write(tmp_path, "A1.txt", "Bond duration measures interest rate risk.")
    a2 = _write(tmp_path, "A2.txt", "Equity valuation uses discounted cash flows.")
    b1 = _write(tmp_path, "B1.txt", "Monte Carlo simulation of bankruptcy risk "
                                     "in leveraged buyouts, Monte Carlo bankruptcy.")
    b2 = _write(tmp_path, "B2.txt", "Real options value managerial flexibility.")

    create_course_handler(app_state, "Course A")
    add_file_handler(app_state, [a1, a2], [])
    create_course_handler(app_state, "Course B")
    add_file_handler(app_state, [b1, b2], [])
    assert app_state.doc_choices() == ["B1.txt", "B2.txt"]

    out = create_course_handler(app_state, "Course A")  # already exists: refused
    assert out[1].startswith("⚠️")
    from course_assistant.ui.app import switch_course_handler
    out = switch_course_handler(app_state, "Course A")
    ask_docs, q_docs = out[5], out[6]
    assert ask_docs["choices"] == ["A1.txt", "A2.txt"] == q_docs["choices"]
    assert ask_docs["value"] == []                      # stale picks cleared

    hits = app_state.catalog.search("Monte Carlo bankruptcy simulation", k=10)
    assert hits and {h.doc_name for h in hits} <= {"A1.txt", "A2.txt"}
    assert not list((app_state.settings.materials_dir).glob("B*.txt"))


def test_upload_box_is_emptied_so_a_course_switch_cannot_resend_files(app_state,
                                                                        tmp_path):
    """The reported leak: the multi-file upload box kept every file dropped
    into it and re-sent them all on the next drop, so after switching course
    the previous course's files were added to the new one."""
    pytest.importorskip("gradio")
    from course_assistant.ui.app import build_ui

    demo = build_ui(app_state)
    upload = [f for f in demo.fns.values()
              if any(ev == "upload" for _, ev in f.targets)]
    assert len(upload) == 1
    out = upload[0].fn([_write(tmp_path, "A1.txt", "Bond duration.")])
    assert out[-1] is None                               # box emptied after ingest
