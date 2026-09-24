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
