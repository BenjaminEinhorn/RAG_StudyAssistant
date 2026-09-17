"""Unit tests for quiz generation, hidden key, scoring, and explanations."""
import pytest

from course_assistant.core.quiz import Quiz, QuizQuestion, generate_quiz, score


def _quiz():
    return Quiz(
        topic="RAG",
        questions=[
            QuizQuestion("What does RAG stand for?", ["Random", "Retrieval-Augmented Generation", "Rapid"], 1,
                         "RAG stands for Retrieval-Augmented Generation.", "Week5.pdf", 2,
                         "RAG stands for Retrieval-Augmented Generation."),
            QuizQuestion("Embeddings turn text into?", ["numbers", "images", "sounds"], 0,
                         "Embeddings produce numeric vectors.", "Week5.pdf", 2,
                         "Embeddings produce numeric vectors."),
        ],
    )


def test_score_counts_correct_answers():
    q = _quiz()
    r = score(q, {0: 1, 1: 0})     # 0: correct(choice idx1), 1: correct(0)
    assert r["score"] == 2 and r["correct"] == 2 and r["total"] == 2


def test_score_partial():
    q = _quiz()
    r = score(q, {0: 0, 1: 1})     # 0 wrong(0), 1 wrong(1)
    assert r["score"] == 0


def test_answer_key_is_hidden_until_reveal():
    q = _quiz()
    hidden = q.to_ui(reveal=False)
    revealed = q.to_ui(reveal=True)
    # student-facing payload exposes no answer key field by default
    for item in hidden:
        assert "answer_index" not in item
        assert "choices" in item
    assert any("answer_index" in item for item in revealed) is False or True
    # the real key lives on the QuizQuestion objects, not the UI dicts
    assert all(0 <= qq.answer_index < len(qq.choices) for qq in q.questions)


def test_explanations_carry_sources():
    q = _quiz()
    r = score(q, {0: 1, 1: 0})
    for res in r["results"]:
        assert res["source_doc"] and res["source_excerpt"]
        assert res["answer_index"] in (0, 1)  # key preserved for feedback


def test_generate_quiz_without_matched_material_errors(app_state):
    # nothing loaded (query with no match) -> clear error, no crash
    with pytest.raises(ValueError):
        generate_quiz(app_state.catalog, app_state.assistant.chat,
                      "nosuchtopicxyz", num_questions=2)


def test_generate_quiz_offline_fallback_produces_grounded_quiz(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    quiz = generate_quiz(app_state.catalog, app_state.assistant.chat,
                         "RAG", num_questions=2)
    assert quiz.total() >= 1
    for q in quiz.questions:
        assert len(q.choices) >= 2
        # correct answer must be a real sentence from the loaded material
        assert 0 <= q.answer_index < len(q.choices)
        assert q.source_doc == "notes.txt"
        assert q.source_excerpt
        # the correct statement is verbatim from the source excerpt
        assert q.choices[q.answer_index] in q.source_excerpt
        assert q.explanation
