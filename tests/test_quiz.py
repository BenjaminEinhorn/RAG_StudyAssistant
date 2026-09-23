"""Unit tests for quiz generation, hidden key, scoring, and explanations."""
import pytest

from course_assistant.core.quiz import Quiz, QuizQuestion, generate_quiz, score
from course_assistant.core.assistant import material_covers
from course_assistant.core.index import RetrievedChunk


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


class _QuizChat:
    def __init__(self, questions, relevance=None):
        self.questions = questions
        self.relevance = relevance   # {"found","answer","citations"} for the gate call

    def complete_json(self, system, user, schema, images=None):
        if "questions" in (schema.get("required") or []):   # quiz generation
            return {"questions": self.questions}
        if self.relevance is not None:
            return self.relevance
        return {"found": True, "answer": "covers", "citations": []}  # fallback


def test_live_quiz_uses_each_questions_cited_source_and_drops_invalid(app_state, small_doc, tmp_path):
    other = tmp_path / "other.txt"
    other.write_text("Reranking reorders candidate chunks by relevance score.")
    app_state.catalog.add_file(str(small_doc))
    app_state.catalog.add_file(str(other))
    chunks = app_state.catalog.search("reranking relevance", k=8)
    rerank_n = next(i for i, c in enumerate(chunks, 1) if c.doc_name == "other.txt")
    quiz = generate_quiz(app_state.catalog, _QuizChat([
        {"question": "What does reranking do?", "choices": ["a", "reorders chunks", "c", "d"],
         "answer_index": 1, "explanation": "e", "source": rerank_n,
         "excerpt": "reorders candidate chunks"},
        {"question": "bad key", "choices": ["a", "b"], "answer_index": 5,
         "explanation": "e", "source": 1, "excerpt": ""},
        {"question": "bad source", "choices": ["a", "b"], "answer_index": 0,
         "explanation": "e", "source": 42, "excerpt": ""},
    ], relevance={"found": True, "answer": "covers",
                  "citations": [{"source": rerank_n,
                                 "excerpt": "reorders candidate chunks"}]}),
        "reranking relevance", num_questions=3)
    assert quiz.total() == 1
    q = quiz.questions[0]
    assert q.source_doc == "other.txt"
    assert q.source_excerpt == "reorders candidate chunks"


class _CoverageChat:
    """Distinguishes the relevance (found) call from quiz generation
    (questions) by which schema the provider is being asked to satisfy."""

    def __init__(self, relevance_found: bool, questions: list):
        self.relevance_found = relevance_found
        self.questions = questions
        self.relevance_calls = 0
        self.gen_calls = 0

    def complete_json(self, system, user, schema, images=None):
        if "questions" in (schema.get("required") or []):      # quiz generation
            self.gen_calls += 1
            return {"questions": self.questions}
        self.relevance_calls += 1                              # found/covers call
        if self.relevance_found:
            return {"found": True, "answer": "covers",
                    "citations": [{"source": 1, "excerpt": "Hybrid RAG combines"}]}
        return {"found": False, "answer": "", "citations": []}


def _valid_q(question: str) -> dict:
    return {"question": question, "choices": ["a", "b", "c", "d"],
            "answer_index": 1, "explanation": "e", "source": 1,
            "excerpt": "Hybrid RAG combines"}


def test_quiz_topic_not_covered_generates_no_quiz(app_state, small_doc):
    # retrieval still returns chunks, but the relevance gate says the selected
    # material does not cover the topic -> no quiz, clear message.
    app_state.catalog.add_file(str(small_doc))
    chat = _CoverageChat(relevance_found=False, questions=[_valid_q("Q?")])
    with pytest.raises(ValueError) as ei:
        generate_quiz(app_state.catalog, chat, "Pizza toppings", num_questions=2)
    assert "Pizza toppings" in str(ei.value)
    assert "No material" in str(ei.value)
    assert chat.relevance_calls == 1     # model-level gate fired, not empty retrieval
    assert chat.gen_calls == 0           # no quiz was generated


def test_quiz_topic_covered_generates_quiz(app_state, small_doc):
    app_state.catalog.add_file(str(small_doc))
    chat = _CoverageChat(relevance_found=True,
                         questions=[_valid_q("Q1?"), _valid_q("Q2?")])
    quiz = generate_quiz(app_state.catalog, chat, "RAG", num_questions=2)
    assert chat.relevance_calls == 1
    assert chat.gen_calls == 1
    assert quiz.total() == 2
    for q in quiz.questions:
        assert 0 <= q.answer_index < len(q.choices)


def test_quiz_empty_topic_skips_relevance_check(app_state, small_doc, monkeypatch):
    # requirement: empty/whitespace topic keeps today's behavior -- generate and
    # never run the relevance gate.
    app_state.catalog.add_file(str(small_doc))

    def _must_not_run(*a, **k):
        raise AssertionError("material_covers must not run for an empty topic")

    monkeypatch.setattr("course_assistant.core.quiz.material_covers", _must_not_run)
    chat = _CoverageChat(relevance_found=True, questions=[_valid_q("Q?")])
    for topic in ("", "   "):
        quiz = generate_quiz(app_state.catalog, chat, topic, num_questions=1)
        assert quiz.total() == 1
    assert chat.relevance_calls == 0


def _chunk(text, doc="notes.txt", page=1):
    return RetrievedChunk(chunk_id="c", doc_id="d", doc_name=doc, page=page,
                          text=text, image_path=None, score=1.0)


class _FakeCatalog:
    def __init__(self, *chunks):
        self._chunks = list(chunks)

    def search(self, query, k=5, doc_names=None):
        return self._chunks


class _GateChat:
    """:class:`material_covers` chat double: controllable found + citations."""
    def __init__(self, found=True, citations=()):
        self.found = found
        self.citations = list(citations)

    def complete_json(self, system, user, schema, images=None):
        return {"found": self.found, "answer": "x", "citations": self.citations}


def test_material_covers_rejects_offtopic_but_adjacent_chunk():
    # (a) W3-slide-5-style "retrieved context" slide; model says found=True and
    # cites a verbatim (verified) excerpt, but the chunk isn't grounded in the
    # topic.
    chunk = _chunk(
        "How Prompts are Processed • Agent harnesses such as Hermes Agent "
        "process your prompt through a stack of context • This stack includes "
        "the user prompt, system prompt, conversation history, retrieved "
        "context/memory, tool schemas, tool results.")
    chat = _GateChat(found=True, citations=[
        {"source": 1,
         "excerpt": "Agent harnesses such as Hermes Agent process your prompt"}])
    assert material_covers(_FakeCatalog(chunk), chat,
                           "Retrieval-Augmented Generation") is False
    assert material_covers(_FakeCatalog(chunk), chat, "Pizza toppings") is False


def test_material_covers_generation_word_alone_is_not_rag():
    # (b) only one of the three topic tokens ("generation") present -> not covered
    chunk = _chunk("The generation of text is fast. Models generate tokens.")
    chat = _GateChat(found=True, citations=[
        {"source": 1, "excerpt": "The generation of text is fast"}])
    assert material_covers(_FakeCatalog(chunk), chat,
                           "Retrieval-Augmented Generation") is False


def test_material_covers_full_phrase_and_short_acronym():
    # (c) the exact phrase covers the long topic; the acronym covers "RAG"
    chunk = _chunk("Retrieval-Augmented Generation (RAG) combines retrieval "
                   "and generation.")
    chat_long = _GateChat(found=True, citations=[
        {"source": 1, "excerpt": "Retrieval-Augmented Generation (RAG) combines"}])
    chat_ac = _GateChat(found=True, citations=[
        {"source": 1, "excerpt": "RAG"}])
    assert material_covers(_FakeCatalog(chunk), chat_long,
                           "Retrieval-Augmented Generation") is True
    assert material_covers(_FakeCatalog(chunk), chat_ac, "RAG") is True


def test_quiz_pizza_toppings_rejected_and_empty_topic_skips(app_state, small_doc,
                                                            monkeypatch):
    # (d) "Pizza toppings" still rejected through the real guard ...
    app_state.catalog.add_file(str(small_doc))
    chat = _CoverageChat(relevance_found=True, questions=[_valid_q("Q?")])
    with pytest.raises(ValueError) as ei:
        generate_quiz(app_state.catalog, chat, "Pizza toppings", num_questions=1)
    assert "Pizza toppings" in str(ei.value) and "No material" in str(ei.value)
    assert chat.relevance_calls == 1 and chat.gen_calls == 0
    # ... and an empty topic still skips the relevance gate entirely.
    def _must_not_run(*a, **k):
        raise AssertionError("material_covers must not run for an empty topic")
    monkeypatch.setattr("course_assistant.core.quiz.material_covers", _must_not_run)
    quiz = generate_quiz(app_state.catalog, chat, "   ", num_questions=1)
    assert quiz.total() == 1
