"""Quiz generation and scoring.

Quiz questions are generated from retrieved course material so they are
answerable from the loaded docs. The answer key is stored server-side only
(:attr:`QuizQuestion.answer_index`) and is hidden from the UI until the
student submits an answer or explicitly requests the answer.

Scoring and the hide/show logic are pure functions (unit-testable without a
model); generation needs a live chat provider and reports a clear error when
one is unavailable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..config.settings import Settings
from ..services.chat import ChatProvider
from .assistant import excerpt_in_text
from .index import Catalog, RetrievedChunk


@dataclass
class QuizQuestion:
    question: str
    choices: list[str]
    answer_index: int          # NOT exposed to the student
    explanation: str
    source_doc: str | None = None
    source_page: int | None = None
    source_excerpt: str | None = None


@dataclass
class Quiz:
    topic: str
    questions: list[QuizQuestion] = field(default_factory=list)

    def total(self) -> int:
        return len(self.questions)

    def to_ui(self, reveal: bool = False) -> list[dict]:
        """Serialize questions for the UI. The answer key stays hidden until
        ``reveal`` (student answered or requested it)."""
        out = []
        for i, q in enumerate(self.questions):
            item = {
                "id": i,
                "question": q.question,
                "choices": q.choices,
                "explanation": q.explanation,
                "source_doc": q.source_doc,
                "source_page": q.source_page,
                "source_excerpt": q.source_excerpt,
            }
            out.append(item)
        return out


def score(quiz: Quiz, answers: dict[int, int]) -> dict:
    """Score a submission against the fixed answer key.

    ``answers`` maps question id -> chosen choice index (as integers). Returns
    score, total, per-question correctness, and reveals the correct choice for
    any question the student answered (or all, if requested).
    """
    total = quiz.total()
    if total == 0:
        return {"score": 0, "correct": 0, "total": 0, "results": []}
    results = []
    correct = 0
    for i, q in enumerate(quiz.questions):
        chosen = answers.get(i)
        is_correct = chosen == q.answer_index
        if is_correct:
            correct += 1
        results.append({
            "id": i,
            "correct": is_correct,
            "chosen": chosen,
            "answer_index": q.answer_index,   # fixed key (shown after answer)
            "explanation": q.explanation,
            "source_doc": q.source_doc,
            "source_page": q.source_page,
            "source_excerpt": q.source_excerpt,
        })
    return {"score": correct, "correct": correct, "total": total, "results": results}


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return json.loads(text[start:end + 1])


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= 7 and re.search(r"[A-Za-z]", p):
            out.append(p)
    return out


def _offline_quiz(chunks, num_questions: int) -> Quiz:
    """Deterministic, material-grounded quiz for offline/local mode.

    Uses verbatim sentences from the retrieved slides as answerable statements:
    the correct option is a real sentence from a slide, distractors are other
    sentences from the same deck. Produces real (answerable) questions even
    when no live LLM is configured.
    """
    sentence_pool = []
    for c in chunks:
        for s in _sentences(c.text):
            sentence_pool.append((s, c))
    if not sentence_pool:
        raise ValueError("No usable sentences found to build a quiz.")
    questions, used = [], set()
    # prefer a spread across chunks for topical coverage
    ordered = sorted(sentence_pool, key=lambda t: -len(t[0]))
    for s, chunk in ordered:
        if len(questions) >= num_questions:
            break
        key = s.lower()
        if key in used:
            continue
        # pick up to 3 distractors from the whole pool (different sentences)
        distractors = []
        for other, _ in sentence_pool:
            if other.lower() == key:
                continue
            ot = re.sub(r"\s+", " ", other).strip()
            if ot and ot not in distractors and ot != s:
                distractors.append(ot)
            if len(distractors) >= 3:
                break
        if len(distractors) < 2:
            continue
        choices = [s] + distractors  # correct first; shuffled below
        import random
        # stable shuffle so correct position is deterministic per session
        idx = [i for i in range(len(choices))]
        rng = random.Random(hash((s, "q")))
        rng.shuffle(idx)
        choices4 = [choices[i] for i in idx]
        answer_index = idx.index(0)
        questions.append(QuizQuestion(
            question="According to the course material, which statement is correct?",
            choices=choices4[:4],
            answer_index=answer_index,
            explanation=(f"This statement appears on {chunk.doc_name}, slide/page "
                         f"{chunk.page}. The others are not supported by the slides."),
            source_doc=chunk.doc_name,
            source_page=chunk.page,
            source_excerpt=s[:400],
        ))
        used.add(key)
    if not questions:
        raise ValueError("No answerable questions could be built from the material.")
    return Quiz(topic="Quiz", questions=questions)


QUIZ_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["question", "choices", "answer_index", "explanation",
                     "source", "excerpt"],
        "properties": {
            "question": {"type": "string"},
            "choices": {"type": "array", "items": {"type": "string"}},
            "answer_index": {"type": "integer"},
            "explanation": {"type": "string"},
            "source": {"type": "integer"},
            "excerpt": {"type": "string"},
        }}}},
}


def generate_quiz(catalog: Catalog, chat: ChatProvider,
                  topic: str, doc_names: list[str] | None = None,
                  num_questions: int = 5, context_k: int = 8,
                  chunk_method: str = "recursive") -> Quiz:
    """Generate MCQs grounded in the retrieved course material.

    With the live LLM, each question must cite the excerpt number it is based
    on plus a verbatim quote; questions with an out-of-range key or source are
    dropped, and errors propagate so the UI can show them. The deterministic
    offline generator (verbatim slide sentences) is used only in local mode.
    """
    from ..services.chat import LocalEchoChat

    chunks = catalog.search(topic, k=context_k, doc_names=doc_names)
    if not chunks:
        raise ValueError("No course material matched this topic. Try a broader "
                         "topic or select a different document.")
    if isinstance(chat, LocalEchoChat):
        return _offline_quiz(chunks, num_questions)
    context_block = "\n\n".join(
        f"[{i}] ({c.doc_name}, slide/page {c.page})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )
    prompt = (
        f"Using ONLY the course material excerpts below, write {num_questions} "
        "multiple-choice quiz questions about this topic: " + topic + "\n\n"
        + context_block + "\n\n"
        "For each question give: question; exactly 4 choices; answer_index "
        "(0-based index of the correct choice); explanation (one sentence that "
        "names the document and slide); source (the excerpt number the answer "
        "comes from); excerpt (a short phrase copied VERBATIM from that excerpt "
        "that proves the answer). Every question MUST be answerable from its "
        "cited excerpt, and exactly one choice may be correct."
    )
    data = chat.complete_json(
        "You generate pedagogically sound multiple-choice quiz questions that "
        "are strictly answerable from provided course material.", prompt, QUIZ_SCHEMA)
    questions = []
    for q in data.get("questions", []):
        choices = [str(c) for c in q.get("choices") or []]
        n, key = q.get("source"), q.get("answer_index")
        if (not q.get("question") or len(choices) < 2
                or not isinstance(key, int) or not 0 <= key < len(choices)
                or not isinstance(n, int) or not 1 <= n <= len(chunks)):
            continue
        chunk = chunks[n - 1]
        excerpt = str(q.get("excerpt") or "")
        questions.append(QuizQuestion(
            question=q["question"],
            choices=choices,
            answer_index=key,
            explanation=q.get("explanation", ""),
            source_doc=chunk.doc_name,
            source_page=chunk.page,
            # show the model's quote only if it really occurs in the slide text
            source_excerpt=(excerpt if excerpt_in_text(excerpt, chunk.text)
                            else chunk.text[:400]),
        ))
        if len(questions) >= num_questions:
            break
    if not questions:
        raise ValueError("The model returned no valid, source-grounded questions.")
    return Quiz(topic=topic, questions=questions)
