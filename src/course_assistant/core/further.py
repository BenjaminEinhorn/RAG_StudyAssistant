"""Suggestions for further learning beyond the course material.

These are the one place where the model's general knowledge is used on
purpose: 3-5 related subjects a student could study next, each with a short
reason. They are never mixed into answers, never cited as course sources, and
the UI labels them as coming from outside the loaded documents.

The Ask tab gets them in the same LLM call as the answer (``further_topics``
in the answer schema). The model first judges ``question_is_about_course``;
when it is false the code drops the suggestions, because asked to self-censor
while listing, the model invented links (a Super Bowl question got "Sports
Data Analytics" and "RAG Systems"). The Quiz tab asks for them with
:func:`suggest_further_topics` after the quiz is generated.
"""
from __future__ import annotations

from ..services.chat import ChatProvider, LocalEchoChat

MAX_SUGGESTIONS = 5

FURTHER_ITEMS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["topic", "why"],
        "properties": {"topic": {"type": "string"}, "why": {"type": "string"}},
    },
}

RELATED_INSTRUCTION = (
    "question_is_about_course: true if the question is about the subject of "
    "this course (even if the material does not answer it), false if it is "
    "about something unrelated (e.g. sports results, celebrities, weather).")

FURTHER_INSTRUCTION = (
    "further_topics: 3 to 5 related subjects the student could study next "
    "that go BEYOND the course material, each with a one-sentence reason "
    "(why), related to both the question and the course subject. This is "
    "the only place to use your general knowledge; do not repeat what "
    "the material already covers, and never put these in the answer or cite "
    "them.")


def clean_further(items) -> list[dict]:
    """Keep well-formed, de-duplicated suggestions, at most MAX_SUGGESTIONS."""
    out, seen = [], set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        topic = " ".join(str(it.get("topic") or "").split()).strip(" .")
        why = " ".join(str(it.get("why") or "").split())
        if topic and len(topic) <= 80 and topic.lower() not in seen:
            seen.add(topic.lower())
            out.append({"topic": topic, "why": why[:240]})
    return out[:MAX_SUGGESTIONS]


def suggest_further_topics(chat: ChatProvider, subject: str,
                           covered: list[str] | None = None) -> list[dict]:
    """Further-learning suggestions for a quiz subject. ``covered`` lists
    topics the material already covers, so they are not suggested again."""
    if isinstance(chat, LocalEchoChat) or not subject.strip():
        return []
    covered_line = ("The course material already covers: "
                    + "; ".join(covered) + ".\n") if covered else ""
    data = chat.complete_json(
        "You suggest what a business-school student could study next. "
        "Reply with JSON only.",
        f"A student just took a practice quiz on: {subject}.\n{covered_line}"
        f"List {FURTHER_INSTRUCTION[len('further_topics: '):]}",
        {"type": "object", "additionalProperties": False,
         "required": ["further_topics"],
         "properties": {"further_topics": FURTHER_ITEMS_SCHEMA}})
    return clean_further(data.get("further_topics"))


def further_markdown(items: list[dict]) -> str:
    """Markdown block for the UI; empty when there are no suggestions."""
    if not items:
        return ""
    lines = [f"- **{it['topic']}**" + (f" — {it['why']}" if it["why"] else "")
             for it in items]
    return ("**Explore further — beyond the course material**  \n"
            "<span class=hint>Suggested from the model's general knowledge, "
            "not from your documents; not verified.</span>\n\n"
            + "\n".join(lines))
