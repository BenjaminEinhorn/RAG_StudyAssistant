"""Gradio UI for the Course Assistant.

Three tabs:
- ASK: pick document(s) + type/see a question -> grounded answer with slides shown
  as visual evidence, plus plainly stated "not in material" when nothing matches.
- QUIZ: pick document(s) + topic -> MCQs from the material; answers scored
  against a hidden key; explanations + sources revealed after answering.
- DOCUMENTS: add (dedup) and remove files, list what is loaded.

A shared factory (:mod:`course_assistant.factory`) builds the catalog, service
providers and assistant; handlers reference the global app state so the UI is
thin and the pure logic stays unit-testable.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from ..config.settings import Settings, redact
from ..core.assistant import Assistant
from ..core.index import Catalog
from ..core.quiz import Quiz, generate_quiz, score
from ..factory import AppState

MAX_QUESTIONS = 10

BRAND_CSS = """
<style>
  :root { --brand: #b45309; --brand2: #d97706; }
  .course-hero { background: linear-gradient(120deg,#7c2d12,#b45309 55%,#d97706);
     color:#fff; padding:22px 26px; border-radius:18px; margin-bottom:18px; }
  .course-hero h1, .course-hero p { color:#fff !important; }
  .course-hero h1 { margin:0; font-size:26px; letter-spacing:.4px; }
  .course-hero p { margin:6px 0 0; opacity:.94; font-size:14px; }
  .gr-accordion { border-radius:14px !important; }
  .answer-card { border-left:4px solid var(--brand); background:transparent; }
  .hint { opacity:.65; font-size:12px; }
</style>
"""


def refresh_docs(app: AppState, clear_picks: bool = False):
    """Updates for (doc_table, remove_dd, ask_docs, q_docs).

    The remove selection is always cleared, and the Ask/Quiz selections too
    when documents went away (``clear_picks``): Gradio rejects the next event
    if a component still holds a value that is no longer among its choices.
    """
    choices = app.doc_choices()
    picks = {"value": []} if clear_picks else {}
    return (gr.update(value=list_docs_df(app)),
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, **picks),
            gr.update(choices=choices, **picks))


def add_file_handler(app: AppState, filepaths, files: list):
    """Ingest one or more uploaded files; one message line per file.

    A failure on one file is reported and the rest are still added.
    """
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    filepaths = [f for f in (filepaths or []) if f]
    if not filepaths:
        return "No file selected.", *[gr.update() for _ in range(4)]
    messages = []
    for filepath in filepaths:
        # copy into the app's materials dir (source of truth for indexing)
        src = Path(filepath)
        dest = app.settings.materials_dir / src.name
        try:
            dest.write_bytes(src.read_bytes())
            messages.append(app.catalog.add_file(str(dest))["message"])
        except Exception as e:  # surface failures (e.g. LibreOffice) in the UI
            messages.append(f"⚠️ Could not add '{src.name}': {redact(str(e))}")
    topic_error = app.sync_topics()   # one LLM call per newly added document
    if topic_error:
        messages.append(f"⚠️ Could not list quiz topics: {topic_error}")
    return ("  \n".join(messages), *refresh_docs(app))


def remove_handler(app: AppState, doc_names):
    """Remove one or more documents and all their searchable content."""
    if isinstance(doc_names, str):
        doc_names = [doc_names]
    doc_names = [d for d in (doc_names or []) if d]
    if not doc_names:
        return "Select at least one document to remove.", *[gr.update() for _ in range(4)]
    messages = [app.catalog.remove_document(name)["message"] for name in doc_names]
    app.sync_topics()
    return ("  \n".join(messages), *refresh_docs(app, clear_picks=True))


def list_docs_df(app: AppState) -> list:
    docs = app.catalog.list_documents()
    if not docs:
        return [["(none loaded yet)", 0, 0]]
    return [[d["doc_name"], d["chunks"], d["slide_count"]] for d in docs]


def ask_handler(app: AppState, question: str, doc_selection: list, ask_images: bool):
    if not question.strip():
        return "Please type a question.", []
    try:
        result = app.assistant.answer(
            question.strip(), doc_names=doc_selection or None, include_images=ask_images)
    except Exception as e:  # service errors: show them, never the key
        return f"⚠️ Could not answer: {redact(str(e))}", []
    answer_md = f"**Answer**  \n{result.answer}\n\n"
    if result.outside_slides:
        label = ("Not covered by the slides" if result.coverage == "none"
                 else "Partly outside the slides")
        answer_md += f"> ⚠️ **{label}:** {result.beyond_slides}\n\n"
    src_md = "**Sources**  \n"
    if not result.found_evidence:
        src_md += "*(no supporting material found — nothing is cited)*"
    else:
        lines = []
        for c in result.citations:
            if c.source is None:
                lines.append(f"- ⚠️ cited evidence [{c.number}] does not exist — ignored")
                continue
            if c.excerpt_found:
                check = f"✅ quote verified: “{c.excerpt}”"
            elif c.supported:
                check = "✅ supported by the slide image shown to the model"
            else:
                check = f"⚠️ quote not found in this slide: “{c.excerpt}”"
            lines.append(f"- **{c.source.doc_name}** — slide/page **{c.source.page}**  \n  {check}")
        if not result.citations:
            lines.append("⚠️ The answer cites no source; treat it as unverified.")
        src_md += "\n".join(lines)
    gallery = []
    if result.found_evidence:
        # cited slides first, then the other slides the model was shown
        cited = [s.image_path for s in result.sources if s.image_path]
        for img in cited + result.used_images:
            if img not in [g for g, _ in gallery]:
                gallery.append((img, _caption_of(result.evidence, img)))
    return answer_md + src_md, gallery


def _caption_of(sources, img_path: str) -> str:
    for s in sources:
        if s.image_path == img_path and s.image_path:
            return f"Slide {s.page} · {s.doc_name}"
    return Path(img_path).parent.name + " slide " + Path(img_path).stem[1:]


def quiz_feedback(quiz: Quiz, answers: dict[int, int], reveal_all: bool = False) -> str:
    """Markdown feedback: score plus, per question, the key, explanation and source."""
    res = score(quiz, answers)
    lines = [f"**Score: {res['score']} / {res['total']}**" if not reveal_all
             else "**Answer key**"]
    for r, q in zip(res["results"], quiz.questions):
        right = q.choices[r["answer_index"]]
        if reveal_all:
            mark = "🔑"
        elif r["chosen"] is None:
            mark = "⏭️ not answered"
        else:
            mark = "✅" if r["correct"] else f"❌ you chose “{q.choices[r['chosen']]}”"
        lines.append(f"{mark} **Q{r['id']+1}** — correct answer: **{right}**  \n"
                     f"{r['explanation']}  \n"
                     f"<span class=hint>Source: {r['source_doc']} — slide/page "
                     f"{r['source_page']} · “{(r['source_excerpt'] or '')[:200]}”</span>")
    return "\n\n".join(lines)


def topic_dropdown_update(app: AppState, doc_names: list[str] | None,
                          current: str | None = None):
    """New quiz-topic choices for the selected documents, keeping the current
    pick only if it is still offered."""
    choices = app.topic_choices(doc_names or None)
    value = current if current in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def q_inputs():
    return [gr.State(), *[gr.Radio(choices=[], label=f"Q{i+1}", visible=False)
                           for i in range(MAX_QUESTIONS)]]


def build_ui(app: AppState) -> gr.Blocks:
    with gr.Blocks(title="MBAX 6418 Course Assistant") as demo:
        gr.HTML(BRAND_CSS)
        gr.HTML("""
        <div class="course-hero">
          <h1>📚 MBAX 6418 Course Assistant</h1>
          <p>Answer questions and generate practice quizzes from your course
          slides, with the original slide shown as visual evidence.</p>
        </div>""")

        # ---------------- ASK ----------------
        with gr.Tab("Ask"):
            ask_docs = gr.CheckboxGroup(label="Course material (optional — ask all)",
                                        choices=app.doc_choices())
            ask_question = gr.Textbox(label="Your question", lines=2,
                                      placeholder="e.g. What does the 'Vibe Coding on Prod' meme mean?")
            ask_imgs = gr.Checkbox(label="Show slide images as visual evidence", value=True)
            ask_btn = gr.Button("Ask", variant="primary")
            ask_answer = gr.Markdown()
            ask_gallery = gr.Gallery(label="Supporting slide images", height=420,
                                     columns=3, object_fit="contain")
            ask_btn.click(
                lambda q, d, i: ask_handler(app, q, d, i),
                inputs=[ask_question, ask_docs, ask_imgs],
                outputs=[ask_answer, ask_gallery], api_name="ask")

        # ---------------- QUIZ ----------------
        with gr.Tab("Quiz") as quiz_tab:
            q_docs = gr.CheckboxGroup(label="Course material", choices=app.doc_choices())
            with gr.Row():
                topics = app.topic_choices()
                q_topic = gr.Dropdown(
                    label="Topic", choices=topics, value=topics[0] if topics else None,
                    info="General topics found in the selected material")
                q_num = gr.Slider(3, MAX_QUESTIONS, value=5, step=1,
                                  label="Number of questions")
            q_gen = gr.Button("Generate quiz", variant="primary")
            q_status = gr.Markdown("")

            # answers: a fixed grid of question radios
            qrows = []
            for i in range(MAX_QUESTIONS):
                radio = gr.Radio(choices=[], label=f"Q{i+1}", visible=False,
                                 interactive=True)
                qrows.append(radio)

            def gen(app, topic, num, docs):
                if not topic:
                    return ("Add course material first, then pick a topic.", "",
                            *[gr.update() for _ in range(MAX_QUESTIONS)])
                try:
                    quiz = generate_quiz(app.catalog, app.assistant.chat,
                                         topic, doc_names=docs or None,
                                         num_questions=int(num))
                    app.quiz = quiz
                except Exception as e:  # surface clear, non-crashing error
                    return (f"⚠️ Could not generate quiz: {redact(str(e))}", "",
                            *[gr.update() for _ in range(MAX_QUESTIONS)])
                updates = []
                for i, q in enumerate(quiz.questions):
                    # value = choice index, so grading never parses choice text
                    updates.append(gr.update(
                        label=f"Q{i+1}: {q.question}", visible=True, value=None,
                        choices=[(c, j) for j, c in enumerate(q.choices)]))
                for _ in range(len(quiz.questions), MAX_QUESTIONS):
                    updates.append(gr.update(visible=False, value=None))
                return (f"✅ Generated {quiz.total()} questions — answer below, "
                        "then **Grade** (or **Show answers**).", "", *updates)

            def grade(app, *radio_answers):
                if app.quiz is None:
                    return "Generate a quiz first."
                submitted = {i: int(v) for i, v in enumerate(radio_answers)
                             if v is not None and i < app.quiz.total()}
                if not submitted:
                    return "Answer at least one question first (or **Show answers**)."
                return quiz_feedback(app.quiz, submitted)

            def reveal(app):
                if app.quiz is None:
                    return "Generate a quiz first."
                return quiz_feedback(app.quiz, {}, reveal_all=True)

            q_result = gr.Markdown("")
            q_gen.click(lambda t, n, d: gen(app, t, n, d),
                        inputs=[q_topic, q_num, q_docs],
                        outputs=[q_status, q_result, *qrows])
            # topics follow the selected decks, and pick up newly added ones
            q_docs.change(lambda d, t: topic_dropdown_update(app, d, t),
                          inputs=[q_docs, q_topic], outputs=q_topic)
            quiz_tab.select(lambda d, t: topic_dropdown_update(app, d, t),
                            inputs=[q_docs, q_topic], outputs=q_topic)
            with gr.Row():
                q_grade = gr.Button("Grade my answers", variant="secondary")
                q_reveal = gr.Button("Show answers")
            q_grade.click(lambda *a: grade(app, *a), inputs=qrows, outputs=q_result)
            q_reveal.click(lambda: reveal(app), inputs=[], outputs=q_result)

        # ---------------- DOCUMENTS ----------------
        with gr.Tab("Documents"):
            file_up = gr.File(label="Add course material (pdf, pptx, docx, txt, md) — "
                                    "drop one or more files",
                              type="filepath", file_count="multiple")
            add_msg = gr.Markdown("")
            doc_table = gr.Dataframe(
                headers=["Document", "Text chunks", "Slides/pages"],
                value=list_docs_df(app), interactive=False, wrap=True)
            remove_dd = gr.Dropdown(choices=app.doc_choices(), value=[],
                                    multiselect=True, label="Remove documents",
                                    info="Pick one or more, then click Remove")
            remove_btn = gr.Button("Remove selected", variant="stop")
            remove_msg = gr.Markdown("")
            refresh_btn = gr.Button("Refresh")

            file_up.change(lambda f: add_file_handler(app, f, []),
                           inputs=file_up, outputs=[add_msg, doc_table, remove_dd,
                                                    ask_docs, q_docs])
            remove_btn.click(lambda d: remove_handler(app, d),
                             inputs=remove_dd,
                             outputs=[remove_msg, doc_table, remove_dd, ask_docs, q_docs])
            refresh_btn.click(lambda: refresh_docs(app),
                              inputs=[], outputs=[doc_table, remove_dd, ask_docs, q_docs])
    return demo


def main() -> None:
    from ..factory import build_app_state

    settings = Settings()
    state = build_app_state(settings)
    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="gray",
                           radius_size=gr.themes.sizes.radius_lg)
    demo = build_ui(state)
    demo.launch(theme=theme, share=False)


if __name__ == "__main__":
    main()
