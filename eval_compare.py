"""A/B comparison of one RAG design choice: reranking ON vs OFF.

Both arms run the *full* answer stage (hybrid retrieval -> evidence ->
class vision LLM -> structured answer + citations) over the same fixed
question set and the same index, which is ingested once. The only difference:

  rerank_on   fused BM25 + text-vector candidates are rescored by the class
              multimodal reranker (text chunks and slide images)
  rerank_off  the reciprocal-rank-fused order is kept as-is

Per question and arm it records, mechanically:
  - target_retrieved      the ground-truth slide is in the evidence given to the LLM
  - target_cited          the answer cites the ground-truth slide
  - all_cited_in_evidence every cited source number is an item retrieval returned
  - all_excerpts_found    every quoted excerpt occurs in the text of its cited chunk
  - latency_s             end-to-end wall clock of Assistant.answer()
  - retrieval_s           share of that spent in text + visual retrieval

and leaves `correct` and `sources_support` EMPTY in the CSV for a human to
grade. An LLM grading its own retrieval is not evidence.

    COURSE_BUILD_MODE=full .venv/bin/python eval_compare.py
    .venv/bin/python eval_compare.py --offline      # plumbing check only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
MATERIALS = BASE / "data" / "materials"
OUT = BASE / "results"

DOC = {"w2": "Week 2", "w4": "Week 4", "w5": "Week 5"}

QUESTIONS = [
    # (question, type, expected_doc, expected_page)  page=None => unanswerable
    ("What is retrieval augmented generation (RAG)?", "text", DOC["w5"], 10),
    ("Why use RAG? Give the benefits over training data.", "text", DOC["w5"], 11),
    ("What two components does hybrid RAG combine?", "text", DOC["w5"], 18),
    ("What does the 'Vibe Coding on Prod' meme show?", "visual", DOC["w2"], 33),
    ("Describe what the hybrid RAG pipeline diagram shows.", "visual", DOC["w5"], 18),
    ("What port does Gradio serve an app on by default?", "text", DOC["w4"], 7),
    ("What is semantic search typically based on?", "text", DOC["w5"], 15),
    ("Name two common chunking strategies.", "text", DOC["w5"], 17),
    ("Who won the 2024 Super Bowl and by how much?", "unanswerable", None, None),
    ("How is RAG context optimization different from fine-tuning?", "text", DOC["w5"], 13),
]

ARMS = ("rerank_on", "rerank_off")


def _is_target(src, doc, page) -> bool:
    return src is not None and doc is not None and doc in src.doc_name and src.page == page


class _Timer:
    """Accumulates wall-clock time spent inside wrapped callables."""

    def __init__(self):
        self.total = 0.0

    def wrap(self, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.total += time.perf_counter() - t0
        return inner


def run_question(app, reranker, arm, qid, q, qtype, doc, page) -> dict:
    app.catalog.reranker = reranker if arm == "rerank_on" else None
    timer = _Timer()
    search, visual = app.catalog.search, app.catalog.visual_search
    app.catalog.search, app.catalog.visual_search = timer.wrap(search), timer.wrap(visual)
    try:
        t0 = time.perf_counter()
        error = None
        try:
            res = app.assistant.answer(q)
        except Exception as e:  # recorded, never hidden
            res, error = None, f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
    finally:
        app.catalog.search, app.catalog.visual_search = search, visual

    row = {"arm": arm, "qid": qid, "question": q, "type": qtype,
           "expected": f"{doc} slide {page}" if page else "(unanswerable)",
           "latency_s": round(latency, 3), "retrieval_s": round(timer.total, 3),
           "error": error}
    if res is None:
        return row
    cites = res.citations
    row.update({
        "answer": res.answer,
        "found": res.found_evidence,
        "evidence": [f"{s.doc_name} p{s.page}" for s in res.evidence],
        "cited": [f"[{c.number}] {c.source.doc_name} p{c.source.page}" if c.source
                  else f"[{c.number}] (not in evidence)" for c in cites],
        "excerpts": [c.excerpt for c in cites],
        "images_shown": len(res.used_images),
        "target_retrieved": (any(_is_target(s, doc, page) for s in res.evidence)
                             if page else None),
        "target_cited": (any(_is_target(c.source, doc, page) for c in cites)
                         if page else None),
        "all_cited_in_evidence": all(c.in_evidence for c in cites) if cites else None,
        "all_excerpts_found": (all(c.excerpt_found for c in cites if c.excerpt.strip())
                               if any(c.excerpt.strip() for c in cites) else None),
        "abstained": not res.found_evidence,
    })
    return row


def summarise(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        ans = [x for x in r if x["type"] != "unanswerable" and not x["error"]]
        una = [x for x in r if x["type"] == "unanswerable" and not x["error"]]
        lat = sorted(x["latency_s"] for x in r)
        ret = sorted(x["retrieval_s"] for x in r)

        def frac(key, rows_):
            vals = [x.get(key) for x in rows_ if x.get(key) is not None]
            return f"{sum(bool(v) for v in vals)}/{len(vals)}"

        out[arm] = {
            "errors": sum(1 for x in r if x["error"]),
            "target_retrieved": frac("target_retrieved", ans),
            "target_cited": frac("target_cited", ans),
            "all_cited_in_evidence": frac("all_cited_in_evidence", r),
            "all_excerpts_found": frac("all_excerpts_found", r),
            "unanswerable_abstained": frac("abstained", una),
            "latency_median_s": lat[len(lat) // 2] if lat else None,
            "latency_mean_s": round(sum(lat) / len(lat), 3) if lat else None,
            "latency_max_s": lat[-1] if lat else None,
            "retrieval_median_s": ret[len(ret) // 2] if ret else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true",
                    help="allow COURSE_BUILD_MODE=local (plumbing check, NOT a result)")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    args = ap.parse_args()

    # isolated index for the run; the mode is whatever the environment says
    os.environ["COURSE_DATA_DIR"] = tempfile.mkdtemp(prefix="eval_")
    from course_assistant.config.settings import Settings
    from course_assistant.factory import build_app_state

    settings = Settings()
    if settings.build_mode != "full":
        if not args.offline:
            sys.exit("Refusing to run: COURSE_BUILD_MODE is not 'full'. Offline hash "
                     "embeddings are not semantic and the LLM is stubbed, so the numbers "
                     "would mean nothing. Pass --offline to check plumbing anyway.")
        print("!" * 72 + "\n!! OFFLINE MODE: stub LLM + hash embeddings. These numbers are "
              "NOT a result.\n" + "!" * 72)

    decks = sorted(MATERIALS.glob("*.pptx"))
    if not decks:
        sys.exit(f"No decks in {MATERIALS}; add the Week 2/4/5 .pptx files.")
    app = build_app_state(settings, rerank_enabled=True)
    reranker = app.catalog.reranker
    t0 = time.perf_counter()
    for d in decks:
        print(app.catalog.add_file(str(d))["message"])
    ingest_s = time.perf_counter() - t0

    # one untimed warm-up so connection setup is not billed to the first arm
    app.assistant.answer("warm-up: what is RAG?")

    rows = []
    for qid, (q, qtype, doc, page) in enumerate(QUESTIONS, 1):
        # alternate arm order per question so neither arm always runs first
        order = ARMS if qid % 2 else tuple(reversed(ARMS))
        for arm in order:
            row = run_question(app, reranker, arm, qid, q, qtype, doc, page)
            rows.append(row)
            print(f"[{arm:10s}] Q{qid:<2d} {row['latency_s']:6.2f}s "
                  f"retrieved={row.get('target_retrieved')} cited={row.get('target_cited')} "
                  f"excerpts_ok={row.get('all_excerpts_found')} err={row['error']}")
    rows.sort(key=lambda r: (r["qid"], r["arm"]))

    meta = {
        "mode": settings.build_mode,
        "valid_result": settings.build_mode == "full",
        "decks": [d.name for d in decks],
        "ingest_s": round(ingest_s, 1),
        "models": settings.service_env,
        "k_text": 5, "k_visual": 3, "candidate_pool_per_retriever": app.catalog._pool_size(5),
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    summary = summarise(rows)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = "comparison" if meta["valid_result"] else "comparison_OFFLINE_not_a_result"
    (out / f"{stem}.json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "rows": rows}, indent=2, default=str))
    cols = ["arm", "qid", "type", "question", "expected", "answer", "cited", "excerpts",
            "target_retrieved", "target_cited", "all_cited_in_evidence",
            "all_excerpts_found", "latency_s", "retrieval_s", "error",
            "correct", "sources_support"]
    with open(out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "cited": "; ".join(r.get("cited", [])),
                        "excerpts": " | ".join(r.get("excerpts", [])),
                        "correct": "", "sources_support": ""})

    print(f"\ningest {meta['ingest_s']}s  mode={meta['mode']}")
    print("| Metric | rerank on | rerank off |\n|---|---|---|")
    for key in summary["rerank_on"]:
        print(f"| {key} | {summary['rerank_on'][key]} | {summary['rerank_off'][key]} |")
    print(f"saved {out / stem}.json and .csv (fill in `correct` and `sources_support`)")


if __name__ == "__main__":
    main()
