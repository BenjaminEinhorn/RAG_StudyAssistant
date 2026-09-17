"""RAG evaluation: answer-quality comparison across two design axes.

For an identical question set (incl. >=2 visual questions and 1 unanswerable),
we record retrieval correctness (top-3 contains the ground-truth slide), source
support, and end-to-end latency for:
  A) rerank ON  vs  rerank OFF
  B) chunking: recursive vs fixed-size

Runs in COURSE_BUILD_MODE=local (deterministic offline embedding) so it is
repeatable without live keys. Results are written to results/comparison.json
and a markdown table.
"""
import json
import os
import time
from pathlib import Path

os.environ["COURSE_BUILD_MODE"] = "local"
import shutil
import tempfile

from course_assistant.config.settings import Settings
from course_assistant.factory import build_app_state

BASE = Path(__file__).resolve().parents[0]
MATERIALS = BASE / "data" / "materials"

# doc-name substrings used to match ground truth
DOC = {
    "w2": "Week 2", "w4": "Week 4", "w5": "Week 5",
}

QUESTIONS = [
    # (q, expected_doc, expected_page)  page=None => unanswerable
    ("What is retrieval augmented generation (RAG)?", DOC["w5"], 10),
    ("Why use RAG? Give the benefits over training data.", DOC["w5"], 11),
    ("What two components does hybrid RAG combine?", DOC["w5"], 18),
    ("What does the 'Vibe Coding on Prod' meme show? (visual)", DOC["w2"], 33),
    ("Describe what the hybrid RAG pipeline diagram shows. (visual)", DOC["w5"], 18),
    ("What port does Gradio serve an app on by default?", DOC["w4"], 7),
    ("What is semantic search typically based on?", DOC["w5"], 15),
    ("Name two common chunking strategies.", DOC["w5"], 17),
    ("Who won the 2024 Super Bowl and by how much?", None, None),  # unanswerable
    ("How is RAG context optimization different from fine-tuning?", DOC["w5"], 13),
]

CONFIGS = [
    ("rerank_on/recursive", dict(rerank_enabled=True, chunk_method="recursive")),
    ("rerank_off/recursive", dict(rerank_enabled=False, chunk_method="recursive")),
    ("rerank_on/fixed", dict(rerank_enabled=True, chunk_method="fixed")),
]


def ingest(app, names):
    for n in names:
        p = MATERIALS / n
        app.catalog.add_file(str(p))


def build_cache_dir(base):
    td = Path(tempfile.mkdtemp(prefix="eval_"))
    os.environ["COURSE_DATA_DIR"] = str(td)
    return td


def run_config(cfg_name, cfg, decks, gt):
    td = build_cache_dir(None)
    s = Settings()
    app = build_app_state(s, chunk_method=cfg["chunk_method"],
                           rerank_enabled=cfg["rerank_enabled"])
    t_ing = time.time()
    ingest(app, decks)
    ingest_s = time.time() - t_ing
    rows = []
    for q, doc_k, page in QUESTIONS:
        t0 = time.time()
        hits = app.catalog.search(q, k=6)
        dt = time.time() - t0
        top3 = hits[:3]
        # correctness: ground-truth slide present among top 3 text sources
        correct = doc_k and page is not None and any(
            doc_k in h.doc_name and h.page == page for h in top3)
        # source support: at least one top source shares substantive tokens
        from course_assistant.services.reranker import _tokens
        qt = _tokens(q)
        support = any(_tokens(h.text) & qt for h in hits[:2])
        rows.append({
            "question": q, "expected_doc": doc_k, "expected_page": page,
            "top1": top3[0].page if top3 else None,
            "top1_doc": top3[0].doc_name if top3 else None,
            "top3_pages": [h.page for h in top3],
            "correct": bool(correct), "source_support": bool(support),
            "latency_ms": round(dt * 1000, 1),
        })
    correct_n = sum(1 for r in rows if r["correct"])
    # unanswerable handled separately (question index 8)
    un_ok = rows[8]  # should NOT find a high-confidence match
    res = {
        "config": cfg_name, "ingest_s": round(ingest_s, 2),
        "retrieval_correct_top3": f"{correct_n}/{len(QUESTIONS)}",
        "mean_latency_ms": round(sum(r['latency_ms'] for r in rows) / len(rows), 1),
        "unanswerable_top1_page": rows[8]["top1"],  # report; no strong match expected
        "rows": rows,
    }
    shutil.rmtree(td, ignore_errors=True)
    return res


def main():
    decks = sorted(p.name for p in MATERIALS.glob("*.pptx"))
    results = {}
    for name, cfg in CONFIGS:
        print(">>>", name)
        results[name] = run_config(name, cfg, decks, None)
        # mirror config id
        results[name]["id"] = name
    out = Path(BASE) / "results"
    out.mkdir(exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(results, indent=2, default=str))
    # markdown table
    md = ["| Config | Retrieval correct (top-3) | Mean latency | Unanswerable→top1 |"]
    md.append("|---|---|---|---|")
    for name, r in results.items():
        md.append(f"| {name} | {r['retrieval_correct_top3']} | "
                  f"{r['mean_latency_ms']} ms | slide {r['unanswerable_top1_page']} |")
    print("\n".join(md))
    print("saved results/comparison.json")


if __name__ == "__main__":
    main()
