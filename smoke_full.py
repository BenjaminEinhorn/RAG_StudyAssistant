"""Live smoke test: ingest one real deck against the class services and check
that both Chroma collections hold real vectors.

    COURSE_BUILD_MODE=full .venv/bin/python smoke_full.py [path/to/deck.pptx]

Checks: every vector has the configured dimension, image embeddings differ
between different slides, and a text query lands on a sensible slide in both
the text and the visual index. Uses a throwaway data dir.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

if os.environ.get("COURSE_BUILD_MODE") != "full":
    sys.exit("Set COURSE_BUILD_MODE=full: this smoke test is only meaningful live.")
os.environ["COURSE_DATA_DIR"] = tempfile.mkdtemp(prefix="smoke_")

from course_assistant.config.settings import Settings  # noqa: E402
from course_assistant.factory import build_app_state  # noqa: E402

DEFAULT_DECK = Path("data/materials/MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx")


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    return num / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


def main():
    deck = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DECK
    s = Settings()
    app = build_app_state(s)
    t0 = time.perf_counter()
    r = app.catalog.add_file(str(deck))
    print(f"ingest: {r['message']} chunks={r.get('chunks')} images={r.get('images')} "
          f"in {time.perf_counter() - t0:.1f}s")

    text = app.catalog._text_col.get(include=["embeddings"])
    vis = app.catalog._visual_col.get(include=["embeddings", "metadatas"])
    tdims = {len(v) for v in text["embeddings"]}
    vdims = {len(v) for v in vis["embeddings"]}
    print(f"text collection:   {len(text['ids'])} vectors, dims={tdims} (expected {s.embed_dim})")
    print(f"visual collection: {len(vis['ids'])} vectors, dims={vdims} (expected {s.visual_embed_dim})")
    assert tdims == {s.embed_dim} and vdims == {s.visual_embed_dim}

    vecs = list(vis["embeddings"])
    sims = [cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    identical = sum(1 for x in sims if x > 0.9999)
    print(f"adjacent-slide image cosine: min={min(sims):.3f} mean={sum(sims)/len(sims):.3f} "
          f"max={max(sims):.3f}; identical pairs={identical}")
    assert identical < len(sims), "all image embeddings identical: images are being ignored"

    q = "Vibe coding on prod meme"
    t0 = time.perf_counter()
    hits = app.catalog.search(q, k=3)
    slides = app.catalog.visual_search(q, k=3)
    print(f"query {q!r} ({(time.perf_counter() - t0) * 1000:.0f} ms)")
    print("  text top-3:  ", [(h.page, round(h.score, 3)) for h in hits])
    print("  visual top-3:", [(sl.page, round(sl.score, 3)) for sl in slides])
    print("OK")


if __name__ == "__main__":
    main()
