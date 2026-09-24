import os
import shutil
import sys
from pathlib import Path

# Run in deterministic local mode with an isolated data dir before importing app
_TEST_DIR = Path(__file__).parent / "_tmp_data"
os.environ["COURSE_BUILD_MODE"] = "local"
os.environ["COURSE_DATA_DIR"] = str(_TEST_DIR)
os.environ["COURSE_ENV_FILE"] = str(_TEST_DIR / "nope.env")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _wipe():
    if _TEST_DIR.exists():
        shutil.rmtree(_TEST_DIR)
    _TEST_DIR.mkdir(parents=True, exist_ok=True)


import pytest  # noqa: E402

from course_assistant.config.settings import Settings  # noqa: E402
from course_assistant.factory import build_app_state  # noqa: E402

REAL_DECKS_DIR = Path(__file__).resolve().parents[1] / "data" / "materials"
REAL_DECK_W2 = REAL_DECKS_DIR / "MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx"
REAL_DECK_W5 = REAL_DECKS_DIR / "MBAX 6418 - Week 5 - Context Engineering and RAG v1.pptx"

# Tests needing the bundled course decks skip on a fresh clone unless the decks
# were placed in {repo}/data/materials (see README "Sample data").
# Slide images come from LibreOffice (PPTX -> PDF); without it decks parse as
# text only, so tests that need slide images skip instead of failing.
needs_soffice = pytest.mark.skipif(
    shutil.which("soffice") is None,
    reason="LibreOffice (soffice) not installed; slide images cannot be rendered")

needs_decks = pytest.mark.skipif(
    not (REAL_DECK_W2.exists() and REAL_DECK_W5.exists()),
    reason="course decks not present in data/materials; place them to run deck tests")


@pytest.fixture()
def app_state():
    # Unique data dir per test avoids chroma persistent-client corruption
    # caused by deleting a live client's directory between tests.
    import tempfile
    from pathlib import Path as _P
    td = _P(tempfile.mkdtemp(prefix="mbax_"))
    os.environ["COURSE_DATA_DIR"] = str(td)
    settings = Settings()          # reads COURSE_DATA_DIR
    app = build_app_state(settings)
    yield app
    try:
        app.catalog._chroma.clear_system_cache()
    except Exception:
        pass
    shutil.rmtree(td, ignore_errors=True)


@pytest.fixture()
def small_doc(tmp_path) -> Path:
    p = tmp_path / "notes.txt"
    p.write_text(
        "Hybrid RAG combines keyword search and vector search. "
        "Embeddings turn text into numeric vectors. "
        "Reranking reorders candidate chunks by relevance. "
        "The Vibe Coding on Prod meme warns against casual coding in production."
    )
    return p
