"""Application settings loaded from environment / .env.

Secrets are read from environment variables (and optionally a local
``.env`` file that is gitignored). Only dummy values are ever committed
(see ``.env.example``).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = lambda *a, **k: None  # type: ignore[assignment]

# Load a local .env if present (secrets stay out of git).
_ENV_FILE = Path(os.environ.get("COURSE_ENV_FILE", ".env")).resolve()
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def _get(key: str, default: str = "") -> str:
    """Read env, falling back to COURSE_ prefixed value, then default."""
    return os.environ.get(key) or os.environ.get("COURSE_" + key) or default


def redact(text: str, *secrets: str) -> str:
    """Strip secrets (the class API key by default) from text shown to users/logs."""
    secrets = secrets or (os.environ.get("COURSE_API_KEY", ""),)
    for s in secrets:
        if s:
            text = text.replace(s, "<REDACTED>")
    return text


class Settings:
    def __init__(self) -> None:
        # No usable default: full mode fails loudly if the key is missing.
        self.api_key = _get("COURSE_API_KEY", "")

        # Defaults below are the live contract verified by probe_endpoints.py.
        # Answering / vision LLM (a reasoning model; thinking disabled per call)
        self.llm_base_url = _get("COURSE_LLM_BASE_URL", "http://dobolyi.com:9001/v1")
        self.llm_model = _get("COURSE_LLM_MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
        self.llm_api_key = self.api_key

        # Text embeddings: Cohere-style POST {base}/v2/embed
        self.text_embed_base_url = _get("COURSE_TEXT_EMBED_BASE_URL", "http://dobolyi.com:9002")
        self.text_embed_model = _get("COURSE_TEXT_EMBED_MODEL", "nvidia/Nemotron-3-Embed-1B-BF16")

        # Visual embeddings: POST {base}/embeddings with chat-style messages
        self.visual_embed_base_url = _get("COURSE_VISUAL_EMBED_BASE_URL", "http://dobolyi.com:9003/v1")
        self.visual_embed_model = _get("COURSE_VISUAL_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")

        # Reranker: POST {base}/rerank
        self.rerank_base_url = _get("COURSE_RERANK_BASE_URL", "http://dobolyi.com:9004")
        self.rerank_model = _get("COURSE_RERANK_MODEL", "Qwen/Qwen3-VL-Reranker-2B")

        # Document OCR (chat completions); the default parser is PyMuPDF
        self.parse_base_url = _get("COURSE_PARSE_BASE_URL", "http://dobolyi.com:9005/v1")
        self.parse_model = _get("COURSE_PARSE_MODEL", "dots.mocr")

        # Embedding dimensions (text and visual spaces are separate collections)
        self.embed_dim = int(_get("COURSE_EMBED_DIM", "2048"))
        self.visual_embed_dim = int(_get("COURSE_VISUAL_EMBED_DIM", "2048"))

        # Text chunking
        self.chunk_size = int(_get("CHUNK_SIZE", "1200"))
        self.chunk_overlap = int(_get("CHUNK_OVERLAP", "150"))

        base_dir = Path(os.environ.get("COURSE_DATA_DIR", "data")).resolve()
        self.data_dir = base_dir
        self.materials_dir = base_dir / "materials"      # original uploaded files
        self.images_dir = base_dir / "images"            # rendered page/slide images
        self.chroma_dir = base_dir / "chroma"            # vector DB persistence
        self.meta_db = base_dir / "index.db"             # SQLite doc/chunk metadata
        self.text_meta = base_dir / "text_chunks.jsonl"
        self.image_meta = base_dir / "images.jsonl"
        for d in (self.materials_dir, self.images_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.build_mode = os.environ.get("COURSE_BUILD_MODE", "local")

    @property
    def service_env(self) -> dict:
        """Service config for OpenAI clients (no secrets printed)."""
        return {
            "llm": {"base_url": self.llm_base_url, "model": self.llm_model},
            "text_embed": {"base_url": self.text_embed_base_url, "model": self.text_embed_model},
            "visual_embed": {"base_url": self.visual_embed_base_url, "model": self.visual_embed_model},
            "rerank": {"base_url": self.rerank_base_url, "model": self.rerank_model},
        }


settings = Settings()
