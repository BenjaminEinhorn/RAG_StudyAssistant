"""Embedding providers.

Text and visual embeddings come from separate class services with different
wire formats (both verified with ``probe_endpoints.py``):

- :class:`ClassTextEmbeddings` — 9002, Cohere-style ``POST /v2/embed`` with
  ``input_type`` ``document`` (chunks) or ``query`` (questions).
- :class:`ClassVisualEmbeddings` — 9003, ``POST /v1/embeddings`` with a
  chat-style ``messages`` payload. The same model embeds slide images *and*
  text queries into one space, which is what visual retrieval relies on. The
  OpenAI ``input=[data-uri]`` form is accepted by the server but tokenises the
  base64 string as text, so it is never used.

Failures raise :class:`ServiceError`; nothing degrades to a fake vector.
A deterministic :class:`LocalHashEmbeddings` is used for offline tests only.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from ..config.settings import Settings
from .http import ServiceError, image_data_uri, post_json


class EmbeddingProvider(ABC):
    dim: int
    name: str = "unknown"   # recorded on the Chroma collection that it fills

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return a [n, dim] list of vectors for documents/chunks."""
        raise NotImplementedError

    @abstractmethod
    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        """Return vectors for images (file paths), used for the visual index."""
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Vector for a search query (asymmetric models embed queries differently)."""
        return self.embed_texts([text])[0]


def _check_dim(vectors: list[list[float]], dim: int, what: str) -> list[list[float]]:
    bad = {len(v) for v in vectors if len(v) != dim}
    if dim and bad:
        raise ServiceError(f"{what} returned dimension {sorted(bad)}, expected {dim} "
                           "(check COURSE_EMBED_DIM / COURSE_VISUAL_EMBED_DIM).")
    return vectors


class ClassTextEmbeddings(EmbeddingProvider):
    """Text embeddings via the class ``/v2/embed`` endpoint (port 9002)."""

    batch_size = 32

    def __init__(self, base_url: str, model: str, api_key: str, dim: int):
        self.url = base_url.rstrip("/") + "/v2/embed"
        self.model, self.api_key, self.dim = model, api_key, dim
        self.name = model

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            data = post_json(self.url, {
                "model": self.model, "texts": batch, "input_type": input_type,
                "embedding_types": ["float"]}, self.api_key)
            vecs = (data.get("embeddings") or {}).get("float")
            if not isinstance(vecs, list) or len(vecs) != len(batch):
                raise ServiceError(f"{self.url} returned an unexpected shape "
                                   f"(keys={sorted(data)}).")
            out.extend(vecs)
        return _check_dim(out, self.dim, "text embedder")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        raise ServiceError("The text embedder cannot embed images; use the visual embedder.")


class ClassVisualEmbeddings(EmbeddingProvider):
    """Image (and query-text) embeddings via the class 9003 endpoint."""

    workers = 8

    def __init__(self, base_url: str, model: str, api_key: str, dim: int):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model, self.api_key, self.dim = model, api_key, dim
        self.name = model

    def _embed_content(self, content: list[dict]) -> list[float]:
        data = post_json(self.url, {
            "model": self.model, "encoding_format": "float",
            "messages": [{"role": "user", "content": content}]}, self.api_key)
        try:
            vec = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError):
            raise ServiceError(f"{self.url} returned an unexpected shape "
                               f"(keys={sorted(data)}).") from None
        return _check_dim([vec], self.dim, "visual embedder")[0]

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        # one image per request (a messages payload is a single input)
        def one(path: str) -> list[float]:
            return self._embed_content(
                [{"type": "image_url", "image_url": {"url": image_data_uri(path)}}])

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(one, image_paths))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_content([{"type": "text", "text": t}]) for t in texts]


class LocalHashEmbeddings(EmbeddingProvider):
    """Deterministic, fast hash-based embeddings for offline testing only.

    Produces a fixed pseudo-random vector per input so retrieval/rerank
    ordering is reproducible even with no live service. NOT semantic.
    """

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.name = f"local-hash-{dim}"

    def _vec(self, s: str) -> list[float]:
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(h, "big") % (2**32)
        # deterministic pseudo-random vector via a simple LCG for stability
        x = seed
        out = []
        for _ in range(self.dim):
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            out.append((x / 0x7FFFFFFF) * 2.0 - 1.0)
        return out

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        return [self._vec(p) for p in image_paths]


def build_text_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.build_mode == "local":
        return LocalHashEmbeddings(settings.embed_dim)
    return ClassTextEmbeddings(settings.text_embed_base_url, settings.text_embed_model,
                               settings.api_key, settings.embed_dim)


def build_visual_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.build_mode == "local":
        return LocalHashEmbeddings(settings.visual_embed_dim)
    return ClassVisualEmbeddings(settings.visual_embed_base_url, settings.visual_embed_model,
                                 settings.api_key, settings.visual_embed_dim)
