"""Embedding providers.

Text and visual embeddings are produced by class services (OpenAI-compatible
/v1 embeddings endpoints). A deterministic :class:`LocalHashEmbeddings` fallback
is used for offline unit/e2e testing until real keys are configured; it is not
(for the comparison) a production embedding — see README.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from ..config.settings import Settings


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return a [n, dim] list of vectors for the given texts."""
        raise NotImplementedError

    @abstractmethod
    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        """Return vectors for images (file paths), used for the visual index."""
        raise NotImplementedError


def _client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=60)


class OpenAIEmbeddings(EmbeddingProvider):
    """Text embeddings via a class OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str, dim: int = 0):
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self._client = _client(base_url, api_key)
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vectors = [d.embedding for d in resp.data]
        # order safety, though the API returns in request order
        vectors = [v for _, v in sorted(zip([d.index for d in resp.data], vectors))]
        if self.dim:
            assert all(len(v) == self.dim for v in vectors), "embedding dim mismatch"
        return vectors

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        # OpenAI's text-embedding API can also take images as data URIs on some
        # class endpoints. Try data-URI first; fall back to a per-file prompt
        # wrapper if the endpoint rejects the image payload.
        import base64

        def uri(p: str) -> str:
            b = Path(p).read_bytes()
            return "data:image/png;base64," + base64.b64encode(b).decode()

        try:
            resp = self._client.embeddings.create(
                model=self.model, input=[uri(p) for p in image_paths], input_type=None
            )
            return [d.embedding for d in resp.data]
        except Exception:
            # Some class endpoints expect a text prompt describing the image.
            return self.embed_texts([f"[image at {p}]" for p in image_paths])


class LocalHashEmbeddings(EmbeddingProvider):
    """Deterministic, fast hash-based embeddings for offline testing only.

    Produces a fixed pseudo-random vector per input so retrieval/rerank
    ordering is reproducible even with no live service. NOT semantic.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim

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
    return OpenAIEmbeddings(settings.text_embed_base_url, settings.text_embed_model,
                            settings.llm_api_key, settings.embed_dim)


def build_visual_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.build_mode == "local":
        return LocalHashEmbeddings(settings.embed_dim)
    return OpenAIEmbeddings(settings.visual_embed_base_url, settings.visual_embed_model,
                            settings.llm_api_key, 0)
