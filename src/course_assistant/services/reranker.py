"""Reranker for combining keyword + vector candidates.

Two implementations:
- :class:`ClassReranker` — the class multimodal reranker (port 9004,
  ``POST /rerank``). Scores text chunks and slide images against the query.
  Errors raise :class:`ServiceError`: a silent fallback would make the
  rerank-on arm of the evaluation quietly measure a different algorithm.
- :class:`LexicalReranker` — term-overlap scoring used only in
  ``COURSE_BUILD_MODE=local`` so the pipeline is testable offline.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..config.settings import Settings
from .http import ServiceError, image_data_uri, post_json


class Reranker(ABC):
    @abstractmethod
    def rescore(self, query: str, documents: list[str]) -> list[float]:
        """Return one score per document (higher = more relevant)."""
        raise NotImplementedError

    def rescore_images(self, query: str, image_paths: list[str]) -> list[float] | None:
        """One score per image, or None if this reranker cannot see images."""
        return None


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.split(r"\W+", text.lower()) if t and len(t) > 1
        and t not in {"and", "the", "for", "with", "that", "this", "what", "which"}
    }


class LexicalReranker(Reranker):
    """Term-overlap + query-length normalization scoring."""

    def rescore(self, query: str, documents: list[str]) -> list[float]:
        qt = _tokens(query)
        if not qt:
            return [0.0] * len(documents)
        scores = []
        for d in documents:
            dt = _tokens(d)
            overlap = len(qt & dt)
            # normalize by query size
            scores.append(overlap / (len(qt) ** 0.6))
        return scores


class ClassReranker(Reranker):
    """The class multimodal reranker (Qwen3-VL-Reranker)."""

    def __init__(self, base_url: str, model: str, api_key: str):
        self.url = base_url.rstrip("/") + "/rerank"
        self.model, self.api_key = model, api_key

    def _scores(self, query: str, documents: list, n: int) -> list[float]:
        data = post_json(self.url, {"model": self.model, "query": query,
                                    "documents": documents}, self.api_key)
        results = data.get("results")
        if not isinstance(results, list) or len(results) != n:
            raise ServiceError(f"{self.url} returned {len(results or [])} results "
                               f"for {n} documents.")
        scores = [0.0] * n
        for item in results:
            scores[int(item["index"])] = float(item["relevance_score"])
        return scores

    def rescore(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return self._scores(query, documents, len(documents))

    def rescore_images(self, query: str, image_paths: list[str]) -> list[float]:
        if not image_paths:
            return []
        # each image must be its own document; a single {"content": [...]}
        # holding several images is scored as ONE document.
        docs = [{"content": [{"type": "image_url", "image_url": {"url": image_data_uri(p)}}]}
                for p in image_paths]
        return self._scores(query, docs, len(image_paths))


def build_reranker(settings: Settings) -> Reranker:
    if settings.build_mode == "local":
        return LexicalReranker()
    return ClassReranker(settings.rerank_base_url, settings.rerank_model,
                         settings.api_key)
