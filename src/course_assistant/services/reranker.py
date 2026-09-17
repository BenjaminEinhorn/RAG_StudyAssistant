"""Reranker for combining keyword + vector candidates.

Two implementations:
- :class:`OpenAIReranker` — hits a class /v1/rerank style endpoint when one is
  configured; falls back to a robust lexical boost when the endpoint isn't a
  rerank service.
- :class:`LexicalReranker` — a lightweight quadratic/lexical rerank used in
  ``COURSE_BUILD_MODE=local`` and as the fallback so the answer quality
  comparison (rerank on vs. off) is testable offline.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..config.settings import Settings


class Reranker(ABC):
    @abstractmethod
    def rescore(self, query: str, documents: list[str]) -> list[float]:
        """Return one score per document (higher = more relevant)."""
        raise NotImplementedError


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
            # normalize by query size; slight boost for exact query phrases
            scores.append(overlap / (len(qt) ** 0.6))
        return scores


class OpenAIReranker(Reranker):
    """Try a dedicated rerank endpoint; fall back to LexicalReranker."""

    def __init__(self, base_url: str, model: str, api_key: str):
        import httpx

        self._http = httpx.Client(timeout=60)
        self.url = base_url.rstrip("/") + "/rerank"
        self.model, self.api_key, self.headers = (
            model, api_key, {"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
        )
        self._fallback = LexicalReranker()

    def rescore(self, query: str, documents: list[str]) -> list[float]:
        try:
            r = self._http.post(
                self.url,
                headers=self.headers,
                json={"model": self.model, "query": query, "documents": documents},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            scores = [0.0] * len(documents)
            for item in data.get("results", []):
                idx, score = item.get("index"), item.get("relevance_score", 0.0)
                if idx is not None and 0 <= int(idx) < len(scores):
                    scores[int(idx)] = float(score)
            return scores
        except Exception:
            # endpoint may be chat-only; degrade gracefully
            return self._fallback.rescore(query, documents)


def build_reranker(settings: Settings) -> Reranker:
    if settings.build_mode == "local":
        return LexicalReranker()
    return OpenAIReranker(settings.rerank_base_url, settings.rerank_model,
                          settings.llm_api_key)
