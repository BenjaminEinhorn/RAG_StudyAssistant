"""LLM chat provider (vision-capable) via an OpenAI-compatible endpoint.

Provides two implementations:
- :class:`OpenAIChat` — real class LLM (port 9001). Accepts text and optional
  slide images, and can return schema-constrained JSON.
- :class:`LocalEchoChat` — offline stand-in for tests: returns a truthful
  acknowledgement that material was (or was not) found, without inventing
  answers. Used only in ``COURSE_BUILD_MODE=local``.

The 9001 model (Qwen3.6) is a reasoning model: by default it spends its token
budget in ``message.reasoning`` and returns ``content=None``. Every request
therefore disables thinking via ``chat_template_kwargs`` (verified with
``probe_endpoints.py``), and an empty completion raises instead of producing a
blank answer.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from ..config.settings import Settings, redact
from .http import ServiceError, image_data_uri


class ChatProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str,
                 images: list[str] | None = None) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema: dict,
                      images: list[str] | None = None) -> dict:
        """Return a JSON object constrained to ``schema``."""
        raise NotImplementedError


class OpenAIChat(ChatProvider):
    def __init__(self, base_url: str, model: str, api_key: str, vision: bool = True,
                 max_tokens: int = 1500):
        from openai import OpenAI

        if not api_key:
            raise ServiceError("COURSE_API_KEY is not set; the class LLM needs it.")
        self._client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=180)
        self._api_key = api_key
        self.model = model
        self.vision = vision
        self.max_tokens = max_tokens

    def _create(self, system: str, user: str, images: list[str] | None, **extra) -> str:
        user_parts = [{"type": "text", "text": user}]
        if self.vision and images:
            user_parts.extend(
                {"type": "image_url", "image_url": {"url": image_data_uri(p)}}
                for p in images
            )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_parts},
                ],
                temperature=0.2,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                **extra,
            )
        except Exception as e:  # openai.APIError and transport errors
            raise ServiceError(redact(f"class LLM call failed: {type(e).__name__}: {e}",
                                      self._api_key)) from None
        choice = resp.choices[0]
        content = choice.message.content
        if not content:
            raise ServiceError(
                f"class LLM returned no content (finish_reason={choice.finish_reason}).")
        return content

    def complete(self, system: str, user: str,
                 images: list[str] | None = None) -> str:
        return self._create(system, user, images)

    def complete_json(self, system: str, user: str, schema: dict,
                      images: list[str] | None = None) -> dict:
        raw = self._create(system, user, images, response_format={
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema}})
        try:
            data = json.loads(raw)
        except ValueError:
            raise ServiceError(f"class LLM returned invalid JSON: {raw[:200]!r}") from None
        if not isinstance(data, dict):
            raise ServiceError("class LLM returned JSON that is not an object.")
        return data


class LocalEchoChat(ChatProvider):
    """Offline stand-in. Never fabricates knowledge; reports only whether the
    retrieved context actually matched the question tokens."""

    NO_EVIDENCE = ("I could not find enough supporting material in the loaded "
                   "documents to answer this. I won't guess. Try asking about "
                   "content in the loaded decks, or upload more materials.")
    DISABLED = ("[local fallback answer] Answering from retrieved course material "
                "is disabled until a live class LLM endpoint is configured. "
                "Sources are listed alongside.")

    def complete(self, system: str, user: str,
                 images: list[str] | None = None) -> str:
        # crude signal: the pipeline inserts a <<no-evidence>> marker to tell
        # the fallback that retrieval found nothing.
        if "<<no-evidence>>" in user:
            return self.NO_EVIDENCE
        return self.DISABLED

    def complete_json(self, system: str, user: str, schema: dict,
                      images: list[str] | None = None) -> dict:
        if "<<no-evidence>>" in user:
            return {"found": False, "answer": self.NO_EVIDENCE, "citations": []}
        # Cite evidence [1] with its first words, copied verbatim from the prompt.
        m = re.search(r"^\[1\] \([^\n]*\)\n([^\n]+)", user, re.MULTILINE)
        excerpt = " ".join(m.group(1).split()[:8]) if m else ""
        return {"found": True, "answer": self.DISABLED,
                "citations": [{"source": 1, "excerpt": excerpt}]}


def build_chat(settings: Settings) -> ChatProvider:
    if settings.build_mode == "local":
        return LocalEchoChat()
    return OpenAIChat(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
