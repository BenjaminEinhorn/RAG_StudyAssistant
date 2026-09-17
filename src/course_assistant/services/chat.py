"""LLM chat provider (vision-capable) via an OpenAI-compatible endpoint.

Provides two implementations:
- :class:`OpenAIChat` — real class LLM (port 9001 default). Accepts text and
  optional image URLs/data-URIs so it can see slides/diagrams.
- :class:`LocalEchoChat` — offline stand-in for tests: returns a truthful
  acknowledgement that material was (or was not) found, without inventing
  answers. Used only in ``COURSE_BUILD_MODE=local``.
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path

from ..config.settings import Settings


class ChatProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str,
                 images: list[str] | None = None) -> str:
        raise NotImplementedError


class OpenAIChat(ChatProvider):
    def __init__(self, base_url: str, model: str, api_key: str, vision: bool = True):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=180)
        self.model = model
        self.vision = vision

    @staticmethod
    def _data_uri(path: str) -> str:
        b = Path(path).read_bytes()
        mime = "image/png"
        if str(path).lower().endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif str(path).lower().endswith(".gif"):
            mime = "image/gif"
        return f"data:{mime};base64," + base64.b64encode(b).decode()

    def complete(self, system: str, user: str,
                 images: list[str] | None = None) -> str:
        user_parts = [{"type": "text", "text": user}]
        if self.vision and images:
            user_parts.extend(
                {"type": "image_url", "image_url": {"url": self._data_uri(p)}}
                for p in images
            )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_parts},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class LocalEchoChat(ChatProvider):
    """Offline stand-in. Never fabricates knowledge; reports only whether the
    retrieved context actually matched the question tokens."""

    def complete(self, system: str, user: str,
                 images: list[str] | None = None) -> str:
        # crude signal: retreival<<context>> markers inserted by pipeline to
        # tell the fallback what evidence was found.
        if "<<no-evidence>>" in user:
            return ("I could not find enough supporting material in the loaded "
                    "documents to answer this. I won't guess. Try asking about "
                    "content in the loaded decks, or upload more materials.")
        return ("[local fallback answer] Answering from retrieved course material "
                "is disabled until a live class LLM endpoint is configured. "
                "Sources are listed alongside.")


def build_chat(settings: Settings) -> ChatProvider:
    if settings.build_mode == "local":
        return LocalEchoChat()
    return OpenAIChat(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
