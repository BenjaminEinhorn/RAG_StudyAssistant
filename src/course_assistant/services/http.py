"""Shared HTTP plumbing for the class services.

Every live call goes through :func:`post_json`, which raises
:class:`ServiceError` with the API key redacted instead of degrading silently.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx

from ..config.settings import redact

# Slides render at 144 dpi (~1920 px wide); 1280 px keeps text legible for the
# vision models at roughly half the image tokens.
MAX_IMAGE_SIDE = 1280


class ServiceError(RuntimeError):
    """A class service rejected a request or returned an unexpected shape."""


def post_json(url: str, body: dict, api_key: str, timeout: float = 120) -> dict:
    if not api_key:
        raise ServiceError("COURSE_API_KEY is not set; cannot call " + url)
    try:
        r = httpx.post(url, json=body, timeout=timeout,
                       headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as e:
        raise ServiceError(redact(f"{url} unreachable: {type(e).__name__}: {e}",
                                  api_key)) from None
    if not r.is_success:
        raise ServiceError(redact(f"{url} returned HTTP {r.status_code}: {r.text[:300]}",
                                  api_key))
    try:
        return r.json()
    except ValueError:
        raise ServiceError(f"{url} returned non-JSON: {r.text[:200]!r}") from None


def image_data_uri(path: str, max_side: int = MAX_IMAGE_SIDE) -> str:
    """PNG data URI of an image file, downscaled so its longest side <= max_side."""
    from PIL import Image

    with Image.open(Path(path)) as img:
        img = img.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
