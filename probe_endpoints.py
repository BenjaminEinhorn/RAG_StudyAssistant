"""Probe the class model endpoints and print their real request/response contract.

For every call this prints the exact request (URL, method, JSON body with the
auth header redacted and base64 payloads truncated), the HTTP status, and the
raw *shape* of the response: keys, nesting, types, list/vector lengths.

The API key is read from COURSE_API_KEY (env or .env) and is never printed;
any occurrence of it in responses or errors is replaced with <REDACTED>.

    .venv/bin/python probe_endpoints.py            # all services
    .venv/bin/python probe_endpoints.py 9003 9004  # selected ports
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

HOST = os.environ.get("COURSE_PROBE_HOST", "http://dobolyi.com")
KEY = os.environ.get("COURSE_API_KEY", "")
EMBED_DIM = int(os.environ.get("COURSE_EMBED_DIM", "0") or 0)
TIMEOUT = 120


def scrub(s: str) -> str:
    return s.replace(KEY, "<REDACTED>") if KEY else s


def out(*parts) -> None:
    print(scrub(" ".join(str(p) for p in parts)), flush=True)


# ------------------------------------------------------------------ fixtures
def _png(text: str, bg: tuple[int, int, int]) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 200), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 300, 180], outline=(0, 0, 0), width=4)
    d.text((40, 90), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=400, height=250)
    page.insert_text((30, 60), "Hybrid RAG combines keyword search", fontsize=14)
    page.insert_text((30, 90), "with vector search, then reranks.", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_page_png(pdf: bytes) -> bytes:
    import fitz

    doc = fitz.open(stream=pdf, filetype="pdf")
    png = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
    doc.close()
    return png


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


IMG_A = data_uri(_png("RED SLIDE: pipeline diagram", (230, 80, 80)))
IMG_B = data_uri(_png("BLUE SLIDE: meme about prod", (80, 120, 230)))
PDF = _pdf()
PDF_PAGE = data_uri(_pdf_page_png(PDF))


# ------------------------------------------------------------------ printing
def _trunc(obj):
    """Copy of a request body with long strings (base64) shortened."""
    if isinstance(obj, dict):
        return {k: _trunc(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trunc(v) for v in obj]
    if isinstance(obj, str) and len(obj) > 120:
        return f"{obj[:60]}...<{len(obj)} chars>"
    return obj


def shape(obj, indent: int = 2, key: str = "$") -> list[str]:
    pad = " " * indent
    if isinstance(obj, dict):
        lines = [f"{pad}{key}: dict[{len(obj)}]"]
        for k, v in obj.items():
            lines += shape(v, indent + 2, repr(k))
        return lines
    if isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            head = ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in obj[:3])
            return [f"{pad}{key}: list[number] len={len(obj)} head=[{head}]"]
        lines = [f"{pad}{key}: list len={len(obj)}"]
        for i, v in enumerate(obj[:3]):
            lines += shape(v, indent + 2, f"[{i}]")
        if len(obj) > 3:
            lines.append(f"{pad}  ... {len(obj) - 3} more")
        return lines
    if isinstance(obj, str):
        s = obj if len(obj) <= 160 else obj[:160] + f"...<{len(obj)} chars>"
        return [f"{pad}{key}: str {json.dumps(s)}"]
    return [f"{pad}{key}: {type(obj).__name__} {obj!r}"]


def vectors(obj) -> list[list[float]]:
    """Collect every numeric list found anywhere in a response."""
    found = []
    if isinstance(obj, dict):
        for v in obj.values():
            found += vectors(v)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, float) for x in obj):
            found.append(obj)
        else:
            for v in obj:
                found += vectors(v)
    return found


def cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


RESULTS: dict[str, dict] = {}


def call(label: str, method: str, url: str, body: dict | None = None):
    out(f"\n--- {label}")
    out(f">>> {method} {url}")
    out(">>> headers: {'Authorization': 'Bearer <REDACTED>', 'Content-Type': 'application/json'}")
    if body is not None:
        out(">>> body:", json.dumps(_trunc(body), indent=2))
    t0 = time.perf_counter()
    try:
        r = httpx.request(method, url, json=body, timeout=TIMEOUT,
                          headers={"Authorization": f"Bearer {KEY}"})
    except httpx.HTTPError as e:
        out(f"<<< UNREACHABLE: {type(e).__name__}: {e}")
        RESULTS[label] = {"status": None, "error": type(e).__name__}
        return None
    ms = (time.perf_counter() - t0) * 1000
    out(f"<<< HTTP {r.status_code}  ({ms:.0f} ms)  content-type={r.headers.get('content-type')}")
    try:
        data = r.json()
    except ValueError:
        out("<<< non-JSON body:", r.text[:500])
        RESULTS[label] = {"status": r.status_code}
        return None
    out("<<< shape:")
    for line in shape(data):
        out(line)
    RESULTS[label] = {"status": r.status_code, "ms": round(ms)}
    return data if r.is_success else None


def models(port: int, prefix: str = "/v1") -> list[str]:
    data = call(f":{port} GET {prefix}/models", "GET", f"{HOST}:{port}{prefix}/models")
    return [m.get("id") for m in (data or {}).get("data", [])]


# ------------------------------------------------------------------ services
def probe_9001():
    port = 9001
    names = models(port)
    model = names[0] if names else "unknown"
    base = f"{HOST}:{port}/v1/chat/completions"
    call(":9001 chat text-only", "POST", base, {
        "model": model, "max_tokens": 60, "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}]})
    call(":9001 chat text-only, thinking disabled", "POST", base, {
        "model": model, "max_tokens": 60, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}]})
    call(":9001 chat with image part (thinking disabled)", "POST", base, {
        "model": model, "max_tokens": 80, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What colour is this image and what text is on it?"},
            {"type": "image_url", "image_url": {"url": IMG_A}}]}]})
    data = call(":9001 chat response_format=json_schema", "POST", base, {
        "model": model, "max_tokens": 200, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Give the capital of France."}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "answer", "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["answer", "sources"],
                "properties": {"answer": {"type": "string"},
                               "sources": {"type": "array", "items": {"type": "integer"}}}}}}})
    if data:
        content = data["choices"][0]["message"].get("content") or ""
        try:
            parsed = json.loads(content)
            out(f"    json_schema honoured: parsed keys={sorted(parsed)}")
        except ValueError:
            out("    json_schema NOT honoured: content is not valid JSON")
    call(":9001 chat response_format=json_object", "POST", base, {
        "model": model, "max_tokens": 200, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content":
                      "Return a JSON object with key 'capital' for France."}],
        "response_format": {"type": "json_object"}})


def probe_9002():
    port = 9002
    names = models(port)
    model = names[0] if names else "unknown"
    texts = ["Hybrid RAG combines keyword and vector search.",
             "The quarterly revenue grew by ten percent."]
    for label, url, body in [
        (":9002 POST /v2/embed (contract endpoint)", f"{HOST}:{port}/v2/embed",
         {"model": model, "texts": texts, "input_type": "search_document",
          "embedding_types": ["float"]}),
        (":9002 POST /v2/embed input_type=document", f"{HOST}:{port}/v2/embed",
         {"model": model, "texts": texts, "input_type": "document",
          "embedding_types": ["float"]}),
        (":9002 POST /v2/embed input_type=query", f"{HOST}:{port}/v2/embed",
         {"model": model, "texts": [texts[0]], "input_type": "query",
          "embedding_types": ["float"]}),
        (":9002 POST /v1/embeddings (OpenAI shape)", f"{HOST}:{port}/v1/embeddings",
         {"model": model, "input": texts}),
    ]:
        data = call(label, "POST", url, body)
        vecs = vectors(data) if data else []
        if vecs:
            dims = sorted({len(v) for v in vecs})
            out(f"    vectors={len(vecs)} dims={dims} COURSE_EMBED_DIM={EMBED_DIM or 'unset'} "
                f"match={EMBED_DIM in dims if EMBED_DIM else 'n/a'}")
            if len(vecs) >= 2:
                out(f"    cosine(text0, text1)={cosine(vecs[0], vecs[1]):.4f}")


def probe_9003():
    port = 9003
    names = models(port)
    model = names[0] if names else "unknown"
    url = f"{HOST}:{port}/v1/embeddings"

    def img_msgs(uri):
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}}]}]

    got: dict[str, list[float]] = {}
    for tag, body in [
        ("text input=str", {"model": model, "input": "a pipeline diagram",
                            "encoding_format": "float"}),
        ("text messages", {"model": model, "encoding_format": "float", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a pipeline diagram"}]}]}),
        ("image A messages", {"model": model, "encoding_format": "float",
                              "messages": img_msgs(IMG_A)}),
        ("image B messages", {"model": model, "encoding_format": "float",
                              "messages": img_msgs(IMG_B)}),
        ("image input=[data-uri]", {"model": model, "input": [IMG_A],
                                    "encoding_format": "float"}),
    ]:
        data = call(f":9003 {tag}", "POST", url, body)
        vecs = vectors(data) if data else []
        if vecs:
            got[tag] = vecs[0]
            out(f"    dim={len(vecs[0])}")
    if "image A messages" in got and "image B messages" in got:
        out(f"    cosine(imageA, imageB)={cosine(got['image A messages'], got['image B messages']):.4f}"
            "  (1.0000 would mean the image is ignored)")
    if "image input=[data-uri]" in got and "text input=str" in got:
        out(f"    cosine(input=[data-uri], text)={cosine(got['image input=[data-uri]'], got['text input=str']):.4f}")
    for t in ("text input=str", "text messages"):
        if t in got and "image A messages" in got:
            same_dim = len(got[t]) == len(got["image A messages"])
            out(f"    TEXT CAN SHARE THE VISUAL SPACE via [{t}]: same dim={same_dim}, "
                f"cosine(text, imageA)={cosine(got[t], got['image A messages']):.4f}, "
                f"cosine(text, imageB)={cosine(got[t], got['image B messages']):.4f}")


def probe_9004():
    port = 9004
    names = models(port) or models(port, "")
    model = names[0] if names else "unknown"
    docs = ["Hybrid RAG merges BM25 keyword results with dense vector results.",
            "Gradio serves apps on port 7860 by default.",
            "The cafeteria menu changes every Tuesday."]
    body = {"model": model, "query": "How does hybrid retrieval work?", "documents": docs}
    for path in ("/rerank", "/v1/rerank"):
        call(f":9004 POST {path} (text docs)", "POST", f"{HOST}:{port}{path}", body)
    # A single {"content": [...]} is ONE multimodal document; each image must be
    # its own document to be scored separately.
    call(":9004 POST /rerank (one doc holding two images)", "POST", f"{HOST}:{port}/rerank", {
        "model": model, "query": "a red pipeline diagram",
        "documents": {"content": [{"type": "image_url", "image_url": {"url": IMG_A}},
                                  {"type": "image_url", "image_url": {"url": IMG_B}}]}})
    call(":9004 POST /rerank (list of image docs)", "POST", f"{HOST}:{port}/rerank", {
        "model": model, "query": "a red pipeline diagram",
        "documents": [{"content": [{"type": "image_url", "image_url": {"url": IMG_A}}]},
                      {"content": [{"type": "image_url", "image_url": {"url": IMG_B}}]}]})


def probe_9005():
    port = 9005
    names = models(port)
    model = names[0] if names else "unknown"
    out(f"\n    (small PDF generated in-memory: {len(PDF)} bytes; page 1 rendered to PNG "
        "because this service is a chat/vision OCR model)")
    call(":9005 chat OCR of PDF page", "POST", f"{HOST}:{port}/v1/chat/completions", {
        "model": model, "max_tokens": 300, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": PDF_PAGE}},
            {"type": "text", "text": "Extract the text content from this image."}]}]})


PROBES = {9001: probe_9001, 9002: probe_9002, 9003: probe_9003,
          9004: probe_9004, 9005: probe_9005}


def main() -> None:
    if not KEY:
        sys.exit("COURSE_API_KEY is not set (put it in .env). Nothing probed.")
    ports = [int(a) for a in sys.argv[1:]] or list(PROBES)
    for p in ports:
        out(f"\n==================== :{p}")
        PROBES[p]()
    out("\n==================== summary (label -> HTTP status)")
    for label, r in RESULTS.items():
        out(f"  {r.get('status')!s:>5}  {label}")


if __name__ == "__main__":
    main()
