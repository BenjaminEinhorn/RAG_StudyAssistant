"""Unit tests for the class-service clients (no network: transport is faked).

They pin down the contract verified by probe_endpoints.py and, above all, that
failures are loud: nothing may silently degrade to a fake vector or score.
"""
from types import SimpleNamespace

import pytest

from course_assistant.config.settings import redact
from course_assistant.services import embeddings as emb_mod
from course_assistant.services import reranker as rr_mod
from course_assistant.services.chat import OpenAIChat
from course_assistant.services.http import ServiceError


@pytest.fixture()
def png(tmp_path):
    from PIL import Image

    p = tmp_path / "slide.png"
    Image.new("RGB", (2000, 1000), (200, 50, 50)).save(p)
    return str(p)


def test_visual_embedder_sends_image_as_messages(monkeypatch, png):
    sent = []

    def fake_post(url, body, api_key, timeout=120):
        sent.append(body)
        return {"data": [{"embedding": [0.1] * 4}]}

    monkeypatch.setattr(emb_mod, "post_json", fake_post)
    e = emb_mod.ClassVisualEmbeddings("http://x:9003/v1", "m", "k", dim=4)
    assert e.embed_images([png]) == [[0.1] * 4]
    body = sent[0]
    assert "input" not in body, "input=[data-uri] embeds the base64 string as text"
    part = body["messages"][0]["content"][0]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_visual_embedder_failure_is_loud(monkeypatch, png):
    def boom(*a, **k):
        raise ServiceError("HTTP 400")

    monkeypatch.setattr(emb_mod, "post_json", boom)
    e = emb_mod.ClassVisualEmbeddings("http://x:9003/v1", "m", "k", dim=4)
    with pytest.raises(ServiceError):
        e.embed_images([png])


def test_visual_embedder_rejects_wrong_dimension(monkeypatch, png):
    monkeypatch.setattr(emb_mod, "post_json",
                        lambda *a, **k: {"data": [{"embedding": [0.1] * 3}]})
    e = emb_mod.ClassVisualEmbeddings("http://x:9003/v1", "m", "k", dim=4)
    with pytest.raises(ServiceError, match="dimension"):
        e.embed_images([png])


def test_text_embedder_uses_v2_embed_with_input_types(monkeypatch):
    sent = []

    def fake_post(url, body, api_key, timeout=120):
        sent.append((url, body))
        return {"embeddings": {"float": [[0.0, 1.0] for _ in body["texts"]]}}

    monkeypatch.setattr(emb_mod, "post_json", fake_post)
    e = emb_mod.ClassTextEmbeddings("http://x:9002", "m", "k", dim=2)
    e.embed_texts(["a", "b"])
    e.embed_query("q")
    assert sent[0][0] == "http://x:9002/v2/embed"
    assert sent[0][1]["input_type"] == "document"
    assert sent[1][1]["input_type"] == "query"


def test_text_embedder_rejects_unexpected_shape(monkeypatch):
    monkeypatch.setattr(emb_mod, "post_json", lambda *a, **k: {"data": []})
    e = emb_mod.ClassTextEmbeddings("http://x:9002", "m", "k", dim=2)
    with pytest.raises(ServiceError):
        e.embed_texts(["a"])


def test_reranker_maps_scores_by_index_and_images_are_separate_docs(monkeypatch, png):
    sent = []

    def fake_post(url, body, api_key, timeout=120):
        sent.append(body)
        n = len(body["documents"])
        return {"results": [{"index": i, "relevance_score": 1.0 - i / 10}
                            for i in reversed(range(n))]}

    monkeypatch.setattr(rr_mod, "post_json", fake_post)
    r = rr_mod.ClassReranker("http://x:9004", "m", "k")
    assert r.rescore("q", ["a", "b"]) == [1.0, 0.9]
    r.rescore_images("q", [png, png])
    docs = sent[1]["documents"]
    assert len(docs) == 2 and all(d["content"][0]["type"] == "image_url" for d in docs)


def test_reranker_failure_is_loud_not_lexical(monkeypatch):
    def boom(*a, **k):
        raise ServiceError("HTTP 500")

    monkeypatch.setattr(rr_mod, "post_json", boom)
    with pytest.raises(ServiceError):
        rr_mod.ClassReranker("http://x:9004", "m", "k").rescore("q", ["a"])


class _FakeCompletions:
    def __init__(self, content):
        self.content, self.kwargs = content, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def _chat_with(content):
    chat = OpenAIChat("http://x:9001/v1", "m", "k")
    fake = _FakeCompletions(content)
    chat._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return chat, fake


def test_chat_disables_thinking_and_parses_json():
    chat, fake = _chat_with('{"found": true, "answer": "x", "citations": []}')
    out = chat.complete_json("s", "u", {"type": "object"})
    assert out["answer"] == "x"
    assert fake.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert fake.kwargs["response_format"]["type"] == "json_schema"


def test_chat_empty_content_raises():
    chat, _ = _chat_with(None)  # what the reasoning model returns out of budget
    with pytest.raises(ServiceError, match="no content"):
        chat.complete("s", "u")


def test_redact_removes_key():
    assert redact("Bearer sekret-123 failed", "sekret-123") == "Bearer <REDACTED> failed"
