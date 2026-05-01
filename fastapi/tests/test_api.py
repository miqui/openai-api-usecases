"""FastAPI endpoint tests — all OpenAI calls mocked."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings() -> MagicMock:
    s = MagicMock()
    s.openai_api_key = "sk-test"
    s.chat_model = "gpt-4o"
    s.embedding_model = "text-embedding-3-small"
    return s


def _make_chat_completion(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.total_tokens = 10
    comp = MagicMock()
    comp.choices = [choice]
    comp.usage = usage
    return comp


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    data = []
    for i, vec in enumerate(vectors):
        item = MagicMock()
        item.embedding = vec
        item.index = i
        data.append(item)
    usage = MagicMock()
    usage.total_tokens = 5
    resp = MagicMock()
    resp.data = data
    resp.usage = usage
    return resp


client = TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /chat/complete
# ---------------------------------------------------------------------------


@patch("app.routers.chat.get_settings", return_value=_fake_settings())
@patch("app.routers.chat.OpenAI")
def test_chat_complete(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_completion("Paris")

    resp = client.post("/chat/complete", json={"prompt": "Capital of France?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Paris"
    assert data["model"] == "gpt-4o"


@patch("app.routers.chat.get_settings", return_value=_fake_settings())
@patch("app.routers.chat.OpenAI")
def test_chat_complete_with_history(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_completion("Yes!")

    resp = client.post(
        "/chat/complete",
        json={
            "prompt": "Is that right?",
            "history": [
                {"role": "user", "content": "Paris is the capital."},
                {"role": "assistant", "content": "Correct."},
            ],
        },
    )
    assert resp.status_code == 200

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    # system + 2 history + current user = 4
    assert len(call_kwargs["messages"]) == 4


@patch("app.routers.chat.get_settings", return_value=_fake_settings())
@patch("app.routers.chat.OpenAI")
def test_chat_complete_missing_prompt(mock_openai: MagicMock, _settings: MagicMock) -> None:
    resp = client.post("/chat/complete", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /chat/extract
# ---------------------------------------------------------------------------


@patch("app.routers.chat.get_settings", return_value=_fake_settings())
@patch("app.routers.chat.OpenAI")
def test_chat_extract(mock_openai: MagicMock, _settings: MagicMock) -> None:
    from app.schemas.chat import ExtractedInfo

    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    parsed = ExtractedInfo(name="Bob", age=25, summary="Bob is 25.")
    message = MagicMock()
    message.refusal = None
    message.parsed = parsed
    comp = MagicMock()
    comp.choices = [MagicMock(message=message)]
    mock_client.beta.chat.completions.parse.return_value = comp

    resp = client.post("/chat/extract", json={"text": "Bob is 25 years old."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["extracted"]["name"] == "Bob"
    assert data["extracted"]["age"] == 25


# ---------------------------------------------------------------------------
# /vision/url
# ---------------------------------------------------------------------------


@patch("app.routers.vision.get_settings", return_value=_fake_settings())
@patch("app.routers.vision.OpenAI")
def test_vision_url(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_completion("A dog.")

    resp = client.post(
        "/vision/url",
        json={"url": "https://example.com/dog.jpg"},
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "A dog."


@patch("app.routers.vision.get_settings", return_value=_fake_settings())
@patch("app.routers.vision.OpenAI")
def test_vision_url_invalid_body(mock_openai: MagicMock, _settings: MagicMock) -> None:
    resp = client.post("/vision/url", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /vision/file
# ---------------------------------------------------------------------------


@patch("app.routers.vision.get_settings", return_value=_fake_settings())
@patch("app.routers.vision.OpenAI")
def test_vision_file_upload(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_completion("A cat.")

    resp = client.post(
        "/vision/file",
        files={"file": ("test.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "A cat."


@patch("app.routers.vision.get_settings", return_value=_fake_settings())
@patch("app.routers.vision.OpenAI")
def test_vision_file_wrong_mime(mock_openai: MagicMock, _settings: MagicMock) -> None:
    resp = client.post(
        "/vision/file",
        files={"file": ("doc.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415


# ---------------------------------------------------------------------------
# /embeddings/embed
# ---------------------------------------------------------------------------


@patch("app.routers.embeddings.get_settings", return_value=_fake_settings())
@patch("app.routers.embeddings.OpenAI")
def test_embed(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.embeddings.create.return_value = _make_embedding_response([[0.1] * 1536])

    resp = client.post("/embeddings/embed", json={"text": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["dimensions"] == 1536
    assert data["model"] == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# /embeddings/similarity
# ---------------------------------------------------------------------------


@patch("app.routers.embeddings.get_settings", return_value=_fake_settings())
@patch("app.routers.embeddings.OpenAI")
def test_similarity(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_client.embeddings.create.return_value = _make_embedding_response(
        [
            [1.0, 0.0],   # query
            [1.0, 0.0],   # identical → 1.0
            [0.0, 1.0],   # orthogonal → 0.0
        ]
    )

    resp = client.post(
        "/embeddings/similarity",
        json={"query": "fast car", "candidates": ["sports car", "slow train"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["score"] == pytest.approx(1.0)
    assert data["results"][1]["score"] == pytest.approx(0.0)
