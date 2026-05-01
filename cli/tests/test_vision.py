"""Tests for usecases/vision.py — all OpenAI calls and settings are mocked."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from usecases.vision import describe_image_file, describe_image_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings() -> MagicMock:
    s = MagicMock()
    s.openai_api_key = "sk-test"
    s.chat_model = "gpt-4o"
    return s


def _make_completion(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.total_tokens = 20
    comp = MagicMock()
    comp.choices = [choice]
    comp.usage = usage
    return comp


# ---------------------------------------------------------------------------
# describe_image_url
# ---------------------------------------------------------------------------


@patch("usecases.vision.get_settings", return_value=_fake_settings())
@patch("usecases.vision.OpenAI")
def test_describe_image_url_returns_content(
    mock_openai: MagicMock, _settings: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_completion("A cat sitting.")

    result = describe_image_url("https://example.com/cat.jpg")

    assert result == "A cat sitting."
    mock_client.chat.completions.create.assert_called_once()


@patch("usecases.vision.get_settings", return_value=_fake_settings())
@patch("usecases.vision.OpenAI")
def test_describe_image_url_sends_correct_content_structure(
    mock_openai: MagicMock, _settings: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_completion("ok")

    url = "https://example.com/photo.png"
    question = "What colour is the sky?"
    describe_image_url(url, question=question)

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    types = [block["type"] for block in content]
    assert "image_url" in types
    assert "text" in types
    img_block = next(b for b in content if b["type"] == "image_url")
    assert img_block["image_url"]["url"] == url
    text_block = next(b for b in content if b["type"] == "text")
    assert text_block["text"] == question


# ---------------------------------------------------------------------------
# describe_image_file
# ---------------------------------------------------------------------------


@patch("usecases.vision.get_settings", return_value=_fake_settings())
@patch("usecases.vision.OpenAI")
def test_describe_image_file_returns_content(
    mock_openai: MagicMock, _settings: MagicMock, tmp_path: Path
) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_completion("A red square.")

    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

    result = describe_image_file(str(img_file))
    assert result == "A red square."


@patch("usecases.vision.get_settings", return_value=_fake_settings())
@patch("usecases.vision.OpenAI")
def test_describe_image_file_sends_base64_data_url(
    mock_openai: MagicMock, _settings: MagicMock, tmp_path: Path
) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_completion("ok")

    img_file = tmp_path / "sample.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

    describe_image_file(str(img_file))

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    img_block = next(b for b in messages[0]["content"] if b["type"] == "image_url")
    data_url = img_block["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")


def test_describe_image_file_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        describe_image_file("/does/not/exist.png")
