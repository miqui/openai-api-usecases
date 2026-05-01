"""Tests for usecases/chat.py — all OpenAI calls and settings are mocked."""

from unittest.mock import MagicMock, patch

import pytest

from usecases.chat import ExtractedInfo, basic_chat, stream_chat, structured_chat


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _fake_settings() -> MagicMock:
    s = MagicMock()
    s.openai_api_key = "sk-test"
    s.chat_model = "gpt-4o"
    return s


def _make_completion(content: str, total_tokens: int = 10) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.total_tokens = total_tokens
    comp = MagicMock()
    comp.choices = [choice]
    comp.usage = usage
    return comp


# ---------------------------------------------------------------------------
# basic_chat
# ---------------------------------------------------------------------------


@patch("usecases.chat.get_settings", return_value=_fake_settings())
@patch("usecases.chat.OpenAI")
def test_basic_chat_returns_content(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_completion("Paris")

    result = basic_chat("What is the capital of France?")

    assert result == "Paris"
    mock_client.chat.completions.create.assert_called_once()


@patch("usecases.chat.get_settings", return_value=_fake_settings())
@patch("usecases.chat.OpenAI")
def test_basic_chat_includes_system_prompt(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_completion("ok")

    basic_chat("Hi", system="You are grumpy.")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are grumpy."}
    assert messages[-1]["role"] == "user"


@patch("usecases.chat.get_settings", return_value=_fake_settings())
@patch("usecases.chat.OpenAI")
def test_basic_chat_includes_history(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_completion("yes")

    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    basic_chat("Follow-up", history=history)

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    # system + 2 history + current user = 4
    assert len(messages) == 4
    assert messages[1] == history[0]
    assert messages[2] == history[1]


@patch("usecases.chat.get_settings", return_value=_fake_settings())
@patch("usecases.chat.OpenAI")
def test_basic_chat_empty_content_returns_empty_string(
    mock_openai: MagicMock, _settings: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_completion("")

    result = basic_chat("Silent?")
    assert result == ""


# ---------------------------------------------------------------------------
# stream_chat
# ---------------------------------------------------------------------------


@patch("usecases.chat.get_settings", return_value=_fake_settings())
@patch("usecases.chat.OpenAI")
def test_stream_chat_yields_tokens(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    def _make_event(content: str | None) -> MagicMock:
        ev = MagicMock()
        ev.type = "content.delta" if content else "other"
        ev.content = content
        return ev

    fake_events = [
        _make_event("Hello"),
        _make_event(None),
        _make_event(" world"),
    ]

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__ = MagicMock(return_value=iter(fake_events))
    mock_stream_cm.__exit__ = MagicMock(return_value=False)
    mock_client.chat.completions.stream.return_value = mock_stream_cm

    tokens = list(stream_chat("Hi"))

    assert tokens == ["Hello", " world"]


# ---------------------------------------------------------------------------
# structured_chat
# ---------------------------------------------------------------------------


@patch("usecases.chat.get_settings", return_value=_fake_settings())
@patch("usecases.chat.OpenAI")
def test_structured_chat_returns_parsed(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    parsed = ExtractedInfo(name="Alice", age=30, summary="Alice is 30 years old.")
    message = MagicMock()
    message.refusal = None
    message.parsed = parsed
    comp = MagicMock()
    comp.choices = [MagicMock(message=message)]
    mock_client.beta.chat.completions.parse.return_value = comp

    result = structured_chat("Alice is 30 years old.")

    assert result.name == "Alice"
    assert result.age == 30
    assert "Alice" in result.summary


@patch("usecases.chat.get_settings", return_value=_fake_settings())
@patch("usecases.chat.OpenAI")
def test_structured_chat_raises_on_refusal(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    message = MagicMock()
    message.refusal = "I can't do that."
    message.parsed = None
    comp = MagicMock()
    comp.choices = [MagicMock(message=message)]
    mock_client.beta.chat.completions.parse.return_value = comp

    with pytest.raises(ValueError, match="refused"):
        structured_chat("some text")


@patch("usecases.chat.get_settings", return_value=_fake_settings())
@patch("usecases.chat.OpenAI")
def test_structured_chat_raises_on_no_parsed(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    message = MagicMock()
    message.refusal = None
    message.parsed = None
    comp = MagicMock()
    comp.choices = [MagicMock(message=message)]
    mock_client.beta.chat.completions.parse.return_value = comp

    with pytest.raises(ValueError, match="no parsed output"):
        structured_chat("some text")
