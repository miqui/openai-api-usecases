"""Use case 1: Chat Completions.

Covers:
  - basic_chat        → single Q&A with optional system prompt
  - stream_chat       → stream tokens as they arrive
  - structured_chat   → parse response into a Pydantic model

Learning notes:
  - client.chat.completions.create()  → standard, blocking
  - client.chat.completions.stream()  → async streaming context manager
  - client.chat.completions.parse()   → structured output via Pydantic
"""

import asyncio
from typing import Iterator

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


def basic_chat(
    prompt: str,
    system: str = "You are a helpful assistant.",
    history: list[dict] | None = None,
) -> str:
    """Basic chat completion — single round-trip.

    Args:
        prompt: The user message.
        system: System prompt that sets the assistant's persona.
        history: Optional prior messages for multi-turn conversations.
                 Each item: {"role": "user"|"assistant", "content": "..."}

    Returns:
        The assistant's response text.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    logger.info("chat_request", model=settings.chat_model, turns=len(messages))

    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,  # type: ignore[arg-type]
    )

    response = completion.choices[0].message.content or ""
    logger.info("chat_response", tokens=completion.usage.total_tokens if completion.usage else 0)
    return response


def stream_chat(
    prompt: str,
    system: str = "You are a helpful assistant.",
) -> Iterator[str]:
    """Stream chat tokens as they arrive (sync generator).

    Yields each text delta so the caller can print in real time.

    Args:
        prompt: The user message.
        system: System prompt.

    Yields:
        Token strings as the model generates them.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("stream_chat_start", model=settings.chat_model)

    # create() with stream=True returns an iterator of ChatCompletionChunk
    with client.chat.completions.stream(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    ) as stream:
        for event in stream:
            if event.type == "content.delta" and event.content:
                yield event.content


# -----------------------------------------------------------------
# Structured output
# -----------------------------------------------------------------

class ExtractedInfo(BaseModel):
    """Schema for structured extraction — name + age from free text."""
    name: str
    age: int
    summary: str


def structured_chat(text: str) -> ExtractedInfo:
    """Extract structured information from free text using Pydantic parsing.

    Uses client.chat.completions.parse() — the SDK auto-converts the
    Pydantic model to a JSON schema and parses the response back.

    Args:
        text: Raw text to extract name and age from.

    Returns:
        An ExtractedInfo Pydantic model with name, age, and a one-line summary.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("structured_chat", model=settings.chat_model)

    completion = client.beta.chat.completions.parse(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": "Extract the person's name, age, and write a one-sentence summary.",
            },
            {"role": "user", "content": text},
        ],
        response_format=ExtractedInfo,
    )

    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"Model refused: {message.refusal}")
    if not message.parsed:
        raise ValueError("Model returned no parsed output")

    return message.parsed
