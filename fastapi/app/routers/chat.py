"""Chat completions router — use case 1."""

import base64
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ExtractedInfo,
    ExtractRequest,
    ExtractResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


@router.post("/complete", response_model=ChatResponse, summary="Single-round chat completion")
async def chat_complete(body: ChatRequest) -> ChatResponse:
    """Send a message and receive a single response.

    Supports optional system prompt and multi-turn history.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages: list[dict] = [{"role": "system", "content": body.system}]
    if body.history:
        messages.extend(body.history)
    messages.append({"role": "user", "content": body.prompt})

    logger.info("chat_complete", model=settings.chat_model, turns=len(messages))

    try:
        completion = client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,  # type: ignore[arg-type]
        )
    except Exception as exc:
        logger.error("chat_complete_error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        response=completion.choices[0].message.content or "",
        model=settings.chat_model,
        total_tokens=completion.usage.total_tokens if completion.usage else 0,
    )


@router.post("/extract", response_model=ExtractResponse, summary="Structured extraction")
async def chat_extract(body: ExtractRequest) -> ExtractResponse:
    """Extract name, age, and summary from free text using structured output."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("chat_extract", model=settings.chat_model)

    try:
        completion = client.beta.chat.completions.parse(
            model=settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the person's name, age, and write a one-sentence summary.",
                },
                {"role": "user", "content": body.text},
            ],
            response_format=ExtractedInfo,
        )
    except Exception as exc:
        logger.error("chat_extract_error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    message = completion.choices[0].message
    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused: {message.refusal}")
    if not message.parsed:
        raise HTTPException(status_code=502, detail="Model returned no parsed output")

    return ExtractResponse(extracted=message.parsed, model=settings.chat_model)
