"""Vision router — use case 5."""

import base64
import mimetypes

from fastapi import APIRouter, HTTPException, UploadFile
from openai import OpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.vision import VisionResponse, VisionUrlRequest

router = APIRouter(prefix="/vision", tags=["vision"])
logger = get_logger(__name__)


@router.post("/url", response_model=VisionResponse, summary="Describe image from URL")
async def vision_url(body: VisionUrlRequest) -> VisionResponse:
    """Ask gpt-4o to describe an image hosted at a public URL."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("vision_url", model=settings.chat_model, url=str(body.url))

    try:
        completion = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": str(body.url), "detail": "auto"},
                        },
                        {"type": "text", "text": body.question},
                    ],
                }
            ],
        )
    except Exception as exc:
        logger.error("vision_url_error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return VisionResponse(
        response=completion.choices[0].message.content or "",
        model=settings.chat_model,
        total_tokens=completion.usage.total_tokens if completion.usage else 0,
    )


@router.post("/file", response_model=VisionResponse, summary="Describe uploaded image")
async def vision_file(
    file: UploadFile,
    question: str = "What is in this image? Describe it in detail.",
) -> VisionResponse:
    """Accept a multipart image upload, base64-encode it, and ask gpt-4o to describe it."""
    settings = get_settings()

    mime = file.content_type or ""
    if not mime.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {mime!r}. Must be an image.",
        )

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

    image_data = base64.standard_b64encode(raw).decode("utf-8")
    data_url = f"data:{mime};base64,{image_data}"

    client = OpenAI(api_key=settings.openai_api_key)
    logger.info("vision_file", model=settings.chat_model, mime=mime, bytes=len(raw))

    try:
        completion = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "auto"},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
    except Exception as exc:
        logger.error("vision_file_error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return VisionResponse(
        response=completion.choices[0].message.content or "",
        model=settings.chat_model,
        total_tokens=completion.usage.total_tokens if completion.usage else 0,
    )
