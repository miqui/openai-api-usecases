"""Use case 5: Vision — images as input to gpt-4o.

Covers:
  - describe_image_url   → pass a public URL directly
  - describe_image_file  → read a local file, base64-encode, send inline

Learning notes:
  - Vision is just chat completions with a richer content list.
  - Each content item is either {"type": "text", ...} or {"type": "image_url", ...}
  - For inline images: data:[mime];base64,<b64>  format required by the API.
  - gpt-4o supports JPEG, PNG, GIF, WEBP; max ~20 MB per image.
"""

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


def describe_image_url(
    url: str,
    question: str = "What is in this image? Describe it in detail.",
) -> str:
    """Ask gpt-4o to describe an image hosted at a public URL.

    Args:
        url: Publicly accessible image URL.
        question: What to ask about the image.

    Returns:
        The model's description.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("vision_url", model=settings.chat_model, url=url)

    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": "auto"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
    )

    response = completion.choices[0].message.content or ""
    logger.info(
        "vision_url_response",
        tokens=completion.usage.total_tokens if completion.usage else 0,
    )
    return response


def describe_image_file(
    path: str | Path,
    question: str = "What is in this image? Describe it in detail.",
) -> str:
    """Ask gpt-4o to describe a local image file (base64-encoded inline).

    Args:
        path: Path to a local image (JPEG, PNG, GIF, or WEBP).
        question: What to ask about the image.

    Returns:
        The model's description.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file's MIME type cannot be determined.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"Cannot determine image MIME type for: {file_path}")

    image_data = base64.standard_b64encode(file_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime};base64,{image_data}"

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info(
        "vision_file",
        model=settings.chat_model,
        path=str(file_path),
        mime=mime,
        bytes=file_path.stat().st_size,
    )

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

    response = completion.choices[0].message.content or ""
    logger.info(
        "vision_file_response",
        tokens=completion.usage.total_tokens if completion.usage else 0,
    )
    return response
