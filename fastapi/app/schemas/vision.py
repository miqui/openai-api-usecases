"""Pydantic schemas for the vision endpoints."""

from pydantic import BaseModel, Field, HttpUrl


class VisionUrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="Public image URL")
    question: str = Field(
        default="What is in this image? Describe it in detail.",
        description="Question to ask about the image",
    )


class VisionResponse(BaseModel):
    response: str
    model: str
    total_tokens: int
