"""Pydantic schemas for the chat endpoints."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User message")
    system: str = Field(
        default="You are a helpful assistant.",
        description="System prompt",
    )
    history: list[dict] | None = Field(
        default=None,
        description="Prior messages for multi-turn. Each: {role, content}",
    )


class ChatResponse(BaseModel):
    response: str
    model: str
    total_tokens: int


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Free text to extract from")


class ExtractedInfo(BaseModel):
    name: str
    age: int
    summary: str


class ExtractResponse(BaseModel):
    extracted: ExtractedInfo
    model: str
