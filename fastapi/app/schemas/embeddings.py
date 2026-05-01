"""Pydantic schemas for the embeddings endpoints."""

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to embed")


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: str
    total_tokens: int


class SimilarityRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query string")
    candidates: list[str] = Field(
        ..., min_length=1, description="Strings to rank against the query"
    )


class SimilarityResult(BaseModel):
    text: str
    score: float


class SimilarityResponse(BaseModel):
    query: str
    results: list[SimilarityResult]
    model: str
