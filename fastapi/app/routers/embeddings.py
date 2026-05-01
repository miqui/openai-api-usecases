"""Embeddings router — use case 6."""

import numpy as np
from fastapi import APIRouter, HTTPException
from openai import OpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.embeddings import (
    EmbedRequest,
    EmbedResponse,
    SimilarityRequest,
    SimilarityResponse,
    SimilarityResult,
)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])
logger = get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


@router.post("/embed", response_model=EmbedResponse, summary="Embed a single text")
async def embed(body: EmbedRequest) -> EmbedResponse:
    """Embed a string and return its vector."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("embed", model=settings.embedding_model, chars=len(body.text))

    try:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=body.text,
        )
    except Exception as exc:
        logger.error("embed_error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    vector = response.data[0].embedding
    return EmbedResponse(
        embedding=vector,
        dimensions=len(vector),
        model=settings.embedding_model,
        total_tokens=response.usage.total_tokens,
    )


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    summary="Rank candidates by similarity to query",
)
async def similarity(body: SimilarityRequest) -> SimilarityResponse:
    """Embed the query and all candidates, then rank by cosine similarity."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    all_texts = [body.query] + body.candidates
    logger.info(
        "similarity", model=settings.embedding_model, candidates=len(body.candidates)
    )

    try:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=all_texts,
        )
    except Exception as exc:
        logger.error("similarity_error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sorted_data = sorted(response.data, key=lambda e: e.index)
    query_vec = sorted_data[0].embedding
    results: list[SimilarityResult] = []

    for text, item in zip(body.candidates, sorted_data[1:]):
        score = _cosine_similarity(query_vec, item.embedding)
        results.append(SimilarityResult(text=text, score=score))

    results.sort(key=lambda r: r.score, reverse=True)

    return SimilarityResponse(
        query=body.query,
        results=results,
        model=settings.embedding_model,
    )
