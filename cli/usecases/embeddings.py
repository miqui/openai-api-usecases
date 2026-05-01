"""Use case 6: Embeddings — text-embedding-3-small.

Covers:
  - embed_text        → embed a single string
  - embed_many        → batch-embed a list of strings (one API call)
  - cosine_similarity → compare two embedding vectors
  - find_most_similar → rank a list of candidates against a query

Learning notes:
  - client.embeddings.create() accepts a string or list of strings.
  - Each Embedding object has .embedding (list[float]) and .index.
  - Always L2-normalise before cosine similarity — text-embedding-3 vectors
    are unit-norm by default, but numpy dot product is fastest when guaranteed.
  - Batch calls share the same rate-limit bucket; prefer batching over loops.
"""

import numpy as np
from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


def embed_text(text: str) -> list[float]:
    """Embed a single string using text-embedding-3-small.

    Args:
        text: The string to embed.

    Returns:
        A 1536-dimensional float vector.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("embed_text", model=settings.embedding_model, chars=len(text))

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    vector = response.data[0].embedding
    logger.info("embed_text_done", dims=len(vector))
    return vector


def embed_many(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in a single API call.

    The returned list preserves the same order as the input.

    Args:
        texts: Strings to embed (up to 2048 items per call).

    Returns:
        List of 1536-dimensional float vectors.
    """
    if not texts:
        return []

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("embed_many", model=settings.embedding_model, count=len(texts))

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    # API returns items sorted by .index — re-sort to be safe.
    sorted_data = sorted(response.data, key=lambda e: e.index)
    vectors = [e.embedding for e in sorted_data]

    logger.info(
        "embed_many_done",
        count=len(vectors),
        dims=len(vectors[0]) if vectors else 0,
        tokens=response.usage.total_tokens,
    )
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Similarity score in [-1.0, 1.0]; higher means more similar.
    """
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))


def find_most_similar(
    query: str,
    candidates: list[str],
) -> list[tuple[str, float]]:
    """Rank candidates by semantic similarity to the query.

    Embeds the query + all candidates in one batch call, then sorts by
    cosine similarity descending.

    Args:
        query: The reference string.
        candidates: Strings to rank against the query.

    Returns:
        List of (candidate_text, score) tuples sorted by score descending.
    """
    if not candidates:
        return []

    all_texts = [query] + candidates
    all_vectors = embed_many(all_texts)

    query_vec = all_vectors[0]
    results: list[tuple[str, float]] = []

    for text, vec in zip(candidates, all_vectors[1:]):
        score = cosine_similarity(query_vec, vec)
        results.append((text, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
