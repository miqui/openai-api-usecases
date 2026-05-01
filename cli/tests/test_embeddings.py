"""Tests for usecases/embeddings.py — all OpenAI calls and settings are mocked."""

import math
from unittest.mock import MagicMock, patch

import pytest

from usecases.embeddings import (
    cosine_similarity,
    embed_many,
    embed_text,
    find_most_similar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings() -> MagicMock:
    s = MagicMock()
    s.openai_api_key = "sk-test"
    s.embedding_model = "text-embedding-3-small"
    return s


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    data = []
    for i, vec in enumerate(vectors):
        item = MagicMock()
        item.embedding = vec
        item.index = i
        data.append(item)
    usage = MagicMock()
    usage.total_tokens = len(vectors) * 5
    resp = MagicMock()
    resp.data = data
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------


@patch("usecases.embeddings.get_settings", return_value=_fake_settings())
@patch("usecases.embeddings.OpenAI")
def test_embed_text_returns_vector(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    fake_vec = [0.1] * 1536
    mock_client.embeddings.create.return_value = _make_embedding_response([fake_vec])

    result = embed_text("hello world")

    assert len(result) == 1536
    assert result[0] == pytest.approx(0.1)
    mock_client.embeddings.create.assert_called_once()


# ---------------------------------------------------------------------------
# embed_many
# ---------------------------------------------------------------------------


@patch("usecases.embeddings.get_settings", return_value=_fake_settings())
@patch("usecases.embeddings.OpenAI")
def test_embed_many_returns_correct_count(mock_openai: MagicMock, _settings: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    vecs = [[float(i)] * 4 for i in range(3)]
    mock_client.embeddings.create.return_value = _make_embedding_response(vecs)

    results = embed_many(["a", "b", "c"])

    assert len(results) == 3
    assert results[0] == [0.0, 0.0, 0.0, 0.0]
    assert results[1] == [1.0, 1.0, 1.0, 1.0]


@patch("usecases.embeddings.OpenAI")
def test_embed_many_empty_returns_empty(mock_openai: MagicMock) -> None:
    result = embed_many([])
    assert result == []
    mock_openai.assert_not_called()


@patch("usecases.embeddings.get_settings", return_value=_fake_settings())
@patch("usecases.embeddings.OpenAI")
def test_embed_many_preserves_order_even_if_api_returns_shuffled(
    mock_openai: MagicMock, _settings: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    item0 = MagicMock()
    item0.embedding = [1.0, 0.0]
    item0.index = 0
    item2 = MagicMock()
    item2.embedding = [0.0, 0.0]
    item2.index = 2
    item1 = MagicMock()
    item1.embedding = [0.0, 1.0]
    item1.index = 1

    resp = MagicMock()
    resp.data = [item2, item0, item1]  # shuffled
    resp.usage = MagicMock()
    resp.usage.total_tokens = 15
    mock_client.embeddings.create.return_value = resp

    results = embed_many(["x", "y", "z"])

    assert results[0] == [1.0, 0.0]
    assert results[1] == [0.0, 1.0]
    assert results[2] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_known_angle() -> None:
    a = [1.0, 0.0]
    b = [1.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(math.sqrt(2) / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# find_most_similar
# ---------------------------------------------------------------------------


@patch("usecases.embeddings.embed_many")
def test_find_most_similar_sorted_descending(mock_embed_many: MagicMock) -> None:
    mock_embed_many.return_value = [
        [1.0, 0.0],   # query
        [0.0, 1.0],   # orthogonal → score 0
        [1.0, 0.0],   # identical → score 1
        [-1.0, 0.0],  # opposite → score -1
    ]

    results = find_most_similar("query", ["orthogonal", "identical", "opposite"])

    texts = [r[0] for r in results]
    scores = [r[1] for r in results]

    assert texts[0] == "identical"
    assert scores[0] == pytest.approx(1.0)
    assert scores[-1] == pytest.approx(-1.0)


@patch("usecases.embeddings.embed_many")
def test_find_most_similar_empty_candidates(mock_embed_many: MagicMock) -> None:
    result = find_most_similar("query", [])
    assert result == []
    mock_embed_many.assert_not_called()
