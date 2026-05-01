# openai-api-learning

A structured learning project for the OpenAI Python SDK, covering three use cases across three delivery formats.

## Use Cases Covered

| # | Topic | Endpoint / Method |
|---|-------|-------------------|
| 1 | Chat Completions | `client.chat.completions.create()` · `.stream()` · `.parse()` |
| 5 | Vision | `image_url` content blocks (URL + base64) |
| 6 | Embeddings | `client.embeddings.create()` · cosine similarity |

## Subprojects

| Directory | Format | Description |
|-----------|--------|-------------|
| `cli/` | CLI (typer + rich) | Seven commands: chat, stream, extract, vision-url, vision-file, embed, similar |
| `fastapi/` | REST API (FastAPI) | Six endpoints; hardened distroless container |
| `jupyter/` | Notebooks | Three annotated notebooks with prose explanations |

Each subproject is a standalone uv project with its own `pyproject.toml`, `venv`, and `README.md`.

## Prerequisites

- Python 3.11+
- uv 0.11+
- Docker (for fastapi container)
- An OpenAI API key

## Quick Start

```bash
# CLI
cd cli
uv sync
cp .env.example .env  # fill in OPENAI_API_KEY
uv run python -m cli.main chat "What is Python?"

# FastAPI
cd fastapi
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload
# → http://localhost:8000/docs

# Jupyter
cd jupyter
uv sync
uv run jupyter notebook notebooks/
```

## Running Tests

```bash
# CLI — 24 tests
cd cli && PYTHONPYCACHEPREFIX=/tmp/pycache uv run pytest tests/ -v

# FastAPI — 11 tests
cd fastapi && PYTHONPYCACHEPREFIX=/tmp/pycache uv run pytest tests/ -v
```

## Docker (FastAPI)

```bash
cd fastapi
hadolint Dockerfile
docker build -t openai-fastapi-demo:latest .
docker buildx build -t openai-fastapi-demo:latest .
docker run -p 8000:8000 --env-file .env openai-fastapi-demo:latest
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (all subprojects) |
| `CHAT_MODEL` | No | `gpt-4o` | Model for chat and vision |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embeddings model |

## Repository Layout

```
openai-api-learning/
  cli/
    cli/main.py              # typer commands
    core/{config,logging}.py
    usecases/{chat,vision,embeddings}.py
    tests/
    pyproject.toml
  fastapi/
    app/
      main.py
      core/{config,logging}.py
      routers/{chat,vision,embeddings}.py
      schemas/{chat,vision,embeddings}.py
    tests/
    Dockerfile
    pyproject.toml
  jupyter/
    notebooks/
      01_chat_completions.ipynb
      02_vision.ipynb
      03_embeddings.ipynb
    pyproject.toml
  README.md
```
