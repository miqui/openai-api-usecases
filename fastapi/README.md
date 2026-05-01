# openai-api-learning / fastapi

FastAPI service that exposes OpenAI use cases 1 (chat), 5 (vision), and 6 (embeddings) as REST endpoints. Runs in a hardened distroless container.

## Prerequisites

- Python 3.11+
- uv 0.11+
- Docker (for container build)
- An OpenAI API key

## Installation

```bash
uv sync
cp .env.example .env
# edit .env — set OPENAI_API_KEY
```

## Running locally

```bash
uv run uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

## Running Tests

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache uv run pytest tests/ -v
```

## Docker

```bash
# Lint Dockerfile
hadolint Dockerfile

# Build
docker build -t openai-fastapi-demo:latest .
docker buildx build -t openai-fastapi-demo:latest .

# Run
docker run -p 8000:8000 --env-file .env openai-fastapi-demo:latest
```

## API Reference

### Health

```
GET /health
→ {"status": "ok"}
```

### Chat (use case 1)

```
POST /chat/complete
{
  "prompt": "What is the capital of France?",
  "system": "You are a helpful assistant.",   // optional
  "history": [...]                             // optional multi-turn
}
→ {"response": "Paris", "model": "gpt-4o", "total_tokens": 42}
```

```
POST /chat/extract
{"text": "Alice is 32 years old."}
→ {"extracted": {"name": "Alice", "age": 32, "summary": "..."}, "model": "gpt-4o"}
```

### Vision (use case 5)

```
POST /vision/url
{"url": "https://example.com/photo.jpg", "question": "What is in this image?"}
→ {"response": "...", "model": "gpt-4o", "total_tokens": 300}
```

```
POST /vision/file      (multipart/form-data)
file: <image upload>
question: "What do you see?"   (form field, optional)
→ {"response": "...", "model": "gpt-4o", "total_tokens": 300}
```

### Embeddings (use case 6)

```
POST /embeddings/embed
{"text": "Hello, world"}
→ {"embedding": [...], "dimensions": 1536, "model": "text-embedding-3-small", "total_tokens": 4}
```

```
POST /embeddings/similarity
{"query": "fast car", "candidates": ["sports automobile", "slow train", "rocket"]}
→ {"query": "fast car", "results": [{"text": "...", "score": 0.92}, ...], "model": "..."}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `CHAT_MODEL` | No | `gpt-4o` | Model for chat and vision |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model for embeddings |

## Project Structure

```
fastapi/
  app/
    main.py
    core/
      config.py
      logging.py
    routers/
      chat.py
      vision.py
      embeddings.py
    schemas/
      chat.py
      vision.py
      embeddings.py
  tests/
    test_api.py
  Dockerfile
  .dockerignore
  .env.example
  pyproject.toml
```
