# openai-api-learning / cli

Command-line interface for exploring OpenAI use cases 1, 5, and 6 using the Python SDK.

## Prerequisites

- Python 3.11+
- uv 0.11+
- An OpenAI API key

## Installation

```bash
uv sync
cp .env.example .env
# edit .env — set OPENAI_API_KEY
```

## Commands

| Command | Use case | Description |
|---------|----------|-------------|
| `chat` | 1 | Single-round chat completion |
| `stream` | 1 | Stream tokens as they arrive |
| `extract` | 1 | Structured extraction (name, age, summary) |
| `vision-url` | 5 | Describe a public image URL |
| `vision-file` | 5 | Describe a local image file (base64) |
| `embed` | 6 | Embed a string, show vector dimensions |
| `similar` | 6 | Rank strings by semantic similarity |

## Usage

```bash
# Basic chat
uv run python -m cli.main chat "What is the capital of France?"

# With a custom system prompt
uv run python -m cli.main chat "Translate to Spanish" --system "You are a translator."

# Stream tokens live
uv run python -m cli.main stream "Tell me a haiku about Python."

# Structured extraction
uv run python -m cli.main extract "Alice is 32 years old and works as a pilot."

# Vision — public URL
uv run python -m cli.main vision-url "https://example.com/photo.jpg"

# Vision — local file with custom question
uv run python -m cli.main vision-file ./screenshot.png --question "What text is visible?"

# Embed a string
uv run python -m cli.main embed "Hello, world"

# Semantic similarity ranking
uv run python -m cli.main similar "fast car" "sports automobile" "slow train" "rocket ship"
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `CHAT_MODEL` | No | `gpt-4o` | Model for chat and vision |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model for embeddings |

## Project Structure

```
cli/
  cli/
    __init__.py
    main.py          # typer app — all commands
  core/
    config.py        # pydantic-settings
    logging.py       # structlog
  usecases/
    chat.py          # basic_chat, stream_chat, structured_chat
    vision.py        # describe_image_url, describe_image_file
    embeddings.py    # embed_text, embed_many, cosine_similarity, find_most_similar
  tests/
    test_chat.py
    test_vision.py
    test_embeddings.py
  pyproject.toml
  .env.example
```

## Running Tests

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache uv run pytest tests/ -v
```
