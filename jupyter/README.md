# openai-api-usecases / jupyter

Interactive Jupyter notebooks for exploring OpenAI use cases 1, 5, and 6.

## Prerequisites

- Python 3.11+
- uv 0.11+
- An OpenAI API key

## Installation

```bash
uv sync
cp ../.env.example .env
# edit .env — set OPENAI_API_KEY
```

## Running Notebooks

```bash
uv run jupyter notebook notebooks/
```

Then open any notebook in the browser UI. The notebooks read `OPENAI_API_KEY` from the environment or a `.env` file via `python-dotenv`.

## Notebooks

| Notebook | Use Case | Topics |
|----------|----------|--------|
| `01_chat_completions.ipynb` | 1 | Basic chat, multi-turn history, streaming, structured output |
| `02_vision.ipynb` | 5 | Image from URL, base64 local file, detail levels |
| `03_embeddings.ipynb` | 6 | Single/batch embed, cosine similarity, semantic search |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `CHAT_MODEL` | No | `gpt-4o` | Model for chat and vision |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model for embeddings |
