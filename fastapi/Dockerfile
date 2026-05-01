# syntax=docker/dockerfile:1

# ── builder stage ─────────────────────────────────────────────────────────────
FROM python:3.11.13-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --compile-bytecode

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM gcr.io/distroless/python3-debian12:nonroot AS runtime

WORKDIR /app

COPY --from=builder --chown=65532:65532 /app/.venv/lib /app/.venv/lib
COPY --chown=65532:65532 app/ ./app/

ENV PYTHONPATH="/app/.venv/lib/python3.11/site-packages" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.base.name="gcr.io/distroless/python3-debian12:nonroot" \
      org.opencontainers.image.source="https://github.com/miqui/openai-api-usecases"

EXPOSE 8000

USER 65532

CMD ["/usr/bin/python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
