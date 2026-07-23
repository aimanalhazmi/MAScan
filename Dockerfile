FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# uv writes its bytecode + venv here.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY pyproject.toml uv.lock ./


# Install dependencies only (no project source yet).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# Build the web UI into static assets served by the API.
FROM node:20-slim AS ui-builder
WORKDIR /ui/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
# Copy the resolved virtualenv from the builder.
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy the source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/

# Built web UI, served same-origin at :8000.
COPY --from=ui-builder --chown=app:app /ui/src/mascan/app/static /app/static
ENV MASCAN_STATIC_DIR=/app/static

# Writable dir for figures extracted from uploaded PDFs (RAG_IMAGE_DIR default).
RUN mkdir -p /app/rag_images /app/rag_uploads && chown app:app /app/rag_images /app/rag_uploads

ENV PYTHONPATH=/app/src

USER app

EXPOSE 8000

# Default command. docker-compose overrides this with --reload for dev.
CMD ["uvicorn", "mascan.app.api:app", "--host", "0.0.0.0", "--port", "8000"]