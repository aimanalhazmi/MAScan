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

ENV PYTHONPATH=/app/src

USER app

EXPOSE 8000

# Default command. docker-compose overrides this with --reload for dev.
CMD ["uvicorn", "mascan.app.api:app", "--host", "0.0.0.0", "--port", "8000"]