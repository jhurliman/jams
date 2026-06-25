# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Essentia + librosa runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Resolve + install into a project venv (cached layer when sources are unchanged).
RUN uv sync --no-dev --frozen || uv sync --no-dev

EXPOSE 8000
ENV JAMS_HOST=0.0.0.0 JAMS_PORT=8000
CMD ["uv", "run", "jams"]
