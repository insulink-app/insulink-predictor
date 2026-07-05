# Glucose-forecast inference service (FastAPI). Internal only — the Spring
# backend calls it. The model is trained on the HOST
# (`uv run python scripts/train_and_save.py`) and mounted at /app/artifacts, so
# retraining is a host step + a container restart (no image rebuild).
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir uv

# Deps first (cached), then the source that the local package builds from.
# README.md is referenced by pyproject (readme = ...), so the build needs it.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY config ./config
COPY serve ./serve

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
