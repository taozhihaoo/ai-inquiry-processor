FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so Docker layer caching works
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (secrets are NOT copied: .env is excluded via .dockerignore)
COPY . .

# Run as the unprivileged "app" user instead of root
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/output \
    && chown -R app:app /app
USER app

# SQLite data + reports live on volumes so they survive container restarts
VOLUME ["/app/data", "/app/output"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# Default provider is mock (no credentials needed). For real runs inject:
#   -e LLM_PROVIDER=openai -e OPENAI_API_KEY=...
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
