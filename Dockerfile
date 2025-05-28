# Use Python 3.11 as base image
FROM python:3.11-slim 


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    UV_CACHE_DIR=/tmp/uv-cache \
    FLASK_APP=api.py \
    FLASK_ENV=development \
    FLASK_DEBUG=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY requirements.txt ./

RUN uv venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app && \
    chmod -R 755 /app

USER app

EXPOSE 5000



HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1


CMD ["python", "-m", "app.api"]