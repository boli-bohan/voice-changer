# Python Voice Changer Worker Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY voice_changer.py ./

# Expose worker port
EXPOSE 8001

# Run the FastAPI application
CMD ["uv", "run", "uvicorn", "voice_changer:app", "--host", "0.0.0.0", "--port", "8001"]
