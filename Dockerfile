# ==============================================================================
# Lumen Prediction API — Production Dockerfile for Google Cloud Run
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logs for Cloud Run
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install runtime system dependencies:
# - ffmpeg: required by scenedetect for video shot boundaries
# - libgl1 & libglib2.0-0: required for headless OpenCV image processing
# - curl: for Cloud Run container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for optimal layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY api.py .
COPY shot_pipeline.py .
COPY src/ ./src/
COPY data/ ./data/

# Cloud Run healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/v1/diagnostics/model || exit 1

EXPOSE 8080

# Cloud Run passes PORT as an environment variable (default: 8080)
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1
