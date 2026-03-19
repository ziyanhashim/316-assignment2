FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (model downloads at startup from HuggingFace Hub)
COPY configs/ configs/
COPY scripts/ scripts/

# Expose API port
EXPOSE 5000

# Health check (python:3.10-slim doesn't include curl)
HEALTHCHECK --start-period=300s --interval=30s --timeout=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# HF_TOKEN is loaded from .env file: docker run --env-file .env ...
CMD ["python", "scripts/serve.py"]
