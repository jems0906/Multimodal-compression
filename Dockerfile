# Use Python 3.10 slim image
FROM python:3.10-slim

# Best-practice env vars: disable .pyc files, force stdout/stderr flush
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Optional HuggingFace token (pass via --build-arg or docker-compose env)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Use CPU-only PyTorch wheels in containers to keep image size manageable
# and avoid pulling large CUDA runtimes during build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.5.1+cpu" \
    "torchvision==0.20.1+cpu" \
    "torchaudio==2.5.1+cpu"
RUN pip install --no-cache-dir --default-timeout=600 --retries 10 -r requirements.txt
# Work around occasional corrupted wheels in cached environments.
RUN pip install --no-cache-dir --force-reinstall \
    "uvicorn==0.30.6" \
    "fastapi==0.121.1" \
    "pydantic==2.12.5" \
    "typing-inspection==0.4.2" \
    "PyYAML==6.0.3" \
    "omegaconf==2.3.0" \
    "hydra-core==1.3.2"

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Pre-create output directories so they exist even without a volume mount
RUN mkdir -p experiments/compressed_model experiments/finetuned_model \
             experiments/onnx experiments/reports

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Write startup script using printf so \n produces real newlines
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'if [ "$1" = "dashboard" ]; then' \
    '    exec streamlit run src/ui/dashboard.py --server.port 8501 --server.address 0.0.0.0' \
    'elif [ "$1" = "api" ]; then' \
    '    exec uvicorn src.ui.api:app --host 0.0.0.0 --port 8000' \
    'else' \
    '    echo "Usage: docker run <image> [dashboard|api]"' \
    '    exit 1' \
    'fi' > /app/start.sh && chmod +x /app/start.sh

# Health check that works for either runtime mode:
# - API mode responds on 8000 (/health)
# - Dashboard mode responds on 8501 (/)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || \
        python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501', timeout=3)" || exit 1

# Default command
CMD ["/app/start.sh", "dashboard"]
