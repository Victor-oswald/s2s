# Use PyTorch runtime base image with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface \
    TORCH_HOME=/runpod-volume/torch \
    VOLUME_PATH=/runpod-volume

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip & build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies including FastAPI, async HTTP, omnivoice-server, and runpod
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    transformers \
    accelerate \
    safetensors \
    soundfile \
    numpy \
    python-multipart \
    omnivoice-server \
    runpod

# Copy application script
COPY server.py /app/server.py

# Create mount points for RunPod network volume
RUN mkdir -p /runpod-volume /workspace

EXPOSE 8000

# Launch combined entrypoint script
CMD ["python", "-u", "server.py"]