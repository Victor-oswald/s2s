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

# --- CRITICAL: re-pin torch + torchvision to a matched, CUDA-built pair ---
# One of the packages above (transformers/accelerate/omnivoice-server) pulls
# in a torchvision version that doesn't match the CUDA-compiled torch this
# base image ships with, which breaks custom-op registration
# (RuntimeError: operator torchvision::nms does not exist) and crashes the
# process at import time, before any application code even runs.
# Reinstalling explicitly from the official PyTorch CUDA wheel index, as the
# LAST pip step, guarantees a compatible pair and stops anything upstream
# from silently swapping it out again.
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Sanity check at build time: fail the build loudly here instead of
# discovering the mismatch later in a crash-looping worker.
RUN python -c "import torch, torchvision; \
    print('torch', torch.__version__, 'torchvision', torchvision.__version__); \
    print('CUDA available:', torch.cuda.is_available()); \
    from torchvision.ops import nms; \
    print('torchvision::nms OK')"

# Copy application script
COPY server.py /app/server.py

# Create mount points for RunPod network volume
RUN mkdir -p /runpod-volume /workspace

EXPOSE 8000

# Launch combined entrypoint script
CMD ["python", "-u", "server.py"]