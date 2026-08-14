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

# --- CRITICAL: re-pin torch + torchvision + torchaudio to a matched triplet ---
# torch==2.5.1's own pinned nvidia-cudnn-cu12==9.1.0.70 has been pruned from
# the cu121 wheel index — PyTorch's cu121 channel is being deprecated
# (current stable installer no longer even lists cu121 as an option, only
# 11.8/12.6/12.8). Move to the current stable triplet on a maintained
# channel instead of chasing an aging pin.
# cu126 wheels need host GPU driver >=525 — any RunPod GPU host in 2026
# will comfortably satisfy this; verify with `nvidia-smi` in the pod if
# torch.cuda.is_available() ever comes back False.
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Sanity check at build time: fail the build loudly here instead of
# discovering the mismatch later in a crash-looping worker. This imports
# every path that crashed in prior attempts: torchvision's custom ops
# (transformers' AutoProcessor path) and torchaudio's compiled extension
# (omnivoice's path).
RUN python -c "import torch, torchvision, torchaudio; \
    print('torch', torch.__version__); \
    print('torchvision', torchvision.__version__); \
    print('torchaudio', torchaudio.__version__); \
    print('CUDA available:', torch.cuda.is_available()); \
    from torchvision.ops import nms; \
    print('torchvision::nms OK'); \
    import transformers; \
    from transformers import AutoModelForSpeechSeq2Seq; \
    print('transformers AutoModelForSpeechSeq2Seq import OK')"

# Copy application script
COPY server.py /app/server.py

# Create mount points for RunPod network volume
RUN mkdir -p /runpod-volume /workspace

EXPOSE 8000

# Launch combined entrypoint script
CMD ["python", "-u", "server.py"]