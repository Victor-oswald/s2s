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
# Two separate problems showed up here:
#   1. transformers (installed unpinned above) requires torch>=2.5, so
#      pinning torch to the base image's 2.3.0 makes transformers disable
#      its PyTorch backend entirely.
#   2. omnivoice-server pulls in its own torchaudio dependency; left
#      unpinned, pip resolved a torchaudio build compiled against CUDA 13
#      (libcudart.so.13), which doesn't exist in this CUDA 12.1 image.
# Fix: pin all THREE packages together as an official matched triplet built
# for cu121, as the LAST pip step, so nothing upstream can swap any one of
# them out independently again.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

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