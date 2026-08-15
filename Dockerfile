# Use PyTorch runtime base image with CUDA 12.8
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

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

# --- CRITICAL: Install PyTorch 2.8.0 with CUDA 12.8 support for RTX 5090 ---
# RTX 5090 requires compute capability sm_120 which is supported in PyTorch 2.7.1+
# PyTorch 2.8.0 with CUDA 12.8 provides full support for RTX 5090
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Sanity check at build time: fail the build loudly here instead of
# discovering the mismatch later in a crash-looping worker.
RUN python -c "import torch, torchvision, torchaudio; \
    print('torch', torch.__version__); \
    print('torchvision', torchvision.__version__); \
    print('torchaudio', torchaudio.__version__); \
    print('CUDA available:', torch.cuda.is_available()); \
    if torch.cuda.is_available(): \
        print('CUDA compute capability:', torch.cuda.get_device_capability(0)); \
        print('CUDA device count:', torch.cuda.device_count()); \
        print('CUDA device name:', torch.cuda.get_device_name(0)); \
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