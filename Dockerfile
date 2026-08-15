# Use a clean, non-conda Python image
FROM python:3.11-slim-bookworm

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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --- Install PyTorch FIRST (CUDA 12.6, includes sm_120 / Blackwell support) ---
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# --- Now install downstream packages; they will see torch is already satisfied ---
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

# Sanity check: imports must succeed. CUDA available will be False at build time
# (CPU builder), but on the GPU pod it will pick up the correct 2.7.0+cu126.
RUN python -c "import torch, torchvision, torchaudio; \
    print('torch', torch.__version__); \
    print('torchvision', torchvision.__version__); \
    print('torchaudio', torchaudio.__version__); \
    print('CUDA available at build:', torch.cuda.is_available()); \
    from torchvision.ops import nms; \
    print('torchvision::nms OK'); \
    from transformers import AutoModelForSpeechSeq2Seq; \
    print('transformers AutoModelForSpeechSeq2Seq import OK')"

COPY server.py /app/server.py

RUN mkdir -p /runpod-volume /workspace

EXPOSE 8000

CMD ["python", "-u", "server.py"]