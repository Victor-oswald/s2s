FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface \
    TORCH_HOME=/runpod-volume/torch \
    VOLUME_PATH=/runpod-volume

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

# --- CRITICAL: CUDA 12.8 (cu128) is the minimum for sm_120 / RTX 5090 ---
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install the rest. Because torch is already present, pip should not downgrade it.
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

# Build-time sanity check: on a CPU builder this will show cuda=False,
# but the arch list should include sm_120 if the wheel is correct.
RUN python -c "import torch; \
    print('torch', torch.__version__); \
    print('cuda', torch.version.cuda); \
    print('arch list', torch.cuda.get_arch_list() if torch.cuda.is_available() else 'N/A (CPU)'); \
    import torchvision, torchaudio, transformers; \
    print('imports OK')"

COPY server.py /app/server.py
RUN mkdir -p /runpod-volume /workspace

EXPOSE 8000
CMD ["python", "-u", "server.py"]