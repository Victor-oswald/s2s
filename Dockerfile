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

# --- 4090 / L40S are both Ada Lovelace (sm_89) ---
# cu121 is plenty here; no need for the bleeding-edge cu128 build the 5090
# (sm_120 / Blackwell) required. This is also less likely to trip whatever
# kernel/wheel mismatch you hit on the 5090.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Everything else EXCEPT omnivoice-server first.
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    accelerate \
    safetensors \
    soundfile \
    numpy \
    python-multipart \
    faster-whisper

# Install omnivoice-server with --no-deps: it declares its own (unpinned)
# torch requirement, and letting pip resolve that is exactly what silently
# swapped out your pinned cu128 torch on the 5090 build with no error and
# no warning. --no-deps means it can only use the torch already installed
# above.
RUN pip install --no-cache-dir --no-deps omnivoice-server

# Re-assert the pinned torch build in case anything above touched it, then
# fail the BUILD loudly (pip check) instead of finding out at runtime.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip check

# Build-time sanity check: this runs on a CPU builder so cuda=False here is
# expected and NOT proof the GPU build is correct — the real check is
# log_gpu_diagnostics() in server.py at container startup, which runs with
# the actual GPU attached and will warn loudly if the arch doesn't match.
RUN python -c "import torch, torchvision, torchaudio, faster_whisper; \
    print('torch', torch.__version__, 'cuda build', torch.version.cuda); \
    print('imports OK')"

COPY server.py /app/server.py
RUN mkdir -p /runpod-volume /workspace

EXPOSE 8000
CMD ["python", "-u", "server.py"]