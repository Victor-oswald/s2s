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

# A pip CONSTRAINTS file (not --no-deps) is the right tool here: pip still
# resolves and installs every real dependency of omnivoice-server normally
# (omnivoice, platformdirs, pydantic-settings, etc.) but is *forbidden* from
# picking a different torch/torchvision/torchaudio than what's pinned below,
# no matter what version omnivoice-server itself asks for. --no-deps was
# too blunt — it blocked torch but also silently skipped every other
# dependency, which is what broke the last build.
RUN printf "torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\n" > /tmp/torch-constraints.txt

# Everything except omnivoice-server (which we build from patched source
# below), resolved together under the torch constraint. --extra-index-url
# so cu121 wheels are still reachable for torch's own transitive re-check,
# while everything else still comes from PyPI as usual.
RUN pip install --no-cache-dir \
    --constraint /tmp/torch-constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    accelerate \
    safetensors \
    soundfile \
    numpy \
    python-multipart \
    faster-whisper

# --- omnivoice-server, patched with voice-clone embedding caching ---
# Stock omnivoice-server re-runs OmniVoice's full reference-audio encode
# (silence trim + RMS norm + a real GPU audio_tokenizer.encode() pass) on
# EVERY /v1/audio/speech call, even for the same cached voice profile —
# this was the fixed ~1.1-1.2s tax on every TTS request regardless of text
# length. The three files below add a VoiceClonePrompt cache (in-memory +
# persisted to disk) keyed by profile_id, built once and reused on every
# subsequent call, and de-throttle the previous "torch.cuda.empty_cache()
# after every single request" behavior to a periodic cleanup instead.
# Shipped as full file replacements (not a .patch) — git apply's patch
# format is fragile across git versions/line-ending handling, and a full
# file swap has nothing to corrupt.
RUN git clone --depth 1 https://github.com/maemreyo/omnivoice-server.git /opt/omnivoice-server-src
COPY omnivoice_server_patch/profiles.py /opt/omnivoice-server-src/omnivoice_server/services/profiles.py
COPY omnivoice_server_patch/inference.py /opt/omnivoice-server-src/omnivoice_server/services/inference.py
COPY omnivoice_server_patch/speech.py /opt/omnivoice-server-src/omnivoice_server/routers/speech.py
RUN cd /opt/omnivoice-server-src \
    && pip install --no-cache-dir \
        --constraint /tmp/torch-constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -e .

# Fail the BUILD loudly if the constraint didn't hold, instead of finding
# out at runtime from garbled audio.
RUN pip check && python -c "import torch; assert torch.__version__.startswith('2.5.1'), torch.__version__"

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