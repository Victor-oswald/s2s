import os
import sys
import time
import io
import subprocess
import threading
import traceback
import warnings
import asyncio
import base64
from typing import Optional

import torch
import numpy as np
import httpx
import uvicorn
import runpod
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

# --- PATH & STORAGE CONFIGURATION ---
VOLUME_PATH = os.getenv("VOLUME_PATH", "/runpod-volume")
MODEL_DIR = os.getenv("MODEL_DIR", f"{VOLUME_PATH}/hub/models--openai--whisper-large-v3-turbo")
VOICE_DIR = os.getenv("VOICE_DIR", f"{VOLUME_PATH}/voices")

SERVER_BASE = os.getenv("OMNIVOICE_BASE_URL", "http://127.0.0.1:8880")
OMNIVOICE_URL = f"{SERVER_BASE}/v1/audio/speech"
PROFILE_URL = f"{SERVER_BASE}/v1/voices/profiles"

SAMPLE_RATE = 16000
SILENCE_DURATION_SEC = 0.3

# Diffusion step count for OmniVoice synthesis. Their docs: "Use 16 for
# faster inference" (default is 32). This is the single biggest lever for
# per-utterance GPU time — tune via env without a code change/redeploy.
TTS_NUM_STEP = int(os.getenv("OMNIVOICE_NUM_STEP", "16"))

# Ensure storage subdirectories exist (wrapped — a missing/unwritable volume
# mount should not crash the whole process at import time)
try:
    os.makedirs(VOICE_DIR, exist_ok=True)
except Exception:
    print(f"[Startup] WARNING: could not create VOICE_DIR '{VOICE_DIR}':")
    traceback.print_exc()

app = FastAPI(title="RunPod Speech-to-Speech Engine with Voice Cloning")

# Global variables
asr_pipe = None
http_client: Optional[httpx.AsyncClient] = None
omnivoice_process: Optional[subprocess.Popen] = None
omnivoice_available = False
asr_ready = False


# --- HEALTH CHECK (required for RunPod Load Balancer endpoints) ---

@app.get("/ping")
async def health_check():
    """
    RunPod's Load Balancer worker type polls this route to decide whether
    the worker is healthy enough to receive traffic. Must return 200 quickly.
    We report 200 as soon as the process is up — ASR/TTS readiness is
    reported separately in the body so you can debug without failing health.
    """
    return {
        "status": "healthy",
        "asr_ready": asr_ready,
        "omnivoice_available": omnivoice_available,
    }


# --- SUBPROCESS MANAGEMENT FOR OMNIVOICE-SERVER ---

def start_omnivoice_subprocess():
    """
    Launches omnivoice-server in the background as a subprocess.
    Never raises — a missing binary or bad launch should degrade the
    service (no TTS) rather than crash the whole FastAPI startup.
    """
    global omnivoice_process
    cmd = ["omnivoice-server", "--port", "8880","--device", "cuda"]
    print(f"[Subprocess] Starting OmniVoice server: {' '.join(cmd)}")

    try:
        omnivoice_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print("[Subprocess] FATAL: 'omnivoice-server' executable not found on PATH.")
        print("[Subprocess] Check that the omnivoice-server pip package installed a")
        print("[Subprocess] console-script entry point into an active/PATH'd env.")
        omnivoice_process = None
        return
    except Exception:
        print("[Subprocess] Unexpected error launching omnivoice-server:")
        traceback.print_exc()
        omnivoice_process = None
        return

    def stream_logs():
        try:
            for line in omnivoice_process.stdout:
                print(f"[OmniVoice] {line}", end='')
        except Exception:
            pass

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()


async def wait_for_omnivoice_ready(timeout_s=90):
    """Polls the local OmniVoice HTTP endpoint until up. Returns False on timeout
    or if the subprocess never started — never raises."""
    if omnivoice_process is None:
        print("[Subprocess] Skipping readiness check — omnivoice-server did not start.")
        return False

    print("[Subprocess] Waiting for OmniVoice server to become reachable...")
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            await http_client.get(SERVER_BASE, timeout=2.0)
            print("[Subprocess] OmniVoice server is reachable.")
            return True
        except (httpx.RequestError, httpx.HTTPError):
            await asyncio.sleep(1)
    print(f"[Subprocess] Failed to reach OmniVoice within {timeout_s}s.")
    return False


async def register_voice_profile(profile_id: str, wav_path: str) -> bool:
    """Registers an audio sample file with OmniVoice voice registry."""
    if not os.path.exists(wav_path):
        print(f"[Voice Registry] Reference audio file missing: {wav_path}")
        return False

    print(f"[Voice Registry] Registering voice profile: '{profile_id}'...")
    try:
        with open(wav_path, "rb") as f:
            files = {"ref_audio": (os.path.basename(wav_path), f, "audio/wav")}
            data = {"profile_id": profile_id, "overwrite": "true"}
            response = await http_client.post(PROFILE_URL, data=data, files=files, timeout=15.0)
            if response.status_code in [200, 201]:
                print(f"[Voice Registry] Profile '{profile_id}' successfully registered.")
                return True
            else:
                print(f"[Voice Registry] Server error ({response.status_code}): {response.text}")
                return False
    except Exception as e:
        print(f"[Voice Registry] Failed to register voice profile: {e}")
        return False


def load_local_whisper():
    print(f"[Whisper] Loading model from path: {MODEL_DIR}...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_identifier = MODEL_DIR if os.path.exists(MODEL_DIR) else "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_identifier,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=f"{VOLUME_PATH}/huggingface"
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_identifier, cache_dir=f"{VOLUME_PATH}/huggingface")

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        batch_size=1,
    )


# --- LIFECYCLE HOOKS ---

@app.on_event("startup")
async def startup_event():
    """
    IMPORTANT: an unhandled exception anywhere in this function will crash
    uvicorn's startup and exit the process — which is exactly what produces
    a fast, repeating "worker exited with exit code 1" crash loop on RunPod.
    Every stage below is isolated in its own try/except so a single failing
    component degrades the service instead of taking the whole worker down.
    """
    global asr_pipe, http_client, omnivoice_available, asr_ready

    http_client = httpx.AsyncClient(timeout=60.0)

    # 1. Boot omnivoice-server CLI tool in background thread (non-fatal)
    try:
        start_omnivoice_subprocess()
        omnivoice_available = await wait_for_omnivoice_ready()
    except Exception:
        print("[Startup] Non-fatal error during OmniVoice startup:")
        traceback.print_exc()
        omnivoice_available = False

    if not omnivoice_available:
        print("[Startup] Continuing without TTS support (OmniVoice unavailable).")

    # 2. Warm up ASR pipeline on CUDA (fatal if it fails — but at least we
    #    print a full traceback to the worker logs before exiting, instead
    #    of a bare exit code 1 with no context)
    try:
        asr_pipe = load_local_whisper()
        asr_ready = True
    except Exception:
        print("[Startup] FATAL: failed to load Whisper ASR pipeline:")
        traceback.print_exc()
        raise

    print("[Startup] System fully initialized and ready.")


@app.on_event("shutdown")
async def shutdown_event():
    global omnivoice_process
    if http_client:
        await http_client.aclose()

    if omnivoice_process and omnivoice_process.poll() is None:
        print("[Shutdown] Terminating OmniVoice process...")
        omnivoice_process.terminate()
        try:
            omnivoice_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            omnivoice_process.kill()


# --- VOICE MANAGEMENT ENDPOINTS ---

@app.post("/v1/voices/upload")
async def upload_custom_voice(
    profile_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Accepts multiform voice samples (.mp3, .wav, .m4a, .flac), saves them to network volume,
    and registers the voice with omnivoice-server.
    """
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )

    saved_filename = f"{profile_id}{file_ext}"
    local_file_path = os.path.join(VOICE_DIR, saved_filename)

    # Save audio stream to disk volume
    try:
        contents = await file.read()
        with open(local_file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write audio file to storage: {e}")

    # Register with the local OmniVoice registry
    success = await register_voice_profile(profile_id, local_file_path)

    if not success:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "File saved to volume, but voice profile registration failed."}
        )

    return {
        "status": "success",
        "profile_id": profile_id,
        "saved_path": local_file_path,
        "filename": saved_filename
    }


# --- STREAMING WEBSOCKET ENDPOINT ---

@app.websocket("/ws/speech-to-speech")
async def speech_to_speech_endpoint(websocket: WebSocket, voice_id: str = "eg"):
    """
    WebSocket endpoint accepting raw float32 PCM bytes.
    Routes transcribed text to the requested voice_id.

    Structured as two concurrent coroutines instead of one sequential loop:

      - `receiver()` does nothing but keep calling `websocket.receive_bytes()`
        and pushing frames onto a queue. This is what lets the client keep
        streaming mic audio (i.e. "listen") the entire time a previous
        utterance is being transcribed/synthesized/played back, instead of
        the socket read stalling until that turn finishes.
      - `processor()` pulls frames off the queue and does the actual
        silence-detection / ASR / TTS work, exactly as before. The Whisper
        call now runs in a worker thread (`asyncio.to_thread`) rather than
        inline on the event loop — previously that single blocking call
        froze the *entire* server (every connection, this one included)
        for its whole duration, which was the biggest chunk of the dead
        time between "you stop talking" and "it starts responding".

    A `send_lock` guards the socket so `send_json`/`send_bytes` calls from
    `processor()` never interleave with each other (there's only ever one
    writer, but the lock keeps this safe if you later add e.g. barge-in
    interrupts that write from elsewhere).
    """
    await websocket.accept()
    print(f"[WebSocket] Client connected. Active Voice ID: '{voice_id}'")

    audio_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
    send_lock = asyncio.Lock()

    async def send_json_safe(payload):
        async with send_lock:
            await websocket.send_json(payload)

    async def send_bytes_safe(payload):
        async with send_lock:
            await websocket.send_bytes(payload)

    async def receiver():
        try:
            while True:
                data = await websocket.receive_bytes()
                if data:
                    await audio_queue.put(data)
        except WebSocketDisconnect:
            pass
        finally:
            # Sentinel — unblocks processor()'s queue.get() so it can exit
            # cleanly once the client disconnects mid-turn.
            await audio_queue.put(None)

    async def processor():
        audio_buffer = []
        silent_chunks = 0
        SILENCE_THRESHOLD = 0.035

        while True:
            data = await audio_queue.get()
            if data is None:
                return

            chunk = np.frombuffer(data, dtype=np.float32)
            clean_chunk = chunk - np.mean(chunk)
            volume_norm = np.linalg.norm(clean_chunk) / np.sqrt(clean_chunk.size) if clean_chunk.size > 0 else 0
            chunk_duration = len(clean_chunk) / SAMPLE_RATE

            if volume_norm < SILENCE_THRESHOLD:
                if len(audio_buffer) > 0:
                    silent_chunks += 1
            else:
                silent_chunks = 0
                audio_buffer.append(chunk)

            current_silence_duration = silent_chunks * chunk_duration

            # Process when speech stops
            if len(audio_buffer) > 0 and current_silence_duration >= SILENCE_DURATION_SEC:
                t_utterance_end = time.time()
                recording = np.concatenate(audio_buffer, axis=0).flatten()
                audio_buffer.clear()
                silent_chunks = 0

                if len(recording) < SAMPLE_RATE * 0.3:
                    continue

                if asr_pipe is None:
                    await send_json_safe({
                        "event": "error",
                        "message": "ASR pipeline is not ready yet."
                    })
                    continue

                # 1. Transcribe audio using local Whisper — run off the
                # event loop so `receiver()` keeps draining the socket
                # (i.e. you can keep talking) while this turn transcribes.
                t_asr_start = time.time()
                result = await asyncio.to_thread(
                    asr_pipe,
                    recording,
                    generate_kwargs={
                        "language": "en",
                        "task": "transcribe",
                        "num_beams": 1,
                        "max_new_tokens": 128,
                        "repetition_penalty": 1.3,
                        "no_repeat_ngram_size": 3,
                    }
                )
                spoken_text = result["text"].strip()
                t_asr_end = time.time()

                if not spoken_text:
                    continue

                print(f"[STT] ({t_asr_end - t_asr_start:.2f}s): \"{spoken_text}\"")

                if not omnivoice_available:
                    await send_json_safe({
                        "event": "transcription",
                        "text": spoken_text,
                        "voice_id": voice_id,
                        "warning": "OmniVoice TTS is unavailable; transcript only.",
                        "latency": {"asr_sec": round(t_asr_end - t_asr_start, 3)}
                    })
                    continue

                # 2. Dispatch to OmniVoice HTTP engine with streaming
                # enabled — sentences land on the wire as each one finishes
                # generating instead of waiting for the whole reply. Chunks
                # come back as headerless raw PCM (matching the ASSUMED_*
                # constants in wavUtils.js), so we forward each one straight
                # to the client the moment it arrives — no more buffering
                # the full response before sending anything.
                payload = {
                    "model": "omnivoice",
                    "input": spoken_text,
                    "voice": voice_id,
                    "response_format": "pcm",
                    "stream": True,
                    "num_step": TTS_NUM_STEP,
                    # Keeps timbre stable across streamed sentence chunks —
                    # OmniVoice synthesizes each sentence independently when
                    # streaming, and non-zero temperature can drift voice
                    # from chunk to chunk otherwise.
                    "position_temperature": 0.0,
                }

                t_tts_start = time.time()
                t_first_chunk = None
                chunk_count = 0
                sent_transcription = False

                try:
                    async with http_client.stream("POST", OMNIVOICE_URL, json=payload) as response:
                        if response.status_code != 200:
                            body = await response.aread()
                            await send_json_safe({
                                "event": "error",
                                "message": f"OmniVoice server returned status {response.status_code}: {body[:200]!r}"
                            })
                            continue

                        async for pcm_chunk in response.aiter_bytes():
                            if not pcm_chunk:
                                continue
                            if t_first_chunk is None:
                                t_first_chunk = time.time()
                                # Send the transcript as soon as the first
                                # chunk lands rather than after everything
                                # finishes — the client shows it immediately.
                                await send_json_safe({
                                    "event": "transcription",
                                    "text": spoken_text,
                                    "voice_id": voice_id,
                                    "latency": {
                                        "asr_sec": round(t_asr_end - t_asr_start, 3),
                                        "tts_first_chunk_sec": round(t_first_chunk - t_tts_start, 3),
                                    }
                                })
                                sent_transcription = True
                            chunk_count += 1
                            await send_bytes_safe(pcm_chunk)
                except httpx.RequestError as e:
                    await send_json_safe({"event": "error", "message": f"OmniVoice request failed: {e}"})
                    continue

                t_tts_end = time.time()

                if not sent_transcription:
                    # Streamed zero bytes — surface something rather than
                    # leaving the client with no transcript at all.
                    await send_json_safe({
                        "event": "transcription",
                        "text": spoken_text,
                        "voice_id": voice_id,
                        "warning": "TTS produced no audio for this utterance.",
                        "latency": {"asr_sec": round(t_asr_end - t_asr_start, 3)}
                    })

                # Full per-stage breakdown — this is what tells you whether
                # a slow turn is socket/VAD, ASR, or TTS-bound instead of
                # guessing. "silence_wait" is the fixed SILENCE_DURATION_SEC
                # hangover baked into how utterance-end is detected.
                print(
                    "[Latency] "
                    f"silence_wait={SILENCE_DURATION_SEC:.2f}s "
                    f"asr={t_asr_end - t_asr_start:.2f}s "
                    f"tts_first_chunk={((t_first_chunk or t_tts_end) - t_tts_start):.2f}s "
                    f"tts_total={t_tts_end - t_tts_start:.2f}s "
                    f"chunks={chunk_count} "
                    f"end_of_speech_to_first_audio={((t_first_chunk or t_tts_end) - t_utterance_end):.2f}s"
                )

    receiver_task = asyncio.create_task(receiver())
    try:
        await processor()
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        traceback.print_exc()
    finally:
        receiver_task.cancel()
        try:
            await receiver_task
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass


# --- RUNPOD SERVERLESS HANDLER (queue-based mode only) ---

async def runpod_handler(job):
    """
    Handles payload execution when triggered via RunPod Async Serverless API (/run).
    Expected input format:
    {
        "input": {
            "text": "Text to synthesize",
            "voice_id": "eg"
        }
    }
    """
    job_input = job.get("input", {})
    text_input = job_input.get("text")
    voice_id = job_input.get("voice_id", "eg")

    if not text_input:
        return {"error": "No 'text' field provided in job payload input."}

    payload = {
        "model": "omnivoice",
        "input": text_input,
        "voice": voice_id,
        "response_format": "wav"
    }

    try:
        response = await http_client.post(OMNIVOICE_URL, json=payload, timeout=30.0)
        if response.status_code == 200:
            audio_b64 = base64.b64encode(response.content[44:]).decode('utf-8')
            return {
                "status": "success",
                "text": text_input,
                "voice_id": voice_id,
                "audio_base64": audio_b64
            }
        else:
            return {"error": f"OmniVoice engine returned HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # RunPod injects PORT (and PORT_HEALTH) for Load Balancer endpoints.
    # Always read it instead of hardcoding — a mismatched port is a common
    # cause of workers that "start fine" but never receive traffic.
    PORT = int(os.getenv("PORT", 8000))

    # If explicitly set to Serverless Execution Mode via Environment Variable
    if os.getenv("SERVE_RUNPOD_HANDLER", "false").lower() == "true":
        print("[RunPod] Booting in Serverless Worker Mode (queue-based)...")

        # Run startup lifecycle manually for serverless environments
        loop = asyncio.get_event_loop()
        loop.run_until_complete(startup_event())

        runpod.serverless.start({"handler": runpod_handler})
    else:
        # Default Mode: Load Balancer / dedicated HTTP+WebSocket server
        print(f"[RunPod] Booting in Load Balancer FastAPI/WebSocket Mode on port {PORT}...")
        uvicorn.run("server:app", host="0.0.0.0", port=PORT, workers=1)