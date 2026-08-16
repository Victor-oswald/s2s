import os
import time
import subprocess
import threading
import traceback
import warnings
import asyncio
from typing import Optional, List

import torch
import numpy as np
import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

# --- PATH & STORAGE CONFIGURATION ---
VOLUME_PATH = os.getenv("VOLUME_PATH", "/runpod-volume")
MODEL_DIR = os.getenv("MODEL_DIR", "large-v3-turbo")  # faster-whisper shorthand or local CT2 dir
VOICE_DIR = os.getenv("VOICE_DIR", f"{VOLUME_PATH}/voices")

SERVER_BASE = os.getenv("OMNIVOICE_BASE_URL", "http://127.0.0.1:8880")
OMNIVOICE_URL = f"{SERVER_BASE}/v1/audio/speech"
PROFILE_URL = f"{SERVER_BASE}/v1/voices/profiles"

SAMPLE_RATE = 16000
SILENCE_DURATION_SEC = 0.3

# --- INCREMENTAL / PARTIAL TRANSCRIPTION TUNING ---
# How often to re-transcribe the growing buffer while the user is still
# speaking. Lower = faster to start speaking back, but more GPU churn.
PARTIAL_INTERVAL_SEC = float(os.getenv("PARTIAL_INTERVAL_SEC", "0.7"))
# Trailing words treated as "not yet safe to commit" since they're the most
# likely to be rewritten by the next partial pass. Higher = fewer ASR
# revision artifacts, but later first-audio.
UNSTABLE_TAIL_WORDS = int(os.getenv("UNSTABLE_TAIL_WORDS", "3"))
# Don't bother running a partial pass on a sliver of audio.
MIN_PARTIAL_AUDIO_SEC = float(os.getenv("MIN_PARTIAL_AUDIO_SEC", "0.6"))
SENTENCE_END_CHARS = (".", "!", "?")

# --- PCM STREAM ALIGNMENT ---
# Bytes per sample in the PCM OmniVoice streams back. int16 -> 2, float32 -> 4.
# If audio still glitches after this fix, this is the first thing to check
# against whatever OmniVoice actually emits for response_format="pcm".
PCM_SAMPLE_WIDTH_BYTES = int(os.getenv("PCM_SAMPLE_WIDTH_BYTES", "2"))

# Diffusion step count for OmniVoice synthesis. Their docs: "Use 16 for
# faster inference" (default is 32). This is the single biggest lever for
# per-utterance GPU time — tune via env without a code change/redeploy.
TTS_NUM_STEP = int(os.getenv("OMNIVOICE_NUM_STEP", "16"))

try:
    os.makedirs(VOICE_DIR, exist_ok=True)
except Exception:
    print(f"[Startup] WARNING: could not create VOICE_DIR '{VOICE_DIR}':")
    traceback.print_exc()

app = FastAPI(title="Pod Speech-to-Speech Engine with Voice Cloning")

# Global variables
asr_model: Optional[WhisperModel] = None
http_client: Optional[httpx.AsyncClient] = None
omnivoice_process: Optional[subprocess.Popen] = None
omnivoice_available = False
asr_ready = False


# --- HEALTH CHECK ---

@app.get("/ping")
async def health_check():
    return {
        "status": "healthy",
        "asr_ready": asr_ready,
        "omnivoice_available": omnivoice_available,
    }


# --- GPU DIAGNOSTICS (runtime, not build time — build has no GPU) ---

def log_gpu_diagnostics():
    """
    Build-time checks can only prove the imports don't crash — they run on a
    CPU builder and always report cuda=False. This runs on the actual pod
    with the actual GPU attached, so it's the only place that can confirm
    you're really getting kernels for the GPU you rented, and not a torch
    build that got silently swapped in by some other package's dependency
    resolution (e.g. omnivoice-server pulling in its own torch).
    """
    print(f"[GPU] torch version: {torch.__version__}")
    print(f"[GPU] torch CUDA build: {torch.version.cuda}")
    print(f"[GPU] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        arch_list = torch.cuda.get_arch_list()
        print(f"[GPU] device: {name} (compute capability sm_{major}{minor})")
        print(f"[GPU] torch built with arch support: {arch_list}")
        # Only the MAJOR compute-capability generation needs a matching (or
        # lower, same-major) cubin/PTX entry — e.g. an Ada card (sm_89) runs
        # fine on a wheel that only lists sm_80/sm_86, because those are
        # forward-compatible within the same major (8.x) generation. Stock
        # PyTorch wheels routinely omit sm_89 entirely for this reason, so
        # checking for an EXACT "sm_89" match (as before) produces a false
        # positive on every 40-series/L40/L40S card even when torch is
        # perfectly correctly installed. Only warn if there's no same- or
        # lower-major entry at all, which would mean the wheel genuinely
        # can't target this GPU.
        same_major_supported = any(
            arch.startswith(f"sm_{major}") for arch in arch_list
        ) or any(
            int(arch.split("_")[1][0]) < major for arch in arch_list if arch.startswith("sm_")
        )
        if not same_major_supported:
            print(
                "[GPU] WARNING: this torch build has no cubin/PTX compatible "
                f"with compute capability {major}.{minor}. This usually means "
                "a dependency (often omnivoice-server) pulled in a mismatched "
                "torch wheel and overwrote the one you intended to run. "
                "Check `pip show torch` inside the pod."
            )
    else:
        print("[GPU] WARNING: CUDA not available — running on CPU.")


# --- SUBPROCESS MANAGEMENT FOR OMNIVOICE-SERVER ---

def start_omnivoice_subprocess():
    global omnivoice_process
    cmd = ["omnivoice-server", "--port", "8880", "--device", "cuda"]
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


def load_asr_model() -> WhisperModel:
    """
    faster-whisper / CTranslate2 backend instead of the raw transformers
    generate() loop — noticeably faster on GPU for the same weights, and
    matches the backend you're already using in the local Windows pipeline.
    """
    print(f"[Whisper] Loading faster-whisper model: {MODEL_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(
        MODEL_DIR,
        device=device,
        compute_type=compute_type,
        download_root=f"{VOLUME_PATH}/huggingface",
    )
    print("[Whisper] Model loaded.")
    return model


def _transcribe_sync(audio: np.ndarray) -> str:
    """Runs in a worker thread. Full transcript text for the given buffer."""
    segments, _info = asr_model.transcribe(
        audio,
        language="en",
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        vad_filter=False,  # we already do our own VAD/silence detection
    )
    return "".join(seg.text for seg in segments).strip()


def compute_committed_text(full_text: str, unstable_tail_words: int) -> str:
    """
    Splits off the trailing `unstable_tail_words` words (likely to be
    revised by the next partial pass) and returns only the portion of what's
    left up through the last completed clause — so we never hand OmniVoice
    a sentence fragment that ends mid-thought.
    """
    words = full_text.split()
    if len(words) <= unstable_tail_words:
        return ""
    committed = " ".join(words[:-unstable_tail_words]) if unstable_tail_words else " ".join(words)

    last_end = -1
    for ch in SENTENCE_END_CHARS:
        last_end = max(last_end, committed.rfind(ch))
    if last_end == -1:
        # No full sentence yet — allow a comma boundary so long run-on
        # sentences still get spoken incrementally instead of stalling.
        last_end = committed.rfind(",")
    if last_end == -1:
        return ""
    return committed[: last_end + 1].strip()


# --- LIFECYCLE HOOKS ---

@app.on_event("startup")
async def startup_event():
    global asr_model, http_client, omnivoice_available, asr_ready

    http_client = httpx.AsyncClient(timeout=60.0)
    log_gpu_diagnostics()

    try:
        start_omnivoice_subprocess()
        omnivoice_available = await wait_for_omnivoice_ready()
    except Exception:
        print("[Startup] Non-fatal error during OmniVoice startup:")
        traceback.print_exc()
        omnivoice_available = False

    if not omnivoice_available:
        print("[Startup] Continuing without TTS support (OmniVoice unavailable).")

    try:
        asr_model = load_asr_model()
        asr_ready = True
    except Exception:
        print("[Startup] FATAL: failed to load Whisper ASR model:")
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
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )

    saved_filename = f"{profile_id}{file_ext}"
    local_file_path = os.path.join(VOICE_DIR, saved_filename)

    try:
        contents = await file.read()
        with open(local_file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write audio file to storage: {e}")

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
    Three concurrent coroutines:

      - `receiver()` keeps draining the socket into `audio_queue` no matter
        what stage the rest of the pipeline is in.
      - `partial_transcriber()` owns VAD/silence detection AND periodically
        re-transcribes the growing buffer *while you're still talking*.
        As soon as a clause is "stable" (won't be revised by more audio),
        it's pushed onto `tts_queue` immediately — this is what lets
        synthesis start before you finish your sentence, instead of
        waiting for SILENCE_DURATION_SEC.
      - `tts_worker()` drains `tts_queue` in order and streams each segment
        to OmniVoice, forwarding PCM to the client as it arrives, with a
        byte-alignment buffer so stream chunk boundaries never split a
        sample across two WS frames (this was the source of the
        intermittent noise/garbage audio).
    """
    await websocket.accept()
    print(f"[WebSocket] Client connected. Active Voice ID: '{voice_id}'")

    audio_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
    tts_queue: "asyncio.Queue[Optional[tuple]]" = asyncio.Queue()
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
            await audio_queue.put(None)

    async def partial_transcriber():
        audio_buffer: List[np.ndarray] = []
        silent_chunks = 0
        SILENCE_THRESHOLD = 0.035
        last_partial_at = 0.0
        dispatched_text = ""
        utterance_id = 0
        partial_lock = asyncio.Lock()  # prevent overlapping passes on this connection

        async def run_partial_pass(is_final: bool):
            nonlocal dispatched_text
            if not audio_buffer:
                return
            recording = np.concatenate(audio_buffer, axis=0).flatten()
            if len(recording) < SAMPLE_RATE * MIN_PARTIAL_AUDIO_SEC and not is_final:
                return

            async with partial_lock:
                full_text = await asyncio.to_thread(_transcribe_sync, recording)
                if not full_text:
                    return

                if is_final:
                    committed = full_text  # full context, no tail trimming
                else:
                    committed = compute_committed_text(full_text, UNSTABLE_TAIL_WORDS)

                if len(committed) <= len(dispatched_text):
                    return
                if not committed.startswith(dispatched_text):
                    # ASR revised something already spoken — can't unspeak
                    # audio, so just skip this pass and try again next time.
                    print("[Partial] ASR revision conflict, skipping this pass.")
                    return

                delta = committed[len(dispatched_text):].strip()
                if delta:
                    print(f"[Partial{'/final' if is_final else ''}] +\"{delta}\"")
                    await tts_queue.put(("text", utterance_id, delta))
                    dispatched_text = committed

        while True:
            data = await audio_queue.get()
            if data is None:
                if audio_buffer:
                    await run_partial_pass(is_final=True)
                    await tts_queue.put(("turn_complete", utterance_id, None))
                await tts_queue.put(None)
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
            now = time.time()

            # Fire a partial pass periodically while still talking.
            if audio_buffer and (now - last_partial_at) >= PARTIAL_INTERVAL_SEC:
                last_partial_at = now
                asyncio.create_task(run_partial_pass(is_final=False))

            # End of utterance — final pass catches whatever the partial
            # passes hadn't committed yet, then reset for the next turn.
            if len(audio_buffer) > 0 and current_silence_duration >= SILENCE_DURATION_SEC:
                await run_partial_pass(is_final=True)
                await tts_queue.put(("turn_complete", utterance_id, None))
                audio_buffer.clear()
                silent_chunks = 0
                dispatched_text = ""
                utterance_id += 1

    async def tts_worker():
        while True:
            item = await tts_queue.get()
            if item is None:
                return

            kind, utterance_id, text = item

            if kind == "turn_complete":
                await send_json_safe({"event": "turn_complete", "utterance_id": utterance_id})
                continue

            if not omnivoice_available:
                await send_json_safe({
                    "event": "transcription",
                    "text": text,
                    "voice_id": voice_id,
                    "utterance_id": utterance_id,
                    "warning": "OmniVoice TTS is unavailable; transcript only.",
                })
                continue

            payload = {
                "model": "omnivoice",
                "input": text,
                "voice": voice_id,
                "response_format": "pcm",
                "stream": True,
                "num_step": TTS_NUM_STEP,
                "position_temperature": 0.0,
            }

            t_tts_start = time.time()
            t_first_chunk = None
            chunk_count = 0
            sent_transcription = False
            # Carries any trailing partial sample across chunk boundaries so
            # a sample is never split across two WS binary frames.
            pending_remainder = b""

            try:
                async with http_client.stream("POST", OMNIVOICE_URL, json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        await send_json_safe({
                            "event": "error",
                            "message": f"OmniVoice server returned status {response.status_code}: {body[:200]!r}",
                            "utterance_id": utterance_id,
                        })
                        continue

                    async for raw_chunk in response.aiter_bytes():
                        if not raw_chunk:
                            continue

                        buf = pending_remainder + raw_chunk
                        aligned_len = len(buf) - (len(buf) % PCM_SAMPLE_WIDTH_BYTES)
                        to_send, pending_remainder = buf[:aligned_len], buf[aligned_len:]
                        if not to_send:
                            continue

                        if t_first_chunk is None:
                            t_first_chunk = time.time()
                            await send_json_safe({
                                "event": "transcription",
                                "text": text,
                                "voice_id": voice_id,
                                "utterance_id": utterance_id,
                                "latency": {
                                    "tts_first_chunk_sec": round(t_first_chunk - t_tts_start, 3),
                                },
                            })
                            sent_transcription = True

                        chunk_count += 1
                        await send_bytes_safe(to_send)

                    if pending_remainder:
                        # A trailing partial sample with nowhere left to go —
                        # dropping a couple of bytes is inaudible; forwarding
                        # it unaligned is what caused the glitches.
                        print(f"[TTS] Dropped {len(pending_remainder)} trailing unaligned byte(s).")

            except httpx.RequestError as e:
                await send_json_safe({"event": "error", "message": f"OmniVoice request failed: {e}", "utterance_id": utterance_id})
                continue

            t_tts_end = time.time()
            if not sent_transcription:
                await send_json_safe({
                    "event": "transcription",
                    "text": text,
                    "voice_id": voice_id,
                    "utterance_id": utterance_id,
                    "warning": "TTS produced no audio for this segment.",
                })

            print(
                "[Latency] "
                f"segment=\"{text[:40]}\" "
                f"tts_first_chunk={((t_first_chunk or t_tts_end) - t_tts_start):.2f}s "
                f"tts_total={t_tts_end - t_tts_start:.2f}s "
                f"chunks={chunk_count}"
            )

    receiver_task = asyncio.create_task(receiver())
    transcriber_task = asyncio.create_task(partial_transcriber())
    try:
        await tts_worker()
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        traceback.print_exc()
    finally:
        receiver_task.cancel()
        transcriber_task.cancel()
        for t in (receiver_task, transcriber_task):
            try:
                await t
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass


if __name__ == "__main__":
    # Pod mode: dedicated long-running HTTP+WebSocket server. No RunPod
    # Serverless queue handler here — that code path (runpod.serverless.start)
    # only applies to Serverless Workers, not a persistent Pod, and pulling
    # in the `runpod` package for a Pod deployment is one more place a stray
    # dependency resolution could silently swap out your pinned torch build.
    PORT = int(os.getenv("PORT", 8000))
    print(f"[Server] Booting FastAPI/WebSocket server on port {PORT}...")
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, workers=1)