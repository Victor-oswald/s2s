import os
import sys
import time
import io
import subprocess
import threading
import warnings
import asyncio
from typing import Optional

import torch
import numpy as np
import httpx
import uvicorn
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

# Ensure storage subdirectories exist
os.makedirs(VOICE_DIR, exist_ok=True)

app = FastAPI(title="RunPod Speech-to-Speech Engine with Voice Cloning")

# Global variables
asr_pipe = None
http_client: Optional[httpx.AsyncClient] = None
omnivoice_process: Optional[subprocess.Popen] = None


# --- SUBPROCESS MANAGEMENT FOR OMNIVOICE-SERVER ---

def start_omnivoice_subprocess():
    """Launches omnivoice-server in the background as a subprocess."""
    global omnivoice_process
    cmd = ["omnivoice-server", "--port", "8880"]
    print(f"[Subprocess] Starting OmniVoice server: {' '.join(cmd)}")

    omnivoice_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def stream_logs():
        for line in omnivoice_process.stdout:
            print(f"[OmniVoice] {line}", end='')

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()


async def wait_for_omnivoice_ready(timeout_s=90):
    """Polls the local OmniVoice HTTP endpoint until up."""
    print("[Subprocess] Waiting for OmniVoice server to become reachable...")
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = await http_client.get(SERVER_BASE, timeout=2.0)
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
    global asr_pipe, http_client
    http_client = httpx.AsyncClient(timeout=60.0)

    # 1. Boot omnivoice-server CLI tool in background thread
    start_omnivoice_subprocess()
    server_ready = await wait_for_omnivoice_ready()

    if not server_ready:
        print("[Error] OmniVoice server failed to start. Continuing without TTS support.")

    # 2. Warm up ASR pipeline on CUDA
    asr_pipe = load_local_whisper()
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
    WebSocket endpoint accepting raw float32/int16 PCM bytes.
    Routes transcribed text to the requested voice_id.
    """
    await websocket.accept()
    print(f"[WebSocket] Client connected. Active Voice ID: '{voice_id}'")

    audio_buffer = []
    silent_chunks = 0
    SILENCE_THRESHOLD = 0.035

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

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
                recording = np.concatenate(audio_buffer, axis=0).flatten()
                audio_buffer.clear()
                silent_chunks = 0

                if len(recording) < SAMPLE_RATE * 0.3:
                    continue

                # 1. Transcribe audio using local Whisper
                t_asr_start = time.time()
                result = asr_pipe(
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

                # 2. Dispatch to OmniVoice HTTP engine
                payload = {
                    "model": "omnivoice",
                    "input": spoken_text,
                    "voice": voice_id,
                    "response_format": "wav"
                }

                t_tts_start = time.time()
                response = await http_client.post(OMNIVOICE_URL, json=payload)
                t_tts_end = time.time()

                if response.status_code == 200:
                    # Send transcript back to client
                    await websocket.send_json({
                        "event": "transcription",
                        "text": spoken_text,
                        "voice_id": voice_id,
                        "latency": {
                            "asr_sec": round(t_asr_end - t_asr_start, 3),
                            "tts_sec": round(t_tts_end - t_tts_start, 3)
                        }
                    })

                    # Send raw audio binary (stripping 44-byte WAV header)
                    audio_payload = response.content[44:]
                    await websocket.send_bytes(audio_payload)
                else:
                    await websocket.send_json({
                        "event": "error",
                        "message": f"OmniVoice server returned status {response.status_code}"
                    })

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket Error] {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)