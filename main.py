import numpy as np
import requests
import sounddevice as sd
import torch
import queue
import sys
import time
import os
import warnings
from collections import deque
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

# --- CONFIGURATION ---
SERVER_BASE = "http://127.0.0.1:8880"
OMNIVOICE_URL = f"{SERVER_BASE}/v1/audio/speech"
SAMPLE_RATE = 16000
SILENCE_DURATION_SEC = 0.3          # was 1.2 — VAD-like responsiveness with lookback
POST_PLAY_COOLDOWN_SEC = 0.5        # ignore mic for 500 ms after TTS finishes

# --- VOICE CLONING PARAMETERS ---
VOICE_ID = "eg"
REF_WAV = r"C:\Users\oswal\Downloads\test.wav"

# --- PATH TO YOUR LOCAL WHISPER MODEL ---
LOCAL_MODEL_PATH = r"C:\Users\oswal\.cache\huggingface\hub\models--openai--whisper-large-v3-turbo\snapshots\41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

audio_queue = queue.Queue()
http_session = requests.Session()


def load_local_whisper():
    print(f"Loading local Whisper Turbo from: {LOCAL_MODEL_PATH}...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        batch_size=1,
    )
    return asr_pipe


def setup_cloned_voice(server_base_url, profile_id, wav_path):
    profile_url = f"{server_base_url}/v1/voices/profiles"

    if not os.path.exists(wav_path):
        print(f"Reference audio file not found at: '{wav_path}'")
        print("The system will run using the OmniVoice server's default fallback voice.")
        return False

    print(f"Registering reference voice profile: '{profile_id}'...")
    with open(wav_path, "rb") as f:
        files = {"ref_audio": (os.path.basename(wav_path), f, "audio/wav")}
        data = {"profile_id": profile_id, "overwrite": "true"}
        try:
            response = http_session.post(profile_url, data=data, files=files, timeout=10)
            if response.status_code in [200, 201]:
                print(f"Voice profile '{profile_id}' ready.")
                return True
            else:
                print(f"Voice profile server alert: {response.text}")
                return False
        except Exception as e:
            print(f"Failed to connect to OmniVoice voice registry endpoint: {e}")
            return False


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Hardware Status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def main():
    pipe = load_local_whisper()
    setup_cloned_voice(SERVER_BASE, VOICE_ID, REF_WAV)

    # --- CALIBRATION (shortened to 2 s) ---
    print("Calibration: please remain quiet for 2 seconds...")
    calib_volumes = []

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=1024):
        start_time = time.time()
        while time.time() - start_time < 2.0:
            try:
                data_chunk = audio_queue.get(timeout=0.1)
                clean_chunk = data_chunk - np.mean(data_chunk)
                volume_norm = np.linalg.norm(clean_chunk) / np.sqrt(clean_chunk.size)
                calib_volumes.append(volume_norm)
            except queue.Empty:
                continue

        while not audio_queue.empty():
            audio_queue.get_nowait()

    avg_noise = np.mean(calib_volumes) if calib_volumes else 0.02
    SILENCE_THRESHOLD = round(avg_noise + 0.015, 4)
    print(f"Calibration complete. Noise floor: {avg_noise:.4f}. Threshold: {SILENCE_THRESHOLD}")

    print("System ready. Start speaking...")

    # Ring buffer holds last 0.5 s so we don't chop the start of speech
    lookback_buffer = deque(maxlen=int(SAMPLE_RATE * 0.5 / 1024))
    audio_buffer = []
    silent_chunks = 0
    is_recording = False
    tts_cooldown_until = 0.0

    last_status_print = 0
    STATUS_PRINT_INTERVAL = 0.5

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=1024):
        while True:
            try:
                data_chunk = audio_queue.get(timeout=0.1)

                # Discard mic input while TTS echo is decaying
                if time.time() < tts_cooldown_until:
                    continue

                clean_chunk = data_chunk - np.mean(data_chunk)
                volume_norm = np.linalg.norm(clean_chunk) / np.sqrt(clean_chunk.size)
                chunk_duration = len(clean_chunk) / SAMPLE_RATE

                if not is_recording:
                    lookback_buffer.append(data_chunk)

                current_silence_duration = silent_chunks * chunk_duration

                now = time.time()
                if now - last_status_print > STATUS_PRINT_INTERVAL:
                    print(
                        f"Vol: {volume_norm:.3f}  Thresh: {SILENCE_THRESHOLD}  "
                        f"Rec: {is_recording}  Quiet: {current_silence_duration:.1f}s",
                        end='\r'
                    )
                    last_status_print = now

                if volume_norm < SILENCE_THRESHOLD:
                    if is_recording:
                        silent_chunks += 1
                        audio_buffer.append(data_chunk)
                else:
                    if not is_recording:
                        print("\nListening...")
                        is_recording = True
                        # Prepend lookback so we keep the audio that triggered detection
                        audio_buffer = list(lookback_buffer)
                        lookback_buffer.clear()
                        silent_chunks = 0
                    silent_chunks = 0
                    audio_buffer.append(data_chunk)

                if is_recording and (current_silence_duration >= SILENCE_DURATION_SEC):
                    t_finalize = time.time()

                    recording = np.concatenate(audio_buffer, axis=0).flatten()
                    audio_buffer = []
                    silent_chunks = 0
                    is_recording = False
                    lookback_buffer.clear()

                    # Skip near-empty clips
                    if len(recording) < SAMPLE_RATE * 0.3:
                        continue

                    # --- TRANSCRIBE ---
                    t_transcribe_start = time.time()
                    result = pipe(
                        recording,
                        generate_kwargs={
                            "language": "en",
                            "task": "transcribe",
                            "num_beams": 1,
                            "max_new_tokens": 128,
                            "max_length": None,          # <-- kills the warning
                            "repetition_penalty": 1.3,
                            "no_repeat_ngram_size": 3,
                        }
                    )
                    t_transcribe_end = time.time()
                    spoken_text = result["text"].strip()

                    if not spoken_text:
                        continue

                    print(f'\nSTT: "{spoken_text}"')

                    # --- SEND TO TTS SERVER ---
                    t_send = time.time()
                    payload = {
                        "model": "omnivoice",
                        "input": spoken_text,
                        "voice": VOICE_ID,
                        "response_format": "wav"
                    }

                    try:
                        t0 = time.time()
                        response = http_session.post(
                            OMNIVOICE_URL, json=payload, timeout=60
                        )
                        t1 = time.time()

                        # This line proves the text is dispatched instantly
                        print(
                            f"[TIMING] whisper={(t_transcribe_end - t_transcribe_start):.2f}s | "
                            f"post_dispatch={(t0 - t_transcribe_end):.3f}s | "
                            f"server+network={(t1 - t0):.2f}s | "
                            f"turn_total={(t1 - t_finalize):.2f}s"
                        )

                        if response.status_code == 200:
                            print(f"TTS ok in {(t1 - t0):.2f}s, playing...")

                            audio_data = np.frombuffer(response.content[44:], dtype=np.int16)

                            sd.play(audio_data, samplerate=24000)
                            sd.wait()

                            # Clear anything the mic recorded during playback
                            while not audio_queue.empty():
                                audio_queue.get_nowait()

                            # Brief cooldown so TTS tail doesn't re-trigger
                            tts_cooldown_until = time.time() + POST_PLAY_COOLDOWN_SEC
                        else:
                            print(f"OmniVoice server error: {response.status_code} {response.text}")
                    except requests.exceptions.RequestException as req_err:
                        print(f"Network error: {req_err}")
                    except Exception as e:
                        print(f"API error: {e}")

            except queue.Empty:
                continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSession terminated.")
        sys.exit(0)