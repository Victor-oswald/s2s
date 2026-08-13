"""
Single entry point for the whole pipeline:

  1. Starts `omnivoice-server` as a subprocess with the tuned settings.
  2. Waits until it's actually ready to receive requests.
  3. Sets up the voice profile (record new / use existing / uploaded file).
  4. Sends one warmup request so the ASR submodel is loaded before real traffic hits it.
  5. Runs the main STT -> TTS loop, routing audio to your virtual cable (for OBS)
     and optionally your speakers too.

Run this instead of main.py directly:

    python launcher.py
"""

import subprocess
import sys
import time
import threading
import warnings

import requests
import numpy as np
import sounddevice as sd
import torch
import queue
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False
    print("pycaw not installed - volume ducking disabled. Run: pip install pycaw comtypes")

import config
from voice_setup import get_reference_wav_path, setup_cloned_voice

warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

LOCAL_MODEL_PATH = r"C:\Users\oswal\.cache\huggingface\hub\models--openai--whisper-large-v3-turbo\snapshots\41f01f3fe87f28c78e2fbf8b568835947dd65ed9"

audio_queue = queue.Queue()
http_session = requests.Session()

server_process = None

# Set while TTS audio is playing back. The mic callback checks this and
# drops incoming audio entirely during that window - otherwise the mic
# picks up the speaker output (or cable bleed) and re-transcribes its
# own voice, causing an endless feedback loop.
is_playing_back = threading.Event()


def start_server():
    """Launches omnivoice-server as a subprocess and streams its output to this console."""
    global server_process
    cmd = ["omnivoice-server"] + config.SERVER_STARTUP_ARGS
    print(f"Starting server: {' '.join(cmd)}")

    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream server output on a background thread so we can see logs
    # while still being able to poll for readiness in the main thread.
    def stream_output():
        for line in server_process.stdout:
            print(f"[server] {line}", end='')

    t = threading.Thread(target=stream_output, daemon=True)
    t.start()
    return server_process


def wait_for_server_ready(timeout_s=90):
    """Polls the server's HTTP endpoint until it responds, instead of guessing based on logs."""
    print("Waiting for server to become reachable...")
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            # Any response (even 404/405) means the HTTP server is up.
            r = requests.get(config.SERVER_BASE, timeout=2)
            print("Server is reachable.")
            return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    print(f"Server did not become reachable within {timeout_s}s.")
    return False


def warmup_server(voice_id):
    """Sends one throwaway request so the ASR submodel loads before real traffic arrives."""
    print("Warming up server (first request can take 30-60s)...")
    t0 = time.time()
    try:
        r = http_session.post(
            config.OMNIVOICE_URL,
            json={"model": "omnivoice", "input": "warmup", "voice": voice_id, "response_format": "wav"},
            timeout=120,
        )
        print(f"Warmup done in {time.time()-t0:.1f}s, status {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"Warmup failed: {e}")
        return False


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


def get_speaker_volume_interface():
    """Gets the Windows volume control interface for the default playback device."""
    if not PYCAW_AVAILABLE:
        return None
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return interface.QueryInterface(IAudioEndpointVolume)
    except Exception as e:
        print(f"Could not get volume interface: {e}")
        return None


def duck_speaker_volume(target_level=0.05):
    """
    Lowers the default speaker's system volume to `target_level` (0.0-1.0).
    Returns the previous volume so it can be restored later.
    """
    vol = get_speaker_volume_interface()
    if vol is None:
        return None
    try:
        previous = vol.GetMasterVolumeLevelScalar()
        vol.SetMasterVolumeLevelScalar(target_level, None)
        return previous
    except Exception as e:
        print(f"Could not duck volume: {e}")
        return None


def restore_speaker_volume(previous_level):
    """Restores the speaker volume to what it was before ducking."""
    if previous_level is None:
        return
    vol = get_speaker_volume_interface()
    if vol is None:
        return
    try:
        vol.SetMasterVolumeLevelScalar(previous_level, None)
    except Exception as e:
        print(f"Could not restore volume: {e}")


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Hardware Status: {status}", file=sys.stderr)
    if is_playing_back.is_set():
        # Drop mic input entirely while TTS is playing - prevents the
        # mic from hearing the speaker output and re-transcribing it.
        return
    audio_queue.put(indata.copy())


def play_audio(audio_data):
    """Plays to the virtual cable (for OBS) and optionally to local speakers too."""
    is_playing_back.set()

    # Extra layer of protection on top of mic-muting: physically lower the
    # speaker volume while this plays, in case something outside our
    # control (e.g. OBS's own audio monitoring) is also pushing sound out.
    previous_volume = duck_speaker_volume(target_level=0.05)

    try:
        sd.play(audio_data, samplerate=config.TTS_SAMPLE_RATE, device=config.CABLE_OUTPUT_DEVICE_INDEX)
        sd.wait()

        if config.PLAY_TO_LOCAL_MONITOR:
            sd.play(audio_data, samplerate=config.TTS_SAMPLE_RATE, device=config.LOCAL_MONITOR_DEVICE_INDEX)
            sd.wait()
    finally:
        restore_speaker_volume(previous_volume)
        # Small grace period after playback stops - mic/room echo tail
        # and buffered driver latency can otherwise still leak through
        # for a few hundred ms right as we re-enable capture.
        time.sleep(0.3)
        is_playing_back.clear()
        # Drop anything that slipped into the queue during playback.
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break


def run_main_loop(pipe, voice_id):
    silence_threshold = 0.02

    print("Calibration: please remain quiet for 3 seconds...")
    calib_volumes = []

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=config.SAMPLE_RATE, blocksize=1024):
        start_time = time.time()
        while time.time() - start_time < 3.0:
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
    silence_threshold = round(avg_noise + 0.025, 4)
    print(f"Calibration complete. Noise floor: {avg_noise:.4f}. Threshold: {silence_threshold}")
    print("System ready. Start speaking...")

    audio_buffer = []
    silent_chunks = 0
    is_recording = False
    last_status_print = 0
    STATUS_PRINT_INTERVAL = 0.5

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=config.SAMPLE_RATE, blocksize=1024):
        while True:
            try:
                data_chunk = audio_queue.get(timeout=0.1)
                clean_chunk = data_chunk - np.mean(data_chunk)
                volume_norm = np.linalg.norm(clean_chunk) / np.sqrt(clean_chunk.size)
                chunk_duration = len(clean_chunk) / config.SAMPLE_RATE
                current_silence_duration = silent_chunks * chunk_duration

                now = time.time()
                if now - last_status_print > STATUS_PRINT_INTERVAL:
                    print(
                        f"Vol: {volume_norm:.3f} Thresh: {silence_threshold} "
                        f"Rec: {is_recording} Quiet: {current_silence_duration:.1f}s",
                        end='\r'
                    )
                    last_status_print = now

                if volume_norm < silence_threshold:
                    if is_recording:
                        silent_chunks += 1
                        audio_buffer.append(data_chunk)
                else:
                    if not is_recording:
                        print("\nListening...")
                        is_recording = True
                    silent_chunks = 0
                    audio_buffer.append(data_chunk)

                if is_recording and (current_silence_duration >= config.SILENCE_DURATION_SEC):
                    recording = np.concatenate(audio_buffer, axis=0).flatten()
                    audio_buffer = []
                    silent_chunks = 0
                    is_recording = False

                    if len(recording) < config.SAMPLE_RATE * 0.3:
                        continue

                    result = pipe(
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
                    if not spoken_text:
                        continue

                    print(f'STT: "{spoken_text}"')

                    payload = {
                        "model": "omnivoice",
                        "input": spoken_text,
                        "voice": voice_id,
                        "response_format": "wav"
                    }

                    try:
                        t0 = time.time()
                        response = http_session.post(config.OMNIVOICE_URL, json=payload, timeout=60)
                        elapsed = time.time() - t0

                        if response.status_code == 200:
                            print(f"TTS ok in {elapsed:.2f}s, playing...")
                            audio_data = np.frombuffer(response.content[44:], dtype=np.int16)

                            while not audio_queue.empty():
                                audio_queue.get_nowait()

                            play_audio(audio_data)
                        else:
                            print(f"OmniVoice server error: {response.status_code} {response.text}")
                    except requests.exceptions.RequestException as req_err:
                        print(f"Network error: {req_err}")
                    except Exception as e:
                        print(f"API error: {e}")

            except queue.Empty:
                continue


def shutdown():
    global server_process
    if server_process and server_process.poll() is None:
        print("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()


def main():
    start_server()

    if not wait_for_server_ready(config.SERVER_READY_TIMEOUT_S):
        print("Aborting - server never came up.")
        shutdown()
        sys.exit(1)

    ref_wav = get_reference_wav_path()
    setup_cloned_voice(config.SERVER_BASE, config.VOICE_ID, ref_wav, session=http_session)

    if not warmup_server(config.VOICE_ID):
        print("Warmup failed - the server may still work, but the first real request may be slow or fail.")

    pipe = load_local_whisper()

    try:
        run_main_loop(pipe, config.VOICE_ID)
    except KeyboardInterrupt:
        print("\nSession terminated.")
    finally:
        shutdown()


if __name__ == "__main__":
    main()