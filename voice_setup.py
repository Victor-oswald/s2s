"""
Handles getting a reference voice sample - either a path the user provides
to an existing file, or a fresh recording from their microphone - and
registers it with the OmniVoice server as a named profile.
"""

import os
import time
import wave
import numpy as np
import sounddevice as sd
import requests

import config


def record_voice_sample(duration_sec=8, samplerate=16000, save_path=None):
    """Records `duration_sec` seconds from the default microphone and saves it as a WAV file."""
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "recorded_voice_sample.wav")

    print(f"\nRecording will start in 2 seconds. Speak clearly for {duration_sec} seconds.")
    print("(Read a sentence or two out loud - a clear, natural sample works best.)")
    time.sleep(2)
    print("Recording...")

    audio = sd.rec(int(duration_sec * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")

    with wave.open(save_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    print(f"Saved sample to: {save_path}")
    return save_path


def get_reference_wav_path():
    """
    Determines the reference WAV to use:
    - If config.REF_WAV is set, use it directly (no prompt).
    - Otherwise, ask the user whether to record a new sample or provide a path.
    """
    if config.REF_WAV:
        if os.path.exists(config.REF_WAV):
            return config.REF_WAV
        print(f"Configured REF_WAV not found at '{config.REF_WAV}' - falling back to prompt.")

    print("\nVoice profile setup")
    print("  1) Record a new sample from my microphone")
    print("  2) Provide a path to an existing .wav file")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        return record_voice_sample()

    while True:
        path = input("Enter full path to your .wav file: ").strip().strip('"')
        if os.path.exists(path):
            return path
        print(f"File not found: '{path}'. Try again.")


def setup_cloned_voice(server_base_url, profile_id, wav_path, session=None):
    """Registers the reference audio file as a named profile on the OmniVoice server."""
    session = session or requests
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
            response = session.post(profile_url, data=data, files=files, timeout=15)
            if response.status_code in [200, 201]:
                print(f"Voice profile '{profile_id}' ready.")
                return True
            else:
                print(f"Voice profile server alert: {response.text}")
                return False
        except Exception as e:
            print(f"Failed to connect to OmniVoice voice registry endpoint: {e}")
            return False