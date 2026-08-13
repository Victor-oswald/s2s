"""
Central configuration for the voice pipeline.
Edit values here rather than digging through the other scripts.
"""

# --- SERVER ---
SERVER_BASE = "http://127.0.0.1:8880"
OMNIVOICE_URL = f"{SERVER_BASE}/v1/audio/speech"
SERVER_STARTUP_ARGS = ["--device", "cuda", "--num-step", "16", "--max-concurrent", "1"]
SERVER_READY_TIMEOUT_S = 90   # how long to wait for the server to come up before giving up

# --- AUDIO CAPTURE (microphone) ---
SAMPLE_RATE = 16000
SILENCE_DURATION_SEC = 1.2

# --- VOICE PROFILE ---
VOICE_ID = "eg"
# Leave as None to be prompted at startup (record or supply a path).
# Set to a fixed path to always use the same reference file without prompting.
REF_WAV = None

# --- PLAYBACK ROUTING ---
# From your `python query.py` device list:
#   9  CABLE Input (VB-Audio Virtual Cable), MME   <- feed audio INTO the cable (OBS listens on the CABLE Output side)
#   7  Speaker (Realtek(R) Audio), MME              <- your normal speakers/headphones
CABLE_OUTPUT_DEVICE_INDEX = 9      # what OBS's "Audio Output Capture" should be set to read: "CABLE Output"
LOCAL_MONITOR_DEVICE_INDEX = 7     # your speakers, so you can hear it live too

# If True, plays to both the virtual cable (for OBS) AND your local speakers (so you can monitor).
# If False, only sends to the cable - OBS hears it, but you won't hear it live through your speakers.
PLAY_TO_LOCAL_MONITOR = True

TTS_SAMPLE_RATE = 24000