import requests
import os

SERVER_BASE = "http://127.0.0.1:8880"
PROFILE_ID = "eg"
WAV_PATH = r"C:\Users\oswal\Downloads\test.wav"

if not os.path.exists(WAV_PATH):
    print(f"❌ File not found: {WAV_PATH}")
else:
    profile_url = f"{SERVER_BASE}/v1/voices/profiles"
    with open(WAV_PATH, "rb") as f:
        files = {"ref_audio": (os.path.basename(WAV_PATH), f, "audio/wav")}
        data = {"profile_id": PROFILE_ID, "overwrite": "true"}
        response = requests.post(profile_url, data=data, files=files, timeout=15)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")