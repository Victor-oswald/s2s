import requests, time

print("Warming up (this can take 30-60s on first call)...")
start = time.time()
r = requests.post(
    "http://127.0.0.1:8880/v1/audio/speech",
    json={"model": "omnivoice", "input": "warmup test", "voice": "eg", "response_format": "wav"},
    timeout=120
)
print(f"Done in {time.time()-start:.1f}s — status {r.status_code}")