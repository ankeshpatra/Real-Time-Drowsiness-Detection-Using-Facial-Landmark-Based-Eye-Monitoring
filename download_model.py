"""
download_model.py — Download the MediaPipe Face Landmarker model.

Downloads `face_landmarker.task` from Google's model storage.
This model is required for the MediaPipe Tasks API.

Usage:
    python download_model.py
"""

import os
import sys
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_FILE = "face_landmarker.task"

def download_file(url: str, dest: str) -> None:
    """Download *url* to *dest*."""
    print(f"[INFO] Downloading {url} ...")
    
    def _report(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(downloaded / total_size * 100, 100)
            sys.stdout.write(f"\r  Progress: {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    print()  # newline after progress

def main() -> None:
    if os.path.isfile(MODEL_FILE):
        print(f"[INFO] '{MODEL_FILE}' already exists -- skipping download.")
        return

    try:
        download_file(MODEL_URL, MODEL_FILE)
        if os.path.exists(MODEL_FILE):
             print(f"[INFO] '{MODEL_FILE}' is ready ({os.path.getsize(MODEL_FILE) / 1e6:.1f} MB).")
        else:
             print(f"[ERROR] '{MODEL_FILE}' was not created.")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
