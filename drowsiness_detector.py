"""
drowsiness_detector.py — Real-Time Driver Drowsiness Detection System
======================================================================

Uses MediaPipe Tasks API (Face Landmarker) and the Eye Aspect Ratio (EAR) 
to determine drowsiness. Handles cases where the legacy mp.solutions 
is unavailable.

Usage
-----
    python drowsiness_detector.py [--webcam 0]
                                  [--ear-threshold 0.25]
                                  [--consec-frames 25]
                                  [--alarm alarm.wav]
                                  [--calibrate]
                                  [--no-sound]
"""

import argparse
import os
import sys
import time
import threading

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import (
    eye_aspect_ratio,
    get_ear_from_landmarks,
    resize_frame,
    draw_eye_contours,
)

# ── Defaults ──
DEFAULT_EAR_THRESHOLD = 0.25
DEFAULT_CONSEC_FRAMES = 25
DEFAULT_ALARM_WAV = "alarm.wav"
DEFAULT_MODEL = "face_landmarker.task"
FRAME_WIDTH = 640

# Colors (BGR)
GREEN, RED, WHITE, YELLOW = (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 255, 255)

# ── Alarm ──
_alarm_on = False
_alarm_lock = threading.Lock()

def _play_alarm(path):
    global _alarm_on
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(-1)
        while True:
            with _alarm_lock:
                if not _alarm_on: break
            time.sleep(0.1)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except ImportError:
        print("[WARN] Pygame not found. Sound alerts are disabled for CLI.")
        print("       To enable sound, run: pip install pygame")
    except Exception as e:
        print(f"[WARN] Alarm error: {e}")

def start_alarm(path):
    global _alarm_on
    with _alarm_lock:
        if _alarm_on: return
        _alarm_on = True
    threading.Thread(target=_play_alarm, args=(path,), daemon=True).start()

def stop_alarm():
    global _alarm_on
    with _alarm_lock: _alarm_on = False

# ── HUD ──
def draw_hud(frame, ear, frame_counter, drowsy, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    color = RED if drowsy else GREEN
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"Closed frames: {frame_counter}", (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1)

    if drowsy:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 60), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        text = "!! DROWSINESS ALERT !!"
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0][0]
        cv2.putText(frame, text, ((w - tw)//2, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 3)
    return frame

# ── Main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0)
    ap.add_argument("-t", "--ear-threshold", type=float, default=DEFAULT_EAR_THRESHOLD)
    ap.add_argument("-f", "--consec-frames", type=int, default=DEFAULT_CONSEC_FRAMES)
    ap.add_argument("-a", "--alarm", type=str, default=DEFAULT_ALARM_WAV)
    ap.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--no-sound", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"[ERROR] Model file '{args.model}' not found. Run download_model.py first.")

    # ── Init Landmarker ──
    print("[INFO] Initialising MediaPipe Face Landmarker …")
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened(): sys.exit(f"[ERROR] Cannot open webcam {args.webcam}")

    ear_threshold = args.ear_threshold
    consec_frames = args.consec_frames
    has_alarm_file = not args.no_sound and os.path.isfile(args.alarm)

    frame_counter, alarm_active, prev_time, fps = 0, False, time.time(), 0.0
    print("[INFO] Running. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = resize_frame(frame, FRAME_WIDTH)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Tasks API requires timestamp in ms
            ts = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts)

            drowsy, ear = False, 0.0
            if result.face_landmarks:
                ear, left, right = get_ear_from_landmarks(result.face_landmarks[0], w, h)
                draw_eye_contours(frame, left, right, GREEN)

                if ear < ear_threshold:
                    frame_counter += 1
                    if frame_counter >= consec_frames:
                        drowsy = True
                        if not alarm_active and has_alarm_file:
                            alarm_active = True
                            start_alarm(args.alarm)
                else:
                    if alarm_active:
                        alarm_active = False
                        stop_alarm()
                    frame_counter = 0

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            frame = draw_hud(frame, ear, frame_counter, drowsy, fps)
            cv2.imshow("Drowsiness Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        stop_alarm()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__": main()
