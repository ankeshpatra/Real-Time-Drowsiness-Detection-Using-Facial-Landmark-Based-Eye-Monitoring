"""
utils.py — Utility functions for the Drowsiness Detection System.

Uses MediaPipe Tasks API (Face Landmarker) for facial landmark detection.
Maps landmark indices to the 6-point eye model used by EAR.

Contains:
    - eye_aspect_ratio(): Computes EAR from 6 landmark points.
    - get_ear_from_landmarks(): Extracts eye landmarks from MediaPipe
      task results and computes average EAR.
    - resize_frame(): Resizes a frame while preserving aspect ratio.
    - draw_eye_contours(): Draws convex hull contours around both eyes.
"""

import numpy as np
from scipy.spatial import distance as dist
import cv2

# ────────────────────────────────────────────────────────────────────
# MediaPipe Face Mesh landmark indices for the 6-point eye model.
# Left eye (subject's perspective): 33, 160, 158, 133, 153, 144
# Right eye (subject's perspective): 362, 385, 387, 263, 373, 380
# ────────────────────────────────────────────────────────────────────
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])

    if C == 0.0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear

def get_ear_from_landmarks(landmarks, frame_w: int, frame_h: int):
    """
    Given MediaPipe Task landmarks (list of NormalizedLandmark), extract 
    eye regions and compute average EAR.
    """
    # Convert normalised (0-1) landmarks to pixel coordinates
    left_eye = np.array([
        [int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)]
        for idx in LEFT_EYE_INDICES
    ], dtype=np.int32)

    right_eye = np.array([
        [int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)]
        for idx in RIGHT_EYE_INDICES
    ], dtype=np.int32)

    left_ear = eye_aspect_ratio(left_eye.astype(float))
    right_ear = eye_aspect_ratio(right_eye.astype(float))

    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear, left_eye, right_eye

def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """Resize frame while keeping aspect ratio."""
    h, w = frame.shape[:2]
    ratio = width / float(w)
    new_height = int(h * ratio)
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)

def draw_eye_contours(frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray, color=(0, 255, 0)):
    """Draw eye contours on frame."""
    left_hull = cv2.convexHull(left_eye)
    right_hull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [left_hull], -1, color, 1)
    cv2.drawContours(frame, [right_hull], -1, color, 1)
    return frame
