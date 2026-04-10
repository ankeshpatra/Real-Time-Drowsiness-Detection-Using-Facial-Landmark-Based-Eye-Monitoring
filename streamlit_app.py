"""
streamlit_app.py - Web-Based Real-Time Driver Drowsiness Detection Dashboard
=============================================================================

All detection logic (MediaPipe, EAR computation) runs in Python.
The browser only handles the webcam stream and CSS styling.

Launch:
    streamlit run streamlit_app.py
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import time
import threading
import math
import struct
import io
import base64
import os

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from utils import (
    eye_aspect_ratio,
    draw_eye_contours,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)

# ══════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════
MODEL_PATH = "face_landmarker.task"
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def build_rtc_configuration():
    """Build ICE server configuration with optional TURN support via env vars."""
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]

    turn_url = os.getenv("TURN_SERVER_URL", "").strip()
    turn_username = os.getenv("TURN_USERNAME", "").strip()
    turn_password = os.getenv("TURN_PASSWORD", "").strip()
    if turn_url and turn_username and turn_password:
        ice_servers.append(
            {
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_password,
            }
        )

    return {"iceServers": ice_servers}

# ══════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# Custom CSS — styling only, no logic
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Hide Streamlit chrome for clean dashboard look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tighten padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
    }

    /* Title */
    .main-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.4;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #666;
        margin-top: 2px;
        margin-bottom: 12px;
    }

    /* Badge (fps / online) */
    .badge-row {
        text-align: right;
        font-size: 0.85rem;
        color: #888;
        padding-top: 6px;
    }

    /* Stats panel labels and values */
    .stat-label {
        font-size: 0.82rem;
        color: #777;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .stat-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 0;
        line-height: 1.3;
    }
    .stat-divider {
        border: none;
        border-top: 1px solid #eee;
        margin: 6px 0;
    }

    /* Green progress bar for EAR */
    .stProgress > div > div > div > div {
        background-color: #2e7d32 !important;
        background-image: linear-gradient(90deg, #2e7d32, #43a047) !important;
    }

    /* Status messages */
    .status-ready {
        padding: 8px 14px;
        background-color: #e8f5e9;
        color: #2e7d32;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .status-active {
        padding: 8px 14px;
        background-color: #e3f2fd;
        color: #1565c0;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .status-alert {
        padding: 8px 14px;
        background-color: #ffebee;
        color: #c62828;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# Model check
# ══════════════════════════════════════════════════════════════════
if not os.path.isfile(MODEL_PATH):
    st.error(
        f"Model file `{MODEL_PATH}` not found. "
        f"Run `python download_model.py` first."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════
# Alarm sound generator (pure Python, no JS logic)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def generate_alarm_b64(frequency=800, duration=1.5, sample_rate=22050):
    """Generate a beep tone and return it as a base64 WAV string."""
    n = int(sample_rate * duration)
    buf = io.BytesIO()
    data_size = n * 2  # 16-bit mono
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate,
                          sample_rate * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    for i in range(n):
        t = i / sample_rate
        envelope = min(t * 8, 1.0) * max(0.0, 1.0 - t * 0.3)
        val = int(32767 * 0.5 * envelope *
                  math.sin(2 * math.pi * frequency * t))
        val = max(-32768, min(32767, val))
        buf.write(struct.pack('<h', val))
    return base64.b64encode(buf.getvalue()).decode()

alarm_b64 = generate_alarm_b64()

# ══════════════════════════════════════════════════════════════════
# Shared state object (persists across Streamlit reruns)
# ══════════════════════════════════════════════════════════════════
class SharedState:
    """Thread-safe state shared between the video callback and the UI."""
    def __init__(self):
        self.ear: float = 0.0
        self.frame_counter: int = 0
        self.alert_count: int = 0
        self.drowsy: bool = False
        self.fps: float = 0.0
        self.ear_threshold: float = 0.22
        self.consec_frames_limit: int = 20
        self._prev_time: float = time.time()
        self._prev_drowsy: bool = False
        self._landmarker = None
        self.last_error: str = ""

    def get_landmarker(self):
        """Lazy-initialise the MediaPipe Face Landmarker (once)."""
        if self._landmarker is None:
            base_opts = BaseOptions(model_asset_path=MODEL_PATH)
            opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = (
                vision.FaceLandmarker.create_from_options(opts)
            )
        return self._landmarker

if "shared" not in st.session_state:
    st.session_state.shared = SharedState()
if "start_time" not in st.session_state:
    st.session_state.start_time = None

shared = st.session_state.shared

# ══════════════════════════════════════════════════════════════════
# Video frame callback — ALL detection logic is Python
# ══════════════════════════════════════════════════════════════════
def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Process one video frame: detect face, compute EAR, draw overlays."""
    img = frame.to_ndarray(format="bgr24")
    h, w = img.shape[:2]

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        landmarker = shared.get_landmarker()
        result = landmarker.detect(mp_image)
        shared.last_error = ""
    except Exception as e:
        shared.last_error = str(e)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    ear = 0.0

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # Extract 6-point eye coordinates (pixel space)
        left_eye = np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)]
             for i in LEFT_EYE_INDICES],
            dtype=np.int32,
        )
        right_eye = np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)]
             for i in RIGHT_EYE_INDICES],
            dtype=np.int32,
        )

        # EAR calculation
        left_ear = eye_aspect_ratio(left_eye.astype(float))
        right_ear = eye_aspect_ratio(right_eye.astype(float))
        ear = (left_ear + right_ear) / 2.0

        # Choose contour color: GREEN when open, RED when closed
        eyes_closed = ear < shared.ear_threshold
        contour_color = RED if eyes_closed else GREEN
        # Draw eye contours (thicker when closed for visibility)
        thickness = 2 if eyes_closed else 1
        left_hull = cv2.convexHull(left_eye)
        right_hull = cv2.convexHull(right_eye)
        cv2.drawContours(img, [left_hull], -1, contour_color, thickness)
        cv2.drawContours(img, [right_hull], -1, contour_color, thickness)

        # Threshold logic
        if ear < shared.ear_threshold:
            shared.frame_counter += 1
            if shared.frame_counter >= shared.consec_frames_limit:
                shared.drowsy = True
                if not shared._prev_drowsy:
                    shared.alert_count += 1
                    shared._prev_drowsy = True
        else:
            shared.frame_counter = 0
            shared.drowsy = False
            shared._prev_drowsy = False
    else:
        # No face detected — reset
        shared.frame_counter = 0
        shared.drowsy = False
        shared._prev_drowsy = False

    # FPS
    now = time.time()
    shared.fps = 1.0 / max(now - shared._prev_time, 1e-6)
    shared._prev_time = now
    shared.ear = ear

    # Draw red alert banner on frame when drowsy
    if shared.drowsy:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        text = "!! DROWSINESS ALERT !!"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = (w - ts[0]) // 2
        cv2.putText(img, text, (tx, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ══════════════════════════════════════════════════════════════════
# UI LAYOUT — matches the reference image exactly
# ══════════════════════════════════════════════════════════════════

# ── Row 1: Title + FPS badge ──
title_col, badge_col = st.columns([5, 1])
with title_col:
    st.markdown(
        '<p class="main-title">'
        'Real-time driver drowsiness detection system '
        'using face mesh eye landmark analysis</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-title">Driver Drowsiness Monitor</p>',
        unsafe_allow_html=True,
    )
with badge_col:
    fps_ph = st.empty()
    fps_ph.markdown(
        '<p class="badge-row">&mdash; fps &nbsp;&nbsp; Offline</p>',
        unsafe_allow_html=True,
    )

# ── Row 2: Video feed + Stats panel ──
col_video, col_stats = st.columns([2, 1], gap="large")

with col_video:
    ctx = webrtc_streamer(
        key="drowsiness-detector",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_callback,
        rtc_configuration=build_rtc_configuration(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_stats:
    st.markdown('<p class="stat-label">Eye Aspect Ratio</p>',
                unsafe_allow_html=True)
    ear_ph = st.empty()
    ear_ph.markdown('<p class="stat-value">&mdash;</p>',
                    unsafe_allow_html=True)

    st.markdown('<hr class="stat-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p class="stat-label">Eyes Closed (consecutive frames)</p>',
        unsafe_allow_html=True,
    )
    closed_ph = st.empty()
    closed_ph.markdown('<p class="stat-value">0</p>',
                       unsafe_allow_html=True)

    st.markdown('<hr class="stat-divider">', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Alerts triggered</p>',
                unsafe_allow_html=True)
    alerts_ph = st.empty()
    alerts_ph.markdown('<p class="stat-value">0</p>',
                       unsafe_allow_html=True)

    st.markdown('<hr class="stat-divider">', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Session duration</p>',
                unsafe_allow_html=True)
    duration_ph = st.empty()
    duration_ph.markdown('<p class="stat-value">0:00</p>',
                         unsafe_allow_html=True)

# ── Row 3: EAR progress bar ──
ear_bar_label_col, ear_bar_val_col = st.columns([6, 1])
with ear_bar_label_col:
    st.markdown("**Eye Aspect Ratio (EAR)**")
    ear_bar_ph = st.empty()
    ear_bar_ph.progress(0.0)
with ear_bar_val_col:
    st.markdown("&nbsp;")   # spacer to align vertically
    ear_bar_val_ph = st.empty()
    ear_bar_val_ph.markdown("&mdash;")

# ── Row 4: Sliders ──
s1, s2 = st.columns([1, 1])
with s1:
    ear_threshold = st.slider(
        "EAR threshold", 0.10, 0.40, 0.22, 0.01,
    )
with s2:
    consec_frames_limit = st.slider(
        "Alert after (frames)", 5, 50, 20, 1,
    )

# Push slider values into the shared state
shared.ear_threshold = ear_threshold
shared.consec_frames_limit = consec_frames_limit

# ── Row 5: Status bar + Alarm placeholder ──
status_ph = st.empty()
alarm_ph = st.empty()
status_ph.markdown(
    '<p class="status-ready">'
    'System ready. Awaiting camera start.</p>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════
# Real-time polling loop — updates the dashboard while stream is on
# ══════════════════════════════════════════════════════════════════
if ctx.state.playing:
    # Record session start time
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    while ctx.state.playing:
        ear = shared.ear
        fc = shared.frame_counter
        alerts = shared.alert_count
        drowsy = shared.drowsy
        fps = shared.fps

        # FPS badge
        fps_ph.markdown(
            f'<p class="badge-row">{fps:.0f} fps '
            f'&nbsp;&nbsp; &#x1F7E2; Online</p>',
            unsafe_allow_html=True,
        )

        # Stats
        ear_ph.markdown(
            f'<p class="stat-value">{ear:.3f}</p>',
            unsafe_allow_html=True,
        )
        closed_ph.markdown(
            f'<p class="stat-value">{fc}</p>',
            unsafe_allow_html=True,
        )
        alerts_ph.markdown(
            f'<p class="stat-value">{alerts}</p>',
            unsafe_allow_html=True,
        )

        # Session duration
        elapsed = int(time.time() - st.session_state.start_time)
        mins, secs = divmod(elapsed, 60)
        duration_ph.markdown(
            f'<p class="stat-value">{mins}:{secs:02d}</p>',
            unsafe_allow_html=True,
        )

        # EAR bar (scale: 0.0 – 0.4 maps to 0% – 100%)
        bar_val = min(max(ear / 0.4, 0.0), 1.0)
        ear_bar_ph.progress(bar_val)
        ear_bar_val_ph.markdown(f"**{ear:.2f}**")

        # Status + alarm
        if drowsy:
            status_ph.markdown(
                '<p class="status-alert">'
                '&#9888;&#65039; DROWSINESS DETECTED! Wake up!</p>',
                unsafe_allow_html=True,
            )
            alarm_ph.markdown(
                f'<audio autoplay>'
                f'<source src="data:audio/wav;base64,{alarm_b64}" '
                f'type="audio/wav"></audio>',
                unsafe_allow_html=True,
            )
        else:
            status_ph.markdown(
                '<p class="status-active">'
                '&#9989; Monitoring active. Eyes detected.</p>',
                unsafe_allow_html=True,
            )
            alarm_ph.empty()

        if shared.last_error:
            st.warning(f"Video processing error: {shared.last_error}")

        time.sleep(0.1)
else:
    # Stream is not playing — reset timer
    st.session_state.start_time = None
    st.info(
        "If camera does not start on Streamlit Cloud, allow browser camera access. "
        "If your network blocks direct WebRTC, set TURN_SERVER_URL, TURN_USERNAME, "
        "and TURN_PASSWORD in app secrets/environment."
    )
