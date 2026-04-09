# Real-Time Driver Drowsiness Detection System

> **Using Facial Landmark Detection and Eye Aspect Ratio (EAR)**

A real-time computer vision system that detects driver drowsiness by monitoring eye closure patterns through a webcam. The system uses **MediaPipe Face Mesh** (478-point facial landmark model) to extract eye regions, computes the Eye Aspect Ratio (EAR), and triggers audio-visual alerts when prolonged eye closure is detected.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Algorithm — Eye Aspect Ratio](#algorithm--eye-aspect-ratio)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration & Tuning](#configuration--tuning)
- [Challenges & Solutions](#challenges--solutions)
- [Evaluation Metrics](#evaluation-metrics)
- [Dataset References](#dataset-references)
- [Possible Improvements](#possible-improvements)
- [License](#license)

---

## Features

- **Real-time processing** from webcam at 20–30+ FPS.
- **EAR-based drowsiness metric** — lightweight, explainable, no GPU required.
- **MediaPipe Tasks API** — Robust, modern facial landmarking with 478 points.
- **Lightweight Model** — Uses a 3.8 MB landmarker task model.
- **Adaptive calibration** — per-user EAR threshold tuning.
- **Audio + visual alerts** via pygame and on-screen HUD.
- **Heads-up display** showing live EAR, FPS, and closed-frame counter.
- **Modular codebase** with separated utilities for easy testing.

---

## System Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐
│  Webcam  │──▶│  MediaPipe   │──▶│  478-Point    │──▶│  EAR Compute  │──▶│  Alert   │
│  Input   │    │  Face Mesh   │    │  Landmarks    │    │  + Threshold  │    │  Module  │
└──────────┘    └──────────────┘    └──────────────┘    └───────────────┘    └──────────┘
```

**Pipeline detail:**
1. **Capture** — Read BGR frame from webcam via OpenCV.
2. **Preprocess** — Resize to 640px width, convert to RGB for MediaPipe.
3. **Face Mesh** — Extract 478 facial landmarks (including iris).
4. **Eye Extraction** — Map 6 key landmarks per eye (corners + upper/lower lids).
5. **EAR Calculation** — Compute the ratio of vertical to horizontal eye distances.
6. **Decision** — If average EAR < threshold for N consecutive frames → alert.
7. **Alert** — Flash a red banner + play an audio alarm (if `.wav` provided).

---

## Algorithm — Eye Aspect Ratio

For each eye, 6 facial landmarks (p₁ to p₆) define the contour:

```
        p2    p3
  p1 ──────────── p4
        p6    p5
```

The EAR is computed as:

```
EAR = (||p2 − p6|| + ||p3 − p5||) / (2 × ||p1 − p4||)
```

| Eye State | Typical EAR |
|-----------|-------------|
| Wide open | 0.30 – 0.40 |
| Normal    | 0.25 – 0.30 |
| Closed    | 0.05 – 0.15 |

**Decision rule**: If the *average EAR* (left + right eye) stays below the threshold for *N* consecutive frames (~1–1.5 seconds), the system raises a **DROWSINESS ALERT**.

### MediaPipe Landmark Mapping

The 478-point Face Mesh is mapped to 6 EAR points per eye:

| Point | Left Eye Index | Right Eye Index | Anatomical Position |
|-------|---------------|-----------------|---------------------|
| p1    | 33            | 362             | Inner/outer corner  |
| p2    | 160           | 385             | Upper-left lid      |
| p3    | 158           | 387             | Upper-right lid     |
| p4    | 133           | 263             | Outer/inner corner  |
| p5    | 153           | 373             | Lower-right lid     |
| p6    | 144           | 380             | Lower-left lid      |

---

## Project Structure

```
.
├── streamlit_app.py               # Web dashboard (Streamlit — recommended)
├── drowsiness_detector.py         # CLI detection script (OpenCV window)
├── utils.py                       # EAR computation, drawing helpers
├── download_model.py              # Downloads the MediaPipe model
├── test_ear.py                    # Unit tests for EAR logic
├── face_landmarker.task           # MediaPipe model (downloaded via script)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE
```

---

## Prerequisites

- **Python** 3.8 or later
- A working **webcam**
- **Windows / Linux / macOS**

> **Note**: Unlike dlib-based approaches, this project uses MediaPipe which
> installs as a pure pip package — no CMake or C++ compiler needed.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ankeshpatra/Real-Time-Drowsiness-Detection-Using-Facial-Landmark-Based-Eye-Monitoring.git
cd Real-Time-Drowsiness-Detection-Using-Facial-Landmark-Based-Eye-Monitoring
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Face Landmarker model

```bash
python download_model.py
```

This downloads `face_landmarker.task` (~3.8 MB) required by the MediaPipe Tasks API.

---

## Usage

### Web Dashboard (Recommended)

```bash
streamlit run streamlit_app.py
```

This opens a browser-based dashboard at `http://localhost:8501` with:
- Live webcam feed with eye landmark overlays
- Real-time EAR value, frame counter, alert counter, and session timer
- Adjustable EAR threshold and consecutive-frame sliders
- Audio + visual alert when drowsiness is detected

### CLI Version (OpenCV Window)

```bash
python drowsiness_detector.py
```

### CLI with calibration (Recommended for First Use)

```bash
python drowsiness_detector.py --calibrate
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `-w`, `--webcam` | `0` | Webcam device index |
| `-t`, `--ear-threshold` | `0.25` | EAR value below which eyes are "closed" |
| `-f`, `--consec-frames` | `25` | Frames below threshold before alert |
| `-a`, `--alarm` | `alarm.wav` | Path to alarm sound file |
| `--calibrate` | off | Run EAR calibration at startup |
| `--no-sound` | off | Disable audio alarm (visual only) |

### Keyboard shortcuts while running

- **`q`** — Quit the application.
- **`r`** — Reset the frame counter and silence the alarm.

---

## Configuration & Tuning

### EAR Threshold

- **Default**: `0.25` (works well for most people).
- **Too many false alarms?** Lower it → e.g. `0.22`.
- **Missing real drowsiness?** Raise it → e.g. `0.28`.
- **Best approach**: Use `--calibrate` to let the system measure your personal open-eye EAR and set the threshold to 80% of the mean.

### Consecutive Frames

- At 20 FPS, `25` frames ≈ 1.25 seconds.
- Increase for fewer alerts; decrease for faster response.

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Glasses / reflections** | MediaPipe's ML-based landmark detector handles glasses better than HOG-based approaches; it was trained on diverse face data including eyewear. |
| **Variable lighting** | MediaPipe internally normalises inputs; additionally the EAR ratio is scale-invariant. |
| **Head tilt / pose variation** | MediaPipe Face Mesh is robust to ±30° yaw/pitch. EAR is normalised by horizontal eye width, absorbing mild scaling. |
| **Natural blinks vs. drowsiness** | Temporal filtering (consecutive-frames threshold) distinguishes short blinks (~200 ms) from prolonged closure (~1.25 s). |
| **Alarm latency** | Audio runs in a background thread; visual alert renders in the same frame loop. |
| **Cross-platform install** | MediaPipe installs via pip on all platforms — no CMake or C++ compiler needed (unlike dlib). |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of frames correctly classified as open/closed. |
| **Precision** | Of all "drowsy" alerts, how many were genuine? |
| **Recall** | Of all genuine drowsy episodes, how many were caught? |
| **F1 Score** | Harmonic mean of precision and recall. |
| **Latency** | Time from eye closure to alert trigger (aim: < 2 s). |
| **FPS** | Frames per second on the target hardware (aim: ≥ 15). |

Run the unit tests to verify EAR math correctness:

```bash
python test_ear.py
```

---

## Dataset References

| Dataset | Purpose | Link |
|---------|---------|------|
| **MediaPipe Face Mesh** | 478-point landmark model (bundled) | [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| **iBUG 300-W** | Original facial landmark benchmark | [iBUG 300-W](https://ibug.doc.ic.ac.uk/resources/300-W/) |
| **MRL Eye Dataset** | Open/closed eye images for offline validation | [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) |
| **NTHU-DDD** | Driver Drowsiness Detection benchmark | [NTHU-DDD](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/) |

---

## Possible Improvements

1. **Blink Rate Analysis** — Track blinks per minute as an early fatigue indicator.
2. **Yawn Detection** — Use mouth landmarks (MediaPipe indices 13, 14, 78, 308) to compute Mouth Aspect Ratio (MAR).
3. **Head Pose Estimation** — Detect nodding off using `solvePnP` with 6 canonical face points from Face Mesh.
4. **CNN Classifier** — Replace or supplement EAR with a small CNN trained on cropped eye images for higher accuracy.
5. **PERCLOS Metric** — Percentage of eye closure over a rolling time window (widely used in drowsiness research).
6. **Mobile Deployment** — MediaPipe already supports Android/iOS for smartphone deployment.
7. **Data Logging** — Record EAR time series to CSV for post-hoc analysis and research.

---

## License

This project is licensed under the terms of the [GNU 2.0 License](LICENSE).
