# Multimodal Fatigue Detection System

## Overview

A real-time multimodal behavioral fatigue detection system using:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head posture and nod analysis
- Privacy-safe keyboard telemetry

The system fuses camera and keyboard signals into 10-second aggregation windows, feeds them into a trained Random Forest classifier (~75% accuracy, ROC-AUC 0.73), and exposes live predictions via a REST API for frontend consumption.

---

## Features

### Camera Node
- Real-time eye fatigue detection
- Blink and microsleep analysis
- Real-time yawn detection
- Adaptive mouth fatigue tracking
- Forward head-drop and nod detection
- Visual facial landmark overlays
- Unified synchronized FaceMesh pipeline
- Live ML prediction overlay (fuses camera + keyboard CSV rows)
- Optional headless mode (no preview window) for background operation

### Keyboard Node
- Privacy-safe behavioral telemetry
- Typing speed deviation analysis
- Pause behavior analysis
- Backspace deviation tracking
- Adaptive personal baseline modeling
- Rolling behavioral degradation metrics

### ML & API
- Random Forest classifier (19 multimodal features)
- Trained model: `fusion_node/fatigue_model.pkl`
- 3-minute sustained-fatigue alarm (rolling-window ratio, not a single-reading trigger)
- REST API for frontend integration (`backend/api_server.py`)
- Optional live camera preview with prediction overlay for demos

### System Features
- ML-ready CSV dataset generation
- Modular detector architecture
- Real-time behavioral aggregation
- Multimodal telemetry synchronization

---

## Tech Stack

- Python
- OpenCV
- MediaPipe FaceMesh
- NumPy
- pynput
- scikit-learn / joblib (Random Forest classifier)
- Flask (REST API for frontend integration)
- CSV dataset pipeline

---

## Project Structure

```text
Multimodal_Fatigue_Detection_System/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main_system.py                # Orchestrator — run this for data collection
│
├── camera_node/
│   ├── ear_detector.py
│   ├── mar_detector.py
│   ├── head_detector.py
│   ├── main_camera_system.py     # includes live ML overlay + window toggle
│   └── camera_fatigue_dataset.csv
│
├── keyboard_node/
│   ├── keyboard_detector.py
│   ├── main_keyboard_system.py
│   └── keyboard_fatigue_dataset.csv
│
├── fusion_node/
│   ├── dataset_fusion.py
│   ├── multimodal_fatigue_dataset.csv
│   ├── fatigue_model.pkl         # trained Random Forest model
│   └── fatigue_model_meta.pkl    # feature_order + clip_caps
│
└── backend/
    └── api_server.py             # Flask REST API for the frontend
```

---

## Installation & Usage

```bash
# 1. Install dependencies (Python 3.10–3.12 recommended)
pip install -r requirements.txt
```

### Data collection (camera + keyboard nodes)

```bash
# 2. Run the full pipeline (launches camera + keyboard nodes, then fuses)
python main_system.py
```

At startup you'll be asked:

```text
Show camera preview window? [Y/n]:
```

- **Y (or just press Enter)** — camera window is shown, exactly as below.
- **N** — runs headless (no window). The session can then only be stopped
  via the keyboard node's `ESC` key (see below) — there's no window to
  press `Q` in.

Controls during a session:

- **Camera window** (if shown) — `0`/`1` set the manual fatigue label, `Q` quits.
- **Keyboard node** — `F1`/`F2` set the label (ALERT/FATIGUED), `ESC` quits
  **both** the keyboard node and the camera node (ESC writes the shared
  shutdown signal, so it works even when the camera is running headless).

When both nodes exit, the fusion node automatically merges the two CSVs into
`fusion_node/multimodal_fatigue_dataset.csv`. Let both nodes run for at least
~10 s so the first aggregation window is written.

### API server (for frontend)

```bash
# Run from the repo root — do NOT cd into backend/
python backend/api_server.py
```

The server opens the webcam, starts a keyboard listener, and begins producing
10-second predictions immediately. No other nodes need to be running —
**don't run this at the same time as `main_system.py`**, since both will try
to open the same webcam.

By default this runs fully headless (no window) — it's designed as a
background API server. For a live demo where you want to see the camera feed
with the prediction overlaid, set the environment variable first:

```bash
# PowerShell
$env:SHOW_CAMERA_WINDOW="1"
python backend/api_server.py

# Command Prompt
set SHOW_CAMERA_WINDOW=1
python backend/api_server.py
```

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness probe. Returns `{"status": "ok", "model_available": true/false}` |
| `/api/status` | GET | Latest 10-second window prediction (see fields below) |
| `/api/history` | GET | Last 60 window readings (oldest first) |

#### `/api/status` response fields

| Field | Type | Values |
|---|---|---|
| `status` | string | `ALERT` / `FATIGUED` / `NO_FACE_DETECTED` / `starting` |
| `fatigue_label` | string \| null | `ALERT` / `FATIGUED` / null |
| `fatigue_probability` | float \| null | 0.0–1.0, probability of FATIGUED class |
| `features` | object | Raw 19-feature dict for this window |
| `timestamp` | float \| null | Unix epoch of the window |
| `alarm_triggered` | bool | True once sustained fatigue is detected (see below) |
| `consecutive_fatigued_windows` | int | Running count of back-to-back FATIGUED windows (informational only — does not gate the alarm) |
| `fatigued_ratio_in_window` | float | 0.0–1.0, fraction of the last 18 windows (≈3 min) that were FATIGUED — **this is what actually triggers the alarm** |
| `alarm_threshold_windows` | int | 18 (= 3 min ÷ 10 s per window) |

The alarm fires when `fatigued_ratio_in_window >= 0.70`, i.e. at least 70% of
the last 3 minutes of readings were FATIGUED. This is a rolling ratio, not a
streak counter — a single normal blink or brief look-away only nudges the
ratio slightly, rather than resetting accumulated evidence to zero.

#### `/api/history` response

Array of objects: `{timestamp, fatigue_probability, fatigue_label}`.
Oldest entry first, capped at 60 entries.

All endpoints include `Access-Control-Allow-Origin: *` so a browser frontend
on a different port can call them directly.

---

## Validation & Findings

- Compared 3 models (Logistic Regression, Decision Tree, Random Forest) via
  session-aware cross-validation (split by recording session, not by row, to
  avoid leakage between adjacent windows from the same sitting). Random
  Forest was selected for the best overall balance and ROC-AUC (0.729).
- A naive random train/test split overstates performance (86.7% accuracy)
  because adjacent windows from one session leak between train and test.
  The session-aware result (74.9%) is the honest estimate reported here.
- Tested cross-subject generalization: a model trained exclusively on one
  subject performs close to chance (53.0% accuracy) on a second subject it
  never saw. Per-session normalization of camera features improves this to
  69.5%; a small personal-calibration sample from the new subject alone
  outperforms both (82.9%).
- These results indicate the system benefits from either a brief per-user
  calibration period or feature normalization rather than relying on a
  single universal pre-trained model — see Future Work.

## Future Work

- Expand the dataset across more subjects and sessions to reduce the
  session/subject confound observed during validation
- Implement an explicit per-user calibration flow (a brief baseline-building
  session before live use), motivated by the personalization finding above
- Extend the rolling-window alarm logic with trend analysis (e.g.
  rate-of-change across windows) rather than a flat ratio threshold