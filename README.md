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
- 3-minute sustained-fatigue alarm
- REST API for frontend integration (`backend/api_server.py`)

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
multimodal-fatigue-detection/
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
│   ├── main_camera_system.py     # includes live ML overlay
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

Controls during a session:

- **Camera window** — `0`/`1` set the manual fatigue label, `Q` quits.
- **Keyboard node** — `F1`/`F2` set the label (ALERT/FATIGUED), `ESC` quits.

When both nodes exit, the fusion node automatically merges the two CSVs into
`fusion_node/multimodal_fatigue_dataset.csv`. Let both nodes run for at least
~10 s so the first aggregation window is written.

### API server (for frontend)

```bash
# Run from the repo root — do NOT cd into backend/
python backend/api_server.py
```

The server opens the webcam, starts a keyboard listener, and begins producing
10-second predictions immediately. No other nodes need to be running.

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
| `alarm_triggered` | bool | True once 3 min of sustained fatigue detected |
| `consecutive_fatigued_windows` | int | Running count of back-to-back FATIGUED windows |
| `alarm_threshold_windows` | int | 18 (= 3 min ÷ 10 s per window) |

#### `/api/history` response

Array of objects: `{timestamp, fatigue_probability, fatigue_label}`.
Oldest entry first, capped at 60 entries.

All endpoints include `Access-Control-Allow-Origin: *` so a browser frontend
on a different port can call them directly.

---

## Future Work

- Dataset expansion and cross-session evaluation
- Model benchmarking and validation across users
- Cross-user fatigue generalization
