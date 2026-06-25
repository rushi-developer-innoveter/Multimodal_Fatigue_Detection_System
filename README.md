# Multimodal Fatigue Detection System

## Overview

A real-time multimodal behavioral fatigue detection system using:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head posture and nod analysis
- Privacy-safe keyboard telemetry

The system performs synchronized behavioral feature extraction from camera and keyboard signals to generate ML-ready multimodal fatigue datasets through temporal aggregation windows.

The architecture is designed for future fatigue classification using machine learning and multimodal behavioral fusion.

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

### Keyboard Node
- Privacy-safe behavioral telemetry
- Typing speed deviation analysis
- Pause behavior analysis
- Backspace deviation tracking
- Adaptive personal baseline modeling
- Rolling behavioral degradation metrics

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
- CSV dataset pipeline

---

## Project Structure

```text
multimodal-fatigue-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main_system.py                # Orchestrator — run this
│
├── camera_node/
│   ├── ear_detector.py
│   ├── mar_detector.py
│   ├── head_detector.py
│   ├── main_camera_system.py
│   └── camera_fatigue_dataset.csv
│
├── keyboard_node/
│   ├── keyboard_detector.py
│   ├── main_keyboard_system.py
│   └── keyboard_fatigue_dataset.csv
│
└── fusion_node/
    ├── dataset_fusion.py
    └── multimodal_fatigue_dataset.csv
```

---

## Installation & Usage

```bash
# 1. Install dependencies (Python 3.10–3.12 recommended)
pip install -r requirements.txt

# 2. Run the full pipeline (launches camera + keyboard nodes, then fuses)
python main_system.py
```

Controls during a session:

- **Camera window** — `0`/`1` set the manual fatigue label, `Q` quits.
- **Keyboard node** — `F1`/`F2` set the label (ALERT/FATIGUED), `ESC` quits.

When both nodes exit, the fusion node automatically merges the two CSVs into
`fusion_node/multimodal_fatigue_dataset.csv`. Let both nodes run for at least
~10 s so the first aggregation window is written.

---

## Future Work

- Multimodal feature fusion
- Fatigue classification using machine learning
- Real-time fatigue prediction
- Dataset expansion and evaluation
- Model benchmarking and validation
- Cross-user fatigue generalization
- Cross-user fatigue generalization