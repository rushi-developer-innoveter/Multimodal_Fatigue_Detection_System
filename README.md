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
├── LICENSE
├── .gitignore
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
└── datasets/
```

---

## Future Work

- Multimodal feature fusion
- Fatigue classification using machine learning
- Real-time fatigue prediction
- Dataset expansion and evaluation
- Model benchmarking and validation
- Cross-user fatigue generalization
- Cross-user fatigue generalization

