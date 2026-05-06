# Multimodal Fatigue Detection System
<br>
A real-time behavioral fatigue detection system using:
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head posture/nod analysis

The system performs synchronized multimodal feature extraction using MediaPipe FaceMesh and generates ML-ready fatigue datasets through 10-second behavioral aggregation windows.


## Features

- Real-time eye fatigue detection
- Blink and microsleep analysis
- Real-time yawn detection
- Adaptive mouth fatigue tracking
- Forward head-drop and nod detection
- Unified synchronized camera pipeline
- Visual landmark tracking overlays
- ML-ready CSV dataset generation
- Modular detector architecture
- Real-time behavioral feature aggregation

## Tech Stack

- Python
- OpenCV
- MediaPipe FaceMesh
- NumPy
- CSV-based dataset pipeline

## Project Structure

project/
│
├── ear_detector.py
├── mar_detector.py
├── head_detector.py
├── main_camera_system.py
│
└── camera_fatigue_dataset.csv

## Future Work

- Keyboard behavior analysis
- Multimodal feature fusion
- Fatigue classification using ML
- Real-time fatigue prediction
- Dataset expansion and evaluation
