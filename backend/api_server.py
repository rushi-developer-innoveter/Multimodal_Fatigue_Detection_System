"""
Flask API server for real-time fatigue detection.
Runs a background camera + keyboard capture loop and exposes three endpoints
for a separate frontend to consume.

Run from the repo root:
    python backend/api_server.py

Endpoints:
    GET /api/health   -- liveness + model availability
    GET /api/status   -- latest 10-second window prediction
    GET /api/history  -- last 60 window readings
"""

import os
import sys
import time
import threading
import warnings

import cv2
import mediapipe as mp
from collections import deque
from flask import Flask, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# Detectors live in sibling dirs; insert them so imports resolve without moving
# any existing files.
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

sys.path.insert(0, os.path.join(_ROOT, "camera_node"))
sys.path.insert(0, os.path.join(_ROOT, "keyboard_node"))

from ear_detector import EyeFatigueDetector          # noqa: E402
from mar_detector import MouthFatigueDetector        # noqa: E402
from head_detector import HeadPostureDetector        # noqa: E402
from keyboard_detector import KeyboardTelemetryDetector  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAP_WIDTH = 1280
CAP_HEIGHT = 720
FPS = 30

BUFFER_WINDOW = 10.0               # seconds per aggregation window
ALARM_THRESHOLD_MINUTES = 3.0     # sustained fatigue before alarm fires
HISTORY_MAXLEN = 60               # rolling history window (readings)
CAP_MAX_CONSECUTIVE_FAILS = 30    # camera read failures before exit

_ALARM_THRESHOLD_WINDOWS = int(ALARM_THRESHOLD_MINUTES * 60 / BUFFER_WINDOW)  # 18

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — graceful degradation if files are missing or sklearn
# version mismatches; server still starts and serves /api/status.
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(_ROOT, "fusion_node", "fatigue_model.pkl")
_META_PATH = os.path.join(_ROOT, "fusion_node", "fatigue_model_meta.pkl")

try:
    import joblib
    _model = joblib.load(_MODEL_PATH)
    _meta = joblib.load(_META_PATH)
    MODEL_AVAILABLE = True
    print("[ML] Fatigue model loaded — live predictions enabled.")
except Exception as _e:
    _model = None
    _meta = None
    MODEL_AVAILABLE = False
    print(f"[ML] Model not loaded ({_e}) — running without predictions.")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE — all writes must hold _lock; Flask reads on a different thread.
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()

_state: dict = {
    "status": "starting",                          # ALERT | FATIGUED | NO_FACE_DETECTED | starting
    "fatigue_label": None,                         # ALERT | FATIGUED | None
    "fatigue_probability": None,                   # float [0, 1] | None
    "features": {},                                # merged feature dict for this window
    "timestamp": None,                             # float (Unix epoch)
    "alarm_triggered": False,                      # True once rolling-window threshold hit
    "consecutive_fatigued_windows": 0,
    "alarm_threshold_windows": _ALARM_THRESHOLD_WINDOWS,
    "fatigued_ratio_in_window": 0.0,              # fraction of last 18 windows that were FATIGUED
}

_history: list = []   # list of {timestamp, fatigue_probability, fatigue_label}


# ─────────────────────────────────────────────────────────────────────────────
# ALARM
# ─────────────────────────────────────────────────────────────────────────────
def _play_alarm() -> None:
    """Play a system beep. Failure never crashes detection."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 600)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def _predict(merged: dict):
    """
    Build the feature vector in model-expected order, apply clip caps,
    and return (label, probability) or (None, None) on any failure.
    """
    if not MODEL_AVAILABLE:
        return None, None
    try:
        feature_order = _meta["feature_order"]
        clip_caps = _meta["clip_caps"]
        vector = []
        for feat in feature_order:
            val = float(merged.get(feat, 0.0))
            if feat in clip_caps:
                val = min(val, clip_caps[feat])
            vector.append(val)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            proba = _model.predict_proba([vector])[0]
        fatigue_prob = float(proba[1])
        label = "FATIGUED" if fatigue_prob >= 0.5 else "ALERT"
        return label, fatigue_prob
    except Exception as e:
        print(f"[ML] Prediction failed: {type(e).__name__}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND CAPTURE LOOP
# ─────────────────────────────────────────────────────────────────────────────
def _capture_loop() -> None:
    """
    Opens the webcam, runs MediaPipe FaceMesh, and starts a pynput keyboard
    listener — all in this single process. Every BUFFER_WINDOW seconds it
    aggregates features, runs the model, updates shared state, and manages
    the sustained-fatigue alarm counter.
    """
    # ── Camera init ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[FATAL] Cannot open webcam.")
        return

    # Set resolution BEFORE the probe read — wrong ordering caused a native
    # buffer-size crash on some Windows webcam drivers (seen with MSMF backend).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if sys.platform == "win32":
        try:
            ret_probe, _ = cap.read()
        except cv2.error:
            ret_probe = False
        if not ret_probe:
            cap.release()
            print("[SYSTEM] Default backend failed to read a frame, retrying with DSHOW...")
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("[FATAL] Cannot open webcam with DSHOW backend.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    )

    # ── Detectors ─────────────────────────────────────────────────────────────
    eye_detector = EyeFatigueDetector()
    mouth_detector = MouthFatigueDetector()
    head_detector = HeadPostureDetector()
    kbd_detector = KeyboardTelemetryDetector()

    # ── Keyboard listener (pynput manages its own thread internally) ──────────
    kbd_listener = None
    try:
        from pynput.keyboard import Listener as KbdListener
        kbd_listener = KbdListener(on_press=kbd_detector.on_press)
        kbd_listener.daemon = True
        kbd_listener.start()
        print("[KBD] Keyboard listener started.")
    except Exception as e:
        print(f"[KBD] Could not start keyboard listener: {e} — keyboard features will be zero.")

    buffer_start = time.time()
    consecutive_failures = 0
    consecutive_fatigued = 0    # streak counter — for reporting only, no longer gates alarm
    fatigued_window = deque(maxlen=_ALARM_THRESHOLD_WINDOWS)  # rolling 1/0 per window
    alarm_fired = False          # guards against re-firing while ratio stays above threshold

    print("[SYSTEM] Capture loop running.")

    try:
        while True:
            # ── Frame capture ─────────────────────────────────────────────────
            try:
                ret, frame = cap.read()
            except cv2.error:
                ret, frame = False, None

            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= CAP_MAX_CONSECUTIVE_FAILS:
                    print("[ERROR] Camera disconnected or too many consecutive read failures.")
                    break
                continue

            consecutive_failures = 0

            # ── Face mesh ─────────────────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            landmarks = None
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

            eye_detector.process(landmarks, frame.shape)
            mouth_detector.process(landmarks, frame.shape)
            head_detector.process(landmarks, frame.shape)

            # ── 10-second window aggregation ──────────────────────────────────
            if time.time() - buffer_start >= BUFFER_WINDOW:
                eye_f = eye_detector.get_features()
                mouth_f = mouth_detector.get_features()
                head_f = head_detector.get_features()

                # process() updates the personal baseline THEN resets the
                # per-window buffer. Calling get_features() alone would silently
                # break baseline calibration for the deviation features.
                kbd_f = kbd_detector.process(BUFFER_WINDOW)

                eye_detector.reset_buffer()
                mouth_detector.reset_buffer()
                head_detector.reset_buffer()
                buffer_start = time.time()

                ts = round(time.time(), 3)

                # ── No-face window ────────────────────────────────────────────
                if not eye_f and not mouth_f and not head_f:
                    consecutive_fatigued = 0
                    fatigued_window.clear()
                    alarm_fired = False
                    with _lock:
                        _state["status"] = "NO_FACE_DETECTED"
                        _state["fatigue_label"] = None
                        _state["fatigue_probability"] = None
                        _state["features"] = {}
                        _state["timestamp"] = ts
                        _state["alarm_triggered"] = False
                        _state["consecutive_fatigued_windows"] = 0
                        _state["fatigued_ratio_in_window"] = 0.0
                        _history.append({
                            "timestamp": ts,
                            "fatigue_probability": None,
                            "fatigue_label": "NO_FACE_DETECTED",
                        })
                        if len(_history) > HISTORY_MAXLEN:
                            _history.pop(0)
                    continue

                # ── Merge features ────────────────────────────────────────────
                merged: dict = {}
                merged.update(eye_f)
                merged.update(mouth_f)
                merged.update(head_f)
                merged.update(kbd_f)

                # ── Predict ───────────────────────────────────────────────────
                label, prob = _predict(merged)

                # ── Alarm logic ───────────────────────────────────────────────
                # Update streak counter (reporting only).
                if label == "FATIGUED":
                    consecutive_fatigued += 1
                else:
                    consecutive_fatigued = 0

                # Rolling-window alarm: requires a full window of history and
                # at least 70 % of readings FATIGUED. A single stray ALERT
                # reading barely affects the ratio instead of wiping the streak.
                fatigued_window.append(1 if label == "FATIGUED" else 0)
                window_full = len(fatigued_window) == _ALARM_THRESHOLD_WINDOWS
                fatigued_ratio = (
                    sum(fatigued_window) / len(fatigued_window)
                    if fatigued_window else 0.0
                )
                alarm_triggered = window_full and fatigued_ratio >= 0.70

                if alarm_triggered and not alarm_fired:
                    alarm_fired = True
                    threading.Thread(target=_play_alarm, daemon=True).start()
                    print(
                        f"[ALARM] Rolling fatigue ratio {fatigued_ratio:.0%} over last "
                        f"{_ALARM_THRESHOLD_WINDOWS} windows — alarm triggered."
                    )
                elif not alarm_triggered:
                    alarm_fired = False

                status = label if label is not None else "starting"

                # Strip metadata-only keys (timestamp, fatigue_label) from the
                # features dict exposed by the API — those aren't model inputs.
                display_features = {
                    k: v for k, v in merged.items()
                    if k not in ("timestamp", "fatigue_label")
                }

                # ── Write shared state ────────────────────────────────────────
                with _lock:
                    _state["status"] = status
                    _state["fatigue_label"] = label
                    _state["fatigue_probability"] = prob
                    _state["features"] = display_features
                    _state["timestamp"] = ts
                    _state["alarm_triggered"] = alarm_triggered
                    _state["consecutive_fatigued_windows"] = consecutive_fatigued
                    _state["fatigued_ratio_in_window"] = round(fatigued_ratio, 3)
                    _history.append({
                        "timestamp": ts,
                        "fatigue_probability": prob,
                        "fatigue_label": label,
                    })
                    if len(_history) > HISTORY_MAXLEN:
                        _history.pop(0)

                print(
                    f"[WINDOW] ts={ts}  status={status}  prob={prob}  "
                    f"consec={consecutive_fatigued}  ratio={fatigued_ratio:.0%}({len(fatigued_window)}/{_ALARM_THRESHOLD_WINDOWS})"
                )

    finally:
        cap.release()
        try:
            face_mesh.close()
        except Exception:
            pass
        if kbd_listener is not None:
            try:
                kbd_listener.stop()
            except Exception:
                pass
        print("[SYSTEM] Capture loop exited.")


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response


@app.route("/api/health")
def health():
    """Liveness probe. model_available reflects whether .pkl files loaded."""
    return jsonify({"status": "ok", "model_available": MODEL_AVAILABLE})


@app.route("/api/status")
def status():
    """
    Latest single 10-second window reading.

    Fields:
        status                    -- ALERT | FATIGUED | NO_FACE_DETECTED | starting
        fatigue_label             -- ALERT | FATIGUED | null
        fatigue_probability       -- float [0, 1] | null
        features                  -- dict of raw feature values for this window
        timestamp                 -- Unix epoch float | null
        alarm_triggered           -- bool, true when ≥70 % of last 18 windows are FATIGUED
        consecutive_fatigued_windows -- int (streak counter, informational only)
        alarm_threshold_windows   -- int (18 for 3-min default)
        fatigued_ratio_in_window  -- float [0, 1], fraction of last N windows that were FATIGUED
    """
    with _lock:
        return jsonify(dict(_state))


@app.route("/api/history")
def history():
    """
    Last 60 window readings, oldest first.
    Each entry: {timestamp, fatigue_probability, fatigue_label}
    """
    with _lock:
        return jsonify(list(_history))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    capture_thread.start()

    print("[SERVER] Starting Flask on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
