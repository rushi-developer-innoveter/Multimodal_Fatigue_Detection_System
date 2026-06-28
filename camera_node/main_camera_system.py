import csv
import os
import sys
import time
import warnings

import cv2
import mediapipe as mp

from ear_detector import EyeFatigueDetector
from mar_detector import MouthFatigueDetector
from head_detector import HeadPostureDetector

# ─────────────────────────────────────────────
# CAMERA CONFIG
# ─────────────────────────────────────────────
CAMERA_INDEX = 0

CAP_WIDTH = 1280
CAP_HEIGHT = 720

FPS = 30
SHOW_CAMERA_WINDOW = os.environ.get("SHOW_CAMERA_WINDOW", "1") == "1"

# ─────────────────────────────────────────────
# CSV CONFIG
# ─────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(_SCRIPT_DIR, "camera_fatigue_dataset.csv")

# Orchestrator shutdown signal path (one level up from camera_node/)
SHUTDOWN_FLAG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    ".shutdown_flag"
)

# ─────────────────────────────────────────────
# LIVE ML PREDICTION (additive, optional, fails gracefully)
# ─────────────────────────────────────────────
# Reads the keyboard node's CSV directly to fuse a feature vector live, using
# the same sync tolerance as fusion_node/dataset_fusion.py. Does not modify
# any detector logic, thresholds, or the camera CSV schema -- this only adds
# an on-screen overlay. If the model files are missing or joblib isn't
# installed, the system runs exactly as before with no overlay shown.
KEYBOARD_CSV = os.path.join(_SCRIPT_DIR, "..", "keyboard_node", "keyboard_fatigue_dataset.csv")
MODEL_PATH = os.path.join(_SCRIPT_DIR, "..", "fusion_node", "fatigue_model.pkl")
META_PATH = os.path.join(_SCRIPT_DIR, "..", "fusion_node", "fatigue_model_meta.pkl")
ML_SYNC_TOLERANCE_SECONDS = 7.0  # widened from 5.0: MediaPipe's import/init time
# delays the camera node's window timer relative to the keyboard node's by
# ~5s in practice (observed consistently, not random jitter), so 5.0s was
# too tight for the live overlay. The saved/fused dataset is unaffected --
# dataset_fusion.py uses its own separate tolerance and full nearest-match
# search across all rows, not just the latest one.

try:
    import joblib
    _ml_model = joblib.load(MODEL_PATH)
    _ml_meta = joblib.load(META_PATH)
    ML_AVAILABLE = True
    print("[ML] Fatigue model loaded -- live predictions enabled.")
except Exception as e:
    _ml_model = None
    _ml_meta = None
    ML_AVAILABLE = False
    print(f"[ML] Model not loaded ({e}) -- running without live predictions.")


def get_latest_keyboard_row():
    """
    Read the last row of the keyboard node's CSV. Returns a dict or None if
    the file doesn't exist yet or has no data rows. Read-only, never writes.
    """
    if not os.path.exists(KEYBOARD_CSV):
        return None
    try:
        with open(KEYBOARD_CSV, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        if not reader:
            return None
        return reader[-1]
    except Exception:
        return None


def predict_live(camera_row):
    """
    Attempt a live ML prediction by fusing the just-computed camera window
    with the most recent keyboard window, if it's within sync tolerance.
    Returns (label_text, color, confidence) or (None, None, None) if a
    prediction isn't currently possible (no model, no keyboard data yet,
    or the two windows are too far apart in time).

    Prints a one-line diagnostic to the console explaining exactly why a
    prediction wasn't made, so this is debuggable instead of silently
    showing "waiting for sync" with no clue why.
    """
    if not ML_AVAILABLE:
        return None, None, None

    kbd_row = get_latest_keyboard_row()
    if kbd_row is None:
        print("[ML DEBUG] No keyboard data found yet at:", KEYBOARD_CSV)
        return None, None, None

    try:
        delta = abs(camera_row["timestamp"] - float(kbd_row["timestamp"]))
    except (KeyError, ValueError, TypeError) as e:
        print(f"[ML DEBUG] Could not read timestamp from keyboard row: {e} -- row was: {kbd_row}")
        return None, None, None

    if delta > ML_SYNC_TOLERANCE_SECONDS:
        print(f"[ML DEBUG] Camera/keyboard windows too far apart: {delta:.1f}s (tolerance is {ML_SYNC_TOLERANCE_SECONDS}s)")
        return None, None, None

    try:
        feature_order = _ml_meta["feature_order"]
        clip_caps = _ml_meta["clip_caps"]

        merged = dict(camera_row)
        merged.update(kbd_row)

        vector = []
        for feat in feature_order:
            val = float(merged[feat])
            if feat in clip_caps:
                val = min(val, clip_caps[feat])
            vector.append(val)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            proba = _ml_model.predict_proba([vector])[0]
        pred = int(proba[1] >= 0.5)
        confidence = proba[pred]

        label_text = "FATIGUED" if pred == 1 else "ALERT"
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        return label_text, color, confidence
    except Exception as e:
        print(f"[ML DEBUG] Prediction failed unexpectedly: {type(e).__name__}: {e}")
        return None, None, None



CSV_COLUMNS = [
    "timestamp",

    "avg_ear",
    "min_ear",
    "blink_rate",
    "fatigue_eye_events",

    "avg_mar",
    "yawn_count",
    "max_yawn_duration",

    "avg_head_metric",
    "nod_count",
    "max_head_drop_duration",

    "fatigue_label"
]

BUFFER_WINDOW = 10.0

CAP_MAX_CONSECUTIVE_FAILS = 30


def initialize_csv():

    if os.path.exists(CSV_FILE):
        return

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS
        )

        writer.writeheader()


def append_csv_row(row):

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS
        )

        writer.writerow(row)


def main():

    initialize_csv()

    # CAMERA NODE INTEGRATION
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        print("[FATAL] Cannot open webcam.")

        return

    # Apply target resolution/FPS BEFORE any read happens. Changing these
    # properties immediately after a read (rather than before) caused a
    # native OpenCV buffer-size crash on some Windows webcam drivers --
    # setting them first, then probing, avoids that entirely.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Windows-specific safety net: some cameras "open" successfully but never
    # deliver frames on the default MSMF backend (handle opens, cap.read()
    # fails repeatedly). DSHOW is the standard fix for that exact symptom.
    # Only triggers if a real read failure occurs — has no effect when the
    # camera already works, so behavior is unchanged for working setups.
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
                print("[FATAL] Cannot open webcam.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)

    # CAMERA NODE INTEGRATION
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    eye_detector = EyeFatigueDetector()

    mouth_detector = MouthFatigueDetector()

    head_detector = HeadPostureDetector()

    buffer_start = time.time()

    consecutive_failures = 0

    manual_label = 0

    # Live ML prediction overlay state -- updated once per 10s window,
    # drawn every frame until the next update.
    ml_label_text = "ML: warming up..."
    ml_color = (200, 200, 200)

    print("[SYSTEM] Camera fatigue system running.")

    try:
        while True:

            try:
                ret, frame = cap.read()
            except cv2.error:
                ret, frame = False, None

            if not ret:

                consecutive_failures += 1

                if (
                    consecutive_failures
                    >= CAP_MAX_CONSECUTIVE_FAILS
                ):

                    print("[ERROR] Camera disconnected.")

                    break

                continue

            consecutive_failures = 0

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = face_mesh.process(rgb)

            landmarks = None

            if result.multi_face_landmarks:

                landmarks = (
                    result
                    .multi_face_landmarks[0]
                    .landmark
                )

            # CAMERA NODE INTEGRATION
            eye_detector.process(
                landmarks,
                frame.shape
            )

            mouth_detector.process(
                landmarks,
                frame.shape
            )

            head_detector.process(
                landmarks,
                frame.shape
            )

            # SYSTEM FATIGUE STATUS
            system_fatigued = (
                eye_detector.fatigue_event_count >= 2
                or mouth_detector.yawn_count >= 2
                or head_detector.nod_count >= 2
            )

            system_status = (
                "FATIGUED"
                if system_fatigued
                else "ALERT"
            )

            if SHOW_CAMERA_WINDOW:
                # OVERLAY
                eye_detector.draw_overlay(frame)

                mouth_detector.draw_overlay(frame)

                head_detector.draw_overlay(frame)

                cv2.putText(
                    frame,
                    "SYSTEM",
                    (850, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Status: {system_status}",
                    (850, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (
                        (0, 0, 255)
                        if system_fatigued
                        else (0, 255, 0)
                    ),
                    2
                )

                cv2.putText(
                    frame,
                    f"Manual Label: {manual_label}",
                    (850, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    ml_label_text,
                    (850, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    ml_color,
                    2
                )

            elapsed = time.time() - buffer_start

            # 10-SECOND AGGREGATION
            if elapsed >= BUFFER_WINDOW:

                eye_features = (
                    eye_detector.get_features()
                )

                mouth_features = (
                    mouth_detector.get_features()
                )

                head_features = (
                    head_detector.get_features()
                )

                row = {

                    "timestamp":
                        round(time.time(), 3),

                    "avg_ear":
                        eye_features.get(
                            "avg_ear",
                            0.0
                        ),

                    "min_ear":
                        eye_features.get(
                            "min_ear",
                            0.0
                        ),

                    "blink_rate":
                        eye_features.get(
                            "blink_rate",
                            0.0
                        ),

                    "fatigue_eye_events":
                        eye_features.get(
                            "fatigue_eye_events",
                            0
                        ),

                    "avg_mar":
                        mouth_features.get(
                            "avg_mar",
                            0.0
                        ),

                    "yawn_count":
                        mouth_features.get(
                            "yawn_count",
                            0
                        ),

                    "max_yawn_duration":
                        mouth_features.get(
                            "max_yawn_duration",
                            0.0
                        ),

                    "avg_head_metric":
                        head_features.get(
                            "avg_head_metric",
                            0.0
                        ),

                    "nod_count":
                        head_features.get(
                            "nod_count",
                            0
                        ),

                    "max_head_drop_duration":
                        head_features.get(
                            "max_head_drop_duration",
                            0.0
                        ),

                    "fatigue_label":
                        manual_label
                }

                append_csv_row(row)

                print("\n[DATASET ROW]")
                print(row)

                pred_label, pred_color, pred_conf = predict_live(row)
                if pred_label is not None:
                    ml_label_text = f"ML: {pred_label} ({pred_conf*100:.0f}%)"
                    ml_color = pred_color
                else:
                    ml_label_text = "ML: waiting for sync..."
                    ml_color = (200, 200, 200)

                eye_detector.reset_buffer()
                mouth_detector.reset_buffer()
                head_detector.reset_buffer()

                buffer_start = time.time()

            if SHOW_CAMERA_WINDOW:
                cv2.imshow(
                    "Unified Fatigue Camera System",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                # manual labels
                if key == ord("0"):
                    manual_label = 0

                elif key == ord("1"):
                    manual_label = 1

                elif key == ord("q"):
                    print("[INFO] Quit signal received.")
                    break

            # Orchestrator-controlled shutdown
            if os.path.exists(SHUTDOWN_FLAG):
                print("[INFO] Orchestrator shutdown signal detected.")
                break

        # ── POST-LOOP: flush partial window then release ────────────────

        elapsed = time.time() - buffer_start

        if elapsed >= 2.0:

            eye_features = eye_detector.get_features()
            mouth_features = mouth_detector.get_features()
            head_features = head_detector.get_features()

            if eye_features:

                row = {

                    "timestamp":
                        round(time.time(), 3),

                    "avg_ear":
                        eye_features.get(
                            "avg_ear",
                            0.0
                        ),

                    "min_ear":
                        eye_features.get(
                            "min_ear",
                            0.0
                        ),

                    "blink_rate":
                        eye_features.get(
                            "blink_rate",
                            0.0
                        ),

                    "fatigue_eye_events":
                        eye_features.get(
                            "fatigue_eye_events",
                            0
                        ),

                    "avg_mar":
                        mouth_features.get(
                            "avg_mar",
                            0.0
                        ),

                    "yawn_count":
                        mouth_features.get(
                            "yawn_count",
                            0
                        ),

                    "max_yawn_duration":
                        mouth_features.get(
                            "max_yawn_duration",
                            0.0
                        ),

                    "avg_head_metric":
                        head_features.get(
                            "avg_head_metric",
                            0.0
                        ),

                    "nod_count":
                        head_features.get(
                            "nod_count",
                            0
                        ),

                    "max_head_drop_duration":
                        head_features.get(
                            "max_head_drop_duration",
                            0.0
                        ),

                    "fatigue_label":
                        manual_label
                }

                append_csv_row(row)

                print("[DATASET] Final partial window flushed to CSV.")


    finally:
        # Guaranteed cleanup — runs even if a detector or frame op raises,
        # so the webcam handle and MediaPipe graph are always released.
        cap.release()
        cv2.destroyAllWindows()
        try:
            face_mesh.close()
        except Exception:
            pass
        print("[SYSTEM] Camera node exited cleanly.")


if __name__ == "__main__":

    main()