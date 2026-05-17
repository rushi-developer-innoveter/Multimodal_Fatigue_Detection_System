import csv
import os
import time

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

# ─────────────────────────────────────────────
# CSV CONFIG
# ─────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(_SCRIPT_DIR, "camera_fatigue_dataset.csv")

# Orchestrator shutdown signal path (one level up from camera_node/)
SHUTDOWN_FLAG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".shutdown_flag"
)

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

    print("[SYSTEM] Camera fatigue system running.")

    while True:

        ret, frame = cap.read()

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

            eye_detector.reset_buffer()
            mouth_detector.reset_buffer()
            head_detector.reset_buffer()

            buffer_start = time.time()

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

        # Flush partial telemetry window before releasing resources
        elapsed = time.time() - buffer_start
        if elapsed >= 2.0:
            eye_features = eye_detector.get_features()
            mouth_features = mouth_detector.get_features()
            head_features = head_detector.get_features()

            if eye_features:
                row = {
                    "timestamp": round(time.time(), 3),
                    "avg_ear": eye_features.get("avg_ear", 0.0),
                    "min_ear": eye_features.get("min_ear", 0.0),
                    "blink_rate": eye_features.get("blink_rate", 0.0),
                    "fatigue_eye_events": eye_features.get("fatigue_eye_events", 0),
                    "avg_mar": mouth_features.get("avg_mar", 0.0),
                    "yawn_count": mouth_features.get("yawn_count", 0),
                    "max_yawn_duration": mouth_features.get("max_yawn_duration", 0.0),
                    "avg_head_metric": head_features.get("avg_head_metric", 0.0),
                    "nod_count": head_features.get("nod_count", 0),
                    "max_head_drop_duration": head_features.get("max_head_drop_duration", 0.0),
                    "fatigue_label": manual_label,
                }
                append_csv_row(row)
                print("[DATASET] Final partial window flushed to CSV.")

        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Camera node exited cleanly.")


if __name__ == "__main__":
    main()