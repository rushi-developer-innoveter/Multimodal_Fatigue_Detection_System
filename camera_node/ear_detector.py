import time
from collections import deque

import cv2
import numpy as np

# ─────────────────────────────────────────────
# EYE LANDMARKS
# ─────────────────────────────────────────────
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
BASE_EAR_THRESHOLD = 0.20

BLINK_MAX_DURATION = 0.30
FATIGUE_MIN_DURATION = 0.50

WARMUP_TIME = 2.0

BASELINE_EAR_MIN = 0.15
BASELINE_EAR_MAX = 0.45

MIN_EYE_SPAN_PX = 1e-4
EAR_BUFFER_MAXLEN = 600


class EyeFatigueDetector:

    def __init__(self):

        self.start_time = time.time()
        self.in_warmup = True

        self.face_detected = False

        self.ear_raw = 0.0
        self.ear = 0.0

        self.baseline_ear = None
        self.ear_threshold = BASE_EAR_THRESHOLD

        self.eye_closed = False
        self.closed_duration = 0.0

        self.status = "WARMUP"

        self.blink_count = 0

        self._was_closed = False
        self._close_start = 0.0

        self.ear_buffer = deque(maxlen=EAR_BUFFER_MAXLEN)
        self.ear_smooth_buffer = deque(maxlen=5)

        self.blink_durations = []
        self.fatigue_durations = []

        self.fatigue_event_count = 0
        self.max_closure_duration = 0.0

        self._buffer_blinks = 0
        self._buffer_start = time.time()

        self.left_eye_pts = None
        self.right_eye_pts = None

    def compute_ear(self, eye):

        p1, p2, p3, p4, p5, p6 = eye

        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)

        h = np.linalg.norm(p1 - p4)

        if h < MIN_EYE_SPAN_PX:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)

        return float(np.clip(ear, 0.0, 1.0))

    # CAMERA NODE INTEGRATION
    def process(self, landmarks, frame_shape):

        if landmarks is None:
            self.face_detected = False
            self.status = "NO FACE"

            if self._was_closed:
                self._was_closed = False
                self.closed_duration = 0.0

            return

        self.face_detected = True

        h, w = frame_shape[:2]

        left = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in LEFT_EYE_IDX
        ])

        right = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in RIGHT_EYE_IDX
        ])

        if (
            np.any(~np.isfinite(left))
            or np.any(~np.isfinite(right))
        ):
            return

        # CAMERA NODE INTEGRATION
        self.left_eye_pts = left.astype(int)
        self.right_eye_pts = right.astype(int)

        ear_l = self.compute_ear(left)
        ear_r = self.compute_ear(right)

        self.ear_raw = (ear_l + ear_r) / 2.0

        self.ear_smooth_buffer.append(self.ear_raw)

        if len(self.ear_smooth_buffer) < 5:
            return

        self.ear = float(np.mean(self.ear_smooth_buffer))

        # warmup
        if time.time() - self.start_time < WARMUP_TIME:

            self.status = "WARMUP"

            self.ear_buffer.append(self.ear)

            if self.baseline_ear is None:
                self.baseline_ear = self.ear
            else:
                self.baseline_ear = (
                    0.9 * self.baseline_ear
                    + 0.1 * self.ear
                )

            self.baseline_ear = float(
                np.clip(
                    self.baseline_ear,
                    BASELINE_EAR_MIN,
                    BASELINE_EAR_MAX
                )
            )

            return

        self.in_warmup = False

        if self.baseline_ear:

            raw_threshold = self.baseline_ear * 0.75

            self.ear_threshold = float(
                np.clip(
                    raw_threshold,
                    BASE_EAR_THRESHOLD * 0.5,
                    BASE_EAR_THRESHOLD * 1.5
                )
            )

        self.eye_closed = self.ear < self.ear_threshold

        self.update_state()

        self.ear_buffer.append(self.ear)

    def update_state(self):

        if not self.face_detected or self.in_warmup:
            return

        if self.eye_closed:

            if not self._was_closed:
                self._close_start = time.time()
                self._was_closed = True

            self.closed_duration = (
                time.time() - self._close_start
            )

        else:

            if self._was_closed:

                duration = (
                    time.time() - self._close_start
                )

                if 0 < duration < 60.0:

                    if duration >= FATIGUE_MIN_DURATION:

                        self.fatigue_durations.append(duration)
                        self.fatigue_event_count += 1

                    if duration < BLINK_MAX_DURATION:

                        self.blink_count += 1
                        self._buffer_blinks += 1

                        self.blink_durations.append(duration)

                    self.max_closure_duration = max(
                        self.max_closure_duration,
                        duration
                    )

            self.closed_duration = 0.0
            self._was_closed = False

        if (
            self.max_closure_duration > 0.8
            or self.fatigue_event_count >= 2
        ):
            self.status = "FATIGUE"

        elif self.eye_closed:
            self.status = "CLOSED"

        else:
            self.status = "NORMAL"

    def get_features(self):

        elapsed = time.time() - self._buffer_start
        minutes = elapsed / 60.0

        snap = list(self.ear_buffer)

        if not snap:
            return {}

        return {
            "avg_ear":
                round(float(np.mean(snap)), 4),

            "min_ear":
                round(float(np.min(snap)), 4),

            "blink_rate":
                round(
                    self._buffer_blinks /
                    max(minutes, 1e-9),
                    2
                ),

            "fatigue_eye_events":
                self.fatigue_event_count,
        }

    def reset_buffer(self):

        self.ear_buffer.clear()

        self._buffer_blinks = 0

        self.fatigue_event_count = 0

        self.max_closure_duration = 0.0

        self.blink_durations = []
        self.fatigue_durations = []

        self._buffer_start = time.time()

    def draw_overlay(self, frame, x=20, y=30):

        color = (
            (0, 0, 255)
            if self.eye_closed
            else (0, 255, 0)
        )

        # ─────────────────────────────────────
        # VISUAL TRACKING
        # ─────────────────────────────────────
        if self.left_eye_pts is not None:

            for pt in self.left_eye_pts:
                cv2.circle(
                    frame,
                    tuple(pt),
                    2,
                    color,
                    -1
                )

            cv2.polylines(
                frame,
                [self.left_eye_pts],
                True,
                color,
                1
            )

        if self.right_eye_pts is not None:

            for pt in self.right_eye_pts:
                cv2.circle(
                    frame,
                    tuple(pt),
                    2,
                    color,
                    -1
                )

            cv2.polylines(
                frame,
                [self.right_eye_pts],
                True,
                color,
                1
            )

        # ─────────────────────────────────────
        # TEXT OVERLAY
        # ─────────────────────────────────────
        blink_rate = 0.0

        elapsed = time.time() - self._buffer_start

        if elapsed > 0:
            blink_rate = (
                    self._buffer_blinks /
                    (elapsed / 60.0)
            )

        lines = [
            "EYES",
            f"EAR: {self.ear:.3f}",
            f"Blinks: {self.blink_count}",
            f"Blink Rate: {blink_rate:.1f}/min",
            f"State: {self.status}"
        ]

        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x, y + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color if i > 0 else (255, 255, 0),
                2
            )