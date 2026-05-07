import time
from collections import deque

import cv2
import numpy as np

# ─────────────────────────────────────────────
# MOUTH LANDMARKS
# ─────────────────────────────────────────────
MOUTH_IDX = {
    "left_corner": 61,
    "right_corner": 291,

    "upper_outer_1": 37,
    "lower_outer_1": 267,

    "upper_inner_1": 13,
    "lower_inner_1": 14,

    "upper_inner_2": 82,
    "lower_inner_2": 87,
}

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
MAR_THRESHOLD = 0.60

YAWN_MIN_DURATION = 1.0
TALKING_IGNORE_DURATION = 0.50

WARMUP_TIME = 2.0

MAR_CLIP_MAX = 1.0

MIN_MOUTH_WIDTH_PX = 1e-4

BASELINE_BUFFER_MIN = 20
BASELINE_CLIP_MIN = 0.20
BASELINE_CLIP_MAX = 0.45

ADAPTIVE_THRESHOLD_MIN = 0.45
ADAPTIVE_THRESHOLD_MAX = 0.75
ADAPTIVE_THRESHOLD_SCALE = 2.0

MAX_FRAME_DELTA_MAR = 0.18

MAR_BUFFER_MAXLEN = 600


class MouthFatigueDetector:

    def __init__(self):

        self.start_time = time.time()

        self.in_warmup = True

        self.face_detected = False

        self.mar_raw = 0.0
        self.mar = 0.0

        self.mar_smooth = deque(maxlen=5)

        self.prev_mar_raw = None

        self.baseline_mar = 0.30

        self.baseline_samples = []

        self.adaptive_mar_threshold = MAR_THRESHOLD

        self.mouth_open = False
        self.open_duration = 0.0

        self.status = "WARMUP"

        self._was_open = False
        self._open_start = 0.0

        self.yawn_count = 0

        self.yawn_durations = []

        self.max_yawn_duration = 0.0

        self.mar_buffer = deque(maxlen=MAR_BUFFER_MAXLEN)

        self._buf_yawn_count = 0
        self._buf_yawn_dur = []

        self._buffer_start = time.time()

        self.mouth_pts = None

    def compute_mar(self, pts):

        p1 = pts["left_corner"]
        p4 = pts["right_corner"]

        v_pairs = [
            (
                pts["upper_outer_1"],
                pts["lower_outer_1"]
            ),
            (
                pts["upper_inner_1"],
                pts["lower_inner_1"]
            ),
            (
                pts["upper_inner_2"],
                pts["lower_inner_2"]
            ),
        ]

        h = np.linalg.norm(p1 - p4)

        if h < MIN_MOUTH_WIDTH_PX:
            return 0.0

        verticals = [
            np.linalg.norm(top - bottom)
            for top, bottom in v_pairs
        ]

        mar = np.mean(verticals) / h

        return float(
            np.clip(
                mar,
                0.0,
                MAR_CLIP_MAX
            )
        )

    # CAMERA NODE INTEGRATION
    def process(self, landmarks, frame_shape):

        if landmarks is None:

            self.face_detected = False

            self.status = "NO FACE"

            if self._was_open:
                self._was_open = False
                self.open_duration = 0.0

            return

        self.face_detected = True

        h, w = frame_shape[:2]

        pts = {}

        for name, idx in MOUTH_IDX.items():

            pts[name] = np.array([
                landmarks[idx].x * w,
                landmarks[idx].y * h
            ])

        coords = np.stack(list(pts.values()))

        if not np.all(np.isfinite(coords)):
            return

        # CAMERA NODE INTEGRATION
        self.mouth_pts = coords.astype(int)

        raw_mar = self.compute_mar(pts)

        if self.prev_mar_raw is not None:

            delta = raw_mar - self.prev_mar_raw

            if abs(delta) > MAX_FRAME_DELTA_MAR:

                raw_mar = (
                    self.prev_mar_raw
                    + np.sign(delta)
                    * MAX_FRAME_DELTA_MAR
                )

        self.prev_mar_raw = raw_mar

        self.mar_raw = raw_mar

        self.mar_smooth.append(self.mar_raw)

        if len(self.mar_smooth) < 5:
            return

        self.mar = float(np.mean(self.mar_smooth))

        if time.time() - self.start_time < WARMUP_TIME:

            self.status = "WARMUP"

            if 0.10 <= self.mar <= 0.50:
                self.baseline_samples.append(self.mar)

            if (
                len(self.baseline_samples)
                >= BASELINE_BUFFER_MIN
            ):

                baseline = float(
                    np.median(self.baseline_samples)
                )

                self.baseline_mar = float(
                    np.clip(
                        baseline,
                        BASELINE_CLIP_MIN,
                        BASELINE_CLIP_MAX
                    )
                )

                adaptive = (
                    self.baseline_mar
                    * ADAPTIVE_THRESHOLD_SCALE
                )

                self.adaptive_mar_threshold = float(
                    np.clip(
                        adaptive,
                        ADAPTIVE_THRESHOLD_MIN,
                        ADAPTIVE_THRESHOLD_MAX
                    )
                )

            self.mar_buffer.append(self.mar)

            return

        self.in_warmup = False

        self.mouth_open = (
            self.mar >
            self.adaptive_mar_threshold
        )

        self.update_state()

        self.mar_buffer.append(self.mar)

    def update_state(self):

        if not self.face_detected or self.in_warmup:
            return

        if self.mouth_open:

            if not self._was_open:

                self._open_start = time.time()

                self._was_open = True

            self.open_duration = (
                time.time() - self._open_start
            )

        else:

            if self._was_open:

                duration = (
                    time.time() - self._open_start
                )

                if 0 < duration < 60.0:

                    if duration >= YAWN_MIN_DURATION:

                        self.yawn_count += 1

                        self.yawn_durations.append(duration)

                        self.max_yawn_duration = max(
                            self.max_yawn_duration,
                            duration
                        )

                        self._buf_yawn_count += 1

                        self._buf_yawn_dur.append(duration)

            self.open_duration = 0.0
            self._was_open = False

        if (
            self.mouth_open
            and self.open_duration >= YAWN_MIN_DURATION
        ):
            self.status = "YAWN"

        elif self.mouth_open:
            self.status = "OPEN"

        else:
            self.status = "NORMAL"

    def get_features(self):

        snap = list(self.mar_buffer)

        if not snap:
            return {}

        return {

            "avg_mar":
                round(float(np.mean(snap)), 4),

            "yawn_count":
                self._buf_yawn_count,

            "max_yawn_duration":
                round(
                    float(
                        np.max(self._buf_yawn_dur)
                    )
                    if self._buf_yawn_dur
                    else 0.0,
                    4
                )
        }

    def reset_buffer(self):

        self.mar_buffer.clear()

        self._buf_yawn_count = 0
        self._buf_yawn_dur = []

        self._buffer_start = time.time()

    def draw_overlay(self, frame, x=20, y=220):

        color = (
            (0, 0, 255)
            if self.mouth_open
            else (0, 255, 0)
        )

        # ─────────────────────────────────────
        # VISUAL TRACKING
        # ─────────────────────────────────────
        if self.mouth_pts is not None:

            for pt in self.mouth_pts:
                cv2.circle(
                    frame,
                    tuple(pt),
                    2,
                    color,
                    -1
                )

            hull = cv2.convexHull(
                self.mouth_pts
            )

            cv2.polylines(
                frame,
                [hull],
                True,
                color,
                1
            )

        # ─────────────────────────────────────
        # TEXT OVERLAY
        # ─────────────────────────────────────
        lines = [
            "MOUTH",
            f"MAR: {self.mar:.3f}",
            f"Yawns: {self.yawn_count}",
            f"Yawn Dur: {self.open_duration:.2f}s",
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