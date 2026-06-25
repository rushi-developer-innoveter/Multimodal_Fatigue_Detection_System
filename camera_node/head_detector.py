import time
from collections import deque

import cv2
import numpy as np

# ─────────────────────────────────────────────
# LANDMARKS
# ─────────────────────────────────────────────
IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
HEAD_DROP_THRESHOLD = 0.08

NOD_MIN_DURATION = 1.0

WARMUP_TIME = 2.0

METRIC_CLIP_MIN = 0.0
METRIC_CLIP_MAX = 1.5

MAX_FRAME_METRIC_JUMP = 0.15

ANGLE_BUFFER_MAXLEN = 600


class HeadPostureDetector:

    def __init__(self):

        self.start_time = time.time()

        self.in_warmup = True

        self.face_detected = False

        self.raw_metric = 0.0
        self.metric = 0.0

        self.baseline_metric = None

        self._prev_raw = None

        self._metric_smooth = deque(maxlen=5)

        self.head_dropped = False

        self.head_drop = 0.0


        self.drop_duration = 0.0

        self.status = "WARMUP"

        self._was_dropped = False
        self._drop_start = 0.0

        self.nod_count = 0

        self.metric_buffer = deque(
            maxlen=ANGLE_BUFFER_MAXLEN
        )

        self.drop_buffer = deque(
            maxlen=ANGLE_BUFFER_MAXLEN
        )

        self._buf_nod_count = 0

        self._buf_drop_durs = []

        self._buffer_start = time.time()

        self.max_head_drop_dur = 0.0

        self.nose_pt = None
        self.left_eye_pt = None
        self.right_eye_pt = None
        self.chin_pt = None

    def compute_head_drop_metric(
        self,
        nose,
        leye,
        reye,
        chin
    ):

        eye_mid_y = (
            leye[1] + reye[1]
        ) / 2.0

        nose_y = nose[1]
        chin_y = chin[1]

        face_h = chin_y - eye_mid_y

        if face_h < 1e-4:
            return (
                self.metric
                if self.metric != 0.0
                else 0.5
            )

        nose_eye = (
            (nose_y - eye_mid_y)
            / face_h
        )

        nose_chin = (
            (chin_y - nose_y)
            / face_h
        )

        compression_signal = 1.0 - nose_chin

        metric = (
            0.72 * nose_eye
            + 0.28 * compression_signal
        )

        return float(
            np.clip(
                metric,
                METRIC_CLIP_MIN,
                METRIC_CLIP_MAX
            )
        )

    # CAMERA NODE INTEGRATION
    def process(self, landmarks, frame_shape):

        if landmarks is None:

            self.face_detected = False

            self.status = "NO FACE"

            if self._was_dropped:

                self._was_dropped = False

                self.drop_duration = 0.0

            return

        self.face_detected = True

        h, w = frame_shape[:2]

        nose = np.array([
            landmarks[IDX_NOSE_TIP].x * w,
            landmarks[IDX_NOSE_TIP].y * h
        ])

        chin = np.array([
            landmarks[IDX_CHIN].x * w,
            landmarks[IDX_CHIN].y * h
        ])

        leye = np.array([
            landmarks[IDX_LEFT_EYE].x * w,
            landmarks[IDX_LEFT_EYE].y * h
        ])

        reye = np.array([
            landmarks[IDX_RIGHT_EYE].x * w,
            landmarks[IDX_RIGHT_EYE].y * h
        ])

        pts = np.stack([
            nose,
            chin,
            leye,
            reye
        ])

        if not np.all(np.isfinite(pts)):
            return

        # CAMERA NODE INTEGRATION
        self.nose_pt = nose.astype(int)
        self.left_eye_pt = leye.astype(int)
        self.right_eye_pt = reye.astype(int)
        self.chin_pt = chin.astype(int)

        raw = self.compute_head_drop_metric(
            nose,
            leye,
            reye,
            chin
        )

        if self._prev_raw is not None:

            jump = abs(raw - self._prev_raw)

            if jump > MAX_FRAME_METRIC_JUMP:
                raw = self._prev_raw

        self._prev_raw = raw

        self.raw_metric = raw

        self._metric_smooth.append(raw)

        if len(self._metric_smooth) < 5:
            return

        self.metric = float(
            np.mean(self._metric_smooth)
        )

        if time.time() - self.start_time < WARMUP_TIME:

            self.status = "WARMUP"

            self.metric_buffer.append(self.metric)

            if self.baseline_metric is None:

                self.baseline_metric = self.metric

            else:

                self.baseline_metric = (
                    0.9 * self.baseline_metric
                    + 0.1 * self.metric
                )

            return

        self.in_warmup = False

        # Safety: if no face was seen during the warmup window, baseline_metric
        # was never initialised. Seed it from the first post-warmup sample —
        # exactly what the warmup branch above would have done on its first hit.
        # Prevents a `float - None` TypeError that crashes the camera node.
        if self.baseline_metric is None:
            self.baseline_metric = self.metric

        if abs(self.metric - self.baseline_metric) < 0.02:

            self.baseline_metric = (
                0.995 * self.baseline_metric
                + 0.005 * self.metric
            )

        self.head_drop = (
            self.metric
            - self.baseline_metric
        )

        self.head_dropped = (
            self.head_drop
            > HEAD_DROP_THRESHOLD
        )

        self.update_state()


        self.metric_buffer.append(self.metric)

        self.drop_buffer.append(
            max(self.head_drop, 0.0)
        )

    def update_state(self):

        if not self.face_detected or self.in_warmup:
            return

        if self.head_dropped:

            if not self._was_dropped:

                self._drop_start = time.time()

                self._was_dropped = True

            self.drop_duration = (
                time.time() - self._drop_start
            )

        else:

            if self._was_dropped:

                duration = (
                    time.time() - self._drop_start
                )

                if 0 < duration < 60.0:

                    if duration >= NOD_MIN_DURATION:

                        self.nod_count += 1

                        self._buf_nod_count += 1

                        self._buf_drop_durs.append(
                            duration
                        )

                        self.max_head_drop_dur = max(
                            self.max_head_drop_dur,
                            duration
                        )

            self.drop_duration = 0.0
            self._was_dropped = False

        if (
            self.head_dropped
            and self.drop_duration >= NOD_MIN_DURATION
        ):
            self.status = "FATIGUE_NOD"

        elif self.head_dropped:
            self.status = "DROPPED"

        else:
            self.status = "NORMAL"

    def get_features(self):

        metric_snap = list(self.metric_buffer)

        if not metric_snap:
            return {}

        return {

            "avg_head_metric":
                round(float(np.mean(metric_snap)), 4),

            "nod_count":
                self._buf_nod_count,

            "max_head_drop_duration":
                round(
                    float(np.max(self._buf_drop_durs))
                    if self._buf_drop_durs
                    else 0.0,
                    4
                )
        }

    def reset_buffer(self):

        self.metric_buffer.clear()

        self.drop_buffer.clear()

        self._buf_nod_count = 0

        self._buf_drop_durs = []

        self._buffer_start = time.time()

    def draw_overlay(self, frame, x=20, y=410):

        color = (
            (0, 0, 255)
            if self.head_dropped
            else (0, 255, 0)
        )

        # ─────────────────────────────────────
        # VISUAL TRACKING
        # ─────────────────────────────────────
        if (
                self.nose_pt is not None
                and self.left_eye_pt is not None
                and self.right_eye_pt is not None
                and self.chin_pt is not None
        ):

            eye_mid = (
                    (
                            self.left_eye_pt +
                            self.right_eye_pt
                    ) / 2
            ).astype(int)

            # eye center line
            cv2.line(
                frame,
                tuple(self.left_eye_pt),
                tuple(self.right_eye_pt),
                (255, 200, 0),
                1
            )

            # posture tracking line
            cv2.line(
                frame,
                tuple(eye_mid),
                tuple(self.nose_pt),
                color,
                2
            )

            # vertical face line
            cv2.line(
                frame,
                tuple(self.nose_pt),
                tuple(self.chin_pt),
                (200, 200, 200),
                1
            )

            for pt in [
                self.nose_pt,
                self.left_eye_pt,
                self.right_eye_pt,
                self.chin_pt
            ]:
                cv2.circle(
                    frame,
                    tuple(pt),
                    3,
                    color,
                    -1
                )

        # ─────────────────────────────────────
        # TEXT OVERLAY
        # ─────────────────────────────────────
        lines = [
            "HEAD",
            f"Metric: {self.metric:.3f}",
            f"Drop: {self.head_drop:.3f}",
            f"Nods: {self.nod_count}",
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