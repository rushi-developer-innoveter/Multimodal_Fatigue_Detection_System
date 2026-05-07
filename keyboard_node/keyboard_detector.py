"""
keyboard_detector.py
Privacy-Safe Adaptive Keyboard Telemetry Engine

PRIVACY GUARANTEE:
- NO typed content stored
- NO key sequences stored
- NO passwords captured
- ONLY timing intervals and statistical aggregates
"""

import time
import statistics
from collections import deque


class KeyboardTelemetryDetector:
    """
    Adaptive behavioral telemetry engine.
    Detects fatigue via DEVIATION from personal baseline — not absolute thresholds.
    """

    def __init__(self, baseline_window: int = 200, buffer_window: int = 100):
        # ── Rolling Baseline Storage (long-term personal behavior) ──────────
        self._baseline_intervals    = deque(maxlen=baseline_window)
        self._baseline_bs_rates     = deque(maxlen=baseline_window)
        self._baseline_pause_counts = deque(maxlen=50)

        # ── Per-Window Buffers (reset every 10s) ────────────────────────────
        self._window_intervals  = deque(maxlen=buffer_window)
        self._window_pauses     = []          # pause durations this window
        self._window_backspaces = 0
        self._window_keystrokes = 0

        # ── Timing State ────────────────────────────────────────────────────
        self._last_press_time  = None
        self._window_start     = None

        # ── Calibration Config ──────────────────────────────────────────────
        self._min_baseline_samples    = 20
        self._pause_multiplier        = 3.0   # interval > 3× baseline avg → pause
        self._startup_pause_threshold = 2.0   # fallback before baseline built
        # Baseline contamination guard — set True during FATIGUED sessions
        self.baseline_locked: bool = False

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────────────

    def on_press(self, key) -> None:
        """
        Process a single key event.
        Extracts ONLY timing data — key identity is discarded immediately
        after checking if it is a backspace.
        """
        now = time.perf_counter()

        # Identify backspace by type only — content never stored
        is_backspace = self._is_backspace(key)

        if self._window_start is None:
            self._window_start = now

        if self._last_press_time is not None:
            interval = now - self._last_press_time

            # Clamp extreme outliers (>30s) — likely system sleep / lock screen
            if interval <= 30.0:
                threshold = self._dynamic_pause_threshold()

                if interval >= threshold:
                    # This is a behavioral pause — store duration only
                    self._window_pauses.append(interval)
                else:
                    # Normal keystroke interval
                    self._window_intervals.append(interval)
                    if not self.baseline_locked:
                        self._baseline_intervals.append(interval)

        if is_backspace:
            self._window_backspaces += 1

        self._window_keystrokes += 1
        self._last_press_time = now

    def process(self, window_seconds: float = 10.0) -> dict:
        """
        Compute features for the completed window, update baselines,
        reset buffers. Returns ML-ready feature row.
        """
        features = self.get_features(window_seconds)
        self._update_rolling_baselines(features)
        self.reset_buffer()
        return features

    def get_features(self, window_seconds: float = 10.0) -> dict:
        """
        Generate ML-ready feature row representing behavioral deviation.
        All values are relative to personal baseline — NOT absolute.
        """
        now = time.time()
        ws  = max(window_seconds, 0.001)  # guard div-by-zero

        # ── 1. TYPING SPEED ─────────────────────────────────────────────────
        typing_speed    = self._window_keystrokes / ws
        baseline_speed  = self._baseline_speed()
        speed_deviation = (typing_speed / baseline_speed) if baseline_speed > 0 else 1.0

        # ── 2. KEY INTERVAL METRICS ─────────────────────────────────────────
        intervals = list(self._window_intervals)
        if len(intervals) >= 2:
            avg_key_interval = statistics.mean(intervals)
            typing_variance  = statistics.variance(intervals)
        elif len(intervals) == 1:
            avg_key_interval = intervals[0]
            typing_variance  = 0.0
        else:
            avg_key_interval = 0.0
            typing_variance  = 0.0

        interval_deviation = self._interval_deviation(avg_key_interval)

        # ── 3. PAUSE METRICS ─────────────────────────────────────────────────
        pause_count     = len(self._window_pauses)
        pause_deviation = self._pause_deviation(pause_count)

        # ── 4. BACKSPACE METRICS ─────────────────────────────────────────────
        backspace_rate      = self._window_backspaces / ws
        backspace_deviation = self._backspace_deviation(backspace_rate)

        return {
            "timestamp":           round(now, 3),
            "typing_speed":        round(typing_speed, 4),
            "speed_deviation":     round(speed_deviation, 4),
            "avg_key_interval":    round(avg_key_interval, 4),
            "interval_deviation":  round(interval_deviation, 4),
            "typing_variance":     round(typing_variance, 6),
            "pause_count":         pause_count,
            "pause_deviation":     round(pause_deviation, 4),
            "backspace_rate":      round(backspace_rate, 4),
            "backspace_deviation": round(backspace_deviation, 4),
            "fatigue_label":       -1,   # Applied externally via keyboard control
        }

    def reset_buffer(self) -> None:
        """
        Reset per-window counters. PRESERVES rolling baselines.
        Call after process() — baselines must be updated before reset.
        """
        self._window_intervals.clear()
        self._window_pauses     = []
        self._window_backspaces = 0
        self._window_keystrokes = 0
        self._window_start      = None

    # ────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_backspace(key) -> bool:
        """Detect backspace by type only. Never stores or logs the key."""
        try:
            from pynput.keyboard import Key
            return key == Key.backspace
        except Exception:
            return False

    def _has_baseline(self) -> bool:
        return len(self._baseline_intervals) >= self._min_baseline_samples

    def _baseline_avg_interval(self) -> float:
        if self._has_baseline():
            return statistics.mean(self._baseline_intervals)
        return 0.0

    def _baseline_speed(self) -> float:
        avg = self._baseline_avg_interval()
        return (1.0 / avg) if avg > 0 else 0.0

    def _dynamic_pause_threshold(self) -> float:
        avg = self._baseline_avg_interval()
        if avg > 0:
            return avg * self._pause_multiplier
        return self._startup_pause_threshold

    def _interval_deviation(self, current_avg: float) -> float:
        baseline_avg = self._baseline_avg_interval()
        if baseline_avg > 0 and current_avg > 0:
            return current_avg / baseline_avg
        return 1.0

    def _pause_deviation(self, current_pause_count: int) -> float:
        if len(self._baseline_pause_counts) >= 3:
            baseline_avg = statistics.mean(self._baseline_pause_counts)
            return (current_pause_count / baseline_avg) if baseline_avg > 0 else 1.0
        return 1.0

    def _backspace_deviation(self, current_rate: float) -> float:
        if len(self._baseline_bs_rates) >= self._min_baseline_samples:
            baseline_avg = statistics.mean(self._baseline_bs_rates)
            return (current_rate / baseline_avg) if baseline_avg > 0 else 1.0
        return 1.0

    def _update_rolling_baselines(self, features: dict) -> None:
        """Feed completed window metrics into rolling baselines.
        ONLY updates during ALERT state to prevent baseline contamination."""
        if self.baseline_locked:
            return
        self._baseline_pause_counts.append(features["pause_count"])
        self._baseline_bs_rates.append(features["backspace_rate"])