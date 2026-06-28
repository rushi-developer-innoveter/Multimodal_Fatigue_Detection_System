

import time
import csv
import os
import threading
import sys

from pynput import keyboard as kb
from keyboard_detector import KeyboardTelemetryDetector

# ── Configuration ─────────────────────────────────────────────────────────────
WINDOW_SECONDS = 10
# Orchestrator shutdown signal path (one level up from keyboard_node/)
SHUTDOWN_FLAG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".shutdown_flag"
)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(_SCRIPT_DIR, "keyboard_fatigue_dataset.csv")
FIELDNAMES     = [
    "timestamp",
    "typing_speed",
    "speed_deviation",
    "avg_key_interval",
    "interval_deviation",
    "typing_variance",
    "pause_count",
    "pause_deviation",
    "backspace_rate",
    "backspace_deviation",
    "fatigue_label",
]

# ── Shared State ──────────────────────────────────────────────────────────────
_current_label  = 0          # 0 = ALERT, 1 = FATIGUED
_label_lock     = threading.Lock()
_shutdown_event = threading.Event()


# ── CSV Utilities ─────────────────────────────────────────────────────────────

def ensure_csv_header() -> None:
    """Create CSV with header if it does not exist or is empty."""
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        print(f"[CSV] Created: {CSV_FILE}")
    else:
        print(f"[CSV] Appending to: {CSV_FILE}")


def write_row(features: dict) -> None:
    """Append a single feature row to CSV. Thread-safe via GIL + append mode."""
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({k: features.get(k, "") for k in FIELDNAMES})
    except OSError as e:
        print(f"[CSV ERROR] Write failed: {e}")


# ── Display ───────────────────────────────────────────────────────────────────

def print_header() -> None:
    print("\n" + "═" * 56)
    print("  KEYBOARD TELEMETRY NODE  —  ACTIVE")
    print("═" * 56)
    print(f"  Window   : {WINDOW_SECONDS}s")
    print(f"  Output   : {CSV_FILE}")
    print(f"  F1       : Label → ALERT (0)")
    print(f"  F2       : Label → FATIGUED (1)")
    print(f"  ESC      : Shutdown")
    print("═" * 56 + "\n")


def print_snapshot(features: dict) -> None:
    label_str = "ALERT" if features["fatigue_label"] == 0 else "FATIGUED"
    ts = time.strftime("%H:%M:%S", time.localtime(features["timestamp"]))

    # Color codes for terminal
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RESET  = "\033[0m"

    label_color = GREEN if features["fatigue_label"] == 0 else RED

    print(f"\n{'─'*56}")
    print(f"  [{ts}]  Label: {label_color}{label_str}{RESET}")
    print(f"{'─'*56}")

    def deviation_marker(val: float) -> str:
        """Flag significant deviations."""
        if val > 1.5 or val < 0.5:
            return f" {YELLOW}⚠{RESET}"
        return ""

    print(f"  Typing Speed      : {features['typing_speed']:.3f} kps"
          f"  | Dev: {features['speed_deviation']:.3f}"
          f"{deviation_marker(features['speed_deviation'])}")

    print(f"  Avg Key Interval  : {features['avg_key_interval']:.4f}s"
          f"  | Dev: {features['interval_deviation']:.3f}"
          f"{deviation_marker(features['interval_deviation'])}")

    print(f"  Typing Variance   : {features['typing_variance']:.6f}")

    print(f"  Pauses            : {features['pause_count']}"
          f"         | Dev: {features['pause_deviation']:.3f}"
          f"{deviation_marker(features['pause_deviation'])}")

    print(f"  Backspace Rate    : {features['backspace_rate']:.4f}/s"
          f"  | Dev: {features['backspace_deviation']:.3f}"
          f"{deviation_marker(features['backspace_deviation'])}")
    print(f"{'─'*56}")


# ── Telemetry Loop ────────────────────────────────────────────────────────────

def telemetry_loop(detector: KeyboardTelemetryDetector) -> None:
    """
    Background thread: fires every WINDOW_SECONDS.
    Collects features, stamps label, writes CSV row.
    """
    global _current_label

    while not _shutdown_event.is_set():
        # Wait for window duration, but check shutdown every 0.5s
        for _ in range(WINDOW_SECONDS * 2):
            if _shutdown_event.is_set():
                break
            if os.path.exists(SHUTDOWN_FLAG):
                _shutdown_event.set()
                break
            time.sleep(0.5)

            # Flush final partial window before exit
        if _shutdown_event.is_set():
            with _label_lock:
                label = _current_label
            detector.baseline_locked = (label == 1)
            features = detector.process(WINDOW_SECONDS)
            features["fatigue_label"] = label
            write_row(features)
            print("[SYSTEM] Final keyboard telemetry window flushed.")
            return

        with _label_lock:
            label = _current_label

        detector.baseline_locked = (label == 1)
        features = detector.process(WINDOW_SECONDS)
        features["fatigue_label"] = label

        write_row(features)
        print_snapshot(features)


# ── Key Event Handler ─────────────────────────────────────────────────────────

def build_press_handler(detector: KeyboardTelemetryDetector):
    """
    Returns the on_press callback.
    F1/F2 update label. ESC triggers shutdown.
    All other keys: pass timing to detector — key identity discarded.
    """
    global _current_label

    def on_press(key):
        global _current_label

        try:
            # ── Control Keys ────────────────────────────────────────────────
            if key == kb.Key.f1:
                with _label_lock:
                    _current_label = 0
                print("\n[LABEL] → ALERT (0)")
                return  # Do NOT feed control keys into telemetry

            if key == kb.Key.f2:
                with _label_lock:
                    _current_label = 1
                print("\n[LABEL] → FATIGUED (1)")
                return

            if key == kb.Key.esc:
                print("\n[SYSTEM] ESC received — shutting down...")
                _shutdown_event.set()
                try:
                    with open(SHUTDOWN_FLAG, "w") as f:
                        f.write("shutdown")
                except OSError as e:
                    print(f"[WARN] Could not write shutdown flag: {e}")
                return False   # Stops pynput listener

        except AttributeError:
            pass  # Special key with no .char — normal, continue

        # ── Behavioral Telemetry (timing only) ──────────────────────────────
        detector.on_press(key)

    return on_press


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    ensure_csv_header()
    print_header()

    detector = KeyboardTelemetryDetector(
        baseline_window=200,
        buffer_window=100,
    )

    # Start background telemetry aggregation thread
    telem_thread = threading.Thread(
        target=telemetry_loop,
        args=(detector,),
        daemon=True,
        name="TelemetryLoop",
    )
    telem_thread.start()
    print("[SYSTEM] Telemetry loop started.")
    print("[SYSTEM] Keyboard listener active — type normally.\n")

    # Blocking keyboard listener — exits on ESC or exception
    try:
        with kb.Listener(on_press=build_press_handler(detector)) as listener:
            while listener.running:
                if _shutdown_event.is_set():
                    listener.stop()
                    break
                time.sleep(0.5)
    except Exception as e:
        print(f"[LISTENER ERROR] {e}")
    finally:
        _shutdown_event.set()
        telem_thread.join(timeout=2.0)
        print("[SYSTEM] Keyboard node terminated cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()