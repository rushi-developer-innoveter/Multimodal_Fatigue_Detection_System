"""
dataset_fusion.py
─────────────────────────────────────────────────────────────────────────────
Multimodal Fusion Layer — Behavioral Fatigue Detection
─────────────────────────────────────────────────────────────────────────────

PURPOSE:
    Synchronizes and merges camera and keyboard behavioral telemetry into a
    single ML-ready multimodal dataset.

INPUT:
    camera_fatigue_dataset.csv   — camera node output (10s windows)
    keyboard_fatigue_dataset.csv — keyboard node output (10s windows)

OUTPUT:
    multimodal_fatigue_dataset.csv — fused, timestamp-synchronized dataset

SYNCHRONIZATION STRATEGY:
    Nearest-timestamp matching within a configurable tolerance window.
    Both nodes operate on 10-second behavioral windows; matches within
    SYNC_TOLERANCE_SECONDS are accepted. Unmatched rows are discarded to
    preserve label integrity.

RULES:
    ✔ Camera node is NOT modified.
    ✔ Keyboard node is NOT modified.
    ✔ Detector logic, thresholds, baselines — all untouched.
    ✔ This file is the ONLY new artifact.
─────────────────────────────────────────────────────────────────────────────
"""

import csv
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_CSV = os.path.join(
    _SCRIPT_DIR,
    "..",
    "camera_node",
    "camera_fatigue_dataset.csv"
)

KEYBOARD_CSV = os.path.join(
    _SCRIPT_DIR,
    "..",
    "keyboard_node",
    "keyboard_fatigue_dataset.csv"
)

OUTPUT_CSV = os.path.join(
    _SCRIPT_DIR,
    "multimodal_fatigue_dataset.csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# SYNC CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Maximum allowed timestamp delta (seconds) for a camera↔keyboard pair to be
# considered "the same behavioral window".  Both nodes tick every 10 s, so a
# tolerance of 5 s comfortably covers any clock skew or startup offset while
# still being strict enough to avoid cross-window contamination.
SYNC_TOLERANCE_SECONDS = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Camera source columns  →  fused output column names
# Names are preserved exactly as emitted by the camera node.
# No semantic aliasing: each column means precisely what the detector computed.
CAMERA_COLUMN_MAP = {
    "avg_ear":                "avg_ear",
    "min_ear":                "min_ear",
    "blink_rate":             "blink_rate",
    "fatigue_eye_events":     "fatigue_eye_events",     # count of fatigue-threshold eye events
    "avg_mar":                "avg_mar",
    "yawn_count":             "yawn_count",
    "max_yawn_duration":      "max_yawn_duration",
    "avg_head_metric":        "avg_head_metric",        # normalized head-drop magnitude
    "nod_count":              "nod_count",              # discrete nod events, not a distance
    "max_head_drop_duration": "max_head_drop_duration",
}

# Keyboard source columns  →  fused output column names (already match spec)
KEYBOARD_COLUMN_MAP = {
    "typing_speed":       "typing_speed",
    "speed_deviation":    "speed_deviation",
    "avg_key_interval":   "avg_key_interval",
    "interval_deviation": "interval_deviation",
    "typing_variance":    "typing_variance",
    "pause_count":        "pause_count",
    "pause_deviation":    "pause_deviation",
    "backspace_rate":     "backspace_rate",
    "backspace_deviation":"backspace_deviation",
}

# Final fused CSV column order
OUTPUT_COLUMNS = [
    "timestamp",
    # ── Camera features (names match camera node output exactly) ──────────
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
    # ── Keyboard features ─────────────────────────────────────────────────
    "typing_speed",
    "speed_deviation",
    "avg_key_interval",
    "interval_deviation",
    "typing_variance",
    "pause_count",
    "pause_deviation",
    "backspace_rate",
    "backspace_deviation",
    # ── Label ─────────────────────────────────────────────────────────────
    "fatigue_label",
]


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    """
    Load a telemetry CSV into a list of row dicts.

    Handles:
      - Missing file              → returns []
      - Empty file                → returns []
      - Malformed / non-numeric timestamps → row silently skipped
      - Corrupted numeric values  → field set to 0.0, row retained
    """
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return []

    rows = []

    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                print(f"[WARN] Empty or headerless file: {path}")
                return []

            for line_num, raw in enumerate(reader, start=2):
                # ── Timestamp: must be a valid float ──────────────────────
                try:
                    ts = float(raw.get("timestamp", "").strip())
                except (ValueError, AttributeError):
                    print(
                        f"[SKIP] {path} line {line_num}: "
                        f"unparseable timestamp → '{raw.get('timestamp')}'"
                    )
                    continue

                # ── Numeric fields: coerce; keep row even if one is bad ───
                clean = {"timestamp": ts}
                for key, value in raw.items():
                    if key == "timestamp":
                        continue
                    try:
                        clean[key] = float(str(value).strip())
                    except (ValueError, AttributeError):
                        clean[key] = 0.0  # safe fallback

                rows.append(clean)

    except OSError as exc:
        print(f"[ERROR] Cannot read {path}: {exc}")
        return []

    print(f"[LOAD] {path}: {len(rows)} rows loaded")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SYNCHRONIZATION
# ─────────────────────────────────────────────────────────────────────────────

def sync_nearest(
    camera_rows: list[dict],
    keyboard_rows: list[dict],
    tolerance: float,
) -> list[tuple[dict, dict]]:
    """
    Nearest-timestamp matching between two sorted telemetry streams.

    Algorithm:
      - Sort both lists by timestamp (defensive; nodes already emit in order).
      - For each camera row, find the keyboard row with the smallest |Δt|.
      - Accept the pair only if |Δt| ≤ tolerance.
      - Each keyboard row may be matched to at most ONE camera row
        (greedy, first-wins) to prevent duplicate training samples.

    Returns:
      List of (camera_row, keyboard_row) pairs, ordered by camera timestamp.
    """
    if not camera_rows or not keyboard_rows:
        return []

    # Sort defensively
    cam_sorted  = sorted(camera_rows,   key=lambda r: r["timestamp"])
    kbd_sorted  = sorted(keyboard_rows, key=lambda r: r["timestamp"])

    matched: list[tuple[dict, dict]] = []
    used_kbd_indices: set[int] = set()

    kbd_len = len(kbd_sorted)

    for cam_row in cam_sorted:
        cam_ts = cam_row["timestamp"]

        best_idx   = None
        best_delta = float("inf")

        # Linear scan is acceptable for typical dataset sizes (hours of data
        # ≈ thousands of rows).  A binary-search optimisation would add
        # complexity with no practical benefit at this scale.
        for i, kbd_row in enumerate(kbd_sorted):
            if i in used_kbd_indices:
                continue

            delta = abs(kbd_row["timestamp"] - cam_ts)

            if delta < best_delta:
                best_delta = delta
                best_idx   = i

            # Since both lists are sorted and we've passed the tolerance
            # window, further rows can only be farther away.
            if kbd_row['timestamp'] > cam_ts + tolerance:
                if i not in used_kbd_indices:
                    break
                continue

        if best_idx is not None and best_delta <= tolerance:
            matched.append((cam_row, kbd_sorted[best_idx]))
            used_kbd_indices.add(best_idx)

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# ROW BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_fused_row(cam: dict, kbd: dict) -> dict:
    """
    Merge one camera row and one keyboard row into a single fused row.

    Feature naming:
      All camera features are passed through under their original source names.
      No semantic aliasing is applied.

    Label strategy:
      Camera label is the sole ground truth.  The camera node captures direct
      physiological fatigue signals (eye closure, yawns, head nodding).
      Keyboard telemetry is supporting behavioral evidence — it must NOT
      override or contaminate the physiological label.
    """
    row = {
        "timestamp": cam["timestamp"],

        # ── Camera features (source names preserved exactly) ──────────────
        "avg_ear":                cam.get("avg_ear",                0.0),
        "min_ear":                cam.get("min_ear",                0.0),
        "blink_rate":             cam.get("blink_rate",             0.0),
        "fatigue_eye_events":     cam.get("fatigue_eye_events",     0.0),
        "avg_mar":                cam.get("avg_mar",                0.0),
        "yawn_count":             cam.get("yawn_count",             0.0),
        "max_yawn_duration":      cam.get("max_yawn_duration",      0.0),
        "avg_head_metric":        cam.get("avg_head_metric",        0.0),
        "nod_count":              cam.get("nod_count",              0.0),
        "max_head_drop_duration": cam.get("max_head_drop_duration", 0.0),

        # ── Keyboard features ─────────────────────────────────────────────
        "typing_speed":        kbd.get("typing_speed",        0.0),
        "speed_deviation":     kbd.get("speed_deviation",     0.0),
        "avg_key_interval":    kbd.get("avg_key_interval",    0.0),
        "interval_deviation":  kbd.get("interval_deviation",  0.0),
        "typing_variance":     kbd.get("typing_variance",     0.0),
        "pause_count":         kbd.get("pause_count",         0.0),
        "pause_deviation":     kbd.get("pause_deviation",     0.0),
        "backspace_rate":      kbd.get("backspace_rate",       0.0),
        "backspace_deviation": kbd.get("backspace_deviation",  0.0),

        # ── Label: camera node is sole ground truth ───────────────────────
        "fatigue_label": int(cam.get("fatigue_label", 0)),
    }

    return row


# ─────────────────────────────────────────────────────────────────────────────
# CSV WRITING
# ─────────────────────────────────────────────────────────────────────────────

def write_fused_csv(rows: list[dict], output_path: str) -> None:
    """
    Write fused rows to output CSV.  Creates a fresh file (overwrites if
    exists) so each fusion run is fully reproducible.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            # Round floats to 6 decimal places for clean ML input
            clean = {}
            for col in OUTPUT_COLUMNS:
                val = row.get(col, 0.0)
                if isinstance(val, float):
                    clean[col] = round(val, 6)
                else:
                    clean[col] = val
            writer.writerow(clean)

    print(f"[OUTPUT] Written: {output_path}  ({len(rows)} fused rows)")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    cam_count:  int,
    kbd_count:  int,
    fused_count: int,
) -> None:
    """Print a concise post-fusion summary."""
    print()
    print("═" * 54)
    print("  FUSION REPORT")
    print("═" * 54)
    print(f"  Camera rows loaded    : {cam_count}")
    print(f"  Keyboard rows loaded  : {kbd_count}")
    print(f"  Fused rows output     : {fused_count}")
    if cam_count > 0:
        match_rate = fused_count / cam_count * 100
        print(f"  Match rate            : {match_rate:.1f}%")
    print(f"  Output file           : {OUTPUT_CSV}")
    print("═" * 54)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("═" * 54)
    print("  MULTIMODAL FUSION NODE  —  STARTING")
    print("═" * 54)
    print(f"  Sync tolerance        : ±{SYNC_TOLERANCE_SECONDS}s")
    print(f"  Camera input          : {CAMERA_CSV}")
    print(f"  Keyboard input        : {KEYBOARD_CSV}")
    print(f"  Output                : {OUTPUT_CSV}")
    print("═" * 54)
    print()

    # ── Load ─────────────────────────────────────────────────────────────────
    camera_rows   = load_csv(CAMERA_CSV)
    keyboard_rows = load_csv(KEYBOARD_CSV)

    # ── Guard: empty inputs ───────────────────────────────────────────────────
    if not camera_rows and not keyboard_rows:
        print("[ABORT] Both input datasets are empty. Nothing to fuse.")
        print_report(0, 0, 0)
        sys.exit(0)

    if not camera_rows:
        print("[ABORT] Camera dataset is empty. Cannot fuse.")
        print_report(0, len(keyboard_rows), 0)
        sys.exit(0)

    if not keyboard_rows:
        print("[ABORT] Keyboard dataset is empty. Cannot fuse.")
        print_report(len(camera_rows), 0, 0)
        sys.exit(0)

    # ── Synchronize ───────────────────────────────────────────────────────────
    print("[SYNC] Running nearest-timestamp matching ...")
    matched_pairs = sync_nearest(camera_rows, keyboard_rows, SYNC_TOLERANCE_SECONDS)

    if not matched_pairs:
        print(
            "[WARN] No synchronized pairs found within tolerance "
            f"({SYNC_TOLERANCE_SECONDS}s).  Check that both CSVs cover "
            "overlapping time ranges."
        )
        print_report(len(camera_rows), len(keyboard_rows), 0)
        sys.exit(0)

    print(f"[SYNC] Matched {len(matched_pairs)} window pairs")

    # ── Build fused rows ─────────────────────────────────────────────────────
    fused_rows = [build_fused_row(cam, kbd) for cam, kbd in matched_pairs]

    # ── Write output ──────────────────────────────────────────────────────────
    write_fused_csv(fused_rows, OUTPUT_CSV)

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(len(camera_rows), len(keyboard_rows), len(fused_rows))

    print("[DONE] Multimodal dataset is ML-training ready.")


if __name__ == "__main__":
    main()