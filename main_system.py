"""
main_system.py
══════════════════════════════════════════════════════════════════
Orchestration Launcher — Multimodal Fatigue Detection System

Execution Flow:
    1. Validate all node scripts exist
    2. Launch camera node  (camera_node/main_camera_system.py)
    3. Launch keyboard node (keyboard_node/main_keyboard_system.py)
    4. Both run simultaneously in separate visible terminals
    5. Wait until BOTH nodes terminate
    6. Auto-trigger fusion (fusion_node/dataset_fusion.py)
    7. Print final execution report

Exit Controls (in the node windows):
    Camera   → press Q
    Keyboard → press ESC
    Emergency → Ctrl+C in this window (Recommended)

Author  : Rushikesh Aryaveer (Bhujbal)
Project : Multimodal Fatigue Detection System
══════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import platform
import subprocess
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))

CAMERA_NODE_DIR   = os.path.join(BASE_DIR, "camera_node")
KEYBOARD_NODE_DIR = os.path.join(BASE_DIR, "keyboard_node")
FUSION_NODE_DIR   = os.path.join(BASE_DIR, "fusion_node")

CAMERA_SCRIPT     = os.path.join(CAMERA_NODE_DIR,   "main_camera_system.py")
KEYBOARD_SCRIPT   = os.path.join(KEYBOARD_NODE_DIR, "main_keyboard_system.py")
FUSION_SCRIPT     = os.path.join(FUSION_NODE_DIR,   "dataset_fusion.py")

OUTPUT_CSV        = os.path.join(FUSION_NODE_DIR, "multimodal_fatigue_dataset.csv")
SHUTDOWN_FLAG     = os.path.join(BASE_DIR, ".shutdown_flag")

# Same Python interpreter / virtual environment that runs this file
PYTHON = sys.executable

IS_WINDOWS = platform.system() == "Windows"

# ══════════════════════════════════════════════════════════════════
# CONSOLE HELPERS
# ══════════════════════════════════════════════════════════════════

DIV  = "=" * 60
SDIV = "-" * 60


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


def banner(title: str) -> None:
    print(f"\n{DIV}")
    print(f"  {title}")
    print(DIV)


def section(title: str) -> None:
    print(f"\n{SDIV}")
    print(f"  {title}")
    print(SDIV)


# ══════════════════════════════════════════════════════════════════
# PRE-FLIGHT VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_paths() -> bool:
    """
    Confirm all required scripts exist before launching anything.
    Fail fast with a clear error rather than a cryptic subprocess crash.
    """
    required = {
        "Camera node"   : CAMERA_SCRIPT,
        "Keyboard node" : KEYBOARD_SCRIPT,
        "Fusion node"   : FUSION_SCRIPT,
    }

    all_ok = True

    for label, path in required.items():
        if os.path.isfile(path):
            log(f"[OK]   {label:<14} -> {path}")
        else:
            log(f"[MISS] {label:<14} -> {path}")
            all_ok = False

    return all_ok


# ══════════════════════════════════════════════════════════════════
# NODE LAUNCHER
# ══════════════════════════════════════════════════════════════════

def launch_node(script_path: str, cwd: str, label: str) -> subprocess.Popen:
    """
    Launch a node script in a separate visible terminal window.

    Windows  -> CREATE_NEW_CONSOLE opens a dedicated cmd window.
    Linux    -> xterm is attempted first; falls back to gnome-terminal,
               xfce4-terminal, konsole, then inline if none found.

    cwd is set to the node's own directory so bare relative imports
    (e.g. `from ear_detector import ...`) resolve correctly.
    """

    if IS_WINDOWS:
        proc = subprocess.Popen(
            [PYTHON, script_path],
            cwd=cwd,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        return proc

    # Linux / macOS: try common terminal emulators in preference order
    terminal_cmds = [
        ["xterm",          "-title", label, "-e",
         PYTHON, script_path],
        ["gnome-terminal", f"--title={label}", "--",
         PYTHON, script_path],
        ["xfce4-terminal", f"--title={label}", "-e",
         f"{PYTHON} {script_path}"],
        ["konsole",        "--title", label, "-e",
         f"{PYTHON} {script_path}"],
    ]

    for cmd in terminal_cmds:
        try:
            proc = subprocess.Popen(cmd, cwd=cwd)
            return proc
        except FileNotFoundError:
            continue

    # Last resort: run without a separate window
    log(f"[WARN] No terminal emulator found — running {label} inline.")
    proc = subprocess.Popen([PYTHON, script_path], cwd=cwd)
    return proc


# ══════════════════════════════════════════════════════════════════
# PROCESS MONITORING
# ══════════════════════════════════════════════════════════════════

def wait_for_both(
    cam_proc: subprocess.Popen,
    kbd_proc: subprocess.Popen,
    poll_interval: float = 0.5,
) -> tuple:
    """
    Block until both nodes have terminated.
    Prints a live status line every 10 s so the user knows the
    orchestrator is alive and which node is still running.

    Returns (camera_exit_code, keyboard_exit_code).
    """
    log("Monitoring nodes — waiting for both to finish ...")
    log("  Camera   -> press Q   in the camera window to stop")
    log("  Keyboard -> press ESC in the keyboard window to stop")
    print()

    status_interval = 10
    last_status_t   = time.time()

    while True:
        cam_done = cam_proc.poll() is not None
        kbd_done = kbd_proc.poll() is not None

        if cam_done and kbd_done:
            break

        now = time.time()

        if now - last_status_t >= status_interval:
            alive = []
            if not cam_done:
                alive.append("camera")
            if not kbd_done:
                alive.append("keyboard")
            log(f"Still running: {', '.join(alive)} ...")
            last_status_t = now

        time.sleep(poll_interval)

    return cam_proc.returncode, kbd_proc.returncode

def _cleanup_flag() -> None:
    """Remove the shutdown flag file if it exists."""
    try:
        if os.path.isfile(SHUTDOWN_FLAG):
            os.remove(SHUTDOWN_FLAG)
    except OSError:
        pass


def graceful_shutdown(processes: list, timeout: float = 15.0) -> None:
    """
    1. Write shutdown flag → nodes detect it, flush telemetry, exit 0.
    2. Wait up to `timeout` seconds for clean exits.
    3. Force-kill any survivors.
    4. Remove the flag.
    """
    log("[SIGNAL] Writing shutdown flag to disk ...")
    try:
        with open(SHUTDOWN_FLAG, "w") as f:
            f.write("shutdown")
    except OSError as e:
        log(f"[WARN] Could not write shutdown flag: {e}")
        # Fall back to immediate termination
        for p in processes:
            if p.poll() is None:
                p.terminate()
        return

    log(f"Waiting up to {int(timeout)}s for nodes to flush and exit cleanly ...")
    deadline = time.time() + timeout

    while time.time() < deadline:
        if all(p.poll() is not None for p in processes):
            break
        time.sleep(0.5)

    # Escalate to force-kill for any survivors
    for proc in processes:
        if proc.poll() is None:
            log(f"  Timeout — force-killing PID {proc.pid}")
            proc.kill()
            proc.wait()

    _cleanup_flag()



def terminate_all(processes: list) -> None:
    """
    Gracefully terminate all living child processes.
    Escalates to kill() if terminate() is ignored after 5 s.
    """
    for proc in processes:
        if proc.poll() is None:
            log(f"  Terminating PID {proc.pid} ...")
            proc.terminate()

    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log(f"  Force-killing PID {proc.pid} ...")
            proc.kill()
            proc.wait()


# ══════════════════════════════════════════════════════════════════
# FUSION RUNNER
# ══════════════════════════════════════════════════════════════════

def run_fusion() -> int:
    """
    Execute dataset_fusion.py in this same terminal window.
    Runs synchronously — orchestrator waits for it to finish.
    fusion_node/dataset_fusion.py uses _SCRIPT_DIR internally,
    so it finds both CSVs correctly regardless of cwd.

    Returns the fusion process exit code.
    """
    section("FUSION NODE — RUNNING")
    log("Merging camera + keyboard datasets ...")
    print()

    fusion_proc = subprocess.Popen(
        [PYTHON, FUSION_SCRIPT],
        cwd=FUSION_NODE_DIR,
    )

    fusion_proc.wait()
    return fusion_proc.returncode


# ══════════════════════════════════════════════════════════════════
# EXECUTION REPORT
# ══════════════════════════════════════════════════════════════════

def print_report(
    session_start : float,
    cam_exit      : int,
    kbd_exit      : int,
    fusion_exit   : int,
    interrupted   : bool,
) -> None:

    elapsed = time.time() - session_start
    mins, secs = divmod(int(elapsed), 60)

    output_exists = os.path.isfile(OUTPUT_CSV)

    banner("FINAL EXECUTION REPORT")

    print(f"  Session Duration  :  {mins}m {secs}s")
    print()
    print(f"  Camera Node       :  exit {cam_exit}  "
          f"{'[OK]' if cam_exit == 0 else '[ERROR]'}")
    print(f"  Keyboard Node     :  exit {kbd_exit}  "
          f"{'[OK]' if kbd_exit == 0 else '[ERROR]'}")
    print(f"  Fusion Node       :  exit {fusion_exit}  "
          f"{'[OK]' if fusion_exit == 0 else '[ERROR]'}")
    print()

    if interrupted:
        print("  Status            :  INTERRUPTED (Ctrl+C)")
        print("  Note              :  Fusion skipped — no data merged")

    elif output_exists:
        print("  Status            :  SUCCESS")
        print(f"  Output CSV        :  {OUTPUT_CSV}")

    else:
        print("  Status            :  COMPLETED (no output CSV generated)")
        print("  Hint              :  Both nodes need >= 10s of activity")
        print("                       before the first CSV row is written")

    print(DIV + "\n")


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main() -> None:

    session_start = time.time()
    interrupted   = False
    cam_exit      = -1
    kbd_exit      = -1
    fusion_exit   = -1

    # ── Banner ─────────────────────────────────────────────────────
    banner("MULTIMODAL FATIGUE DETECTION SYSTEM  —  ORCHESTRATOR")
    print(f"  Python      : {PYTHON}")
    print(f"  Platform    : {platform.system()} {platform.release()}")
    print(f"  Base dir    : {BASE_DIR}")
    print(DIV)

    # ── Pre-flight ─────────────────────────────────────────────────
    section("PRE-FLIGHT CHECK")

    if not validate_paths():
        print()
        log("[ABORT] One or more node scripts are missing.")
        log("        Verify your project structure and retry.")
        sys.exit(1)

    print()
    log("All paths verified. Ready to launch.")

    # ── Launch nodes ───────────────────────────────────────────────
    section("LAUNCHING NODES")

    log("Starting camera node ...")
    cam_proc = launch_node(
        CAMERA_SCRIPT,
        CAMERA_NODE_DIR,
        "Camera Fatigue Node",
    )
    log(f"  Camera node started   (PID {cam_proc.pid})")

    # Brief stagger so both terminal windows open without racing
    time.sleep(0.3)

    log("Starting keyboard node ...")
    kbd_proc = launch_node(
        KEYBOARD_SCRIPT,
        KEYBOARD_NODE_DIR,
        "Keyboard Telemetry Node",
    )
    log(f"  Keyboard node started (PID {kbd_proc.pid})")

    print()
    log("Both nodes are running.")

    # ── Monitor until both exit ────────────────────────────────────
    section("RUNTIME  —  WAITING FOR NODES")

    try:
        cam_exit, kbd_exit = wait_for_both(cam_proc, kbd_proc)

    except KeyboardInterrupt:
        print()
        log("[SIGNAL] Ctrl+C received — initiating graceful shutdown ...")
        graceful_shutdown([cam_proc, kbd_proc])
        cam_exit = cam_proc.returncode if cam_proc.returncode is not None else -1
        kbd_exit = kbd_proc.returncode if kbd_proc.returncode is not None else -1
        # Only treat as interrupted (skip fusion) if force-kill was needed
        interrupted = not (cam_exit == 0 and kbd_exit == 0)

    log("Both nodes have terminated.")
    log(f"  Camera exit code   : {cam_exit}")
    log(f"  Keyboard exit code : {kbd_exit}")

    # ── Fusion ─────────────────────────────────────────────────────
    if not interrupted:
        fusion_exit = run_fusion()
        log(f"Fusion finished. Exit code: {fusion_exit}")
    else:
        section("FUSION SKIPPED")
        log("Session was interrupted — fusion will not run.")
        log("Run a full session (Q + ESC to exit) to generate fused data.")
        fusion_exit = -1

    _cleanup_flag()  # Safety — remove any stale flag before exit

    # ── Final report ───────────────────────────────────────────────
    print_report(
        session_start,
        cam_exit,
        kbd_exit,
        fusion_exit,
        interrupted,
    )


if __name__ == "__main__":
    main()