"""
main.py — Entry Point for the Real-Time Drowsiness Detection System
=====================================================================

Provides a single, unified entry point that lets the user choose
between two modes:

    1. CLI mode  — OpenCV window with live webcam feed (default)
    2. Web mode  — Streamlit dashboard with WebRTC streaming

Usage
-----
    # Launch the CLI detector (default)
    python main.py

    # Launch the CLI detector with custom options
    python main.py cli --ear-threshold 0.22 --consec-frames 20

    # Launch the Streamlit web dashboard
    python main.py web

    # Download the required face landmarker model
    python main.py download-model
"""

import argparse
import os
import sys
import subprocess


# ── Paths ──
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "face_landmarker.task"


def ensure_model() -> bool:
    """Check if the face landmarker model exists. Return True if present."""
    if os.path.isfile(MODEL_FILE):
        return True
    print(f"[WARNING] Model file '{MODEL_FILE}' not found.")
    print("          Run:  python main.py download-model")
    return False


def run_download_model() -> None:
    """Download the MediaPipe Face Landmarker model."""
    from download_model import main as dl_main
    dl_main()


def run_cli(extra_args: list) -> None:
    """Launch the CLI-based drowsiness detector (OpenCV window)."""
    if not ensure_model():
        sys.exit(1)

    # Import and run directly so argparse inside drowsiness_detector
    # can pick up the remaining CLI flags.
    sys.argv = ["drowsiness_detector.py"] + extra_args
    from drowsiness_detector import main as detector_main
    detector_main()


def run_web() -> None:
    """Launch the Streamlit web dashboard."""
    if not ensure_model():
        sys.exit(1)

    app_path = os.path.join(ROOT_DIR, "streamlit_app.py")
    print("[INFO] Starting Streamlit dashboard …")
    print(f"       App: {app_path}")
    print("       Press Ctrl+C to stop.\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_path,
             "--server.headless", "true",
             "--server.address", "localhost"],
            cwd=ROOT_DIR,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Streamlit stopped.")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Real-Time Drowsiness Detection System — "
            "Facial Landmark-Based Eye Monitoring"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── cli sub-command ──
    cli_parser = subparsers.add_parser(
        "cli",
        help="Run the CLI detector with an OpenCV window.",
    )
    cli_parser.add_argument(
        "-w", "--webcam", type=int, default=0,
        help="Webcam device index (default: 0)",
    )
    cli_parser.add_argument(
        "-t", "--ear-threshold", type=float, default=0.25,
        help="EAR threshold below which the eye is considered closed (default: 0.25)",
    )
    cli_parser.add_argument(
        "-f", "--consec-frames", type=int, default=25,
        help="Number of consecutive frames below threshold before alert (default: 25)",
    )
    cli_parser.add_argument(
        "-a", "--alarm", type=str, default="alarm.wav",
        help="Path to alarm sound file (default: alarm.wav)",
    )
    cli_parser.add_argument(
        "--no-sound", action="store_true",
        help="Disable alarm sound.",
    )

    # ── web sub-command ──
    subparsers.add_parser(
        "web",
        help="Launch the Streamlit web dashboard.",
    )

    # ── download-model sub-command ──
    subparsers.add_parser(
        "download-model",
        help="Download the MediaPipe Face Landmarker model.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args()

    if args.command is None:
        # Default: show help and quick-start instructions
        parser.print_help()
        print("\n──────────────────────────────────────────")
        print("  Quick start:")
        print("    1. python main.py download-model")
        print("    2. python main.py cli          # OpenCV window")
        print("    3. python main.py web          # Streamlit dashboard")
        print("──────────────────────────────────────────\n")
        return

    if args.command == "download-model":
        run_download_model()

    elif args.command == "cli":
        # Rebuild sys.argv for the CLI detector's own argparse
        cli_args = []
        if hasattr(args, "webcam"):
            cli_args += ["--webcam", str(args.webcam)]
        if hasattr(args, "ear_threshold"):
            cli_args += ["--ear-threshold", str(args.ear_threshold)]
        if hasattr(args, "consec_frames"):
            cli_args += ["--consec-frames", str(args.consec_frames)]
        if hasattr(args, "alarm"):
            cli_args += ["--alarm", str(args.alarm)]
        if args.no_sound:
            cli_args.append("--no-sound")
        cli_args += remaining
        run_cli(cli_args)

    elif args.command == "web":
        run_web()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
