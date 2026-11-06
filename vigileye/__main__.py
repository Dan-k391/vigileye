"""Run with: python -m vigileye"""

import argparse
import sys
import time

import cv2

from .alerts import Alerter
from .monitor import CALIBRATING_CLOSED, FatigueMonitor, WARMING_UP
from .session import Session
from .tracker import FaceTracker

MAX_READ_FAILURES = 30


def parse_args():
    parser = argparse.ArgumentParser(
        prog="vigileye", description="Detect fatigue from eyelid and eye movement.")
    parser.add_argument("--source", default="0", help="camera index or video file (default 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-window", action="store_true", help="run without the video window")
    parser.add_argument("--mute", action="store_true", help="no alarm sounds")
    parser.add_argument("--no-log", action="store_true", help="do not write a session log")
    return parser.parse_args()


def open_source(args):
    if not args.source.isdigit():
        return cv2.VideoCapture(args.source), True
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(int(args.source), backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return capture, False


def fail_to_open(source):
    print(f"Could not open video source {source!r}.", file=sys.stderr)
    if source.isdigit() and sys.platform == "darwin":
        print("Allow camera access for your terminal in System Settings > "
              "Privacy & Security > Camera, then restart the terminal.", file=sys.stderr)
    return 1


def main():
    args = parse_args()
    capture, is_file = open_source(args)
    if not capture.isOpened():
        return fail_to_open(args.source)

    # A file is timed by its own frame rate, so a run is repeatable and is not
    # affected by how fast this machine can process it.
    file_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = FaceTracker()
    monitor = FatigueMonitor()
    alerter = Alerter(muted=args.mute or is_file)
    session = Session(write_log=not args.no_log)
    display = None
    if not args.no_window:
        from .display import Display
        display = Display(monitor.config)

    print("Calibrating: look straight ahead with your eyes open, then close them "
          "when you hear the chime.")
    started = time.monotonic()
    frames = 0
    failures = 0
    fps = 0.0
    last_clock = started
    previous_state = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                failures += 1
                if is_file or failures >= MAX_READ_FAILURES:
                    break
                continue
            failures = 0

            t = frames / file_fps if is_file else time.monotonic() - started
            frames += 1
            points, sample = tracker.process(frame, t)
            status = monitor.update(t, sample)
            session.update(t, status)

            if not is_file and status.state != previous_state and (
                    status.state == CALIBRATING_CLOSED
                    or (previous_state == CALIBRATING_CLOSED and status.state == WARMING_UP)):
                alerter.cue()
            previous_state = status.state
            alerter.update(status.level)

            if "calibrated" in status.events:
                baseline = monitor.baseline
                print(f"Calibrated: open EAR {baseline.open_ear:.3f}, "
                      f"closed EAR {baseline.closed_ear:.3f}"
                      f"{'' if baseline.closed_measured else ' (estimated, eyes did not close)'}")

            if display is not None:
                clock = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(clock - last_clock, 1e-3)
                last_clock = clock
                key = display.show(frame, points, status, fps, alerter.muted)
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    monitor.recalibrate()
                if key == ord("m"):
                    alerter.toggle_mute()
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        tracker.close()
        alerter.close()
        session.close()
        if display is not None:
            display.close()

    print(session.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
