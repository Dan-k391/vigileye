"""End-to-end checks on real face pixels: tracker, geometry and the full app.

They need the bundled scikit-image photo, and are skipped where it is missing.
"""

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vigileye.geometry import LEFT_EYE, RIGHT_EYE
from vigileye.tracker import PROJECT_ROOT, FaceTracker

PHOTO = Path("/opt/anaconda3/lib/python3.13/site-packages/skimage/data/astronaut.png")
SCRATCH = PROJECT_ROOT / "tests" / ".scratch"
FPS = 30

pytestmark = pytest.mark.skipif(not PHOTO.exists(), reason="test photo not available")


@pytest.fixture(scope="module")
def face():
    image = cv2.imread(str(PHOTO))[0:300, 90:360]
    return cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def measure_still(image):
    tracker = FaceTracker()
    for i in range(5):
        points, sample = tracker.process(image, i / FPS)
    tracker.close()
    return points, sample


def close_eyes(image, points):
    """Paint skin over each eye with a dark lash line, like a shut eyelid."""
    result = image.copy()
    for eye in (RIGHT_EYE, LEFT_EYE):
        corners = points[list(eye), :2]
        centre = corners.mean(axis=0)
        half_width = np.ptp(corners[:, 0]) / 2 * 1.35
        half_height = max(np.ptp(corners[:, 1]) * 1.3, half_width * 0.5)
        patch_centre = (int(centre[0]), int(centre[1] + half_height * 3.0))
        size = (int(half_width * 2.4), int(half_height * 2.4))
        x0, y0 = patch_centre[0] - size[0] // 2, patch_centre[1] - size[1] // 2
        patch = image[y0:y0 + size[1], x0:x0 + size[0]]
        mask = np.zeros(patch.shape[:2], np.uint8)
        cv2.ellipse(mask, (size[0] // 2, size[1] // 2),
                    (size[0] // 2 - 2, size[1] // 2 - 2), 0, 0, 360, 255, -1)
        target = (int(centre[0]), int(centre[1]))
        result = cv2.seamlessClone(patch, result, mask, target, cv2.NORMAL_CLONE)
        left = (int(centre[0] - half_width * 0.75), int(centre[1]))
        right = (int(centre[0] + half_width * 0.75), int(centre[1]))
        cv2.line(result, left, right, (35, 40, 60), 3, cv2.LINE_AA)
    return result


def test_pose_matrix_agrees_with_the_landmarks_in_3d(face):
    """The pose comes from a solved matrix, so check its sign conventions
    against an independent estimate from the raw landmarks."""
    points, sample = measure_still(face)
    # Camera axes: x right, y up, z towards the camera (MediaPipe z is the reverse).
    camera = points * np.array([1.0, -1.0, -1.0])
    up = camera[10] - camera[152]  # forehead over chin
    across = camera[454] - camera[234]  # one cheek to the other
    forward = np.cross(across, up)
    forward /= np.linalg.norm(forward)
    assert np.degrees(np.arcsin(forward[1])) == pytest.approx(sample.pitch, abs=8)
    assert np.degrees(np.arctan2(forward[0], forward[2])) == pytest.approx(sample.yaw, abs=8)


def test_a_shut_eyelid_lowers_ear(face):
    points, open_sample = measure_still(face)
    _, shut_sample = measure_still(close_eyes(face, points))
    assert shut_sample.ear < open_sample.ear * 0.75


def test_app_detects_a_microsleep_in_a_video(face):
    """Eyes open for calibration, shut on cue, then open with a 3 s microsleep."""
    points, _ = measure_still(face)
    shut = close_eyes(face, points)
    timeline = [(face, 8.5), (shut, 3.0), (face, 8.0)]
    for _ in range(5):
        timeline += [(shut, 0.2), (face, 3.0)]
    timeline += [(shut, 3.0), (face, 3.0)]

    SCRATCH.mkdir(exist_ok=True)
    video = SCRATCH / "microsleep.avi"
    height, width = face.shape[:2]
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
    assert writer.isOpened()
    for image, seconds in timeline:
        for _ in range(int(seconds * FPS)):
            writer.write(image)
    writer.release()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "vigileye", "--source", str(video), "--no-window"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120)
        assert result.returncode == 0, result.stderr
        print(result.stdout)
        log = next(line.split("Log: ")[1] for line in result.stdout.splitlines()
                   if line.startswith("Log: "))
        with open(log) as file:
            rows = list(csv.DictReader(file))
        Path(log).unlink()
    finally:
        video.unlink()
        SCRATCH.rmdir()

    assert "(estimated" not in result.stdout  # the closed-eye cue was measured
    assert "microsleeps: 1" in result.stdout
    blinks = [r for r in rows if "blink" in r["events"]]
    assert len(blinks) >= 5
    critical = [r for r in rows if r["level"] == "CRITICAL"]
    assert critical
    alarm_time = float(critical[0]["time_s"])
    assert 36.4 < alarm_time < 37.0  # 1 s into the closure that starts at 35.5 s
