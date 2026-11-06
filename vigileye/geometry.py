"""Turn MediaPipe face landmarks into eyelid, gaze and head-pose measurements."""

from dataclasses import dataclass

import numpy as np

# Six-point eye contours in EAR order: corner, two upper lid, corner, two lower lid.
# Both run image-left to image-right, so gaze signs agree between the eyes.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_IRIS = 468
LEFT_IRIS = 473


@dataclass(frozen=True)
class Sample:
    """One frame's measurements for the analysis pipeline."""

    ear: float
    gaze_x: float
    gaze_y: float
    yaw: float
    pitch: float


def eye_aspect_ratio(points, eye):
    """Lid gap over eye width (Soukupova & Cech, 2016).

    Using 3D points keeps this far steadier under head rotation than the usual
    2D form, because the eye width no longer foreshortens as the head turns.
    """
    p1, p2, p3, p4, p5, p6 = points[list(eye)]
    gap = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    return float(gap / (2.0 * np.linalg.norm(p1 - p4)))


def gaze_offset(points, eye, iris):
    """Iris position relative to the eye centre, in fractions of eye width."""
    corner_a, corner_b = points[eye[0], :2], points[eye[3], :2]
    axis = corner_b - corner_a
    width = np.linalg.norm(axis)
    along = axis / width
    across = np.array([-along[1], along[0]])
    offset = points[iris, :2] - (corner_a + corner_b) / 2.0
    return float(offset @ along / width), float(offset @ across / width)


def head_angles(transform):
    """Yaw and pitch in degrees from MediaPipe's face-to-camera matrix.

    The third rotation column is where the nose points in camera space (y up,
    z towards the camera), so pitch is negative when the head tips down.
    """
    nose = np.asarray(transform)[:3, 2]
    yaw = np.degrees(np.arctan2(nose[0], nose[2]))
    pitch = np.degrees(np.arctan2(nose[1], np.hypot(nose[0], nose[2])))
    return float(yaw), float(pitch)


def measure(points, transform):
    """Build a Sample from an (N, 3) landmark array and a 4x4 pose matrix."""
    right_x, right_y = gaze_offset(points, RIGHT_EYE, RIGHT_IRIS)
    left_x, left_y = gaze_offset(points, LEFT_EYE, LEFT_IRIS)
    yaw, pitch = head_angles(transform)
    return Sample(
        ear=(eye_aspect_ratio(points, RIGHT_EYE) + eye_aspect_ratio(points, LEFT_EYE)) / 2.0,
        gaze_x=(right_x + left_x) / 2.0,
        gaze_y=(right_y + left_y) / 2.0,
        yaw=yaw,
        pitch=pitch,
    )
