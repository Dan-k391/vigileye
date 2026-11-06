"""MediaPipe face tracking: 478 landmarks (including irises) and head pose."""

import hashlib
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .geometry import measure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_MD5 = "b0e7274907a1644404fef66b28dd6d85"


def ensure_model(path=MODEL_PATH):
    """Fetch the model into the project on first run and verify it."""
    if not path.exists():
        print(f"Downloading face model to {path} ...")
        path.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
        if hashlib.md5(data).hexdigest() != MODEL_MD5:
            raise RuntimeError("Downloaded face model failed its checksum")
        path.write_bytes(data)
    return path


class FaceTracker:
    def __init__(self, model_path=MODEL_PATH):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_ms = -1

    def process(self, frame, t):
        """Returns (points in pixels, Sample), or (None, None) with no face."""
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # MediaPipe requires strictly increasing timestamps.
        self.last_ms = max(self.last_ms + 1, int(t * 1000))
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.last_ms)
        if not result.face_landmarks:
            return None, None
        points = np.array([(p.x * width, p.y * height, p.z * width)
                           for p in result.face_landmarks[0]])
        return points, measure(points, result.facial_transformation_matrixes[0])

    def close(self):
        self.landmarker.close()
