"""Run the face tracker over videos once and keep only the numbers.

Tracking a clip takes minutes and replaying its numbers takes seconds, so this
split lets the analysis be rerun freely. The cache holds no images, which also
means the videos can be deleted afterwards.
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .dataset import Clip

CACHE_DIR = Path(__file__).resolve().parent / "cache"
FIELDS = ("t", "ear", "gaze_x", "gaze_y", "yaw", "pitch")
MAX_WIDTH = 1280
PROGRESS_EVERY = 1


def cache_path(clip, cache_dir=CACHE_DIR):
    return Path(cache_dir) / f"{clip.key}.npz"


def extract_clip(clip, cache_dir=CACHE_DIR):
    """Track one clip and save its per-frame measurements. Frames with no face
    are stored as NaN, so lost tracking stays visible to the analysis."""
    import cv2

    from vigileye.tracker import FaceTracker

    capture = cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {clip.path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = FaceTracker()
    rows = []
    frames = 0
    last_t = -1.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        # Phone videos can have a variable frame rate, so prefer the file's own
        # clock and fall back to counting frames if it does not advance.
        t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= last_t:
            t = frames / fps
        last_t = t
        frames += 1
        if frame.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, sample = tracker.process(frame, t)
        if sample is None:
            rows.append((t, *[np.nan] * 5))
        else:
            rows.append((t, sample.ear, sample.gaze_x, sample.gaze_y, sample.yaw, sample.pitch))
    capture.release()
    tracker.close()

    if not rows:
        raise RuntimeError(f"No frames in {clip.path}")
    data = np.array(rows, dtype=np.float32)
    target = cache_path(clip, cache_dir)
    target.parent.mkdir(exist_ok=True)
    # Written then renamed, so an interrupted run never leaves a half-saved clip.
    partial = target.with_suffix(".partial.npz")
    np.savez_compressed(partial, **{name: data[:, i] for i, name in enumerate(FIELDS)})
    partial.rename(target)
    return clip, len(rows)


def extract_all(clips, workers=None, cache_dir=CACHE_DIR):
    """Track every clip not already cached. Safe to stop and rerun."""
    pending = [clip for clip in clips if not cache_path(clip, cache_dir).exists()]
    print(f"{len(clips) - len(pending)} clips cached, {len(pending)} to track")
    failures = []
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(extract_clip, clip, cache_dir): clip for clip in pending}
        for done, job in enumerate(as_completed(jobs), 1):
            clip = jobs[job]
            try:
                _, frames = job.result()
                print(f"[{done}/{len(pending)}] {clip.key}: {frames} frames", flush=True)
            except Exception as error:
                failures.append((clip.key, str(error)))
                print(f"[{done}/{len(pending)}] {clip.key} FAILED: {error}", flush=True)
    return failures


def cached_clips(cache_dir=CACHE_DIR):
    """Every clip in the cache, so reports still run once the videos are gone."""
    clips = []
    for path in sorted(Path(cache_dir).glob("*.npz")):
        match = re.fullmatch(r"(.+)_(\d+)", path.stem)
        if match:
            clips.append(Clip(match[1], int(match[2]), None, None))
    return clips


def load_clip(clip, cache_dir=CACHE_DIR):
    with np.load(cache_path(clip, cache_dir)) as data:
        return {name: data[name] for name in FIELDS}
