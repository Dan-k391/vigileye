"""Feed cached measurements through the real FatigueMonitor, unchanged."""

from dataclasses import dataclass
from statistics import mean

import numpy as np

from vigileye.calibration import Baseline
from vigileye.geometry import Sample
from vigileye.monitor import TRACKING, FatigueMonitor
from vigileye.scoring import Level

CALIBRATION_S = 60.0
FEATURE_SPACING_S = 10.0


@dataclass(frozen=True)
class ClipResult:
    subject: str
    label: int
    minutes: float
    tracked_share: float  # share of frames with a face
    mean_score: float
    peak_score: float
    alarms: int  # entries into DROWSY or CRITICAL
    microsleeps: int
    drowsy_share: float  # share of scored time at DROWSY or above
    perclos: float
    blink_s: float
    long_blinks_per_min: float
    reopen_s: float
    blinks_per_min: float
    saccades_per_min: float


def opening_median(data, name):
    values = data[name][data["t"] <= data["t"][0] + CALIBRATION_S]
    return float(np.nanmedian(values))


def make_baseline(ear_data, pose_data, config):
    """The clips have no closed-eye cue, so closed EAR uses the same assumed
    ratio as the app's fallback."""
    open_ear = opening_median(ear_data, "ear")
    return Baseline(open_ear=open_ear,
                    closed_ear=open_ear * config.fallback_closed_ratio,
                    yaw=opening_median(pose_data, "yaw"),
                    pitch=opening_median(pose_data, "pitch"),
                    closed_measured=False)


def replay(clip, data, baseline, config):
    """Score one clip, or return None if it never reaches a usable score. The
    calibration stretch is skipped in every clip, so all are judged alike and
    none is scored on the data that calibrated it."""
    monitor = FatigueMonitor(config)
    monitor.start_with(baseline)

    usable = data["t"] >= data["t"][0] + CALIBRATION_S
    rows = zip(*(data[name][usable].tolist()
                 for name in ("t", "ear", "gaze_x", "gaze_y", "yaw", "pitch")))

    scores, levels, features = [], [], []
    frames = tracked = alarms = microsleeps = 0
    first_t = last_t = None
    previous = Level.ALERT
    next_feature = float("-inf")

    for t, ear, gaze_x, gaze_y, yaw, pitch in rows:
        sample = None if np.isnan(ear) else Sample(ear, gaze_x, gaze_y, yaw, pitch)
        status = monitor.update(t, sample)

        first_t = t if first_t is None else first_t
        last_t = t
        frames += 1
        tracked += sample is not None
        microsleeps += "microsleep" in status.events
        alarms += status.level >= Level.DROWSY and previous < Level.DROWSY
        previous = status.level

        if status.state == TRACKING:
            scores.append(status.score)
            levels.append(status.level)
            if t >= next_feature:
                features.append(status.features)
                next_feature = t + FEATURE_SPACING_S

    if not scores:
        return None
    return ClipResult(
        subject=clip.subject,
        label=clip.label,
        minutes=(last_t - first_t) / 60.0,
        tracked_share=tracked / frames,
        mean_score=mean(scores),
        peak_score=max(scores),
        alarms=alarms,
        microsleeps=microsleeps,
        drowsy_share=sum(level >= Level.DROWSY for level in levels) / len(levels),
        perclos=mean(f.perclos for f in features),
        blink_s=mean(f.mean_blink_s for f in features),
        long_blinks_per_min=mean(f.long_blinks_per_min for f in features),
        reopen_s=mean(f.mean_reopen_s for f in features),
        blinks_per_min=mean(f.blinks_per_min for f in features),
        saccades_per_min=mean(f.saccades_per_min for f in features),
    )
