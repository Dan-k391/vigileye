"""Synthetic eyelid traces with known ground truth, for testing without a camera."""

import random
from dataclasses import dataclass

from vigileye.geometry import Sample
from vigileye.monitor import FatigueMonitor

OPEN_EAR = 0.30
CLOSED_EAR = 0.10


def ear_for(closure):
    return OPEN_EAR - closure * (OPEN_EAR - CLOSED_EAR)


@dataclass
class Profile:
    blinks_per_min: float = 16.0
    blink_s: tuple = (0.12, 0.22)
    saccades_per_min: float = 40.0
    extra_closures: tuple = ()  # (start time after calibration, duration)


ALERT = Profile()
DROWSY = Profile(blinks_per_min=20.0, blink_s=(0.35, 0.8))


def closure_at(t, blinks):
    """Trapezoid eyelid: 60 ms down, hold, then 100 ms back up."""
    for start, duration in blinks:
        if start <= t <= start + duration:
            if t < start + 0.06:
                return (t - start) / 0.06
            if t > start + duration - 0.10:
                return (start + duration - t) / 0.10
            return 1.0
    return 0.0


def make_blinks(profile, seconds, rng):
    blinks = list(profile.extra_closures)
    t = 1.0
    while t < seconds:
        t += rng.expovariate(profile.blinks_per_min / 60.0)
        blinks.append((t, rng.uniform(*profile.blink_s)))
    return sorted(blinks)


class Run:
    """Drives a monitor through calibration and then a chosen profile."""

    def __init__(self, fps=30.0, seed=1, config=None):
        self.fps = fps
        self.rng = random.Random(seed)
        self.monitor = FatigueMonitor(config)
        self.t = 0.0
        self.gaze = 0.0

    def step(self, sample):
        self.t += 1.0 / self.fps
        return self.monitor.update(self.t, sample)

    def sample(self, closure=0.0, yaw=0.0, pitch=0.0):
        noise = self.rng.gauss(0.0, 0.004)
        return Sample(ear_for(closure) + noise, self.gaze, 0.0, yaw, pitch)

    def calibrate(self, close_eyes=True):
        while self.monitor.baseline is None:
            phase = self.monitor.calibrator.phase
            self.step(self.sample(1.0 if close_eyes and phase == "closed" else 0.0))
        return self.monitor.baseline

    def play(self, profile, seconds):
        """Returns every Status produced, with times relative to the start."""
        blinks = make_blinks(profile, seconds, self.rng)
        start = self.t
        next_saccade = 0.0
        statuses = []
        while self.t - start < seconds:
            now = self.t - start
            if profile.saccades_per_min and now >= next_saccade:
                # Always crosses the centre, so each jump is a true saccade.
                self.gaze = -1.0 if self.gaze > 0 else 1.0
                self.gaze *= self.rng.uniform(0.05, 0.15)
                next_saccade = now + self.rng.expovariate(profile.saccades_per_min / 60.0)
            statuses.append(self.step(self.sample(closure_at(now, blinks))))
        return statuses
