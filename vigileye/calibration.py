"""Measure this person's open and closed eyes and neutral head pose.

Eye shape varies too much for a fixed EAR threshold, so closure is always
expressed relative to these two personal reference points.
"""

from dataclasses import dataclass
from statistics import median

OPEN = "open"
CLOSED = "closed"
DONE = "done"


@dataclass(frozen=True)
class Baseline:
    open_ear: float
    closed_ear: float
    yaw: float
    pitch: float
    closed_measured: bool  # False if the fallback closed value was used

    def closure(self, ear):
        """Eyelid closure from 0 (fully open) to 1 (fully closed)."""
        return min(1.0, max(0.0, (self.open_ear - ear) / (self.open_ear - self.closed_ear)))


class Calibrator:
    def __init__(self, config):
        self.config = config
        self.phase = OPEN
        self.baseline = None
        self.open_progress = 0.0
        self.closed_started = None
        self.open_ears, self.closed_ears = [], []
        self.yaws, self.pitches = [], []

    def update(self, t, dt, sample):
        """Feed one frame. The sample is None when no face was found."""
        if self.phase == OPEN:
            if sample is not None:
                self.open_progress += dt
                self.open_ears.append(sample.ear)
                self.yaws.append(sample.yaw)
                self.pitches.append(sample.pitch)
            if self.open_progress >= self.config.calib_open_s:
                self.phase = CLOSED
                self.closed_started = t
        elif self.phase == CLOSED:
            # Timed by the clock: closed eyes may lose tracking, so waiting for
            # valid frames here could never finish.
            elapsed = t - self.closed_started
            if sample is not None and elapsed >= self.config.calib_settle_s:
                self.closed_ears.append(sample.ear)
            if elapsed >= self.config.calib_closed_s:
                self._finish()

    def _finish(self):
        config = self.config
        open_ear = median(self.open_ears)
        closed_ear = median(self.closed_ears) if self.closed_ears else open_ear
        measured = closed_ear <= open_ear * config.calib_min_closed_ratio
        self.baseline = Baseline(
            open_ear=open_ear,
            closed_ear=closed_ear if measured else open_ear * config.fallback_closed_ratio,
            yaw=median(self.yaws),
            pitch=median(self.pitches),
            closed_measured=measured,
        )
        self.phase = DONE
