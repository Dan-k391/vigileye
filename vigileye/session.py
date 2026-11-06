"""Numeric session log and end-of-run summary. No video is ever stored."""

import csv
from collections import Counter
from datetime import datetime

from .monitor import CALIBRATING_CLOSED, CALIBRATING_OPEN, TRACKING
from .scoring import Level
from .tracker import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
FIELDS = ("time_s", "state", "level", "score", "perclos", "mean_blink_s",
          "long_blinks_per_min", "mean_reopen_s", "blinks_per_min",
          "saccades_per_min", "events")


class Session:
    def __init__(self, write_log=True, interval_s=1.0):
        self.interval_s = interval_s
        self.last_row = float("-inf")
        self.pending_events = []
        self.seconds_at_level = Counter()
        self.seconds_tracked = 0.0
        self.event_counts = Counter()
        self.peak_score = 0.0
        self.last_t = None
        self.path = None
        self.file = None
        if write_log:
            LOG_DIR.mkdir(exist_ok=True)
            self.path = LOG_DIR / f"session-{datetime.now():%Y%m%d-%H%M%S}.csv"
            self.file = open(self.path, "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(FIELDS)

    def update(self, t, status):
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        self.event_counts.update(status.events)
        self.pending_events.extend(status.events)
        if status.state not in (CALIBRATING_OPEN, CALIBRATING_CLOSED):
            self.seconds_at_level[status.level] += dt
            self.peak_score = max(self.peak_score, status.score)
        if status.state == TRACKING:
            self.seconds_tracked += dt
        if self.file and (t - self.last_row >= self.interval_s or status.events):
            self._write(t, status)

    def _write(self, t, status):
        features = status.features
        self.writer.writerow((
            f"{t:.2f}", status.state, status.level.name, f"{status.score:.1f}",
            f"{features.perclos:.3f}", f"{features.mean_blink_s:.3f}",
            f"{features.long_blinks_per_min:.1f}", f"{features.mean_reopen_s:.3f}",
            f"{features.blinks_per_min:.1f}", f"{features.saccades_per_min:.1f}",
            "|".join(self.pending_events)))
        self.last_row = t
        self.pending_events.clear()

    def summary(self):
        lines = [
            f"Peak fatigue score: {self.peak_score:.0f}/100",
            f"Blinks: {self.event_counts['blink']}, "
            f"long blinks: {self.event_counts['long-blink']}, "
            f"microsleeps: {self.event_counts['microsleep']}",
        ]
        lines += [f"  {level.name:<8} {self.seconds_at_level[level]:7.1f} s" for level in Level]
        if self.path:
            lines.append(f"Log: {self.path}")
        return "\n".join(lines)

    def close(self):
        if self.file:
            self.file.close()
