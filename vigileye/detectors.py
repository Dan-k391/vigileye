"""Time-based detectors. Everything uses timestamps, never frame counts, so
results do not change with the camera's frame rate."""

from collections import deque
from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Blink:
    end: float
    duration: float
    reopen: float


class ClosureFraction:
    """Share of valid time spent at or above a closure level (PERCLOS)."""

    def __init__(self, window_s):
        self.window_s = window_s
        self.samples = deque()
        self.total = 0.0
        self.closed = 0.0

    def add(self, t, dt, is_closed):
        self.samples.append((t, dt, is_closed))
        self.total += dt
        self.closed += dt if is_closed else 0.0
        self.expire(t)

    def expire(self, t):
        while self.samples and self.samples[0][0] < t - self.window_s:
            _, dt, is_closed = self.samples.popleft()
            self.total -= dt
            self.closed -= dt if is_closed else 0.0

    @property
    def value(self):
        return self.closed / self.total if self.total > 0 else 0.0


class BlinkDetector:
    """Hysteresis on closure, timing each closure while it is still happening
    so a microsleep can be flagged before the eyes reopen."""

    def __init__(self, config):
        self.config = config
        self.started = None
        self.peak = 0.0
        self.peak_end = 0.0
        self.microsleep_fired = False

    def reset(self):
        """Tracking dropped out, so any closure in progress cannot be timed."""
        self.started = None
        self.microsleep_fired = False

    def closed_for(self, t):
        return t - self.started if self.started is not None else 0.0

    def update(self, t, closure):
        """Returns (finished Blink or None, True if a microsleep just began)."""
        config = self.config
        if self.started is None:
            if closure >= config.blink_enter:
                self.started = t
                self.peak = closure
                self.peak_end = t
            return None, False

        if closure >= self.peak * 0.9:
            self.peak = max(self.peak, closure)
            self.peak_end = t

        if closure > config.blink_exit:
            if not self.microsleep_fired and self.closed_for(t) >= config.microsleep_s:
                self.microsleep_fired = True
                return None, True
            return None, False

        duration = t - self.started
        blink = Blink(t, duration, t - self.peak_end)
        self.reset()
        return (blink if duration >= config.blink_min_s else None), False


class SaccadeDetector:
    """Counts fast gaze shifts by displacement over a short span. Displacement
    is much less sensitive to landmark jitter than frame-to-frame velocity."""

    def __init__(self, config):
        self.config = config
        self.recent = deque()
        self.last_saccade = float("-inf")

    def reset(self):
        self.recent.clear()

    def update(self, t, x, y):
        """Returns True if a saccade was detected on this sample."""
        config = self.config
        self.recent.append((t, x, y))
        while self.recent[0][0] < t - config.saccade_span_s:
            self.recent.popleft()
        _, old_x, old_y = self.recent[0]
        if hypot(x - old_x, y - old_y) < config.saccade_amplitude:
            return False
        if t - self.last_saccade < config.saccade_refractory_s:
            return False
        self.last_saccade = t
        return True
