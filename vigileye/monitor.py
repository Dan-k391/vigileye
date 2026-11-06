"""The analysis pipeline: per-frame samples in, fatigue status out."""

from collections import deque
from dataclasses import dataclass, field
from statistics import mean, median

from . import calibration
from .calibration import Calibrator
from .config import Config
from .detectors import BlinkDetector, ClosureFraction, SaccadeDetector
from .scoring import FatigueScorer, Features, Level

# Tracking states
CALIBRATING_OPEN = "calibrating-open"
CALIBRATING_CLOSED = "calibrating-closed"
WARMING_UP = "warming-up"
TRACKING = "tracking"
NO_FACE = "no-face"
LOOKING_AWAY = "looking-away"


@dataclass(frozen=True)
class Status:
    state: str
    level: Level = Level.ALERT
    score: float = 0.0
    closure: float = 0.0
    closed_for: float = 0.0
    features: Features = field(default_factory=Features)
    calibration_progress: float = 0.0
    events: tuple = ()  # "blink", "long-blink", "microsleep", "calibrated"


class FatigueMonitor:
    def __init__(self, config=None):
        self.config = config or Config()
        self.recalibrate()

    def recalibrate(self):
        config = self.config
        self.calibrator = Calibrator(config)
        self.baseline = None
        self.settle_until = 0.0
        self.last_t = None
        self.ears = deque(maxlen=config.ear_median_samples)
        self.perclos = ClosureFraction(config.window_s)
        self.blink_detector = BlinkDetector(config)
        self.saccade_detector = SaccadeDetector(config)
        self.scorer = FatigueScorer(config)
        self.blinks = deque()
        self.saccades = deque()
        self.gaze_blocked_until = 0.0
        self.saccades_total = 0
        self.valid_s = 0.0
        self.saccade_baseline = None

    def start_with(self, baseline):
        """Skip live calibration and use a baseline measured elsewhere."""
        self.recalibrate()
        self.baseline = baseline

    def update(self, t, sample):
        """Process one frame. The sample is None when no face was found."""
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        if self.baseline is None:
            return self._calibrate(t, dt, sample)
        if t < self.settle_until:
            return Status(WARMING_UP)

        # After a frame hitch the eyelid was unobserved, so a closure in progress
        # cannot be timed, and the gap counts as one normal step at most.
        if dt > self.config.max_gap_s:
            self.blink_detector.reset()
            dt = self.config.max_gap_s
        if sample is None:
            return self._invalid(t, dt, NO_FACE)
        if self._looking_away(sample):
            return self._invalid(t, dt, LOOKING_AWAY)
        return self._track(t, dt, sample)

    def _calibrate(self, t, dt, sample):
        calibrator = self.calibrator
        calibrator.update(t, dt, sample)
        if calibrator.phase == calibration.DONE:
            self.baseline = calibrator.baseline
            self.settle_until = t + self.config.calib_reopen_s
            return Status(WARMING_UP, events=("calibrated",))
        if calibrator.phase == calibration.OPEN:
            progress = calibrator.open_progress / self.config.calib_open_s
            return Status(CALIBRATING_OPEN, calibration_progress=progress)
        return Status(CALIBRATING_CLOSED, calibration_progress=1.0)

    def _looking_away(self, sample):
        config, baseline = self.config, self.baseline
        return (abs(sample.yaw - baseline.yaw) > config.max_yaw_deg
                or sample.pitch - baseline.pitch > config.max_pitch_up_deg)

    def _invalid(self, t, dt, state):
        """No usable eye data: hold the score and judge nothing from this frame."""
        self.blink_detector.reset()
        self.saccade_detector.reset()
        self.ears.clear()
        self.perclos.expire(t)
        self._expire_events(t)
        score, level = self.scorer.apply_events(t)
        return Status(state, level, score, features=self._features())

    def _track(self, t, dt, sample):
        config = self.config
        self.ears.append(sample.ear)
        closure = self.baseline.closure(median(self.ears))
        self.perclos.add(t, dt, closure >= config.perclos_closure)
        self.valid_s += dt

        events = self._update_eyelids(t, closure)
        self._update_gaze(t, sample, closure)
        self._expire_events(t)

        features = self._features()
        if self.perclos.total < config.min_coverage_s:
            # Too little data to score, but a microsleep needs no history.
            score, level = self.scorer.apply_events(t)
            return Status(WARMING_UP, level, score, closure,
                          self.blink_detector.closed_for(t), features, events=events)

        score, level = self.scorer.update(t, dt, features)
        return Status(TRACKING, level, score, closure,
                      self.blink_detector.closed_for(t), features, events=events)

    def _update_eyelids(self, t, closure):
        blink, microsleep = self.blink_detector.update(t, closure)
        events = []
        if blink is not None:
            self.blinks.append(blink)
            events.append("blink")
            if blink.duration >= self.config.long_blink_s:
                events.append("long-blink")
        if microsleep:
            events.append("microsleep")
        # Renewed every frame, so the alarm lasts as long as the eyes stay shut.
        if self.blink_detector.microsleep_fired:
            self.scorer.note_microsleep(t)
        return tuple(events)

    def _update_gaze(self, t, sample, closure):
        config = self.config
        if closure >= config.blink_exit:
            self.gaze_blocked_until = t + config.blink_gaze_margin_s
        if t < self.gaze_blocked_until:
            self.saccade_detector.reset()
            return
        if self.saccade_detector.update(t, sample.gaze_x, sample.gaze_y):
            self.saccades.append(t)
            self.saccades_total += 1
        if self.saccade_baseline is None and self.valid_s >= config.gaze_baseline_s:
            self.saccade_baseline = self.saccades_total * 60.0 / self.valid_s

    def _expire_events(self, t):
        cutoff = t - self.config.window_s
        while self.blinks and self.blinks[0].end < cutoff:
            self.blinks.popleft()
        while self.saccades and self.saccades[0] < cutoff:
            self.saccades.popleft()

    def _features(self):
        config = self.config
        minutes = max(self.perclos.total, 1.0) / 60.0
        durations = [blink.duration for blink in self.blinks]
        saccade_rate = len(self.saccades) / minutes

        drop = None
        if (self.saccade_baseline is not None
                and self.saccade_baseline >= config.gaze_baseline_min_rate):
            drop = max(0.0, 1.0 - saccade_rate / self.saccade_baseline)

        return Features(
            perclos=self.perclos.value,
            mean_blink_s=mean(durations) if durations else 0.0,
            long_blinks_per_min=sum(d >= config.long_blink_s for d in durations) / minutes,
            mean_reopen_s=mean(b.reopen for b in self.blinks) if durations else 0.0,
            blinks_per_min=len(durations) / minutes,
            saccades_per_min=saccade_rate,
            saccade_drop=drop,
        )
