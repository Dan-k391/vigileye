"""Combine the eye measurements into one explainable 0-100 fatigue score."""

from dataclasses import dataclass
from enum import IntEnum
from math import exp


class Level(IntEnum):
    ALERT = 0
    TIRED = 1
    DROWSY = 2
    CRITICAL = 3


@dataclass(frozen=True)
class Features:
    perclos: float = 0.0
    mean_blink_s: float = 0.0
    long_blinks_per_min: float = 0.0
    mean_reopen_s: float = 0.0
    blinks_per_min: float = 0.0
    saccades_per_min: float = 0.0
    saccade_drop: float | None = None  # None until the personal rate is known


def ramp(value, bounds):
    low, high = bounds
    return min(1.0, max(0.0, (value - low) / (high - low)))


def contributions(features, config):
    """Each feature's share of the score, in score points."""
    drop = features.saccade_drop or 0.0
    return {
        "perclos": config.weight_perclos * ramp(features.perclos, config.perclos_ramp),
        "blink length": config.weight_blink_duration
        * ramp(features.mean_blink_s, config.blink_duration_ramp),
        "long blinks": config.weight_long_blinks
        * ramp(features.long_blinks_per_min, config.long_blink_ramp),
        "slow reopen": config.weight_reopen * ramp(features.mean_reopen_s, config.reopen_ramp),
        "gaze slowing": config.weight_saccade_drop * ramp(drop, config.saccade_drop_ramp),
        "blink rate": config.weight_blink_rate
        * ramp(features.blinks_per_min, config.blink_rate_ramp),
    }


class FatigueScorer:
    def __init__(self, config):
        self.config = config
        self.score = 0.0
        self.level = Level.ALERT
        self.last_microsleep = None

    def note_microsleep(self, t):
        self.last_microsleep = t

    def _target(self, features):
        config = self.config
        weighted = 100.0 * sum(contributions(features, config).values())
        # Sustained closure is the strongest single sign, so it must be able
        # to raise an alarm even when the other measurements look normal.
        return max(weighted, config.perclos_floor * ramp(features.perclos, config.perclos_ramp))

    def update(self, t, dt, features):
        # Smoothed, so a single bad frame cannot raise an alarm.
        keep = exp(-dt / self.config.score_smoothing_s)
        self.score = self._target(features) * (1.0 - keep) + self.score * keep
        return self.apply_events(t)

    def apply_events(self, t):
        """Apply microsleep effects, which need no smoothing or history."""
        config = self.config
        if self.last_microsleep is not None:
            fade = 1.0 - (t - self.last_microsleep) / config.microsleep_decay_s
            self.score = max(self.score, config.microsleep_floor * fade)
        self.level = self._next_level(t)
        return self.score, self.level

    def _next_level(self, t):
        config = self.config
        if (self.last_microsleep is not None
                and t - self.last_microsleep <= config.microsleep_hold_s):
            return Level.CRITICAL

        previous = min(self.level, Level.DROWSY)
        margin = config.level_hysteresis
        if self.score >= config.drowsy_score - (margin if previous == Level.DROWSY else 0.0):
            return Level.DROWSY
        if self.score >= config.tired_score - (margin if previous >= Level.TIRED else 0.0):
            return Level.TIRED
        return Level.ALERT
