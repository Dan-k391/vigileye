"""Every tunable threshold in one place. Times are seconds unless noted."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Calibration
    calib_open_s: float = 8.0  # valid seconds of eyes open, looking ahead
    calib_closed_s: float = 3.0  # eyes closed on cue
    calib_settle_s: float = 0.8  # ignored after the cue, while the person reacts
    calib_reopen_s: float = 2.0  # time to react to the chime and open the eyes
    calib_min_closed_ratio: float = 0.75  # closed/open EAR above this = did not close
    fallback_closed_ratio: float = 0.35  # assumed closed/open EAR when not measured

    # Eyelid closure, on the 0 (open) to 1 (closed) scale
    perclos_closure: float = 0.8  # "P80": at least 80% closed
    blink_enter: float = 0.5
    blink_exit: float = 0.3
    blink_min_s: float = 0.05  # shorter than this is landmark noise
    long_blink_s: float = 0.5
    microsleep_s: float = 1.0
    microsleep_hold_s: float = 5.0  # stay CRITICAL this long after reopening

    # Rolling analysis
    window_s: float = 60.0  # PERCLOS is defined over one minute
    min_coverage_s: float = 20.0  # valid data needed before a score is meaningful
    max_gap_s: float = 0.25  # a longer gap between samples breaks a blink
    ear_median_samples: int = 3

    # Head pose gate, relative to the calibrated neutral pose (degrees).
    # Looking down is deliberately not gated: that is how a head nods off.
    max_yaw_deg: float = 30.0
    max_pitch_up_deg: float = 25.0

    # Saccades: a gaze shift of at least this size (fraction of eye width,
    # roughly 6 degrees) inside this time. Webcams undersample saccades, so
    # this counts them rather than measuring their peak velocity.
    saccade_amplitude: float = 0.04
    saccade_span_s: float = 0.12
    saccade_refractory_s: float = 0.15
    blink_gaze_margin_s: float = 0.1  # the iris is unreliable around a blink
    gaze_baseline_s: float = 180.0  # valid time used to learn the personal rate
    gaze_baseline_min_rate: float = 5.0  # per minute; below this it is no signal

    # Score: feature ramps (low = no fatigue, high = full), then weights
    perclos_ramp: tuple = (0.05, 0.25)
    blink_duration_ramp: tuple = (0.20, 0.50)
    long_blink_ramp: tuple = (0.0, 6.0)  # per minute
    reopen_ramp: tuple = (0.12, 0.35)
    blink_rate_ramp: tuple = (25.0, 40.0)  # per minute
    saccade_drop_ramp: tuple = (0.3, 0.8)  # fractional drop from personal rate

    weight_perclos: float = 0.35
    weight_blink_duration: float = 0.20
    weight_long_blinks: float = 0.20
    weight_reopen: float = 0.10
    weight_saccade_drop: float = 0.10
    weight_blink_rate: float = 0.05

    perclos_floor: float = 70.0  # high PERCLOS alone can reach this score
    microsleep_floor: float = 85.0  # score floor right after a microsleep
    microsleep_decay_s: float = 120.0  # that floor fades to zero over this time
    score_smoothing_s: float = 3.0

    # Levels, with hysteresis so the display does not flicker at a boundary
    tired_score: float = 30.0
    drowsy_score: float = 55.0
    level_hysteresis: float = 8.0
