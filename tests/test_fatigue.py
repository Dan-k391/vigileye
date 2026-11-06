import numpy as np
import pytest
from simulate import ALERT, CLOSED_EAR, DROWSY, OPEN_EAR, Profile, Run

from vigileye.config import Config
from vigileye.detectors import BlinkDetector, ClosureFraction, SaccadeDetector
from vigileye.geometry import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, head_angles
from vigileye.monitor import LOOKING_AWAY, NO_FACE, TRACKING, WARMING_UP
from vigileye.scoring import Level


def rotation(yaw=0.0, pitch=0.0):
    y, p = np.radians(yaw), np.radians(pitch)
    around_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    around_x = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    matrix = np.eye(4)
    matrix[:3, :3] = around_x @ around_y
    return matrix


def test_eye_aspect_ratio_of_known_shape():
    points = np.zeros((478, 3))
    width, gap = 10.0, 3.0
    for eye in (RIGHT_EYE, LEFT_EYE):
        points[list(eye)] = [(0, 0, 0), (3, gap / 2, 0), (7, gap / 2, 0),
                             (width, 0, 0), (7, -gap / 2, 0), (3, -gap / 2, 0)]
        assert eye_aspect_ratio(points, eye) == pytest.approx(gap / width)


def test_head_angle_signs():
    assert head_angles(rotation(yaw=20))[0] == pytest.approx(20)
    # A positive rotation about x tips the nose towards -y: head down.
    assert head_angles(rotation(pitch=15))[1] == pytest.approx(-15)


def test_calibration_measures_closed_eyes():
    baseline = Run().calibrate()
    assert baseline.closed_measured
    assert baseline.open_ear == pytest.approx(OPEN_EAR, abs=0.01)
    assert baseline.closed_ear == pytest.approx(CLOSED_EAR, abs=0.01)


def test_calibration_falls_back_if_eyes_never_close():
    baseline = Run().calibrate(close_eyes=False)
    assert not baseline.closed_measured
    assert baseline.closed_ear == pytest.approx(OPEN_EAR * 0.35, abs=0.01)


def test_calibration_finishes_if_closed_eyes_lose_tracking():
    run = Run()
    while run.monitor.calibrator.phase == "open":
        run.step(run.sample())
    for _ in range(200):
        run.step(None)
    assert run.monitor.baseline is not None
    assert not run.monitor.baseline.closed_measured


def test_slow_reaction_to_the_chime_is_not_a_microsleep():
    run = Run()
    run.calibrate()
    statuses = [run.step(run.sample(closure=1.0)) for _ in range(int(1.8 * run.fps))]
    statuses += run.play(ALERT, 5)
    assert all(s.level == Level.ALERT for s in statuses)


def test_eyes_kept_shut_after_the_grace_period_still_alarm():
    run = Run()
    run.calibrate()
    statuses = [run.step(run.sample(closure=1.0)) for _ in range(int(3.5 * run.fps))]
    assert statuses[-1].level == Level.CRITICAL


def test_closure_fraction_is_time_weighted_and_expires():
    perclos = ClosureFraction(window_s=10.0)
    perclos.add(1.0, 1.0, True)
    perclos.add(2.0, 3.0, False)
    assert perclos.value == pytest.approx(0.25)
    perclos.add(20.0, 1.0, False)
    assert perclos.value == 0.0


def test_blink_timing():
    detector = BlinkDetector(Config())
    trace = [(0.0, 0.0), (0.1, 0.9), (0.2, 1.0), (0.3, 1.0), (0.4, 0.6), (0.5, 0.1)]
    blinks = [detector.update(t, closure)[0] for t, closure in trace]
    blink = blinks[-1]
    assert blink.duration == pytest.approx(0.4)
    assert blink.reopen == pytest.approx(0.2)


def test_single_frame_spike_is_not_a_blink():
    detector = BlinkDetector(Config())
    detector.update(0.00, 0.9)
    assert detector.update(0.03, 0.0)[0] is None


def test_saccade_needs_a_fast_shift():
    config = Config()
    detector = SaccadeDetector(config)
    drift = [detector.update(i / 30, 0.0, i * 0.001) for i in range(30)]
    assert not any(drift)
    assert detector.update(1.0, 0.0, 0.2)


@pytest.mark.parametrize("fps", [15.0, 30.0, 60.0])
def test_alert_person_stays_alert(fps):
    run = Run(fps=fps)
    run.calibrate()
    statuses = run.play(ALERT, 300)
    assert max(s.level for s in statuses) == Level.ALERT
    assert max(s.score for s in statuses) < 30


@pytest.mark.parametrize("fps", [15.0, 30.0, 60.0])
def test_drowsy_person_is_flagged(fps):
    run = Run(fps=fps)
    run.calibrate()
    statuses = run.play(DROWSY, 300)
    assert statuses[-1].level >= Level.DROWSY
    assert statuses[-1].features.perclos > 0.05  # an alert person sits near 0.01


@pytest.mark.parametrize("seed", range(10))
def test_scenarios_hold_across_random_seeds(seed):
    alert = Run(seed=seed)
    alert.calibrate()
    assert max(s.level for s in alert.play(ALERT, 180)) == Level.ALERT

    drowsy = Run(seed=seed)
    drowsy.calibrate()
    assert max(s.level for s in drowsy.play(DROWSY, 180)) >= Level.DROWSY


def test_microsleep_alarms_while_eyes_are_still_shut():
    run = Run()
    run.calibrate()
    statuses = run.play(Profile(extra_closures=((30.0, 3.0),)), 40)
    first = next(i for i, s in enumerate(statuses) if s.level == Level.CRITICAL)
    assert 30.9 < first / run.fps < 31.2
    assert statuses[first].closure > 0.8


def test_alarm_lasts_as_long_as_the_eyes_stay_shut():
    run = Run()
    run.calibrate()
    statuses = run.play(Profile(extra_closures=((30.0, 15.0),)), 46)
    during = statuses[int(31.2 * run.fps):int(44.8 * run.fps)]
    assert all(s.level == Level.CRITICAL for s in during)


def test_microsleep_alarms_during_warm_up():
    run = Run()
    run.calibrate()
    statuses = run.play(Profile(extra_closures=((3.0, 2.0),)), 6)
    held = [s for s in statuses if s.level == Level.CRITICAL]
    assert held and all(s.state == WARMING_UP for s in held)
    assert len(held) >= run.fps  # held for the closure, not one flashed frame


def test_alarm_clears_after_recovery():
    run = Run()
    run.calibrate()
    run.play(Profile(extra_closures=((30.0, 2.0),)), 33)
    statuses = run.play(ALERT, 200)
    assert statuses[-1].level == Level.ALERT


def test_score_is_withheld_until_there_is_enough_data():
    run = Run()
    run.calibrate()
    statuses = run.play(ALERT, 25)
    assert statuses[0].state == WARMING_UP
    assert statuses[-1].state == TRACKING


def test_closed_eyes_while_looking_away_are_ignored():
    run = Run()
    run.calibrate()
    run.play(ALERT, 30)
    for _ in range(int(5 * run.fps)):
        status = run.step(run.sample(closure=1.0, yaw=45.0))
    assert status.state == LOOKING_AWAY
    assert status.level == Level.ALERT


def test_closure_with_head_tipped_down_still_alarms():
    run = Run()
    run.calibrate()
    run.play(ALERT, 30)
    for _ in range(int(2 * run.fps)):
        status = run.step(run.sample(closure=1.0, pitch=-35.0))
    assert status.level == Level.CRITICAL


def test_lost_face_pauses_and_does_not_read_as_recovery():
    run = Run()
    run.calibrate()
    run.play(DROWSY, 120)
    score = run.monitor.scorer.score
    for _ in range(int(10 * run.fps)):
        status = run.step(None)
    assert status.state == NO_FACE
    assert status.score == pytest.approx(score)
    assert status.level >= Level.TIRED


def test_gaze_slowing_is_measured_against_the_personal_rate():
    run = Run()
    run.calibrate()
    run.play(ALERT, 200)
    assert run.monitor.saccade_baseline == pytest.approx(40, abs=10)
    statuses = run.play(Profile(saccades_per_min=4.0), 90)
    assert statuses[-1].features.saccade_drop > 0.7
    assert statuses[-1].level == Level.ALERT  # gaze alone must not raise an alarm


def test_recalibrating_clears_the_session():
    run = Run()
    run.calibrate()
    run.play(DROWSY, 90)
    run.monitor.recalibrate()
    assert run.monitor.baseline is None
    assert run.monitor.scorer.score == 0.0
