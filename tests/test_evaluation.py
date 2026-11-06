"""The evaluation must get known answers right before its real output can be trusted."""

import random
import shutil

import cv2
import numpy as np
import pytest
from simulate import ALERT as ALERT_PROFILE
from simulate import DROWSY as DROWSY_PROFILE
from simulate import closure_at, ear_for, make_blinks

from evaluation import metrics
from evaluation.dataset import ALERT, DROWSY, LOW_VIGILANT, Clip, find_clips
from evaluation.extract import FIELDS, cache_path, cached_clips, extract_clip, load_clip
from evaluation.replay import make_baseline, replay
from evaluation.report import (ALERT_CLIP, SAME_CLIP, build_report, score_clips)
from vigileye.config import Config
from vigileye.tracker import PROJECT_ROOT

SCRATCH = PROJECT_ROOT / "tests" / ".scratch-evaluation"
PHOTO = "/opt/anaconda3/lib/python3.13/site-packages/skimage/data/astronaut.png"


@pytest.fixture
def scratch():
    SCRATCH.mkdir(exist_ok=True)
    yield SCRATCH
    shutil.rmtree(SCRATCH)


def synthetic_data(profile, seed, seconds=240, fps=30.0, open_ear_offset=0.0):
    rng = random.Random(seed)
    blinks = make_blinks(profile, seconds, rng)
    t = np.arange(0.0, seconds, 1.0 / fps)
    ear = np.array([ear_for(closure_at(x, blinks)) + rng.gauss(0, 0.004) for x in t])
    gaze = np.cumsum(np.random.default_rng(seed).random(len(t)) < 40 / 60 / fps) % 2 * 0.1
    return dict(zip(FIELDS, (t, ear + open_ear_offset, gaze, np.zeros_like(t),
                             np.zeros_like(t), np.zeros_like(t))))


def save_clip(data, clip, cache_dir):
    np.savez_compressed(cache_path(clip, cache_dir), **data)


def synthetic_cache(cache_dir, profiles, people=12):
    """profiles maps a label to a Profile; each person gets one clip per label."""
    clips = []
    for person in range(people):
        for label, profile in profiles.items():
            clip = Clip(f"{person:02d}", label, cache_dir / "unused", None)
            # Different starting EAR per person, as real faces have.
            offset = (person % 4) * 0.01
            save_clip(synthetic_data(profile, person * 10 + label, open_ear_offset=offset),
                      clip, cache_dir)
            clips.append(clip)
    return clips


def test_auc_by_hand():
    assert metrics.auc([3, 4], [1, 2]) == 1.0
    assert metrics.auc([1, 2], [3, 4]) == 0.0
    assert metrics.auc([1, 3], [2, 2]) == 0.5
    assert metrics.auc([2], [2]) == 0.5  # a tie is worth half


def test_paired_win_rate_by_hand():
    assert metrics.paired_win_rate([(2, 1), (1, 2), (3, 3), (5, 1)]) == pytest.approx(0.625)


def test_sensitivity_and_specificity_by_hand():
    flags = [(True, False), (True, True), (False, False), (True, False)]
    assert metrics.pair_sensitivity(flags) == 0.75
    assert metrics.pair_specificity(flags) == 0.75


def test_bootstrap_interval_contains_the_value_and_is_repeatable():
    rng = np.random.default_rng(1)
    pairs = [(rng.normal(1), rng.normal(0)) for _ in range(40)]
    low, high = metrics.bootstrap_ci(pairs, metrics.pair_auc, draws=300)
    assert low < metrics.pair_auc(pairs) < high
    assert (low, high) == metrics.bootstrap_ci(pairs, metrics.pair_auc, draws=300)


def test_find_clips_reads_the_layout(scratch):
    for name in ("Fold2_part1/07/0.mov", "Fold2_part1/07/10.MOV", "Fold3_part2/44/5.mp4",
                 "Fold2_part1/07/._0.mov", "Fold2_part1/07/7.mov", "Fold2_part1/07/notes.txt"):
        path = scratch / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    clips = {clip.key: clip for clip in find_clips(scratch)}
    assert set(clips) == {"07_0", "07_10", "44_5"}
    assert clips["07_10"].label == DROWSY
    assert clips["07_10"].fold == 2
    assert clips["44_5"].fold == 3


def test_find_clips_rejects_a_duplicate_person(scratch):
    for name in ("Fold1_part1/01/0.mov", "Fold2_part1/01/0.mov"):
        path = scratch / name
        path.parent.mkdir(parents=True)
        path.touch()
    with pytest.raises(ValueError, match="Two clips"):
        find_clips(scratch)


def test_cache_alone_gives_the_same_scores_as_the_videos(scratch):
    config = Config()
    clips = synthetic_cache(scratch, {ALERT: ALERT_PROFILE, DROWSY: DROWSY_PROFILE}, people=2)
    (scratch / "00_0.partial.npz").touch()
    found = cached_clips(scratch)
    assert {clip.key for clip in found} == {clip.key for clip in clips}
    assert (sorted(score_clips(found, ALERT_CLIP, config, scratch)[0], key=str)
            == sorted(score_clips(clips, ALERT_CLIP, config, scratch)[0], key=str))


def test_clips_with_a_clear_difference_score_near_perfect(scratch):
    config = Config()
    clips = synthetic_cache(scratch, {ALERT: ALERT_PROFILE, DROWSY: DROWSY_PROFILE})
    results, unusable = score_clips(clips, ALERT_CLIP, config, scratch)
    assert not unusable
    drowsy = [r.mean_score for r in results if r.label == DROWSY]
    alert = [r.mean_score for r in results if r.label == ALERT]
    assert metrics.auc(drowsy, alert) > 0.95


def test_clips_with_no_difference_score_near_guessing(scratch):
    """Guards against leakage: labels that mean nothing must not look predictive."""
    config = Config()
    clips = synthetic_cache(scratch, {ALERT: ALERT_PROFILE, DROWSY: ALERT_PROFILE}, people=30)
    results, _ = score_clips(clips, ALERT_CLIP, config, scratch)
    drowsy = [r.mean_score for r in results if r.label == DROWSY]
    alert = [r.mean_score for r in results if r.label == ALERT]
    assert 0.3 < metrics.auc(drowsy, alert) < 0.7


def test_calibration_stretch_is_never_scored(scratch):
    config = Config()
    clip = Clip("01", ALERT, scratch / "unused", None)
    data = synthetic_data(ALERT_PROFILE, 1, seconds=120)
    result = replay(clip, data, make_baseline(data, data, config), config)
    assert result.minutes == pytest.approx(1.0, abs=0.02)


def test_clip_too_short_to_score_is_reported_not_dropped(scratch):
    config = Config()
    clips = synthetic_cache(scratch, {ALERT: ALERT_PROFILE}, people=1)
    short = Clip("02", ALERT, scratch / "unused", None)
    save_clip(synthetic_data(ALERT_PROFILE, 3, seconds=70), short, scratch)
    results, unusable = score_clips(clips + [short], ALERT_CLIP, config, scratch)
    assert unusable == ["02_0"]
    assert len(results) == 1


def test_missing_alert_clip_is_reported_under_alert_calibration(scratch):
    config = Config()
    clip = Clip("03", DROWSY, scratch / "unused", None)
    save_clip(synthetic_data(DROWSY_PROFILE, 4), clip, scratch)
    assert score_clips([clip], ALERT_CLIP, config, scratch) == ([], ["03_10"])
    assert len(score_clips([clip], SAME_CLIP, config, scratch)[0]) == 1


def test_low_tracking_clip_is_excluded_and_named(scratch):
    config = Config()
    clip = Clip("04", ALERT, scratch / "unused", None)
    data = synthetic_data(ALERT_PROFILE, 5)
    data["ear"][len(data["ear"]) // 3:] = np.nan
    save_clip(data, clip, scratch)
    assert score_clips([clip], SAME_CLIP, config, scratch) == ([], ["04_0"])


def test_report_states_the_results(scratch):
    config = Config()
    clips = synthetic_cache(scratch, {ALERT: ALERT_PROFILE, LOW_VIGILANT: DROWSY_PROFILE,
                                      DROWSY: DROWSY_PROFILE}, people=4)
    results, unusable = score_clips(clips, ALERT_CLIP, config, scratch)
    report = build_report(results, unusable, ALERT_CLIP, config, "synthetic")
    assert "4 people" in report
    assert "AUC of mean score" in report
    assert "Three-class accuracy" in report


def test_report_refuses_to_claim_accuracy_from_too_few_people(scratch):
    config = Config()
    clips = synthetic_cache(scratch, {ALERT: ALERT_PROFILE, DROWSY: DROWSY_PROFILE}, people=1)
    results, unusable = score_clips(clips, ALERT_CLIP, config, scratch)
    assert "Too few people" in build_report(results, unusable, ALERT_CLIP, config, "synthetic")


@pytest.mark.skipif(not __import__("os").path.exists(PHOTO), reason="test photo not available")
def test_extraction_keeps_face_loss_and_time(scratch):
    image = cv2.resize(cv2.imread(PHOTO), (640, 640))
    blank = np.zeros_like(image)
    video = scratch / "0.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 640))
    for frame in [image] * 30 + [blank] * 30 + [image] * 30:
        writer.write(frame)
    writer.release()

    clip = Clip("05", ALERT, video, None)
    _, frames = extract_clip(clip, scratch)
    data = load_clip(clip, scratch)
    assert frames == 90
    assert np.all(np.diff(data["t"]) > 0)
    assert data["t"][-1] == pytest.approx(89 / 30, abs=0.1)
    assert not np.isnan(data["ear"][:25]).any()
    assert np.isnan(data["ear"][35:55]).all()
    assert 0.2 < np.nanmedian(data["ear"]) < 0.45
