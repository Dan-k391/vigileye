"""Turn per-clip results into the accuracy report."""

import csv
from dataclasses import asdict, fields
from pathlib import Path
from statistics import mean

from vigileye.config import Config

from . import metrics
from .dataset import ALERT, DROWSY, LABELS, LOW_VIGILANT
from .extract import load_clip
from .replay import ClipResult, make_baseline, replay

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_TRACKED_SHARE = 0.5
ALERT_CLIP = "alert-clip"
SAME_CLIP = "same-clip"
FEATURES = (
    ("Fatigue score", "mean_score"),
    ("PERCLOS", "perclos"),
    ("Blink length", "blink_s"),
    ("Long blinks", "long_blinks_per_min"),
    ("Eyelid reopening", "reopen_s"),
    ("Blink rate", "blinks_per_min"),
    ("Saccade rate (lower = more tired)", "saccades_per_min"),
)


def score_clips(clips, calibration, config, cache_dir):
    """Replay every clip. Returns (results, clip keys with no usable score)."""
    by_key = {clip.key: clip for clip in clips}
    results, unusable = [], []
    for clip in clips:
        data = load_clip(clip, cache_dir)
        ear_data = data
        alert = by_key.get(f"{clip.subject}_{ALERT}")
        if calibration == ALERT_CLIP:
            if alert is None:
                unusable.append(clip.key)
                continue
            ear_data = load_clip(alert, cache_dir)
        result = replay(clip, data, make_baseline(ear_data, data, config), config)
        if result is None or result.tracked_share < MIN_TRACKED_SHARE:
            unusable.append(clip.key)
        else:
            results.append(result)
    return results, unusable


def paired(results):
    """{subject: (drowsy result, alert result)} for people with both clips."""
    drowsy = {r.subject: r for r in results if r.label == DROWSY}
    alert = {r.subject: r for r in results if r.label == ALERT}
    return {subject: (drowsy[subject], alert[subject]) for subject in drowsy if subject in alert}


def with_ci(pairs, statistic):
    value = statistic(pairs)
    low, high = metrics.bootstrap_ci(pairs, statistic)
    return f"{value:.2f} (95% CI {low:.2f} to {high:.2f})"


def predicted_label(score, config):
    if score >= config.drowsy_score:
        return DROWSY
    return LOW_VIGILANT if score >= config.tired_score else ALERT


def build_report(results, unusable, calibration, config, source):
    people = paired(results)
    lines = ["# VigilEye accuracy report", "",
             f"Data: {source}. Calibration: {calibration}.", "",
             f"{len(results)} clips scored, {len(people)} people with both an alert "
             f"and a drowsy clip, {len(unusable)} clips unusable"
             f"{': ' + ', '.join(sorted(unusable)) if unusable else ''}.", ""]
    if len(people) < 2:
        return "\n".join(lines + ["Too few people to measure accuracy."]) + "\n"

    pairs = list(people.values())
    scores = [(d.mean_score, a.mean_score) for d, a in pairs]
    flags = [(d.alarms > 0, a.alarms > 0) for d, a in pairs]
    alert_clips = [r for r in results if r.label == ALERT]
    hours = sum(r.minutes for r in alert_clips) / 60.0
    false_alarms = sum(r.alarms for r in alert_clips)

    lines += ["## Drowsy against alert", "",
              "| Measure | Result |", "| --- | --- |",
              f"| AUC of mean score (0.5 = guessing) | {with_ci(scores, metrics.pair_auc)} |",
              "| Drowsy clip beat the same person's alert clip | "
              f"{with_ci(scores, metrics.paired_win_rate)} |",
              f"| Sensitivity (drowsy clips with an alarm) | "
              f"{with_ci(flags, metrics.pair_sensitivity)} |",
              f"| Specificity (alert clips with no alarm) | "
              f"{with_ci(flags, metrics.pair_specificity)} |",
              f"| False alarms per hour on alert clips | {false_alarms / max(hours, 1e-9):.1f} "
              f"({false_alarms} in {hours:.1f} h) |", ""]

    lines += ["## Each signal alone", "", "| Signal | AUC |", "| --- | --- |"]
    for name, field in FEATURES:
        values = [(getattr(d, field), getattr(a, field)) for d, a in pairs]
        lines.append(f"| {name} | {with_ci(values, metrics.pair_auc)} |")

    lines += ["", "## By label", "",
              "| Label | Clips | Mean score | Clips with an alarm |", "| --- | --- | --- | --- |"]
    names = {ALERT: "Alert", LOW_VIGILANT: "Low vigilant", DROWSY: "Drowsy"}
    for label in LABELS:
        group = [r for r in results if r.label == label]
        if group:
            alarmed = mean(r.alarms > 0 for r in group)
            lines.append(f"| {names[label]} | {len(group)} | "
                         f"{mean(r.mean_score for r in group):.1f} | {alarmed:.0%} |")

    correct = mean(predicted_label(r.mean_score, config) == r.label for r in results)
    lines += ["", f"Three-class accuracy from mean score: {correct:.1%} (guessing: 33.3%).", ""]
    return "\n".join(lines)


def write_outputs(results, report, name, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"{name}-clips.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, [field.name for field in fields(ClipResult)])
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    path = results_dir / f"{name}.md"
    path.write_text(report)
    return path
