"""Run with: python -m evaluation extract --data DIR, then report"""

import argparse
import sys

from vigileye.config import Config

from .dataset import find_clips
from .extract import CACHE_DIR, cache_path, cached_clips, extract_all
from .report import (ALERT_CLIP, RESULTS_DIR, SAME_CLIP, build_report, score_clips,
                     write_outputs)


def parse_args():
    parser = argparse.ArgumentParser(prog="evaluation", description=__doc__)
    parser.add_argument("command", choices=("extract", "report"))
    parser.add_argument("--data", help="folder holding the labelled videos; "
                                       "report can leave it out and use every cached clip")
    parser.add_argument("--folds", help="only these folds, for example 1,2")
    parser.add_argument("--workers", type=int, help="tracking processes (default: half the cores)")
    parser.add_argument("--calibration", choices=(ALERT_CLIP, SAME_CLIP), default=ALERT_CLIP,
                        help="take open-eye EAR from the person's alert clip, or each clip's own start")
    args = parser.parse_args()
    if not args.data and (args.command == "extract" or args.folds):
        parser.error("--data is needed to extract or to choose folds")
    return args


def main():
    args = parse_args()
    clips = find_clips(args.data) if args.data else cached_clips()
    if args.folds:
        wanted = {int(fold) for fold in args.folds.split(",")}
        clips = [clip for clip in clips if clip.fold in wanted]
    if not clips:
        print(f"No labelled clips found under {args.data or CACHE_DIR}", file=sys.stderr)
        return 1
    print(f"Found {len(clips)} clips from {len({c.subject for c in clips})} people")

    if args.command == "extract":
        failures = extract_all(clips, args.workers)
        for key, error in failures:
            print(f"Failed: {key}: {error}", file=sys.stderr)
        return 1 if failures else 0

    cached = [clip for clip in clips if cache_path(clip, CACHE_DIR).exists()]
    if not cached:
        print("Nothing cached yet. Run the extract command first.", file=sys.stderr)
        return 1
    if len(cached) < len(clips):
        print(f"Warning: only {len(cached)} of {len(clips)} clips are cached")

    config = Config()
    results, unusable = score_clips(cached, args.calibration, config, CACHE_DIR)
    data = args.data or "cached clips"
    source = f"{data}, folds {args.folds}" if args.folds else data
    report = build_report(results, unusable, args.calibration, config, source)
    print(report)
    print(f"Saved: {write_outputs(results, report, args.calibration, RESULTS_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
