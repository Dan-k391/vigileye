"""Find labelled clips laid out like UTA-RLDD: <FoldN_...>/<subject>/<label>.<ext>."""

import re
from dataclasses import dataclass
from pathlib import Path

ALERT = 0
LOW_VIGILANT = 5
DROWSY = 10
LABELS = (ALERT, LOW_VIGILANT, DROWSY)
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


@dataclass(frozen=True)
class Clip:
    subject: str
    label: int
    path: Path
    fold: int | None

    @property
    def key(self):
        return f"{self.subject}_{self.label}"


def find_clips(root):
    """Every labelled video under root, or an error if a subject is ambiguous."""
    clips = {}
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES or path.name.startswith("."):
            continue
        if not path.stem.isdigit() or int(path.stem) not in LABELS:
            continue
        fold = re.search(r"fold\s*(\d+)", str(path.parent), re.IGNORECASE)
        clip = Clip(path.parent.name, int(path.stem), path,
                    int(fold.group(1)) if fold else None)
        if clip.key in clips:
            raise ValueError(f"Two clips for {clip.key}: {clips[clip.key].path} and {path}")
        clips[clip.key] = clip
    return list(clips.values())
