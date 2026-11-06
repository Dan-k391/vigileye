"""Non-blocking audible alerts, so sound never delays frame processing."""

import shutil
import subprocess
import sys
import time

from .scoring import Level

SOUNDS = "/System/Library/Sounds/"
DROWSY_REPEAT_S = 20.0
CRITICAL_REPEAT_S = 1.5


class Alerter:
    def __init__(self, muted=False):
        self.muted = muted
        self.player = shutil.which("afplay")
        self.speaker = shutil.which("say")
        self.process = None
        self.last_alert = float("-inf")

    def _run(self, *command):
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)

    def _play(self, sound):
        if self.player:
            self._run(self.player, SOUNDS + sound + ".aiff")
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()

    def cue(self):
        """A calibration prompt: played even when muted, as eyes may be shut."""
        self._play("Glass")

    def toggle_mute(self):
        self.muted = not self.muted

    def update(self, level):
        if self.muted or level < Level.DROWSY:
            return
        now = time.monotonic()
        repeat = CRITICAL_REPEAT_S if level == Level.CRITICAL else DROWSY_REPEAT_S
        if now - self.last_alert < repeat:
            return
        self.last_alert = now
        if level == Level.CRITICAL:
            if self.speaker:
                self._run(self.speaker, "-r", "230", "Wake up")
            else:
                self._play("Sosumi")
        else:
            self._play("Funk")

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
