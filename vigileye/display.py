"""Live video overlay."""

from collections import deque

import cv2
import numpy as np

from .geometry import LEFT_EYE, LEFT_IRIS, RIGHT_EYE, RIGHT_IRIS
from .monitor import (CALIBRATING_CLOSED, CALIBRATING_OPEN, LOOKING_AWAY, NO_FACE,
                      TRACKING, WARMING_UP)
from .scoring import Level, contributions

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_WIDTH = 330
WHITE = (240, 240, 240)
GREY = (170, 170, 170)
LEVEL_COLORS = {
    Level.ALERT: (110, 200, 80),
    Level.TIRED: (60, 200, 240),
    Level.DROWSY: (40, 130, 255),
    Level.CRITICAL: (50, 50, 235),
}
STATE_LABELS = {
    WARMING_UP: "Measuring, score not ready",
    TRACKING: "Tracking",
    NO_FACE: "NO FACE: paused",
    LOOKING_AWAY: "Looking away: paused",
}


def text(image, message, x, y, color=WHITE, scale=0.5, thickness=1):
    cv2.putText(image, message, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def centred(image, message, y, color=WHITE, scale=1.0, thickness=2):
    width = cv2.getTextSize(message, FONT, scale, thickness)[0][0]
    text(image, message, (image.shape[1] - width) // 2, y, color, scale, thickness)


class Display:
    def __init__(self, config, window="VigilEye", history=450):
        self.config = config
        self.window = window
        self.closures = deque(maxlen=history)
        self.flash = 0
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def show(self, frame, points, status, fps, muted):
        """Display one frame and return the key pressed, if any."""
        cv2.imshow(self.window, self.render(frame, points, status, fps, muted))
        return cv2.waitKey(1) & 0xFF

    def render(self, frame, points, status, fps, muted):
        image = cv2.flip(frame, 1)
        if points is not None:
            self._draw_eyes(image, points)
        self.closures.append(status.closure)

        if status.state in (CALIBRATING_OPEN, CALIBRATING_CLOSED):
            self._draw_calibration(image, status)
        else:
            self._draw_panel(image, status, fps, muted)
            self._draw_alarm(image, status)
        return image

    def _draw_eyes(self, image, points):
        width = image.shape[1]
        mirrored = points[:, :2].copy()
        mirrored[:, 0] = width - 1 - mirrored[:, 0]
        for eye in (RIGHT_EYE, LEFT_EYE):
            contour = mirrored[list(eye)].astype(np.int32)
            cv2.polylines(image, [contour], True, (110, 230, 120), 1, cv2.LINE_AA)
        for iris in (RIGHT_IRIS, LEFT_IRIS):
            cv2.circle(image, tuple(mirrored[iris].astype(int)), 2, (255, 200, 60), -1,
                       cv2.LINE_AA)

    def _shade(self, image, x0, y0, x1, y1, strength=0.65):
        region = image[y0:y1, x0:x1]
        region[:] = (region * (1.0 - strength)).astype(region.dtype)

    def _draw_calibration(self, image, status):
        height, width = image.shape[:2]
        self._shade(image, 0, height // 2 - 80, width, height // 2 + 80)
        if status.state == CALIBRATING_OPEN:
            centred(image, "Look straight ahead, eyes open", height // 2 - 20)
            bar = int((width - 200) * min(1.0, status.calibration_progress))
            cv2.rectangle(image, (100, height // 2 + 20), (100 + bar, height // 2 + 34),
                          LEVEL_COLORS[Level.ALERT], -1)
        else:
            centred(image, "Close your eyes until the second chime", height // 2 + 10)

    def _draw_panel(self, image, status, fps, muted):
        features = status.features
        self._shade(image, 0, 0, PANEL_WIDTH, min(image.shape[0], 400))
        color = LEVEL_COLORS[status.level]
        text(image, status.level.name, 16, 42, color, 1.0, 2)
        text(image, f"{status.score:3.0f}", 245, 42, color, 1.0, 2)

        cv2.rectangle(image, (16, 56), (PANEL_WIDTH - 16, 66), (70, 70, 70), -1)
        bar = int((PANEL_WIDTH - 32) * min(100.0, status.score) / 100.0)
        cv2.rectangle(image, (16, 56), (16 + bar, 66), color, -1)
        state_color = WHITE if status.state in (TRACKING, WARMING_UP) else (60, 200, 240)
        text(image, STATE_LABELS[status.state], 16, 88, state_color)

        rows = (
            ("PERCLOS (60 s)", f"{features.perclos * 100:.1f} %"),
            ("Blink length", f"{features.mean_blink_s * 1000:.0f} ms"),
            ("Eyelid reopen", f"{features.mean_reopen_s * 1000:.0f} ms"),
            ("Blinks", f"{features.blinks_per_min:.0f} /min"),
            ("Long blinks", f"{features.long_blinks_per_min:.1f} /min"),
            ("Saccades", f"{features.saccades_per_min:.0f} /min"),
        )
        y = 116
        for label, value in rows:
            text(image, label, 16, y, GREY)
            text(image, value, 200, y, WHITE)
            y += 23

        top = max(contributions(features, self.config).items(), key=lambda item: item[1])
        if top[1] > 0:
            text(image, f"Main factor: {top[0]}", 16, y + 2, GREY)
        self._draw_closure_trace(image, 16, y + 16, PANEL_WIDTH - 32, 64)
        hint = f"q quit  c recalibrate  m {'unmute' if muted else 'mute'}  {fps:.0f} fps"
        text(image, hint, 16, y + 100, GREY, 0.4)

    def _draw_closure_trace(self, image, x, y, width, height):
        cv2.rectangle(image, (x, y), (x + width, y + height), (60, 60, 60), 1)
        limit = y + int(height * (1.0 - self.config.perclos_closure))
        cv2.line(image, (x, limit), (x + width, limit), (60, 60, 150), 1)
        if len(self.closures) >= 2:
            xs = np.linspace(x, x + width, self.closures.maxlen)[-len(self.closures):]
            ys = y + height * (1.0 - np.array(self.closures))
            trace = np.stack((xs, ys), axis=1).astype(np.int32)
            cv2.polylines(image, [trace], False, (240, 240, 120), 1, cv2.LINE_AA)
        text(image, "eyelid closure", x + 4, y + height - 4, GREY, 0.35)

    def _draw_alarm(self, image, status):
        height, width = image.shape[:2]
        if status.level == Level.CRITICAL:
            self.flash = (self.flash + 1) % 12
            if self.flash < 6:
                cv2.rectangle(image, (0, 0), (width - 1, height - 1),
                              LEVEL_COLORS[Level.CRITICAL], 26)
            self._shade(image, 0, height - 120, width, height - 30, 0.8)
            centred(image, "WAKE UP - PULL OVER", height - 60,
                    LEVEL_COLORS[Level.CRITICAL], 1.5, 4)
        elif status.level == Level.DROWSY:
            self._shade(image, 0, height - 110, width, height - 40, 0.8)
            centred(image, "DROWSY - take a break", height - 60,
                    LEVEL_COLORS[Level.DROWSY], 1.1, 3)

    def close(self):
        cv2.destroyAllWindows()
