"""P6 (partial) — overlay sink (ARCHITECTURE.md §5, plan.md Phase 1).

Draws detection/track boxes and a diagnostics HUD onto the BGR frame: track id,
class, confidence, live FPS, and a per-stage latency readout. The HUD is not
decoration — it is how you watch the frame budget in real time and catch the
moment a stage starts eating the tail (p99 is what users feel).

Pure OpenCV drawing; no model dependency, so it stays trivially testable.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from core.events import Track

# Deterministic per-id color so a track keeps its color across frames — an id
# that changes color is a visible tracker ID-switch (an active-learning signal).
_PALETTE = [
    (56, 176, 0), (0, 165, 255), (255, 128, 0), (200, 0, 200),
    (0, 0, 255), (255, 0, 128), (0, 200, 200), (128, 0, 255),
]


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


class FpsMeter:
    """Rolling FPS over the last ``window`` frame intervals."""

    def __init__(self, window: int = 30):
        self._ts = deque(maxlen=window)

    def tick(self, now_s: float) -> float:
        self._ts.append(now_s)
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Draw boxes + labels for each track. Bboxes are normalized xyxy; scale to
    pixels here so the rest of the system stays resolution-independent."""
    h, w = frame.shape[:2]
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        col = _color(t.track_id)
        cv2.rectangle(frame, p1, p2, col, 2)
        label = f"#{t.track_id} {t.cls} {t.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(0, p1[1] - th - 4)
        cv2.rectangle(frame, (p1[0], ly), (p1[0] + tw + 4, ly + th + 4), col, -1)
        cv2.putText(frame, label, (p1[0] + 2, ly + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, *, fps: float, n_tracks: int,
             stage_ms: dict[str, float] | None = None,
             seq: int | None = None) -> np.ndarray:
    """Top-left diagnostics panel: FPS, object count, per-stage latency."""
    lines = [f"FPS {fps:5.1f}   objects {n_tracks}"]
    if seq is not None:
        lines[0] += f"   seq {seq}"
    if stage_ms:
        total = sum(stage_ms.values())
        lines.append("  ".join(f"{k} {v:4.1f}ms" for k, v in stage_ms.items()))
        lines.append(f"total {total:5.1f}ms  budget 33.3ms")

    pad = 6
    (tw, th), _ = cv2.getTextSize(max(lines, key=len), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    box_h = pad + len(lines) * (th + 6)
    cv2.rectangle(frame, (0, 0), (tw + 2 * pad, box_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        y = pad + (i + 1) * (th + 4)
        # Red the total line if we blew the 33.3 ms frame budget.
        over = stage_ms and line.startswith("total") and sum(stage_ms.values()) > 33.3
        color = (0, 0, 255) if over else (0, 255, 0)
        cv2.putText(frame, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return frame
