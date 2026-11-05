"""P0 — Video capture (ARCHITECTURE.md §5).

Two rules that this whole file exists to enforce:

1. ``CAP_PROP_BUFFERSIZE = 1``. Without it OpenCV/FFmpeg buffers frames
   internally and you accumulate seconds of hidden latency that no profiler will
   show you. This is *the* day-one bug (ARCHITECTURE.md §10) — a pass-through
   pipeline that reads 300 ms instead of 30 ms has found it.

2. Timestamp at the moment ``read()`` returns, before any preprocessing, using
   the single monotonic clock. That stamp travels with the frame forever and is
   what every downstream latency and alignment computation is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.events import now_ns


@dataclass(slots=True)
class CapturedFrame:
    """A decoded frame plus its capture metadata. ``seq`` is monotonic; a gap in
    ``seq`` downstream means frames were dropped and where."""

    frame: np.ndarray      # BGR HxWxC uint8, as OpenCV delivers it
    ts_mono_ns: int
    seq: int


class VideoCapture:
    """Thin wrapper over ``cv2.VideoCapture`` with the latency-critical settings.

    ``source`` is a camera index (``0``), a file path, or an RTSP URL. For files
    the frames decode faster than real time, which hides every backpressure bug
    — use a real camera for honest measurement (plan.md Phase 1 trap).
    """

    def __init__(self, source: int | str = 0, *, width: int | None = None,
                 height: int | None = None, fps: float | None = None):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {source!r}")

        # The one setting that matters most. Ask for a 1-frame internal buffer so
        # read() always returns the newest frame, not a queued stale one.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._seq = 0

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) actually granted by the device."""
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self) -> CapturedFrame | None:
        """Grab one frame. Returns None at end-of-stream / on read failure.

        The timestamp is taken the instant ``read()`` returns — not after the
        return value is inspected, not after preprocessing.
        """
        ok, frame = self.cap.read()
        ts = now_ns()
        if not ok or frame is None:
            return None
        cf = CapturedFrame(frame=frame, ts_mono_ns=ts, seq=self._seq)
        self._seq += 1
        return cf

    def frames(self):
        """Iterate frames until the stream ends."""
        while True:
            cf = self.read()
            if cf is None:
                return
            yield cf

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
