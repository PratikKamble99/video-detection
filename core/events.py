"""The wire contract.

Every message that crosses a process boundary is one of these frozen
dataclasses. Freeze this early (ARCHITECTURE.md §4): it is the seam every
process talks across, and changing it later means touching every producer and
consumer at once.

Two invariants that the rest of the system depends on:

1. **One clock.** Every timestamp is nanoseconds from ``time.monotonic_ns()``,
   captured at ingest. Never wall-clock: an NTP step makes a wall clock jump
   backwards and the fusion window travels back in time with it.
2. **Vision events are instants, audio events are intervals.** A ``VisionEvent``
   happened at a single ``ts_mono_ns``. An ``AudioEvent`` spans
   ``[ts_start_ns, ts_end_ns]`` because ASR needs ≥0.5-1.0 s of acoustic
   context to say anything. Fusion must respect the asymmetry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def now_ns() -> int:
    """The one clock. Use this everywhere a timestamp is minted."""
    return time.monotonic_ns()


@dataclass(frozen=True, slots=True)
class FrameRef:
    """A 40-byte descriptor that stands in for a full frame on the queue.

    The frame's pixels live in the shared-memory ring (see
    ``core.ringbuffer``); only this reference is pickled and sent, so the video
    path never pays to move 6 MB of bytes it already has.
    """

    slot: int              # index into the ring
    ts_mono_ns: int        # capture timestamp, stamped at read() return
    seq: int               # monotonically increasing frame number
    shape: tuple[int, ...]  # (H, W, C)
    dtype: str


@dataclass(frozen=True, slots=True)
class Track:
    """One tracked object identity for one frame.

    ``track_id`` is stable across frames — that stability is manufactured by the
    tracker's Kalman filter, and it is what lets the reasoning layer assume
    object permanence instead of hallucinating churn from per-frame flicker.
    """

    track_id: int
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]   # xyxy, normalized to [0, 1]
    velocity: tuple[float, float]             # px/s, from the tracker's Kalman state
    age_frames: int


@dataclass(frozen=True, slots=True)
class VisionEvent:
    """An instant. What the camera saw at ``ts_mono_ns``."""

    ts_mono_ns: int
    seq: int
    tracks: list[Track]
    frame_emb: np.ndarray | None = None   # CLIP/SigLIP, L2-normalized, fp16


@dataclass(frozen=True, slots=True)
class AudioEvent:
    """An interval. Spans ``[ts_start_ns, ts_end_ns]``.

    Carries three orthogonal things (ARCHITECTURE.md §5):
      - ``transcript`` — the words said (ASR). May be None if no speech.
      - ``tags``       — non-speech acoustic events (AudioSet-style).
      - ``caption``    — a natural-language description of what is happening
                         acoustically. This is *not* the transcript; "a vehicle
                         skids as a person shouts" vs. "look out".
    """

    ts_start_ns: int
    ts_end_ns: int
    transcript: str | None = None
    is_final: bool = False                    # streaming ASR: partial vs committed
    tags: list[tuple[str, float]] = field(default_factory=list)
    caption: str | None = None
    audio_emb: np.ndarray | None = None       # CLAP, L2-normalized


@dataclass(frozen=True, slots=True)
class Suggestion:
    """The reasoning layer's output. Must cite evidence or abstain."""

    ts_mono_ns: int
    text: str
    confidence: float
    evidence: list[str]              # event/track ids the reasoner cited
    abstained: bool = False
