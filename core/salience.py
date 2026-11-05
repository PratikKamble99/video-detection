"""The salience gate (ARCHITECTURE.md §6) — the highest-leverage component.

A VLM call is ~10^4x more expensive than a detection, and consecutive frames are
near-identical, so reasoning every frame pays 10^4x for ~0 bits of new
information. The gate wakes the reasoner only when the world has actually
changed. Empirically this cuts VLM invocations ~90-95% at essentially no loss in
event recall, because the events you care about *are* the moments of high
salience.

This lands in Phase 4, before the LLM is wired (plan.md), so the cost structure
exists before there is a bill to pay. The scoring here is intentionally cheap
(< 0.5 ms/frame target) and dependency-free — it operates on a small
``WorldState`` snapshot, not on models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class SalienceWeights:
    """Per-term weights. Tune ``threshold`` on a labeled clip by sweeping it
    against (VLM calls saved) vs (annotated events missed) — see
    ``bench/gate_sweep.py``. This knob decides whether the system costs $5/day
    or $500/day."""

    new_track: float = 0.5
    new_class: float = 1.0
    visual: float = 2.0
    audio: float = 1.5
    kinematic: float = 0.8

    threshold: float = 1.2
    tag_thresh: float = 0.3
    heartbeat_ns: int = 20 * 1_000_000_000   # 20 s: never go fully silent


@dataclass(slots=True)
class WorldState:
    """The minimal snapshot the gate compares across time.

    Fusion (P4) produces these; the gate only reads them. Kept deliberately
    small so ``salience()`` stays sub-millisecond.
    """

    ts_mono_ns: int
    track_ids: set[int] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)
    frame_emb: np.ndarray | None = None                 # L2-normalized
    audio_tags: list[tuple[str, float]] = field(default_factory=list)
    accel_mags: list[float] = field(default_factory=list)  # per-track |accel|

    @property
    def audio_tag_set(self) -> set[str]:
        return {t for t, _ in self.audio_tags}


def salience(prev: WorldState, cur: WorldState, w: SalienceWeights) -> float:
    """Score how much the world changed from ``prev`` to ``cur``.

    Each term is a distinct kind of "something happened":
    new identities, a new class of object, visual drift, a new salient sound, or
    hard acceleration (fall / collision / sudden motion).
    """
    s = 0.0

    # 1. New object identities — the tracker hands us this for free.
    s += w.new_track * len(cur.track_ids - prev.track_ids)

    # 2. Class-set change — e.g. a *knife* entered a *kitchen* scene.
    s += w.new_class * len(cur.classes - prev.classes)

    # 3. Visual drift — cosine distance on the frame embedding.
    if cur.frame_emb is not None and prev.frame_emb is not None:
        cos = float(np.dot(cur.frame_emb, prev.frame_emb))
        s += w.visual * (1.0 - cos)

    # 4. Salient audio — any tag above threshold that wasn't there before.
    prev_tags = prev.audio_tag_set
    s += w.audio * sum(
        c for t, c in cur.audio_tags if c > w.tag_thresh and t not in prev_tags
    )

    # 5. Kinematic — the single hardest-accelerating track.
    s += w.kinematic * max(cur.accel_mags, default=0.0)

    return s


def should_reason(score: float, w: SalienceWeights, last_call_ns: int, now_ns: int) -> bool:
    """Wake the reasoner if the world changed enough, or if the heartbeat
    elapsed. The heartbeat guarantees the system still speaks during long static
    stretches instead of going silent."""
    heartbeat = (now_ns - last_call_ns) > w.heartbeat_ns
    return score > w.threshold or heartbeat
