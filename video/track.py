"""P1 — Multi-object tracking (ARCHITECTURE.md §5).

Why this is load-bearing, restated from the architecture: per-frame detection
flickers. An object at conf 0.48 blinks in and out across consecutive frames. The
tracker is a *temporal prior* that turns a noisy per-frame classifier into a
stable object identity — every "object permanence" property the reasoning layer
assumes is manufactured here.

This is a dependency-free IoU tracker with coasting: it associates detections to
tracks by box overlap, keeps a track alive for a few frames when it goes
unmatched (occlusion / a dropped detection), and estimates velocity from center
motion. It operates on raw ``Detection`` objects — not on any model's internal
state — precisely so detection and tracking can live in separate processes with
only the detections crossing the boundary.

It is deliberately simple and honest about it: ByteTrack / BoT-SORT (association
on low-confidence boxes, ReID, camera-motion compensation) are the upgrade and
plug in behind this same ``update() -> list[Track]`` interface. Ultralytics'
built-in ByteTrack is available via ``video.detect`` for the single-process demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.events import Track
from video.detect import Detection


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter)


def _center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


@dataclass(slots=True)
class _TrackState:
    track_id: int
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]
    velocity: tuple[float, float]           # normalized units / second
    age_frames: int = 0
    misses: int = 0                         # consecutive frames unmatched
    last_ts_ns: int = 0
    center: tuple[float, float] = field(default=(0.0, 0.0))


class IouTracker:
    """Greedy IoU tracker with coasting.

    Parameters
    ----------
    iou_thresh : minimum overlap to associate a detection to a track.
    max_age : frames a track survives unmatched before eviction (object
        permanence across brief occlusion / dropped detections).
    new_track_conf : a detection must clear this to *start* a new track (higher
        than the detector's floor, so we don't spawn tracks on marginal noise).
    match_class : if True, only associate detections of the same class.
    """

    def __init__(self, *, iou_thresh: float = 0.3, max_age: int = 30,
                 new_track_conf: float = 0.5, match_class: bool = True):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.new_track_conf = new_track_conf
        self.match_class = match_class
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 1

    def update(self, detections: list[Detection], ts_mono_ns: int) -> list[Track]:
        """Associate ``detections`` to existing tracks and return the current
        set of visible tracks (matched this frame, or coasting within grace)."""
        track_ids = list(self._tracks.keys())

        # Score all (track, detection) pairs, then greedily take the best.
        pairs: list[tuple[float, int, int]] = []
        for ti, tid in enumerate(track_ids):
            tb = self._tracks[tid].bbox
            tcls = self._tracks[tid].cls
            for di, det in enumerate(detections):
                if self.match_class and det.cls != tcls:
                    continue
                iou = _iou(tb, det.bbox)
                if iou >= self.iou_thresh:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for iou, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            self._update_track(self._tracks[track_ids[ti]], detections[di], ts_mono_ns)

        # Unmatched existing tracks: coast (predict forward, count a miss).
        for ti, tid in enumerate(track_ids):
            if ti not in matched_t:
                self._coast(self._tracks[tid], ts_mono_ns)

        # Unmatched detections above the spawn floor: new identities.
        for di, det in enumerate(detections):
            if di not in matched_d and det.conf >= self.new_track_conf:
                self._spawn(det, ts_mono_ns)

        # Evict stale tracks.
        for tid in [t for t, s in self._tracks.items() if s.misses > self.max_age]:
            del self._tracks[tid]

        # Emit everything currently alive, freshest first.
        return [
            Track(
                track_id=s.track_id, cls=s.cls, conf=s.conf, bbox=s.bbox,
                velocity=s.velocity, age_frames=s.age_frames,
            )
            for s in sorted(self._tracks.values(), key=lambda s: s.track_id)
        ]

    # -- internals --------------------------------------------------------

    def _update_track(self, s: _TrackState, det: Detection, ts: int) -> None:
        cx, cy = _center(det.bbox)
        dt = (ts - s.last_ts_ns) / 1e9 if s.last_ts_ns else 0.0
        if dt > 0:
            s.velocity = ((cx - s.center[0]) / dt, (cy - s.center[1]) / dt)
        s.bbox = det.bbox
        s.conf = det.conf
        s.cls = det.cls
        s.center = (cx, cy)
        s.last_ts_ns = ts
        s.age_frames += 1
        s.misses = 0

    def _coast(self, s: _TrackState, ts: int) -> None:
        """Predict the box forward by its velocity so a briefly-occluded object
        keeps a plausible position instead of freezing."""
        dt = (ts - s.last_ts_ns) / 1e9 if s.last_ts_ns else 0.0
        vx, vy = s.velocity
        x1, y1, x2, y2 = s.bbox
        s.bbox = (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)
        s.center = _center(s.bbox)
        s.last_ts_ns = ts
        s.age_frames += 1
        s.misses += 1

    def _spawn(self, det: Detection, ts: int) -> None:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = _TrackState(
            track_id=tid, cls=det.cls, conf=det.conf, bbox=det.bbox,
            velocity=(0.0, 0.0), age_frames=1, misses=0,
            last_ts_ns=ts, center=_center(det.bbox),
        )
