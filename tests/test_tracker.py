"""IoU tracker: stable ids, coasting through gaps, velocity, class gating.

These run without any model — the tracker operates on plain ``Detection``s, which
is exactly why detection and tracking can live in separate processes.
"""

from core.events import now_ns
from video.detect import Detection
from video.track import IouTracker, _iou


def _det(cls, conf, box):
    return Detection(cls=cls, conf=conf, bbox=box)


def test_iou_math():
    assert _iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert _iou((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1)) == 0.0
    assert abs(_iou((0, 0, 2, 2), (1, 1, 3, 3)) - (1 / 7)) < 1e-6


def test_stable_id_across_frames():
    tr = IouTracker(new_track_conf=0.5)
    b0 = (0.10, 0.10, 0.20, 0.20)
    b1 = (0.12, 0.12, 0.22, 0.22)   # small motion -> high overlap
    t0 = now_ns()
    a = tr.update([_det("person", 0.9, b0)], t0)
    b = tr.update([_det("person", 0.9, b1)], t0 + 33_000_000)
    assert len(a) == len(b) == 1
    assert a[0].track_id == b[0].track_id      # identity persists
    assert b[0].age_frames == 2


def test_two_objects_get_distinct_ids():
    tr = IouTracker(new_track_conf=0.5)
    dets = [_det("person", 0.9, (0.0, 0.0, 0.1, 0.1)),
            _det("car", 0.9, (0.8, 0.8, 0.95, 0.95))]
    out = tr.update(dets, now_ns())
    assert len({t.track_id for t in out}) == 2


def test_coasts_through_a_missed_frame_then_evicts():
    tr = IouTracker(new_track_conf=0.5, max_age=2)
    b = (0.4, 0.4, 0.5, 0.5)
    t = now_ns()
    tr.update([_det("cup", 0.9, b)], t)
    # No detection for several frames: it should coast, then be evicted.
    alive_counts = []
    for i in range(1, 6):
        out = tr.update([], t + i * 33_000_000)
        alive_counts.append(len(out))
    assert alive_counts[0] == 1                 # still coasting right after the gap
    assert alive_counts[-1] == 0                 # eventually evicted past max_age


def test_low_conf_detection_does_not_spawn_track():
    tr = IouTracker(new_track_conf=0.5)
    out = tr.update([_det("dog", 0.4, (0.1, 0.1, 0.2, 0.2))], now_ns())
    assert out == []                             # below spawn floor


def test_velocity_sign_follows_motion():
    tr = IouTracker(new_track_conf=0.5)
    t = now_ns()
    tr.update([_det("ball", 0.9, (0.10, 0.10, 0.20, 0.20))], t)
    out = tr.update([_det("ball", 0.9, (0.30, 0.10, 0.40, 0.20))], t + 100_000_000)
    assert out[0].velocity[0] > 0               # moved right -> +x velocity
    assert abs(out[0].velocity[1]) < 1e-6        # no vertical motion


def test_class_gating_prevents_cross_class_association():
    tr = IouTracker(new_track_conf=0.5, match_class=True)
    box = (0.2, 0.2, 0.3, 0.3)
    t = now_ns()
    a = tr.update([_det("person", 0.9, box)], t)
    # Same location, different class -> must be a NEW track, not a reassignment.
    b = tr.update([_det("cat", 0.9, box)], t + 33_000_000)
    assert a[0].track_id != b[0].track_id
