"""The salience gate scores change and decides when to wake the reasoner."""

import numpy as np

from core.salience import SalienceWeights, WorldState, salience, should_reason


def _emb(*vals) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_identical_world_scores_zero():
    w = SalienceWeights()
    e = _emb(1.0, 0.0, 0.0)
    a = WorldState(0, {1, 2}, {"person"}, e, [], [])
    b = WorldState(1, {1, 2}, {"person"}, e, [], [])
    assert salience(a, b, w) == 0.0


def test_new_track_and_class_raise_score():
    w = SalienceWeights()
    e = _emb(1.0, 0.0)
    prev = WorldState(0, {1}, {"person"}, e, [], [])
    cur = WorldState(1, {1, 2}, {"person", "knife"}, e, [], [])
    s = salience(prev, cur, w)
    assert s == w.new_track * 1 + w.new_class * 1


def test_visual_drift_contributes():
    w = SalienceWeights()
    prev = WorldState(0, set(), set(), _emb(1.0, 0.0), [], [])
    cur = WorldState(1, set(), set(), _emb(0.0, 1.0), [], [])  # orthogonal -> cos 0
    s = salience(prev, cur, w)
    assert abs(s - w.visual * 1.0) < 1e-5


def test_new_audio_tag_above_threshold_counts():
    w = SalienceWeights()
    prev = WorldState(0, set(), set(), None, [], [])
    cur = WorldState(1, set(), set(), None, [("glass_breaking", 0.9)], [])
    s = salience(prev, cur, w)
    assert abs(s - w.audio * 0.9) < 1e-6


def test_quiet_audio_tag_ignored():
    w = SalienceWeights(tag_thresh=0.3)
    prev = WorldState(0, set(), set(), None, [], [])
    cur = WorldState(1, set(), set(), None, [("hum", 0.2)], [])  # below thresh
    assert salience(prev, cur, w) == 0.0


def test_should_reason_threshold_and_heartbeat():
    w = SalienceWeights(threshold=1.2, heartbeat_ns=1000)
    # below threshold, within heartbeat -> skip
    assert not should_reason(0.5, w, last_call_ns=0, now_ns=500)
    # above threshold -> wake
    assert should_reason(2.0, w, last_call_ns=0, now_ns=500)
    # below threshold but heartbeat elapsed -> wake (never go fully silent)
    assert should_reason(0.0, w, last_call_ns=0, now_ns=5000)
