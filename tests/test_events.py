"""The event contract is frozen and picklable (it crosses process boundaries)."""

import pickle
import time

import numpy as np
import pytest

from core.events import (
    AudioEvent,
    FrameRef,
    Suggestion,
    Track,
    VisionEvent,
    now_ns,
)


def test_now_ns_is_monotonic():
    a = now_ns()
    b = now_ns()
    assert isinstance(a, int)
    assert b >= a


def test_events_are_frozen():
    t = Track(1, "person", 0.9, (0, 0, 1, 1), (0.0, 0.0), 3)
    with pytest.raises((AttributeError, TypeError)):
        t.conf = 0.5  # type: ignore[misc]


def test_vision_event_roundtrips_through_pickle():
    ev = VisionEvent(
        ts_mono_ns=now_ns(),
        seq=42,
        tracks=[Track(1, "cat", 0.8, (0.1, 0.1, 0.2, 0.2), (1.0, -1.0), 5)],
        frame_emb=np.ones(8, dtype=np.float16),
    )
    back = pickle.loads(pickle.dumps(ev))
    assert back.seq == 42
    assert back.tracks[0].cls == "cat"
    assert np.array_equal(back.frame_emb, ev.frame_emb)


def test_audio_event_is_an_interval():
    t0 = now_ns()
    time.sleep(0.001)
    t1 = now_ns()
    ev = AudioEvent(ts_start_ns=t0, ts_end_ns=t1, transcript="look out",
                    tags=[("car_skid", 0.9)], caption="a vehicle skids")
    assert ev.ts_end_ns > ev.ts_start_ns          # spans time, unlike VisionEvent
    assert ev.transcript != ev.caption            # ASR != captioning
    assert pickle.loads(pickle.dumps(ev)) == ev


def test_suggestion_carries_evidence():
    s = Suggestion(now_ns(), "check the stove", 0.7, evidence=["track#3", "audio#12"])
    assert s.evidence and not s.abstained


def test_frame_ref_is_small_and_frozen():
    ref = FrameRef(slot=1, ts_mono_ns=now_ns(), seq=0, shape=(1080, 1920, 3),
                   dtype="uint8")
    assert pickle.loads(pickle.dumps(ref)) == ref
