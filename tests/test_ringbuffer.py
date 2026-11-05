"""Ring-buffer correctness — the torn-read test comes first (plan.md Phase 0).

The load-bearing property: when a consumer falls more than ``slots`` frames
behind, the producer overwrites a slot mid-read, and the consumer must *detect
and discard* that torn frame rather than return corrupted pixels.
"""

import threading
import time

import numpy as np
import pytest

from core.events import FrameRef, now_ns
from core.ringbuffer import FrameRing, TornReadError


def test_basic_write_read_roundtrip():
    shape = (8, 8, 3)
    with FrameRing(4, shape, create=True) as ring:
        frame = np.full(shape, 7, dtype=np.uint8)
        ref = ring.write(frame, now_ns(), 0)
        out = ring.read(ref)
        assert np.array_equal(out, frame)
        assert not ring.is_stale(ref)


def test_stale_ref_detected_before_read():
    """A consumer holding a ref to a slot that has since been overwritten gets a
    TornReadError on read (the 'fell behind before read' branch)."""
    shape = (8, 8, 3)
    with FrameRing(3, shape, create=True) as ring:
        f = np.zeros(shape, dtype=np.uint8)
        ref0 = ring.write(f, now_ns(), 0)   # slot 0
        ring.write(f, now_ns(), 1)          # slot 1
        ring.write(f, now_ns(), 2)          # slot 2
        assert not ring.is_stale(ref0)
        ring.write(f, now_ns(), 3)          # overwrites slot 0 with seq 3
        assert ring.is_stale(ref0)
        with pytest.raises(TornReadError):
            ring.read(ref0)


def test_rejects_too_few_slots():
    with pytest.raises(ValueError):
        FrameRing(2, (4, 4, 3), create=True)


def test_shape_mismatch_rejected():
    with FrameRing(3, (4, 4, 3), create=True) as ring:
        with pytest.raises(ValueError):
            ring.write(np.zeros((5, 5, 3), np.uint8), now_ns(), 0)


def test_concurrent_torn_reads_are_detected_never_returned():
    """Fast producer, deliberately-slow consumer, threads sharing the ring.

    Every frame's pixels are a single constant equal to ``seq & 0xFF``. So:
      - a partial tear (old bytes + new bytes) is non-uniform -> caught here,
      - a full-slot overwrite is caught by the seqlock meta re-check.
    Any read that returns without raising MUST be internally consistent. We also
    assert that torn reads actually happened, proving detection really fires and
    the test isn't vacuously green.
    """
    shape = (240, 320, 3)          # ~230 KB: big enough that memcpy can tear
    slots = 4
    ring = FrameRing(slots, shape, create=True)
    total = 4000
    stop = threading.Event()
    produced = {"seq": -1}

    def produce():
        buf = np.empty(shape, dtype=np.uint8)
        for seq in range(total):
            buf.fill(seq & 0xFF)
            ring.write(buf, now_ns(), seq)
            produced["seq"] = seq
        stop.set()

    torn = {"count": 0}
    good = {"count": 0}
    corruption = []

    def consume():
        while not stop.is_set() or produced["seq"] > _last_seen[0]:
            seq = produced["seq"]
            _last_seen[0] = seq
            if seq < 0:
                continue
            # Reconstruct a ref to a frame several behind -> forces lag past `slots`.
            target = max(0, seq - (slots + 2))
            ref = FrameRef(slot=target % slots, ts_mono_ns=0, seq=target,
                           shape=shape, dtype="uint8")
            try:
                out = ring.read(ref)
            except TornReadError:
                torn["count"] += 1
                continue
            good["count"] += 1
            expected = target & 0xFF
            if not (out.min() == out.max() == expected):
                corruption.append((target, int(out.min()), int(out.max()), expected))
            time.sleep(0)  # yield, let producer lap us

    _last_seen = [-1]

    tp = threading.Thread(target=produce)
    tc = threading.Thread(target=consume)
    tc.start(); tp.start()
    tp.join(); tc.join()
    ring.close(); ring.unlink()

    assert not corruption, f"undetected torn reads returned corrupt frames: {corruption[:5]}"
    # The consumer intentionally lags past `slots`, so detection must have fired.
    assert torn["count"] > 0, "consumer never fell behind — test did not exercise tearing"
