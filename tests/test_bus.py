"""The event bus publishes and filters typed events over ZeroMQ.

Uses an inproc endpoint with a shared context so the test needs no TCP port and
no second process. The PUB/SUB slow-joiner caveat means we retry the first sends
until the subscription propagates.
"""

import time

import zmq

from core.bus import Publisher, Subscriber
from core.events import AudioEvent, VisionEvent, now_ns


def _make_pair(topics=()):
    ctx = zmq.Context.instance()
    endpoint = f"inproc://bus-test-{time.monotonic_ns()}"
    pub = Publisher(endpoint, bind=True, ctx=ctx)
    sub = Subscriber(endpoint, topics=topics, connect=True, ctx=ctx)
    return pub, sub


def test_publish_and_receive_roundtrip():
    pub, sub = _make_pair()
    ev = VisionEvent(ts_mono_ns=now_ns(), seq=1, tracks=[])
    got = None
    for _ in range(50):                 # ride out the slow-joiner window
        pub.publish(ev)
        got = sub.recv(timeout_ms=20)
        if got is not None:
            break
    assert got is not None and got.seq == 1
    pub.close(); sub.close()


def test_topic_filtering():
    pub, sub = _make_pair(topics=("AudioEvent",))
    # Warm up the subscription.
    for _ in range(50):
        pub.publish(AudioEvent(now_ns(), now_ns()))
        if sub.recv(timeout_ms=20) is not None:
            break
    # A VisionEvent must be filtered out; an AudioEvent must arrive.
    pub.publish(VisionEvent(now_ns(), 9, []))
    pub.publish(AudioEvent(now_ns(), now_ns(), transcript="hi"))
    got = sub.recv(timeout_ms=200)
    assert isinstance(got, AudioEvent)
    assert got.transcript == "hi"
    pub.close(); sub.close()
