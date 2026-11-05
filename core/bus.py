"""The event bus: ZeroMQ PUB/SUB carrying the frozen events from ``core.events``.

Everything published is timestamped against the single monotonic clock
(``core.events.now_ns``). Wall-clock is banned on the bus for the same reason it
is banned everywhere else — an NTP step makes fusion windows jump backwards.

Transport is ZeroMQ so that producers and consumers are fully decoupled: a slow
subscriber never back-pressures a publisher (PUB drops for a subscriber whose
high-water mark is hit), which is exactly the "latest frame wins" semantics the
video path needs. Bounded high-water marks everywhere — no unbounded queues, per
ARCHITECTURE.md §9.

Topics are the event class names (``VisionEvent``, ``AudioEvent`` ...). A
subscriber picks the topics it cares about; SUB filtering happens in the ZeroMQ
library, not in Python.
"""

from __future__ import annotations

import pickle
from typing import Iterator

import zmq

# Default inproc/tcp endpoint for the bus. Override per-deployment via config.
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5599"

# High-water mark: max messages queued per socket before ZeroMQ drops. Bounded
# by design — a stalled consumer loses messages rather than growing memory.
DEFAULT_HWM = 1000


def topic_of(event) -> bytes:
    """The pub/sub topic for an event is its class name."""
    return type(event).__name__.encode("ascii")


class Publisher:
    """Binds a PUB socket. One per producer process.

    Note the classic ZeroMQ "slow joiner" caveat: a PUB socket drops messages
    sent before a SUB has finished connecting and subscribing. For request/reply
    correctness you would not use PUB/SUB; for a live sensor stream, dropping the
    first few frames while a subscriber attaches is exactly right.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, *, hwm: int = DEFAULT_HWM,
                 bind: bool = True, ctx: zmq.Context | None = None):
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.set_hwm(hwm)
        if bind:
            self.sock.bind(endpoint)
        else:
            self.sock.connect(endpoint)
        self.endpoint = endpoint

    def publish(self, event) -> None:
        """Send one event. Multipart: [topic, pickled payload]."""
        self.sock.send_multipart(
            [topic_of(event), pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL)]
        )

    def close(self) -> None:
        self.sock.close(linger=0)

    def __enter__(self) -> "Publisher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Subscriber:
    """Connects a SUB socket and yields deserialized events.

    Pass ``topics`` (event class names) to filter; empty tuple subscribes to
    everything.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, topics: tuple[str, ...] = (),
                 *, hwm: int = DEFAULT_HWM, connect: bool = True,
                 ctx: zmq.Context | None = None):
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.set_hwm(hwm)
        if connect:
            self.sock.connect(endpoint)
        else:
            self.sock.bind(endpoint)
        if topics:
            for t in topics:
                self.sock.setsockopt(zmq.SUBSCRIBE, t.encode("ascii"))
        else:
            self.sock.setsockopt(zmq.SUBSCRIBE, b"")  # everything
        self.endpoint = endpoint

    def recv(self, timeout_ms: int | None = None):
        """Receive one event, or None on timeout. Blocks if timeout is None."""
        if timeout_ms is not None:
            if not self.sock.poll(timeout_ms):
                return None
        _topic, payload = self.sock.recv_multipart()
        return pickle.loads(payload)

    def stream(self) -> Iterator:
        """Yield events forever."""
        while True:
            yield self.recv()

    def close(self) -> None:
        self.sock.close(linger=0)

    def __enter__(self) -> "Subscriber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
