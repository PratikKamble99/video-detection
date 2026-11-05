"""Observability primitives (ARCHITECTURE.md §11).

Instrument from day one — retrofitting timing into a multiprocess pipeline is
miserable. This wraps ``prometheus_client`` so every process registers the same
metric names, and exposes a tiny HTTP endpoint to scrape.

Metrics that matter:
  - per-stage latency histograms (capture, preprocess, infer, nms, track, fuse, reason)
  - queue-depth gauges (sustained depth > 1 = that stage is the bottleneck)
  - seq-gap counter (how many frames dropped, and where)
  - salience-gate hit rate + VLM tokens/minute (the cost dashboards)

If ``prometheus_client`` is not installed the module degrades to no-op stubs so
that importing it never breaks a process that just wants to run.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _HAVE_PROM = True
except Exception:  # pragma: no cover - exercised only without the dep
    _HAVE_PROM = False

    class _Stub:
        def __init__(self, *a, **k): ...
        def labels(self, *a, **k): return self
        def observe(self, *a, **k): ...
        def inc(self, *a, **k): ...
        def set(self, *a, **k): ...

    Counter = Gauge = Histogram = _Stub  # type: ignore

    def start_http_server(*a, **k):  # type: ignore
        return None


# Buckets tuned for a 33 ms frame budget: fine resolution below budget, a couple
# of coarse buckets above to catch the tail (p99 is what users feel).
_LATENCY_BUCKETS = (
    0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.020, 0.025,
    0.033, 0.050, 0.075, 0.100, 0.200, 0.500, 1.0,
)

STAGE_LATENCY = Histogram(
    "rtp_stage_latency_seconds",
    "Per-stage processing latency",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
)

QUEUE_DEPTH = Gauge(
    "rtp_queue_depth",
    "Instantaneous queue depth per link",
    ["link"],
)

SEQ_GAP = Counter(
    "rtp_seq_gap_total",
    "Frames dropped, detected via seq discontinuity",
    ["stage"],
)

SALIENCE_TRIGGERS = Counter(
    "rtp_salience_triggers_total",
    "Salience-gate decisions",
    ["outcome"],  # "wake" | "skip"
)

VLM_TOKENS = Counter(
    "rtp_vlm_tokens_total",
    "Tokens consumed by the reasoning layer",
    ["direction"],  # "in" | "out"
)


@contextmanager
def timed(stage: str):
    """Context manager that records elapsed wall time into ``STAGE_LATENCY``.

    Uses ``perf_counter`` for the measurement itself (highest-resolution timer);
    the monotonic clock in ``events`` is for *ordering* events, this is for
    *measuring durations*.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        STAGE_LATENCY.labels(stage=stage).observe(time.perf_counter() - t0)


def serve(port: int = 9108) -> None:
    """Start the Prometheus scrape endpoint. Safe no-op without the dep."""
    start_http_server(port)
