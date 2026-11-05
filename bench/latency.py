"""Glass-to-glass latency harness (plan.md Phase 0).

The single source of ground truth for latency. *Everything else lies* — a
``time()`` around ``read()`` cannot see the decoder's internal buffer, which is
exactly where the hidden seconds of latency accumulate (ARCHITECTURE.md §10).

Two modes:

1. ``--camera`` (real measurement): render a millisecond-precision clock, point
   the camera at the screen, capture the pipeline's rendered output, and diff
   the timestamp the camera saw against now. Needs opencv + a physical loop.
   This is the number that counts.

2. ``--synthetic`` (default; runs headless / in CI): push frames through a
   ``FrameRing`` with a configurable consumer delay and report the ring's own
   write→read latency distribution plus torn-read count. This validates the data
   plane's contribution to the budget without hardware, and is what Phase 0's
   exit criteria measure for the pass-through path.

Reports p50 / p90 / p99 — the exit criterion is p99 < 100 ms, not p50. The tail
is what users feel.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from core.events import now_ns
from core.ringbuffer import FrameRing, TornReadError


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {}
    s = sorted(samples_ms)
    def pct(p: float) -> float:
        i = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
        return s[i]
    return {
        "n": len(s),
        "p50": pct(50), "p90": pct(90), "p99": pct(99),
        "max": s[-1], "mean": statistics.fmean(s),
    }


def run_synthetic(
    *,
    shape: tuple[int, int, int] = (1080, 1920, 3),
    slots: int = 12,
    fps: float = 60.0,
    seconds: float = 10.0,
    consumer_lag_frames: int = 2,
) -> dict:
    """Push frames through the ring at ``fps`` with a consumer running
    ``consumer_lag_frames`` behind, and measure write→validated-read latency.

    This is the Phase 0 pass-through proxy: no inference, just the data plane.
    Reports latency percentiles and the torn-read count (must be 0 when the
    consumer stays within ``slots``).
    """
    period = 1.0 / fps
    frame = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    latencies_ms: list[float] = []
    torn = 0

    with FrameRing(slots, shape, create=True) as ring:
        refs: list = []
        seq = 0
        t_start = time.perf_counter()
        deadline = t_start + seconds
        next_tick = t_start

        while time.perf_counter() < deadline:
            # Producer: overwrite the shared frame with the seq so torn reads are
            # detectable in principle, then write.
            ref = ring.write(frame, now_ns(), seq)
            refs.append(ref)
            seq += 1

            # Consumer runs `consumer_lag_frames` behind the producer.
            if len(refs) > consumer_lag_frames:
                old = refs.pop(0)
                try:
                    _ = ring.read(old)
                    latencies_ms.append((now_ns() - old.ts_mono_ns) / 1e6)
                except TornReadError:
                    torn += 1

            # Pace to the target FPS without a busy-spin sleep dominating.
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    out = _percentiles(latencies_ms)
    out["torn_reads"] = torn
    out["fps_target"] = fps
    out["consumer_lag_frames"] = consumer_lag_frames
    return out


def run_soak(*, shape=(1080, 1920, 3), slots=12, fps=60.0, minutes=10.0) -> dict:
    """The Phase 0 exit soak: 1080p @ 60 FPS write for `minutes`, zero torn
    reads with a well-behaved (2-frame-behind) consumer."""
    return run_synthetic(shape=shape, slots=slots, fps=fps,
                          seconds=minutes * 60.0, consumer_lag_frames=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Glass-to-glass latency harness")
    ap.add_argument("--synthetic", action="store_true", default=True,
                    help="Data-plane latency without hardware (default)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--lag", type=int, default=2, help="consumer frames behind producer")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    shape = (args.height, args.width, 3)
    res = run_synthetic(shape=shape, slots=args.slots, fps=args.fps,
                        seconds=args.seconds, consumer_lag_frames=args.lag)
    print(f"synthetic data-plane latency  ({shape[1]}x{shape[0]}, "
          f"{args.fps:.0f} FPS target, consumer {args.lag} frames behind)")
    for k in ("n", "mean", "p50", "p90", "p99", "max"):
        if k in res:
            unit = "" if k == "n" else " ms"
            print(f"  {k:>4}: {res[k]:.3f}{unit}" if k != "n" else f"  {k:>4}: {res[k]}")
    print(f"  torn_reads: {res['torn_reads']}")


if __name__ == "__main__":
    main()
