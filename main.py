"""Process supervisor (ARCHITECTURE.md §12, plan.md §Immediate Next Actions).

THE FIRST LINE THAT MATTERS: ``mp.set_start_method("spawn")`` before anything
imports torch. CUDA + ``fork`` silently corrupts the CUDA context — this is
non-negotiable and must run before any worker or CUDA import (plan.md Phase 0).

Right now this is a Phase 0 scaffold: it sets the start method, starts the
metrics endpoint, and describes the process topology it will supervise. As each
phase lands, its worker (P0..P6) is registered here and the supervisor gains the
Phase 7 responsibilities: restart a dead worker without tearing down the ring
buffer, and drive the graceful-degradation ladder.
"""

from __future__ import annotations

import multiprocessing as mp


def _set_spawn() -> None:
    """Force spawn. Idempotent; safe to call from tests."""
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Already set (e.g. re-import in a test). Verify it is actually spawn.
        if mp.get_start_method() != "spawn":
            raise RuntimeError(
                f"start method is {mp.get_start_method()!r}, need 'spawn' "
                "(fork corrupts the CUDA context)"
            )


# The worker topology this supervisor will own (ARCHITECTURE.md §3). Filled in
# phase by phase; kept here as the single map of what runs where.
TOPOLOGY = {
    "P0": "video.capture   - capture + decode, stamp at read() return",
    "P1": "video.detect    - preprocess + detect + track + embed (GPU stream A)",
    "P2": "audio.capture   - 16 kHz capture, VAD gate, lock-free ring",
    "P3": "audio.asr/tag/caption - ASR + tagging + captioning (GPU stream B)",
    "P4": "fusion.state    - temporal align, scene graph, episodic memory",
    "P5": "reasoning.agent - salience-gated ReAct loop (async)",
    "P6": "sinks           - overlay / websocket / recorder",
}


def main() -> None:
    _set_spawn()

    from core import config, metrics

    cfg = config.load()
    metrics.serve(cfg.observability.prometheus_port)

    print("rt-perception supervisor (Phase 0 scaffold)")
    print(f"metrics: http://127.0.0.1:{cfg.observability.prometheus_port}/")
    print("bus:    ", cfg.bus.endpoint)
    print("topology:")
    for pid, desc in TOPOLOGY.items():
        print(f"  {pid}: {desc}")
    print("\nWorkers are not yet wired — build them phase by phase per plan.md.")


if __name__ == "__main__":
    main()
