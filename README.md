# rt-perception

Real-time video object detection + audio captioning + AI reasoning.

Design docs: [`architecture.md`](architecture.md) and [`plan.md`](plan.md). The
architecture's central idea is the **three-clock problem** — video (~30 Hz),
audio (~1 Hz), and reasoning (event-driven) run on physically incompatible
cadences, so the system is not a pipeline but a set of independent producers
writing into a shared, timestamped world-state that consumers sample at their
own rate.

## Status

**Phase 0 — Foundations & Measurement: complete.** The data plane and the
measurement harness exist and are tested; no ML models are wired yet.

| Phase | Deliverable | State |
| ----- | ----------- | ----- |
| 0 | Ring buffer, event contract, bus, salience gate, latency harness | ✅ done |
| 1 | Video detection + tracking + overlay | ⬜ next |
| 2 | ASR + tags + audio captions | ⬜ |
| 3 | Fusion, scene graph, memory | ⬜ |
| 4 | Salience gate sweep + anomaly signals | ⬜ (gate scorer landed early) |
| 5 | Reasoning agent | ⬜ |
| 6 | Optimization | ⬜ |
| 7 | Hardening | ⬜ |

### What Phase 0 delivered

- [`core/events.py`](core/events.py) — the frozen event contract (the wire seam).
- [`core/ringbuffer.py`](core/ringbuffer.py) — shared-memory frame ring with a
  seqlock. Consumers detect and discard torn reads instead of returning corrupt
  frames. Zero-copy: only a 40-byte `FrameRef` crosses the queue.
- [`core/bus.py`](core/bus.py) — ZeroMQ PUB/SUB, monotonic timestamps, bounded
  high-water marks (no unbounded queues).
- [`core/salience.py`](core/salience.py) — the gate that decides *when* to reason
  (landed early so the cost structure exists before the LLM does).
- [`core/metrics.py`](core/metrics.py) — Prometheus per-stage histograms, queue
  gauges, seq-gap + cost counters.
- [`bench/latency.py`](bench/latency.py) — glass-to-glass harness. Synthetic
  data-plane mode runs headless; camera mode is the real ground truth.

## Setup

```bash
uv sync --extra video --group dev     # core + opencv + test deps
uv run pytest                         # 22 tests, incl. the torn-read soak
uv run python main.py                 # supervisor scaffold + metrics endpoint
uv run python -m bench.latency        # data-plane latency percentiles
```

Heavier extras install per phase: `--extra gpu` (Phase 1), `--extra audio`
(Phase 2), `--extra fusion` (Phase 3), `--extra reasoning` (Phase 5). These pull
torch / faster-whisper / faiss / anthropic and are intentionally *not* in the
core install so Phase 0 stays lean and CI stays headless.

## Non-negotiables (from the design docs)

- `mp.set_start_method("spawn")` before any torch import — fork corrupts CUDA.
- `time.monotonic_ns()` everywhere — never wall-clock (NTP steps travel back in time).
- Bounded queues everywhere — an unbounded queue converts fast failure into slow OOM.
- The salience gate, tracking, the latency harness, and bounded queues are
  **never** cut. Each, removed, produces a system that demos for ten minutes and
  then fails expensively.
