# Implementation Plan

**Companion to `ARCHITECTURE.md`.** Eight phases. Each has a hard, measurable exit criterion — if you can't measure it, you haven't finished the phase.

Estimated at ~8 weeks for one engineer working steadily. Compress by cutting Phase 6 and shipping local-VLM-only.

---

## Guiding Principles

1. **Measure before optimizing.** Phase 0 exists so that every later claim ("this made it faster") is checkable.
2. **Video path first, alone, end-to-end.** Do not add audio until video holds 30 FPS with stable latency. Debugging two async subsystems at once is more than twice as hard.
3. **Every phase ends with something runnable.** No phase produces only library code.
4. **The salience gate lands before the LLM.** Otherwise you build a cost structure you can't undo.

---

## Phase 0 — Foundations & Measurement

**Goal:** you can measure latency truthfully before you have anything to measure.

### Tasks

- Repo scaffold per `ARCHITECTURE.md §12`. `uv`; pin CUDA/cuDNN explicitly.
- Implement `core/events.py` — the frozen dataclasses. Freeze this contract early; it's the seam every process talks across.
- Implement `core/ringbuffer.py` with the seqlock pattern. **Write the torn-read test first**: spawn a deliberately slow consumer, confirm it detects and discards overwritten slots.
- Implement `core/bus.py` (ZeroMQ PUB/SUB, `time.monotonic_ns()` stamps).
- Build `bench/latency.py`: renders a millisecond clock, captures the pipeline output, diffs. This is your ground truth. **Everything else lies** — `time()` around `read()` will not show you the decoder's internal buffer.
- Stand up `prometheus_client` histograms + a Grafana panel, even with one stage in it.

### Exit criteria

- [ ] `FrameRing` sustains 1080p @ 60 FPS write with zero torn reads over a 10-minute soak.
- [ ] Glass-to-glass harness reports a number for a pass-through (capture → render, no inference) pipeline. Expect **30–60 ms**; if you see 300 ms you have found the OpenCV buffering bug on day one, which is the entire point of this phase.
- [ ] `pytest` green; CI runs it.

### Risks

- _CUDA + `fork` silently corrupts context._ Set `mp.set_start_method("spawn")` in `main.py` before anything imports torch. Non-negotiable.

---

## Phase 1 — Video Path

**Goal:** stable 30 FPS detection + tracking with an overlay, and you know exactly where every millisecond goes.

### Tasks

- `video/capture.py` (P0). `CAP_PROP_BUFFERSIZE=1`. Timestamp at `read()` return.
- `video/detect.py` (P1). Start with **eager PyTorch YOLOv11s** — correctness first. Get boxes on screen.
- Export to ONNX → TensorRT FP16. Wrap in an engine class with a stable `.infer(tensor) -> boxes` interface so you can swap backends without touching callers.
- Move preprocessing to GPU (fused, or folded into the exported graph).
- `video/track.py`: ByteTrack. Assign stable `track_id`s.
- `sinks/overlay.py`: boxes, ids, class, FPS, per-stage latency HUD.

### Exit criteria

- [ ] **30 FPS sustained for 10 minutes, glass-to-glass p99 < 100 ms.** Not p50 — p99. The tail is what users feel.
- [ ] Per-stage histograms populated; the sum reconciles with measured end-to-end within 15%.
- [ ] Track IDs persist across 5 s of partial occlusion in a hand-recorded test clip.
- [ ] Under artificial load (`stress-ng`), frames drop and latency stays flat. It must **not** accumulate lag.

### Risks

- _TensorRT export fails on some op._ Budget a day. Fallback: ONNX Runtime + CUDA EP (~20% slower, far fewer sharp edges).
- _Trap:_ benchmarking with a video file instead of a camera. Files decode faster than real time and hide every backpressure bug. Use a real camera, or a file gated to wall-clock rate.

---

## Phase 2 — Audio Path

**Goal:** transcript + audio tags + captions streaming on the bus, independent of video.

### Tasks

- `audio/capture.py` (P2): `sounddevice` callback → lock-free ring. **Assert no allocation in the callback** (test with `tracemalloc` around a 60 s run).
- Silero VAD gating. Verify speech/silence classification on a clip with known boundaries.
- `audio/asr.py`: `faster-whisper`, 1.0 s window / 0.3 s overlap. Emit partials (`is_final=False`), commit on 400 ms trailing silence.
- Handle the overlap-dedup problem: consecutive windows re-transcribe shared audio. Dedup by longest-common-suffix/prefix on the token stream, not on characters.
- `audio/tag.py`: BEATs or PANNs → AudioSet classes.
- `audio/caption.py`: audio captioning. **Confirm to yourself this is a different model from ASR** — see `ARCHITECTURE.md §5`.
- Run P3 on a dedicated CUDA stream.

### Exit criteria

- [ ] Transcript appears within **1.5 s** of speech ending; measured, not eyeballed.
- [ ] Zero dropped audio blocks over a 30-minute soak (counter at zero, not "looks fine").
- [ ] Audio tagger fires correctly on a labeled test set of 20 clips (glass, door, dog, alarm, speech...).
- [ ] Captioner produces a sensible sentence for a non-speech clip — the specific thing ASR _cannot_ do. If your captioner just outputs the transcript, you have wired the wrong model.
- [ ] **Video FPS unchanged from Phase 1** with audio running. If it dropped, P1 and P3 are contending; fix the stream separation now.

### Risks

- _GC pause in the audio callback._ Preallocate everything; `gc.freeze()` after warmup. Symptom is intermittent clicks + garbage transcripts, and it will look like a model problem when it is a memory problem.
- _Whisper hallucinates on silence_ (famously emits "Thank you." / subtitle credits into dead air). VAD gating is the fix, and it is why VAD comes before ASR rather than being an optimization.

---

## Phase 3 — Fusion & World State

**Goal:** one coherent, timestamped picture of the world that a model could read.

### Tasks

- `fusion/state.py` (P4): active track table with TTL eviction.
- **Temporal alignment.** Buffer vision events ≥ 2 s. Align `AudioEvent[t0,t1]` against the vision events _inside that interval_. Write the test that proves a 1 s-late audio event finds its correct video partners — this is where the design earns its keep.
- `fusion/scene_graph.py`: spatial predicates (`near`, `above`, `contains`, `approaching`, `occludes`) from bboxes + tracker velocity.
- Deterministic serializer: scene graph → compact text, sorted by `track_id`. Same world state must always produce byte-identical text (required for caching, and for your own sanity when diffing).
- `video/embed.py`: SigLIP on every 3rd frame. `audio/`: CLAP embedding per window.
- `fusion/memory.py`: FAISS index, snapshot on a timer for now.

### Exit criteria

- [ ] Replay a recorded A/V file; the aligned `(vision, audio)` pairs match hand-annotated ground truth on 20 events.
- [ ] Scene graph text for a 10-object scene is **< 300 tokens** (vs. ~1500 for a raw box dump). Measure it.
- [ ] Serializer is deterministic across 1000 runs.
- [ ] Fusion adds **< 3 ms** to the frame budget.

### Risks

- _Alignment off-by-one-window._ Highest-probability bug in the whole project. Build the replay harness with synthetic events at known timestamps before you trust anything.

---

## Phase 4 — Salience Gate

**Goal:** the system decides _when_ it's worth thinking. Do this **before** wiring the LLM.

### Tasks

- `core/salience.py` per `ARCHITECTURE.md §6`.
- Streaming anomaly score: Welford running mean/covariance over frame embeddings → Mahalanobis z-score.
- Cross-modal disagreement signal: rolling `cos(frame_emb, audio_emb)`; sustained drop = off-screen event.
- `bench/gate_sweep.py`: on a 30-minute labeled clip, sweep `threshold` and plot **(VLM calls saved) vs (annotated events missed)**.
- Heartbeat timer so the system never goes fully silent.

### Exit criteria

- [ ] On the labeled clip: **≥ 90% reduction in trigger count at ≥ 95% event recall.** Both numbers, from the sweep, on a chart.
- [ ] Chosen threshold is justified by that chart, not by vibes.
- [ ] Gate costs < 0.5 ms/frame.

### Why this ordering matters

If you build the reasoner first, you will demo it firing on every frame, it will look magical, and you will discover the cost structure only when the bill arrives or the rate limiter engages. The gate is not an optimization bolted on later — it determines whether the system is deployable at all. Build it while it's still cheap to build.

---

## Phase 5 — Reasoning Agent

**Goal:** grounded analysis and suggestions that cite evidence and know when to shut up.

### Tasks

- `reasoning/agent.py` (P5): async ReAct loop, `max_concurrent=2`, hard token budget per minute.
- `reasoning/tools.py`:
    - `query_memory(text, k)` — RAG over the episodic index
    - `zoom_crop(track_id)` — re-run the VLM on a full-res crop pulled from the ring buffer (recovers detail lost at 640²; high value, low effort)
    - `replay_audio(t0, t1)` — second-pass ASR with a bigger model
    - `get_track_history(id)`
    - `defer()` — decline to answer
- Prompt: scene-graph text + recent transcript + audio tags + top-k retrieved memories. Demand structured output with an `evidence` field naming the `track_id`s / event ids used.
- `reasoning/calibration.py`: temperature scaling fit on a held-out set. One scalar. Enforce `abstain_below_confidence`.
- Snapshot to episodic memory on every salience spike.
- Wire `sinks/websocket.py` so suggestions land in a browser UI.

### Exit criteria

- [ ] Suggestions arrive **< 3 s** after the triggering event.
- [ ] **Video FPS still 30.** Run the full Phase 1 soak with the agent hammering. Zero coupling.
- [ ] Every suggestion carries non-empty `evidence`. Reject unevidenced output at the parser — do not let it reach the UI.
- [ ] On an adversarial set of 20 ambiguous clips, the agent calls `defer()` on **≥ 80%**. A system that always answers is not calibrated; it is guessing fluently.
- [ ] Token budget enforced: no minute exceeds the cap.
- [ ] Kill the LLM provider mid-run → video, detection, audio all continue. Suggestions degrade, nothing crashes.

### Risks

- _Hallucinated object references._ Mitigate by validating every `track_id` the model cites against the live track table. If it cites a nonexistent id, that response is discarded and retried once. Log the rate — it is a direct quality metric.
- _Prompt bloat._ Enforce the < 300-token scene-graph budget from Phase 3. It will try to grow.

---

## Phase 6 — Optimization

Only now. You have a working system and a truthful measurement harness; every change is checkable against both.

### Tasks

- INT8 quantization of the detector with calibration. Accept ≤ 1 mAP loss for ~1.4×.
- Batch the audio heads (ASR / tagging / captioning share one forward over the same window where the architecture allows).
- Pin processes to cores; `SCHED_FIFO` for the audio thread.
- CUDA graphs for the detector to eliminate per-launch overhead.
- Cache VLM responses keyed on the deterministic scene-graph hash — static scenes should cost nothing.
- Profile with Nsight Systems, not `print`.

### Exit criteria

- [ ] ≥ 1.5× throughput vs. Phase 5 at ≤ 1 mAP loss, both numbers from `bench/`.
- [ ] Every optimization has a before/after in the benchmark log. Anything unmeasured gets reverted.

---

## Phase 7 — Hardening & Deploy

### Tasks

- Process supervisor: restart a dead worker without tearing down the pipeline; the ring buffer survives.
- Graceful degradation ladder: GPU OOM → drop the embedding model → drop captioning → detection-only. Each step logged and surfaced in the UI.
- Full observability per `ARCHITECTURE.md §11`: queue-depth gauges, `seq`-gap counter, 1% OTel trace sampling, salience hit-rate + tokens/min dashboards.
- Alert on detection-confidence distribution shift (silent model degradation).
- Active-learning sink: log frames with conf ∈ [0.35, 0.55], tracker ID switches, and anomaly spikes to a labeling queue.
- Containerize (`nvidia/cuda` base). Pin every version; document the exact driver.
- Chaos pass: kill each process in turn, saturate the GPU, unplug the camera, sever the network.

### Exit criteria

- [ ] 24-hour soak: no memory growth, no FPS decay, no unbounded queue.
- [ ] Each of the 4 chaos tests recovers automatically.
- [ ] Active-learning queue accumulates ≥ 100 informative frames/hour on real input.

---

## Milestone Summary

| Phase | Deliverable                                  | Hard gate                                     |
| ----- | -------------------------------------------- | --------------------------------------------- |
| 0     | Ring buffer, event contract, latency harness | Pass-through g2g measured                     |
| 1     | Video detection + tracking + overlay         | 30 FPS, p99 < 100 ms                          |
| 2     | ASR + tags + audio captions                  | Caption < 1.5 s; video FPS unchanged          |
| 3     | Fusion, scene graph, memory                  | Alignment correct on 20 annotated events      |
| 4     | Salience gate                                | 90% trigger reduction @ 95% recall            |
| 5     | Reasoning agent                              | Evidence-grounded, defers on 80% of ambiguous |
| 6     | Optimization                                 | 1.5× throughput, ≤ 1 mAP loss                 |
| 7     | Hardening                                    | 24 h soak, 4/4 chaos tests                    |

---

## Cut Lines (if time is short)

Ship in this order; each line below is safe to drop before the one above it.

1. **Phase 7 hardening** → run it attended.
2. **Phase 6 optimization** → accept FP16 and lower resolution.
3. **Audio captioning** → keep ASR + tagging. You lose non-speech understanding, which is a real loss; know that you're taking it.
4. **Episodic memory / RAG** → the agent becomes stateless. Loses "third time this hour," keeps everything else.
5. **Cloud VLM** → swap for a local small VLM (Moondream, Qwen2-VL-2B). Weaker reasoning, zero marginal cost, no network dependency.

**Never cut:** the salience gate, tracking, the latency harness, or bounded queues. Each of those, removed, produces a system that demos well for ten minutes and then fails in a way that is expensive to diagnose.

---

## Immediate Next Actions

1. `mp.set_start_method("spawn")` in `main.py`, before any torch import.
2. Write `core/events.py`. Freeze the contract.
3. Write `core/ringbuffer.py` and its torn-read test.
4. Build `bench/latency.py` and measure your pass-through pipeline **today**. Whatever number comes back will tell you more about your hardware and your camera stack than a week of reading will.

---

## A Note on Model Choices

Every specific model named across these two documents — YOLOv11, RT-DETR, distil-whisper, BEATs, SigLIP, CLAP, Qwen2-VL — reflects a snapshot of a field that moves quickly, and I'd treat the names as _categories_ rather than recommendations. Before committing, check what's current and benchmark on your own hardware at your own input resolution. Vendor latency numbers are best-case, measured on server GPUs, at batch sizes you won't use.

The architecture does not depend on any of these choices. Every model sits behind an interface (`detect`, `transcribe`, `tag`, `caption`, `embed`) precisely so that swapping one is a config change, not a rewrite. That property is worth more than picking the right model today.
