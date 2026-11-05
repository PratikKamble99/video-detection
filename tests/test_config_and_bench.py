"""Config loads, main forces spawn, and the latency harness runs headless."""

from core import config
from bench import latency


def test_default_config_loads_and_dots():
    cfg = config.load()
    assert cfg.video.size == [640, 640]
    assert cfg.fusion.vision_buffer_s == 2.0
    # vision buffer must exceed ASR window or late audio finds no partners.
    assert cfg.fusion.vision_buffer_s > cfg.audio.asr.window_s
    assert cfg.salience.weights.visual == 2.0


def test_main_forces_spawn():
    from main import _set_spawn
    import multiprocessing as mp
    _set_spawn()
    assert mp.get_start_method() == "spawn"


def test_synthetic_latency_harness_runs_clean():
    # Small + short: this is a smoke test of the data plane, not the 10-min soak.
    res = latency.run_synthetic(shape=(120, 160, 3), slots=8, fps=200.0,
                                seconds=0.5, consumer_lag_frames=2)
    assert res["n"] > 10
    # A 2-frame-behind consumer in an 8-slot ring must never tear.
    assert res["torn_reads"] == 0
    # Data-plane latency for a well-behaved consumer is tiny (< 100 ms easily).
    assert res["p99"] < 100.0
