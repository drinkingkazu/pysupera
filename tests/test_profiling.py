"""
Tests for resource accounting.

The interesting behaviour is what happens when the optional pieces are
missing: profiling must never be the reason a training run falls over, so
every absent dependency has to collapse into a silent no-op.
"""
import numpy as np
import pytest

import pysupera.profiling as prof
from pysupera.profiling import (WandbLogger, cpu_memory, current_rank,
                                gpu_memory, memory_snapshot,
                                reset_peak_gpu_memory)


class FakeRun:
    """Stands in for a wandb Run; records what it was told."""

    def __init__(self):
        self.logged = []
        self.summary = {}

    def log(self, payload, step=None):
        self.logged.append((payload, step))


class TestMemory:

    def test_cpu_memory_is_reported(self):
        m = cpu_memory()
        assert m["cpu_rss_mb"] > 0
        assert m["cpu_rss_total_mb"] >= m["cpu_rss_mb"]

    def test_snapshot_merges_both(self):
        m = memory_snapshot()
        assert "cpu_rss_mb" in m
        # GPU keys appear only with CUDA; either way the call must not raise
        assert all(isinstance(v, float) for v in m.values())

    def test_gpu_memory_is_empty_without_cuda(self):
        g = gpu_memory()
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            assert g == {}
        else:
            assert g["gpu_alloc_mb"] >= 0

    def test_reset_peak_never_raises(self):
        reset_peak_gpu_memory()

    def test_repeated_calls_agree(self):
        a, b = cpu_memory(), cpu_memory()
        assert a["cpu_rss_mb"] == pytest.approx(b["cpu_rss_mb"], rel=0.5)


class TestChildCache:
    """
    Enumerating children scans /proc and dominates the snapshot, so the
    handles are cached -- but a stale cache would silently miscount memory.
    """

    def test_cache_is_reused_within_the_window(self, monkeypatch):
        pytest.importorskip("psutil")
        calls = []

        class P:
            def children(self, recursive=False):
                calls.append(1)
                return []
        prof._KIDS, prof._KIDS_T = [], -1.0
        prof._children(P())
        prof._children(P())
        assert len(calls) == 1

    def test_expiry_forces_a_refresh(self):
        pytest.importorskip("psutil")
        calls = []

        class P:
            def children(self, recursive=False):
                calls.append(1)
                return []
        prof._KIDS, prof._KIDS_T = [], -1.0
        prof._children(P(), refresh_s=0.0)
        prof._children(P(), refresh_s=0.0)
        assert len(calls) == 2

    def test_an_empty_cache_expires_sooner(self, monkeypatch):
        # nothing in an empty cache can signal that a worker pool appeared,
        # so emptiness must not be held for the full window
        pytest.importorskip("psutil")
        calls = []

        class P:
            def children(self, recursive=False):
                calls.append(1)
                return []
        monkeypatch.setattr(prof, "EMPTY_REFRESH_S", 0.0)
        prof._KIDS, prof._KIDS_T = [], -1.0
        prof._children(P(), refresh_s=1e9)
        prof._children(P(), refresh_s=1e9)
        assert len(calls) == 2

    def test_a_populated_cache_is_held(self, monkeypatch):
        pytest.importorskip("psutil")
        calls = []

        class Alive:
            def is_running(self): return True

        class P:
            def children(self, recursive=False):
                calls.append(1)
                return [Alive()]
        monkeypatch.setattr(prof, "EMPTY_REFRESH_S", 0.0)
        prof._KIDS, prof._KIDS_T = [Alive()], 1e18
        prof._children(P(), refresh_s=1e9)
        assert len(calls) == 0

    def test_a_dead_child_forces_a_refresh(self):
        pytest.importorskip("psutil")
        calls = []

        class Dead:
            def is_running(self): return False

        class P:
            def children(self, recursive=False):
                calls.append(1)
                return []
        prof._KIDS, prof._KIDS_T = [Dead()], 1e18   # fresh, but stale in fact
        prof._children(P())
        assert len(calls) == 1

    def test_handle_is_rebuilt_after_a_fork(self, monkeypatch):
        pytest.importorskip("psutil")
        prof._self_proc()
        monkeypatch.setattr(prof.os, "getpid", lambda: 999999)
        monkeypatch.setattr(prof.psutil, "Process", lambda pid: f"proc-{pid}")
        assert prof._self_proc() == "proc-999999"
        assert prof._KIDS == []                     # parent's children dropped
        monkeypatch.undo()
        prof._PROC = None                           # restore for other tests


class TestWandbLogger:

    def test_explicit_run_is_used(self):
        r = FakeRun()
        w = WandbLogger(run=r)
        assert w.active
        w.log({"a": 1})
        assert r.logged == [({"a": 1}, None)]

    def test_prefix_is_applied(self):
        r = FakeRun()
        WandbLogger(run=r, prefix="data").log({"read_s": 0.5})
        assert r.logged[0][0] == {"data/read_s": 0.5}

    def test_summary_is_written(self):
        r = FakeRun()
        WandbLogger(run=r, prefix="data").summary({"steps": 3})
        assert r.summary == {"data/steps": 3}

    def test_disabled_is_inactive(self):
        r = FakeRun()
        w = WandbLogger(run=r, enabled=False)
        assert not w.active and w.reason == "disabled"
        w.log({"a": 1})
        assert r.logged == []

    def test_other_ranks_do_not_log(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        r = FakeRun()
        w = WandbLogger(run=r, log_rank=0)
        assert not w.active
        w.log({"a": 1})
        assert r.logged == []

    def test_log_rank_none_logs_everywhere(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        r = FakeRun()
        assert WandbLogger(run=r, log_rank=None).active

    def test_missing_wandb_is_a_silent_noop(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "wandb", None)
        w = WandbLogger()
        # no run could be resolved, so nothing is logged and nothing raises
        assert not w.active and w.reason
        w.log({"a": 1})
        w.summary({"a": 1})

    def test_rank_defaults_to_zero(self, monkeypatch):
        for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
            monkeypatch.delenv(k, raising=False)
        assert current_rank() == 0


torch = pytest.importorskip("torch")
from pysupera.torchdata import StreamMonitor          # noqa: E402


def fake_batch(n_events=2, n_points=100, read_s=0.01, preprocess_s=0.02,
               payload_mb=1.0):
    return {
        "events": [{"points": np.zeros((n_points, 4), np.float32)}
                   for _ in range(n_events)],
        "batch_size": n_events,
        "n_points": [n_points] * n_events,
        "profile": {"read_s": read_s, "preprocess_s": preprocess_s,
                    "collate_s": 0.001, "payload_mb": payload_mb},
    }


class TestStreamMonitor:

    def test_accumulates_across_steps(self):
        m = StreamMonitor(device=None, memory=False)
        for _ in range(3):
            m.record(fake_batch(), wait_s=0.1, stage_s=0.05)
        s = m.summary()
        assert s["steps"] == 3
        assert s["events"] == 6
        assert s["points"] == 600
        assert s["total_read_s"] == pytest.approx(0.03)
        assert s["total_wait_s"] == pytest.approx(0.3)

    def test_stall_fraction_is_against_real_wall_time(self):
        # the training step counts: blocking 0.25 s out of a 1.0 s
        # iteration is a quarter stalled, not three quarters
        m = StreamMonitor(device=None, memory=False)
        m.record(fake_batch(), wait_s=0.25, stage_s=0.25, consume_s=0.5)
        assert m.summary()["stall_fraction"] == pytest.approx(0.25)

    def test_step_is_partitioned_by_its_parts(self):
        m = StreamMonitor(device=None, memory=False)
        met = m.record(fake_batch(), wait_s=0.1, stage_s=0.2, consume_s=0.3)
        assert met["step_s"] == pytest.approx(0.6)
        assert met["step_s"] == pytest.approx(
            met["wait_s"] + met["stage_s"] + met["consume_s"])

    def test_consume_time_is_measured_across_the_yield(self):
        from time import sleep
        m = StreamMonitor(device=None, memory=False)
        for _ in m.iterate([fake_batch()]):
            sleep(0.05)                       # stand-in for a training step
        s = m.summary()
        assert s["total_consume_s"] >= 0.05
        assert s["stall_fraction"] < 0.5      # the step dominated, not the wait

    def test_a_break_still_records_the_last_step(self):
        m = StreamMonitor(device=None, memory=False)
        for _ in m.iterate([fake_batch(), fake_batch()]):
            break
        assert m.summary()["steps"] == 1

    def test_payload_and_waits_are_tracked(self):
        m = StreamMonitor(device=None, memory=False)
        m.record(fake_batch(payload_mb=2.0), wait_s=0.5)
        m.record(fake_batch(payload_mb=3.0), wait_s=0.1)
        s = m.summary()
        assert s["payload_mb"] == pytest.approx(5.0)
        assert s["first_wait_s"] == pytest.approx(0.5)
        assert s["max_wait_s"] == pytest.approx(0.5)

    def test_payload_rate_uses_the_blocked_time(self):
        m = StreamMonitor(device=None, memory=False)
        met = m.record(fake_batch(payload_mb=4.0), wait_s=2.0)
        assert met["payload_mb_per_s"] == pytest.approx(2.0)

    def test_wall_clock_accounts_for_nearly_everything(self):
        from time import sleep
        m = StreamMonitor(device=None, memory=False)
        for _ in m.iterate([fake_batch() for _ in range(3)]):
            sleep(0.01)
        s = m.summary()
        assert s["wall_s"] > 0
        # the per-step breakdown should miss only the monitor's own overhead
        assert abs(s["unaccounted_s"]) < 0.02 * s["wall_s"] + 1e-3

    def test_throughput_uses_step_time(self):
        m = StreamMonitor(device=None, memory=False)
        met = m.record(fake_batch(n_events=2, n_points=50),
                       wait_s=0.5, stage_s=0.25, consume_s=0.25)
        assert met["points_per_s"] == pytest.approx(100.0)
        assert met["events_per_s"] == pytest.approx(2.0)

    def test_memory_keys_present_only_when_asked(self):
        assert "cpu_rss_mb" in StreamMonitor(device=None, memory=True).record(
            fake_batch(), wait_s=0.1)
        assert "cpu_rss_mb" not in StreamMonitor(device=None, memory=False).record(
            fake_batch(), wait_s=0.1)

    def test_log_every_throttles(self):
        r = FakeRun()
        m = StreamMonitor(device=None, memory=False, log_every=3)
        m.logger = WandbLogger(run=r, prefix="data")
        for _ in range(7):
            m.record(fake_batch(), wait_s=0.1)
        assert len(r.logged) == 2                 # steps 3 and 6
        assert m.summary()["steps"] == 7          # but all are accounted for

    def test_reset_clears(self):
        m = StreamMonitor(device=None, memory=False)
        m.record(fake_batch(), wait_s=0.1)
        m.reset()
        assert m.summary()["steps"] == 0
        assert m.summary()["total_wait_s"] == 0.0

    def test_finish_writes_the_summary(self):
        r = FakeRun()
        m = StreamMonitor(device=None, memory=False)
        m.logger = WandbLogger(run=r, prefix="data")
        m.record(fake_batch(), wait_s=0.1)
        out = m.finish()
        assert r.summary["data/steps"] == 1
        assert out["steps"] == 1

    def test_iterate_times_the_wait(self):
        m = StreamMonitor(device=None, memory=False)
        batches = [fake_batch(), fake_batch()]
        got = list(m.iterate(batches))
        assert len(got) == 2
        assert m.summary()["steps"] == 2
        assert m.summary()["total_wait_s"] > 0

    def test_iterate_sets_the_epoch(self):
        class Sampler:
            epoch = None
            def set_epoch(self, e): Sampler.epoch = e

        class Loader(list):
            sampler = Sampler()

        m = StreamMonitor(device=None, memory=False)
        list(m.iterate(Loader([fake_batch()]), epoch=7))
        assert Sampler.epoch == 7

    def test_monitor_works_without_profile_data(self):
        b = fake_batch()
        del b["profile"]
        m = StreamMonitor(device=None, memory=False)
        met = m.record(b, wait_s=0.1)
        assert met["read_s"] == 0.0
