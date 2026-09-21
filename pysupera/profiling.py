"""
Resource accounting for the training-data pipeline.

Answers two questions about :mod:`pysupera.torchdata`: where wall time goes
between reading an event, turning it into model input and moving it to the
GPU, and how much memory each of those costs.  Measurements are optionally
forwarded to Weights & Biases.

Nothing here requires torch or wandb.  GPU figures are reported when torch
sees CUDA and omitted otherwise; :class:`WandbLogger` degrades to a no-op when
wandb is absent, when no run is active, or on a rank that should not log.
"""

from __future__ import annotations

import os
from time import monotonic

_MB = 1.0 / (1024 * 1024)

try:
    import psutil
except Exception:                                            # pragma: no cover
    psutil = None

#: Enumerating children scans /proc and costs ~4.6 ms with four DataLoader
#: workers, against 0.015 ms to read RSS from handles we already hold -- so
#: the handles are cached and only re-enumerated when one dies or this many
#: seconds pass.  Worker sets are stable for at least an epoch.
CHILDREN_REFRESH_S = 10.0

#: A *dead* child invalidates the cache immediately, but an empty cache has
#: nothing to notice a *new* child by -- so emptiness expires far sooner.
#: Without this, a worker pool spawned just after a childless sample stays
#: invisible for the full refresh window and worker memory reads as zero.
EMPTY_REFRESH_S = 1.0

_PROC = None
_PROC_PID = None
_KIDS: list = []
_KIDS_T = -1.0


def _self_proc():
    """This process's psutil handle, rebuilt after a fork."""
    global _PROC, _PROC_PID, _KIDS, _KIDS_T
    if psutil is None:
        return None
    pid = os.getpid()
    if _PROC is None or _PROC_PID != pid:
        _PROC = psutil.Process(pid)
        _PROC_PID = pid
        _KIDS, _KIDS_T = [], -1.0      # a fork inherits none of the parent's
    return _PROC


def _children(proc, refresh_s: float = CHILDREN_REFRESH_S) -> list:
    """
    Cached child handles, re-enumerated on expiry or when one has died.

    An empty cache expires after :data:`EMPTY_REFRESH_S` instead, because
    there is no cheap way to notice a child that did not exist last time.
    """
    global _KIDS, _KIDS_T
    now = monotonic()
    window = refresh_s if _KIDS else min(refresh_s, EMPTY_REFRESH_S)
    if now - _KIDS_T > window or any(not c.is_running() for c in _KIDS):
        try:
            _KIDS = proc.children(recursive=True)
        except Exception:                                    # pragma: no cover
            _KIDS = []
        _KIDS_T = now
    return _KIDS


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def cpu_memory(include_children: bool = True) -> dict:
    """
    Resident set size of this process, in MiB.

    ``cpu_rss_total_mb`` adds the DataLoader worker processes, which is where
    reading and pre-processing actually allocate once ``num_workers > 0`` --
    the parent's own RSS says nothing about them.  Workers share copy-on-write
    pages with the parent, so the total over-counts somewhat; read it as an
    upper bound.

    Child handles are cached (see :data:`CHILDREN_REFRESH_S`); enumerating
    them on every call would cost more than the rest of the snapshot together.

    Falls back to ``/proc`` when psutil is missing, which costs the
    child-process and system-availability figures.
    """
    proc = _self_proc()
    if proc is not None:
        rss = proc.memory_info().rss
        total = rss
        if include_children:
            for c in _children(proc):
                try:
                    total += c.memory_info().rss
                except Exception:
                    pass                      # a worker may exit mid-walk
        return {"cpu_rss_mb": rss * _MB,
                "cpu_rss_total_mb": total * _MB,
                "cpu_available_mb": psutil.virtual_memory().available * _MB}

    try:                                                     # pragma: no cover
        with open("/proc/self/statm") as fh:
            rss = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return {"cpu_rss_mb": rss * _MB, "cpu_rss_total_mb": rss * _MB}
    except Exception:                                        # pragma: no cover
        return {}


def gpu_memory(device=None) -> dict:
    """
    CUDA memory, in MiB.  Empty when torch is absent or no GPU is present.

    ``alloc``/``reserved`` are the allocator's view -- live tensors and the
    cache it holds for them -- while ``free``/``total`` come from the driver
    and so include every process on the device.  ``peak`` is the high-water
    mark since the last :func:`reset_peak_gpu_memory`.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
    except Exception:
        return {}

    dev = torch.device(device) if device is not None else torch.device("cuda")
    if dev.type != "cuda":
        return {}
    out = {
        "gpu_alloc_mb":    torch.cuda.memory_allocated(dev) * _MB,
        "gpu_reserved_mb": torch.cuda.memory_reserved(dev) * _MB,
        "gpu_peak_mb":     torch.cuda.max_memory_allocated(dev) * _MB,
    }
    try:
        free, total = torch.cuda.mem_get_info(dev)
        out["gpu_free_mb"] = free * _MB
        out["gpu_total_mb"] = total * _MB
    except Exception:                                        # pragma: no cover
        pass
    return out


def memory_snapshot(device=None, include_children: bool = True) -> dict:
    """CPU and GPU memory in one dict; see :func:`cpu_memory`, :func:`gpu_memory`."""
    out = cpu_memory(include_children)
    out.update(gpu_memory(device))
    return out


def reset_peak_gpu_memory(device=None) -> None:
    """Restart the ``gpu_peak_mb`` high-water mark, e.g. at the start of an epoch."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:                                        # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

def current_rank() -> int:
    """This process's rank, from torch.distributed or the usual env vars."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(key)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


class WandbLogger:
    """
    Forward metric dicts to Weights & Biases, or to nowhere.

    Instrumented code should not have to ask whether logging is on, so every
    reason not to log collapses into the same silent no-op: wandb not
    installed, no run active, *enabled* false, or a rank other than
    *log_rank*.  :attr:`active` says which way it went.

    In a distributed job only one rank logs by default.  Every rank sees a
    different shard, so letting them all write to one run interleaves
    unrelated series; pass ``log_rank=None`` to log from all of them anyway.

    Parameters
    ----------
    run : wandb Run or None
        An explicit run.  ``None`` picks up the active global run, so
        ``wandb.init()`` before constructing this is enough.
    prefix : str
        Prepended to every key as ``"<prefix>/<key>"``.
    enabled : bool
        Master switch.
    log_rank : int or None
        Only this rank logs; ``None`` means all ranks do.
    """

    def __init__(self, run=None, prefix: str = "", enabled: bool = True,
                 log_rank: int | None = 0):
        self.prefix = prefix
        self._run = None
        self.reason = ""

        if not enabled:
            self.reason = "disabled"
            return
        if log_rank is not None and current_rank() != log_rank:
            self.reason = f"rank {current_rank()} != log_rank {log_rank}"
            return
        if run is None:
            try:
                import wandb
            except ImportError:
                self.reason = "wandb is not installed"
                return
            run = getattr(wandb, "run", None)
            if run is None:
                self.reason = "no active wandb run (call wandb.init() first)"
                return
        self._run = run

    @property
    def active(self) -> bool:
        """True when metrics are actually going somewhere."""
        return self._run is not None

    def _key(self, k: str) -> str:
        return f"{self.prefix}/{k}" if self.prefix else k

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log one step's metrics."""
        if self._run is None:
            return
        payload = {self._key(k): v for k, v in metrics.items()}
        if step is None:
            self._run.log(payload)
        else:
            self._run.log(payload, step=step)

    def summary(self, metrics: dict) -> None:
        """Write run-summary values, shown as a single number per key."""
        if self._run is None:
            return
        for k, v in metrics.items():
            self._run.summary[self._key(k)] = v
