"""
PyTorch data pipeline for panoptic segmentation on pysupera 3.0.0 output.

Provides a :class:`Dataset` over events, a collate that keeps a batch as a
*list* of events (point clouds differ in length, so nothing is padded or
concatenated), and :func:`make_dataloader` to wire both up for single-process
or distributed training.

A note on "DDP": :class:`torch.nn.parallel.DistributedDataParallel` wraps a
*model*, not data.  What a distributed job needs on the input side is a
:class:`~torch.utils.data.distributed.DistributedSampler` so each rank sees a
disjoint shard of events, which is what :func:`make_dataloader` sets up.

Voxel merging
-------------
A voxel can receive energy from several particle instances.  The model sees one
row per voxel, so overlaps are resolved here:

* ``E`` is **summed** over every contribution, so no charge is lost.
* the semantic and instance labels are taken from the **winning** contribution.

Winner selection, in order:

1. **Semantic priority** — ``kTrack > kShower > kDelta > kMichel > kLEScatter``
   (then ``kUnknown``).  A track passing through a shower keeps the voxel.
2. **Earliest time** — among contributions of the same semantic class, the one
   whose voxel time is smallest wins.

Low-energy scatters
-------------------
LE-ness is a property of a *point*, not of its instance: an instance's block
holds its own points in ``[inst_pc_start, inst_pc_end)`` and the LE deposits
absorbed into it in ``[inst_pc_le_start, inst_pc_le_end)``.  The two runs are
adjacent, so ``inst_pc_end == inst_pc_le_start``.  A point in the second
run is labelled ``kLEScatter`` whatever its host instance is, which is what
makes it the lowest-priority contribution to a contested voxel.  Its *instance*
label is still the host's, so an LE deposit stays attributed to the shower or
track that absorbed it.

LE deposits are **excluded by default** (``include_le=False``): they are around
a third of the points and are usually noise to a segmentation model.  The flag
governs the voxel data as a whole, input and labels alike -- an excluded point
contributes no coordinates and no energy, so the model never sees it.

Instrumentation
---------------
Every event records what it cost to read and to pre-process, and
:class:`StreamMonitor` adds the two figures a training loop cannot get from
inside a worker -- how long it blocked waiting for input, and what the
transfer to the GPU cost -- together with CPU and GPU memory.  See
:mod:`pysupera.profiling`.

Labels
------
``voxel_sem``, ``voxel_instance`` and ``voxel_interaction`` are the three
panoptic targets, one per row of ``points``.  ``instances`` and ``interactions``
are the object-level tables read straight from the file; an interaction's id is
its row index, which is what ``voxel_interaction`` holds.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as _e:                                    # pragma: no cover
    raise ImportError(
        "pysupera.torchdata needs PyTorch: pip install torch"
    ) from _e

from .io_v3 import read_events_v3
from .profiling import WandbLogger, memory_snapshot, reset_peak_gpu_memory
from .utils import SemanticType


#: Lower rank wins a contested voxel.  kUnknown is last so it never displaces
#: a classified instance.
SEM_PRIORITY: dict = {
    SemanticType.kTrack.value:      0,
    SemanticType.kShower.value:     1,
    SemanticType.kDelta.value:      2,
    SemanticType.kMichel.value:     3,
    SemanticType.kLEScatter.value:  4,
    SemanticType.kUnknown.value:    5,
}
_MAX_RANK = max(SEM_PRIORITY.values()) + 1

#: Rows of ``points``: x, y, z, E.
FEATURES = ("x", "y", "z", "E")

#: Assigned to a voxel no instance claims.  Should not occur -- instances
#: partition the event's points -- but a label array must say something.
NO_LABEL = -1


def _priority_rank(sem_values: np.ndarray) -> np.ndarray:
    """Map semantic type codes to contention ranks, unknown codes last."""
    out = np.full(len(sem_values), _MAX_RANK, dtype=np.int16)
    for code, rank in SEM_PRIORITY.items():
        out[sem_values == code] = rank
    return out


def build_event(view, include_le: bool = False) -> dict:
    """
    Turn one :class:`~pysupera.io_v3.EventView` into model input and labels.

    Parameters
    ----------
    view : EventView
        One event from a format 3.x file.
    include_le : bool
        Keep low-energy-scatter points.  Off by default; see the module
        docstring.  Applies to the input points and to every label, so the
        two always describe the same set of voxels.

    Returns
    -------
    dict
        ``points``            (V, 4) float32 -- x, y, z, E, one row per voxel
        ``voxel_sem``         (V,) int16   winner's class; ``kLEScatter`` for
                                        an absorbed low-energy deposit
        ``voxel_instance``    (V,) int32   winning instance's particle ``id``
        ``voxel_interaction`` (V,) int32   that instance's ``interaction_id``
        ``instances``         dict of arrays, one entry per instance row
        ``interactions``      dict of arrays, one entry per interaction row
        ``event``             int
    """
    pts = np.asarray(view.points)                     # (P, 6) x,y,z,t,dE,dX
    c = view.columns
    inst_rows = np.flatnonzero(view.is_instance)

    # ---- per-point provenance from the instance blocks --------------------
    # Instances partition the event's points, so one pass labels every point
    # and nothing is double-assigned.
    n = len(pts)
    p_inst = np.full(n, NO_LABEL, dtype=np.int32)
    p_sem = np.full(n, NO_LABEL, dtype=np.int16)
    p_intr = np.full(n, NO_LABEL, dtype=np.int32)
    le_code = SemanticType.kLEScatter.value
    for r in inst_rows:
        # Two adjacent runs: the instance's own points, then the LE deposits
        # absorbed into it.  Either side may be empty (-1).
        nle_a, nle_b = int(c["inst_pc_start"][r]), int(c["inst_pc_end"][r])
        le_a, le_b = int(c["inst_pc_le_start"][r]), int(c["inst_pc_le_end"][r])
        a = nle_a if nle_a >= 0 else le_a
        b = le_b if le_b >= 0 else nle_b
        if a < 0 or b <= a:
            continue
        p_inst[a:b] = int(c["id"][r])
        p_intr[a:b] = int(c["interaction_id"][r])
        if nle_a >= 0 and nle_b > nle_a:
            p_sem[nle_a:nle_b] = int(c["inst_sem_type"][r])
        if le_a >= 0 and le_b > le_a:
            p_sem[le_a:le_b] = le_code

    # ---- drop low-energy scatters -----------------------------------------
    # Done before merging so an excluded point contributes no energy either.
    if not include_le:
        keep = p_sem != SemanticType.kLEScatter.value
        if not keep.all():
            pts = pts[keep]
            p_inst, p_sem, p_intr = p_inst[keep], p_sem[keep], p_intr[keep]
            n = len(pts)

    if n == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return {"points": empty,
                "voxel_sem": np.zeros(0, np.int16),
                "voxel_instance": np.zeros(0, np.int32),
                "voxel_interaction": np.zeros(0, np.int32),
                "instances": _table(c, inst_rows),
                "interactions": _arrays(view.interactions),
                "event": view.event}

    # ---- collapse to unique voxels ----------------------------------------
    # Coordinates are voxel centres produced by a deterministic formula, so
    # two contributions to the same cell are bit-identical and exact
    # comparison is safe -- no rounding needed.
    xyz = pts[:, :3]
    _, first, inverse = np.unique(xyz, axis=0,
                                  return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    n_vox = len(first)

    energy = np.bincount(inverse, weights=pts[:, 4].astype(np.float64),
                         minlength=n_vox).astype(np.float32)

    # ---- resolve contested voxels -----------------------------------------
    # Sort by (voxel, semantic rank, time) and take the first row of each
    # voxel: that is the highest-priority class, earliest within it.
    rank = _priority_rank(p_sem)
    order = np.lexsort((pts[:, 3], rank, inverse))
    vox_sorted = inverse[order]
    winner = order[np.flatnonzero(
        np.r_[True, vox_sorted[1:] != vox_sorted[:-1]])]
    # vox_sorted is ascending and its values are 0..n_vox-1, so the group
    # firsts come out in voxel order: winner[k] is voxel k's winning point.

    out = np.empty((n_vox, 4), dtype=np.float32)
    out[:, :3] = xyz[winner]
    out[:, 3] = energy

    return {
        "points": out,
        "voxel_sem": p_sem[winner].astype(np.int16),
        "voxel_instance": p_inst[winner].astype(np.int32),
        "voxel_interaction": p_intr[winner].astype(np.int32),
        "instances": _table(c, inst_rows),
        "interactions": _arrays(view.interactions),
        "event": view.event,
    }


def _table(columns: dict, rows: np.ndarray) -> dict:
    """The instance-level table, as one array per column."""
    return {k: np.asarray(v)[rows] for k, v in columns.items()}


def _arrays(table: dict) -> dict:
    """A column table as plain arrays."""
    return {k: np.asarray(v) for k, v in table.items()}


class PysuperaEvents(Dataset):
    """
    One pysupera 3.0.0 file as a map-style dataset of events.

    The HDF5 handle is opened lazily *per worker*: h5py handles cannot cross a
    fork, so opening in ``__init__`` breaks silently under
    ``num_workers > 0``.

    Parameters
    ----------
    path : str
        A format 3.x output file.
    include_le : bool
        Keep low-energy-scatter points.  Off by default.
    transform : callable or None
        Applied to each event dict before it is returned.
    profile : bool
        Time the read and the pre-processing of each event into
        ``event["profile"]``.  On by default: it is two ``perf_counter``
        calls per event, and the numbers are otherwise unobtainable because
        the work happens in a worker process.
    """

    def __init__(self, path: str, transform=None, include_le: bool = False,
                 profile: bool = True):
        self.path = str(path)
        self.transform = transform
        self.include_le = bool(include_le)
        self.profile = bool(profile)
        self._store = None
        with read_events_v3(self.path) as s:          # length only; then close
            self._n = len(s)

    def __len__(self) -> int:
        return self._n

    def _lazy(self):
        if self._store is None:
            self._store = read_events_v3(self.path)
        return self._store

    def __getitem__(self, idx: int) -> dict:
        if not self.profile:
            ev = build_event(self._lazy()[int(idx)], include_le=self.include_le)
            return self.transform(ev) if self.transform else ev

        t0 = perf_counter()
        view = self._lazy()[int(idx)]          # h5py: decompress + slice
        t1 = perf_counter()
        ev = build_event(view, include_le=self.include_le)
        if self.transform:
            ev = self.transform(ev)            # counted as pre-processing
        t2 = perf_counter()
        ev["profile"] = {"read_s": t1 - t0, "preprocess_s": t2 - t1}
        return ev

    def __getstate__(self):
        # Never pickle an open handle into a worker process.
        state = self.__dict__.copy()
        state["_store"] = None
        return state

    def close(self):
        if self._store is not None:
            self._store.close()
            self._store = None


def collate_events(batch: list) -> dict:
    """
    Keep a batch as a list of events.

    Events have different point counts, so stacking would force padding.
    Sparse-convolution and transformer backbones both prefer the ragged form,
    and ``batch_size`` then means exactly what was asked for: a number of
    events.

    Per-event timings are summed into ``batch["profile"]`` so they survive the
    trip out of the worker process, along with what this call itself cost and
    how many bytes the batch is about to ship back to the parent.
    """
    t0 = perf_counter()
    read_s = preprocess_s = 0.0
    for e in batch:
        p = e.get("profile")
        if p:
            read_s += p["read_s"]
            preprocess_s += p["preprocess_s"]
    out = {
        "events": batch,
        "batch_size": len(batch),
        "n_points": [len(e["points"]) for e in batch],
        "profile": {"read_s": read_s, "preprocess_s": preprocess_s,
                    "payload_mb": _payload_bytes(batch) / (1024 * 1024)},
    }
    out["profile"]["collate_s"] = perf_counter() - t0
    return out


def _payload_bytes(events: list) -> int:
    """
    Bytes of array data in a batch -- what crosses the worker boundary.

    Divided by the time the parent blocked, this gives the effective rate at
    which batches are arriving, which is the thing to look at when a batch is
    cheap to build but slow to show up.
    """
    total = 0
    for e in events:
        for v in e.values():
            if isinstance(v, np.ndarray):
                total += v.nbytes
            elif isinstance(v, dict):
                total += sum(a.nbytes for a in v.values()
                             if isinstance(a, np.ndarray))
    return total


def to_torch(event: dict, device=None) -> dict:
    """Convert one event's arrays to tensors, leaving the object tables alone."""
    out = dict(event)
    for k in ("points", "voxel_sem", "voxel_instance", "voxel_interaction"):
        out[k] = torch.as_tensor(event[k])
        if device is not None:
            out[k] = out[k].to(device, non_blocking=True)
    return out


def make_dataloader(path: str, batch_size: int = 1, *, distributed: bool = False,
                    shuffle: bool = True, num_workers: int = 0, seed: int = 0,
                    drop_last: bool = False, include_le: bool = False,
                    profile: bool = True, transform=None, **kwargs):
    """
    Build a :class:`~torch.utils.data.DataLoader` over events.

    With *distributed* set, a
    :class:`~torch.utils.data.distributed.DistributedSampler` shards events
    across ranks; call ``loader.sampler.set_epoch(epoch)`` each epoch or every
    rank replays the same order.

    Low-energy-scatter points are dropped unless *include_le* is set.
    Wrap the result in a :class:`StreamMonitor` to time and log the pipeline.

    Returns
    -------
    DataLoader
        Yields ``{"events": [...], "batch_size": int, "n_points": [...]}``.
    """
    ds = PysuperaEvents(path, transform=transform, include_le=include_le,
                        profile=profile)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed,
                                     drop_last=drop_last)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_events,
        drop_last=drop_last,
        **kwargs,
    )


def stage_batch(batch: dict, device, non_blocking: bool = True) -> tuple:
    """
    Move a whole batch's arrays onto *device*.

    Returns ``(batch, seconds)``.  On CUDA the device is synchronised before
    the clock is read, or the copies would still be in flight and the timing
    would flatter itself.

    The arrays are plain numpy, so host memory is not pinned and the copies
    are synchronous whatever *non_blocking* says.  Pinning would need the
    dataset to emit tensors; the cost shows up here either way.
    """
    t0 = perf_counter()
    batch["events"] = [to_torch(e, device) for e in batch["events"]]
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return batch, perf_counter() - t0


class StreamMonitor:
    """
    Time and log the input pipeline around a :class:`DataLoader`.

    ::

        mon = StreamMonitor(device="cuda", prefix="data")
        for batch in mon.iterate(loader):
            train_step(batch)          # batch already staged on the GPU
        print(mon.summary())

    Timings come in two kinds, and mixing them up is the easy mistake.

    **Worker-side cost** -- ``read_s``, ``preprocess_s``, ``collate_s``, summed
    over the batch's events and measured wherever they were produced.  With
    ``num_workers > 0`` these run in other processes, concurrently with the
    training step and with each other, so their sum routinely exceeds the
    step's wall time.  They say what the work costs, not what it delays.

    **Main-process wall time** -- these partition one iteration and add up to
    ``step_s``:

    ``wait_s``
        How long the loop actually blocked in ``next(loader)``: workers not
        yet done, plus the cost of shipping the finished batch back and
        rebuilding it in this process.  The two are not separable from
        outside -- the DataLoader offers no hook between them -- so compare
        ``wait_s`` against ``payload_mb`` to tell a starved pipeline from a
        fat one.
    ``stage_s``
        The host-to-device transfer, when *device* is set.
    ``consume_s``
        Everything the loop body did with the batch -- the training step.
        Measured across the ``yield``, so it needs no cooperation from the
        caller.

    ``stall_fraction = wait_s / step_s`` is the headline: the share of real
    wall time the loop spent waiting on input.  Near zero means the pipeline
    keeps up however large ``read_s`` is; near one means it does not.

    Memory is sampled once per logged step -- see
    :func:`~pysupera.profiling.memory_snapshot`.  ``wandb_run`` (or an active
    global run) turns on Weights & Biases logging; without one the monitor
    still accumulates, and :meth:`summary` still works.

    Parameters
    ----------
    device : torch device or None
        Stage each batch here.  ``None`` leaves batches on the host and
        reports ``stage_s`` as zero.
    wandb_run, prefix, log_rank
        Passed to :class:`~pysupera.profiling.WandbLogger`.
    enabled : bool
        Master switch for logging; accumulation is unaffected.
    log_every : int
        Log every n-th step.  Totals still cover every step.
    memory : bool
        Sample CPU/GPU memory.  Costs a psutil walk over the worker
        processes, so turn it off if steps are very short.
    """

    #: worker-side costs, overlapping and concurrent
    _WORKER = ("read_s", "preprocess_s", "collate_s")
    #: main-process wall time; the first three partition ``step_s``
    _WALL = ("wait_s", "stage_s", "consume_s", "step_s")
    _STAGES = _WORKER + _WALL

    def __init__(self, device=None, wandb_run=None, prefix: str = "data",
                 enabled: bool = True, log_every: int = 1,
                 log_rank: int | None = 0, memory: bool = True):
        self.device = device
        self.log_every = max(1, int(log_every))
        self.memory = bool(memory)
        self.logger = WandbLogger(run=wandb_run, prefix=prefix,
                                  enabled=enabled, log_rank=log_rank)
        self.reset()

    def reset(self) -> None:
        """
        Zero the accumulators and the GPU high-water mark.

        Also performs one throwaway host-to-device copy.  The first real copy
        in a process otherwise pays for lazy CUDA initialisation -- 35 ms
        against the ~1 ms a warm copy costs -- and step 1 would wear the bill.
        Allocating on the device does not trigger it; the transfer path has to
        be exercised.  This removes most of the cost but not all of it: the
        first step still runs several times long while the caching allocator
        grows to the real batch sizes, so discard it when benchmarking.
        """
        self.totals = {k: 0.0 for k in self._STAGES}
        self.steps = 0
        self.events = 0
        self.points = 0
        self.payload_mb = 0.0
        self.max_wait_s = 0.0
        self.first_wait_s = None
        self._t_start = None
        self._t_last = None
        self._warmup()
        reset_peak_gpu_memory(self.device)

    def _warmup(self) -> None:
        if self.device is None:
            return
        try:
            dev = torch.device(self.device)
            if dev.type == "cuda":
                torch.zeros(1).to(dev)       # host -> device, not just alloc
                torch.cuda.synchronize(dev)
        except Exception:                                    # pragma: no cover
            pass

    def iterate(self, loader, epoch: int | None = None):
        """
        Yield staged batches from *loader*, timing and logging each one.

        A step is recorded *after* the loop body returns, so that its
        ``consume_s`` is known; metrics for step *n* therefore reach W&B once
        step *n* has finished.

        Pass *epoch* to forward it to a ``DistributedSampler`` -- without
        ``set_epoch`` every rank replays one order for the whole run.
        """
        if epoch is not None:
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        t_prev = perf_counter()
        for batch in loader:
            wait_s = perf_counter() - t_prev
            stage_s = 0.0
            if self.device is not None:
                batch, stage_s = stage_batch(batch, self.device)
            t_yield = perf_counter()
            try:
                yield batch
            finally:
                # finally, not plain fall-through: a caller that breaks out
                # of the loop still gets its last step accounted for
                self.record(batch, wait_s=wait_s, stage_s=stage_s,
                            consume_s=perf_counter() - t_yield)
            t_prev = perf_counter()

    def record(self, batch: dict, wait_s: float = 0.0, stage_s: float = 0.0,
               consume_s: float = 0.0) -> dict:
        """Account for one batch and log it.  Returns the metrics logged."""
        now = perf_counter()
        if self._t_start is None:
            self._t_start = now - (wait_s + stage_s + consume_s)
        self._t_last = now

        prof = batch.get("profile", {})
        n_pts = int(sum(batch.get("n_points", ())))
        n_ev = int(batch.get("batch_size", 0))
        payload_mb = float(prof.get("payload_mb", 0.0))
        step_s = wait_s + stage_s + consume_s

        step = {
            "read_s":       float(prof.get("read_s", 0.0)),
            "preprocess_s": float(prof.get("preprocess_s", 0.0)),
            "collate_s":    float(prof.get("collate_s", 0.0)),
            "wait_s":       wait_s,
            "stage_s":      stage_s,
            "consume_s":    consume_s,
            "step_s":       step_s,
        }
        for k in self._STAGES:
            self.totals[k] += step[k]
        self.steps += 1
        self.events += n_ev
        self.points += n_pts
        self.payload_mb += payload_mb
        self.max_wait_s = max(self.max_wait_s, wait_s)
        if self.first_wait_s is None:
            self.first_wait_s = wait_s

        metrics = dict(step)
        metrics["batch_size"] = n_ev
        metrics["n_points"] = n_pts
        metrics["payload_mb"] = payload_mb
        if step_s > 0:
            metrics["events_per_s"] = n_ev / step_s
            metrics["points_per_s"] = n_pts / step_s
            metrics["stall_fraction"] = wait_s / step_s
        if wait_s > 0:
            metrics["payload_mb_per_s"] = payload_mb / wait_s
        if self.memory:
            metrics.update(memory_snapshot(self.device))

        if self.steps % self.log_every == 0:
            self.logger.log(metrics)
        return metrics

    def summary(self) -> dict:
        """
        Totals and per-step means over everything recorded since
        :meth:`reset`.

        The headline is ``stall_fraction`` -- the share of real wall time the
        loop spent blocked on input.  ``wall_s`` is measured by one clock
        spanning the whole loop rather than summed, so ``unaccounted_s``
        exposes anything the per-step breakdown missed, the monitor's own
        overhead included; it should stay near zero.

        ``first_wait_s`` is usually much the largest wait, because it pays for
        worker start-up.  Pass ``persistent_workers=True`` through
        :func:`make_dataloader` to stop paying it once per epoch.
        """
        n = max(self.steps, 1)
        out = {f"total_{k}": v for k, v in self.totals.items()}
        out.update({f"mean_{k}": v / n for k, v in self.totals.items()})
        out["steps"] = self.steps
        out["events"] = self.events
        out["points"] = self.points
        out["payload_mb"] = self.payload_mb
        out["max_wait_s"] = self.max_wait_s
        out["first_wait_s"] = self.first_wait_s or 0.0

        elapsed = self.totals["step_s"]
        if elapsed > 0:
            out["events_per_s"] = self.events / elapsed
            out["points_per_s"] = self.points / elapsed
            out["stall_fraction"] = self.totals["wait_s"] / elapsed
        if self._t_start is not None:
            wall = self._t_last - self._t_start
            out["wall_s"] = wall
            out["unaccounted_s"] = wall - elapsed
        return out

    def finish(self) -> dict:
        """Write :meth:`summary` to the W&B run summary and return it."""
        s = self.summary()
        self.logger.summary(s)
        return s
