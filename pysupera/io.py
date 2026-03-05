"""
HDF5 serialisation for collections of :class:`~pysupera.data.Particle`
objects organised as *events* (one event = a list of particles).

File layout
-----------
Two-level CSR (Compressed Sparse Row) encoding avoids both ragged HDF5 groups
and slow variable-length heap allocations.

.. code-block:: text

    /format_version        scalar str  — semver for forward compatibility
    /n_events              scalar int64

    /events/
        offsets            (n_events + 1,)        int64
            events/offsets[i] : events/offsets[i+1]  — slice into /particles/*
            that belongs to event i.

    /particles/
        id                 (n_total_particles,)   int32
        parent_id          (n_total_particles,)   int32
        ancestor_id        (n_total_particles,)   int32
        pdg                (n_total_particles,)   int32
        parent_pdg         (n_total_particles,)   int32
        process_type       (n_total_particles,)   int32
            Raw integer passed to SetSemanticType; sem_type is re-derived on
            read so the stored value is always the ground-truth input.
        pc_offsets         (n_total_particles + 1,) int64
            particles/pc_offsets[j] : particles/pc_offsets[j+1]  — row slice
            into /points/flat that belongs to particle j.

    /points/
        flat               (n_total_points, _PC_NDIM)    float32
            All point clouds concatenated in particle order.

Random access to event i
------------------------
1.  p_start, p_end = events/offsets[i : i+2]              (2 ints)
2.  Read scalar columns particles/*[p_start : p_end]       (6 contiguous reads)
3.  pc_bounds = particles/pc_offsets[p_start : p_end + 1]  (p_end-p_start+1 ints)
4.  pt_start, pt_end = pc_bounds[0], pc_bounds[-1]
5.  flat_chunk = points/flat[pt_start : pt_end]            (1 contiguous read)
6.  Slice flat_chunk using (pc_bounds - pt_start) as local offsets.

Total I/O per event: O(n_particles) metadata + O(n_points) point data.
Both offset arrays (events/offsets, particles/pc_offsets) are tiny and
cached in RAM after the file is opened.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

FORMAT_VERSION = "1.0.0"

# Chunk sizes for compressed HDF5 datasets.
# Tuned for typical event sizes; adjust if your events are significantly
# larger or smaller.
_CHUNK_PARTICLES = 256     # rows per chunk in particle metadata arrays
_CHUNK_POINTS    = 65536   # rows per chunk in the flat point array
_PC_NDIM = 5               # number of columns in the flat point array (x,y,z,t,e)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def write_events(path: str,
                 events: List[List],
                 compression: str = "lz4",
                 compression_opts: Optional[int] = None) -> None:
    """
    Write a list of events to an HDF5 file.

    Each event is a list of :class:`~pysupera.data.Particle` objects.
    The file is created (or overwritten) at *path*.

    Parameters
    ----------
    path : str
        Output file path.  Existing files are overwritten.
    events : list of list of Particle
        Outer list indexes events; inner list contains the particles for
        that event.  An event may be an empty list.
    compression : str, optional
        HDF5 filter name accepted by h5py (``"lz4"``, ``"gzip"``,
        ``"lzf"``, ``None``).  Default ``"lz4"`` requires the
        ``hdf5plugin`` package; fall back to ``"gzip"`` if unavailable.
    compression_opts : int or None, optional
        Compression level (filter-specific).  ``None`` uses the filter's
        default.

    Notes
    -----
    ``sem_type`` is **not** stored: it is deterministically derived from
    ``process_type``, ``pdg``, ``parent_pdg``, and ``point_cloud`` by
    :func:`~pysupera.utils.SetSemanticType` and is re-created on read.
    Storing the raw ``process_type`` integer is both smaller and more
    faithful to the simulation ground truth.

    Examples
    --------
    >>> from pysupera.io import write_events, read_events
    >>> write_events("data.h5", [[p1, p2], [p3]])
    """
    import h5py

    # Resolve compression kwargs once
    ckw = _compress_kwargs(compression, compression_opts)

    # --------------- flatten all particles and point clouds ----------------
    n_events = len(events)

    # Event-level offsets  (fencepost: n_events + 1 entries)
    ev_offsets = np.zeros(n_events + 1, dtype=np.int64)
    for i, ev in enumerate(events):
        ev_offsets[i + 1] = ev_offsets[i] + len(ev)

    n_total_particles = int(ev_offsets[-1])

    # Particle-level scalars
    p_id          = np.empty(n_total_particles, dtype=np.int32)
    p_parent_id   = np.empty(n_total_particles, dtype=np.int32)
    p_ancestor_id = np.empty(n_total_particles, dtype=np.int32)
    p_pdg         = np.empty(n_total_particles, dtype=np.int32)
    p_parent_pdg  = np.empty(n_total_particles, dtype=np.int32)
    p_proc_type   = np.empty(n_total_particles, dtype=np.int32)

    # Particle → point offsets  (fencepost: n_total_particles + 1 entries)
    pc_offsets = np.zeros(n_total_particles + 1, dtype=np.int64)

    j = 0
    for ev in events:
        for p in ev:
            p_id[j]          = p.id
            p_parent_id[j]   = p.parent_id
            p_ancestor_id[j] = p.ancestor_id
            p_pdg[j]         = p.pdg
            p_parent_pdg[j]  = p.parent_pdg
            # process_type is stored as raw int (0-based, as passed to
            # SetSemanticType before the +1 inside that function).
            p_proc_type[j]   = int(p._process_type)
            pc_offsets[j + 1] = pc_offsets[j] + len(p.point_cloud)
            j += 1

    n_total_points = int(pc_offsets[-1])

    # Flat point array
    flat_points = np.empty((n_total_points, _PC_NDIM), dtype=np.float32)
    j = 0
    for ev in events:
        for p in ev:
            if pc_offsets[j] < pc_offsets[j + 1]:
                cloud = _normalize_cloud(p.point_cloud)
                flat_points[pc_offsets[j] : pc_offsets[j + 1]] = (
                    cloud[:, :_PC_NDIM]
                )
            j += 1

    # --------------- write to HDF5 -----------------------------------------
    with h5py.File(path, "w") as f:
        f.create_dataset("format_version", data=FORMAT_VERSION)
        f.create_dataset("n_events",       data=np.int64(n_events))

        # Event offsets (small — no compression needed)
        eg = f.create_group("events")
        eg.create_dataset("offsets", data=ev_offsets)

        # Particle metadata
        pg = f.create_group("particles")
        _mk = lambda name, arr: pg.create_dataset(
            name, data=arr,
            chunks=(min(_CHUNK_PARTICLES, max(1, n_total_particles)),),
            **ckw,
        )
        if n_total_particles > 0:
            _mk("id",           p_id)
            _mk("parent_id",    p_parent_id)
            _mk("ancestor_id",  p_ancestor_id)
            _mk("pdg",          p_pdg)
            _mk("parent_pdg",   p_parent_pdg)
            _mk("process_type", p_proc_type)
        else:
            for name in ("id", "parent_id", "ancestor_id",
                         "pdg", "parent_pdg", "process_type"):
                pg.create_dataset(name, data=np.empty(0, dtype=np.int32))
        pg.create_dataset("pc_offsets", data=pc_offsets)

        # Flat point cloud
        ptg = f.create_group("points")
        if n_total_points > 0:
            ptg.create_dataset(
                "flat", data=flat_points,
                chunks=(min(_CHUNK_POINTS, n_total_points), _PC_NDIM),
                **ckw,
            )
        else:
            ptg.create_dataset("flat",
                               data=np.empty((0, _PC_NDIM), dtype=np.float32))


def open_writer(path: str,
                mode: str = "w",
                compression: str = "lz4",
                compression_opts: Optional[int] = None) -> "EventWriter":
    """
    Open an :class:`EventWriter` for incremental event writing.

    Parameters
    ----------
    path : str
        Output file path.
    mode : {"w", "a"}, optional
        ``"w"`` creates a new file (overwrites if it exists).  ``"a"``
        appends to an existing file created by this library; raises
        ``FileNotFoundError`` if the file does not exist.
    compression : str, optional
        Same as :func:`write_events`.
    compression_opts : int or None, optional
        Same as :func:`write_events`.

    Returns
    -------
    EventWriter

    Examples
    --------
    >>> with open_writer("data.h5") as w:
    ...     for particles in source:
    ...         w.append_event(particles)
    """
    return EventWriter(path, mode=mode,
                       compression=compression,
                       compression_opts=compression_opts)


# ---------------------------------------------------------------------------
# EventWriter
# ---------------------------------------------------------------------------

class EventWriter:
    """
    Incremental writer for HDF5 particle event files.

    Writes one event at a time via :meth:`append_event`.  All datasets are
    created with ``maxshape=(None, ...)`` so they can grow without bound.
    The file is finalised (``n_events`` scalar updated) on :meth:`close`.

    Must be used as a context manager or :meth:`close` must be called
    explicitly — failing to do so will leave ``n_events`` at 0 and the
    cached offset arrays incomplete.

    Parameters
    ----------
    path : str
        Output file path.
    mode : {"w", "a"}, optional
        ``"w"`` (default) creates or overwrites.  ``"a"`` appends to an
        existing file written by this library.
    compression : str, optional
        HDF5 filter name.  Same options as :func:`write_events`.
    compression_opts : int or None, optional
        Compression level.  Same as :func:`write_events`.

    Examples
    --------
    Create a new file and append events one by one::

        with open_writer("out.h5") as w:
            for particles in simulation:
                w.append_event(particles)

    Append to an existing file::

        with open_writer("out.h5", mode="a") as w:
            w.append_event(more_particles)
    """

    # Names of the six particle scalar fields in order
    _SCALAR_FIELDS = (
        "id", "parent_id", "ancestor_id", "pdg", "parent_pdg", "process_type"
    )

    def __init__(self, path: str, mode: str = "w",
                 compression: str = "lz4",
                 compression_opts: Optional[int] = None) -> None:
        import h5py

        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")

        self._ckw = _compress_kwargs(compression, compression_opts)
        self._path = path

        if mode == "a":
            import os
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Append mode requested but file not found: {path!r}.  "
                    f"Use mode='w' to create a new file."
                )
            self._f = h5py.File(path, "r+")
            # Read back running counters from the existing file
            self._n_events      = int(self._f["n_events"][()])
            self._n_particles   = int(self._f["events/offsets"][-1])
            self._n_points      = int(self._f["particles/pc_offsets"][-1])
        else:
            self._f = h5py.File(path, "w")
            self._n_events    = 0
            self._n_particles = 0
            self._n_points    = 0
            self._init_datasets()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Finalise and close the file.  Idempotent."""
        if self._f.id.valid:
            self._f["n_events"][()] = np.int64(self._n_events)
            self._f.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_event(self, particles: List) -> None:
        """
        Append one event (a list of :class:`~pysupera.data.Particle`
        objects) to the file.

        The event may be empty (zero particles).  Each call extends all
        datasets by the required number of rows in O(n_particles +
        n_points) time with no full-file rewrite.

        Parameters
        ----------
        particles : list of Particle
            Particles belonging to this event.  May be empty.
        """
        n_p = len(particles)
        n_pt = sum(len(p.point_cloud) for p in particles)

        # ---- extend events/offsets by one fencepost entry -----------------
        ev_ds = self._f["events/offsets"]
        ev_ds.resize(self._n_events + 2, axis=0)
        ev_ds[self._n_events + 1] = np.int64(self._n_particles + n_p)

        if n_p > 0:
            # ---- extend particle scalar datasets --------------------------
            new_p_start = self._n_particles
            new_p_end   = self._n_particles + n_p
            for name in self._SCALAR_FIELDS:
                ds = self._f[f"particles/{name}"]
                ds.resize(new_p_end, axis=0)

            # Build scalar arrays for this event
            p_id          = np.empty(n_p, dtype=np.int32)
            p_parent_id   = np.empty(n_p, dtype=np.int32)
            p_ancestor_id = np.empty(n_p, dtype=np.int32)
            p_pdg         = np.empty(n_p, dtype=np.int32)
            p_parent_pdg  = np.empty(n_p, dtype=np.int32)
            p_proc_type   = np.empty(n_p, dtype=np.int32)
            pc_lengths    = np.empty(n_p, dtype=np.int64)

            for k, p in enumerate(particles):
                p_id[k]          = p.id
                p_parent_id[k]   = p.parent_id
                p_ancestor_id[k] = p.ancestor_id
                p_pdg[k]         = p.pdg
                p_parent_pdg[k]  = p.parent_pdg
                p_proc_type[k]   = int(p._process_type)
                pc_lengths[k]    = len(p.point_cloud)

            self._f["particles/id"          ][new_p_start:new_p_end] = p_id
            self._f["particles/parent_id"   ][new_p_start:new_p_end] = p_parent_id
            self._f["particles/ancestor_id" ][new_p_start:new_p_end] = p_ancestor_id
            self._f["particles/pdg"         ][new_p_start:new_p_end] = p_pdg
            self._f["particles/parent_pdg"  ][new_p_start:new_p_end] = p_parent_pdg
            self._f["particles/process_type"][new_p_start:new_p_end] = p_proc_type

            # ---- extend pc_offsets (n_particles+1 entries total) ----------
            pc_ds = self._f["particles/pc_offsets"]
            pc_ds.resize(new_p_end + 1, axis=0)
            # cumulative offsets starting from current total point count
            new_pc_offsets = np.empty(n_p, dtype=np.int64)
            running = self._n_points
            for k in range(n_p):
                running += pc_lengths[k]
                new_pc_offsets[k] = running
            pc_ds[new_p_start + 1 : new_p_end + 1] = new_pc_offsets

            # ---- extend flat point array ----------------------------------
            if n_pt > 0:
                pt_ds = self._f["points/flat"]
                new_pt_start = self._n_points
                pt_ds.resize(new_pt_start + n_pt, axis=0)
                cursor = new_pt_start
                for p in particles:
                    n = len(p.point_cloud)
                    if n > 0:
                        cloud = _normalize_cloud(p.point_cloud)
                        pt_ds[cursor : cursor + n] = cloud[:, :_PC_NDIM]
                    cursor += n

        # ---- advance counters ---------------------------------------------
        self._n_events    += 1
        self._n_particles += n_p
        self._n_points    += n_pt

    @property
    def n_events(self) -> int:
        """Number of events written so far."""
        return self._n_events

    def __repr__(self) -> str:
        status = "open" if self._f.id.valid else "closed"
        return (f"EventWriter(path={self._path!r}, "
                f"n_events={self._n_events}, {status})")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_datasets(self) -> None:
        """Create all resizable datasets and write the initial fencepost."""
        f   = self._f
        ckw = self._ckw

        f.create_dataset("format_version", data=FORMAT_VERSION)
        # Placeholder — updated to real count in close()
        f.create_dataset("n_events", data=np.int64(0))

        # events/offsets — starts with a single fencepost 0
        eg = f.create_group("events")
        eg.create_dataset(
            "offsets",
            data=np.zeros(1, dtype=np.int64),
            maxshape=(None,),
            chunks=(_CHUNK_PARTICLES,),
        )

        # particle scalar datasets — start empty
        pg = f.create_group("particles")
        for name in self._SCALAR_FIELDS:
            pg.create_dataset(
                name,
                shape=(0,), maxshape=(None,), dtype=np.int32,
                chunks=(_CHUNK_PARTICLES,),
                **ckw,
            )
        # pc_offsets — single fencepost 0
        pg.create_dataset(
            "pc_offsets",
            data=np.zeros(1, dtype=np.int64),
            maxshape=(None,),
            chunks=(_CHUNK_PARTICLES,),
        )

        # flat point array — start empty
        ptg = f.create_group("points")
        ptg.create_dataset(
            "flat",
            shape=(0, _PC_NDIM), maxshape=(None, _PC_NDIM), dtype=np.float32,
            chunks=(_CHUNK_POINTS, _PC_NDIM),
            **ckw,
        )


def read_events(path: str) -> "EventStore":
    """
    Open an HDF5 event file and return an :class:`EventStore`.

    The returned object supports ``len()`` and integer indexing
    (``store[i]``) and can be used as a context manager::

        with read_events("data.h5") as store:
            particles = store[0]   # list[Particle] for event 0

    Parameters
    ----------
    path : str
        Path to an HDF5 file written by :func:`write_events`.

    Returns
    -------
    EventStore
    """
    return EventStore(path)


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------

class EventStore:
    """
    Random-access reader for an HDF5 particle event file.

    Supports ``len(store)`` and ``store[i]`` (integer index, negative
    indices and slices are not currently supported).  Use as a context
    manager to ensure the file is closed::

        with EventStore("data.h5") as store:
            ev = store[42]   # list[Particle]

    Both offset arrays (``events/offsets`` and ``particles/pc_offsets``)
    are loaded into RAM on construction; they are tiny (O(n_events) and
    O(n_total_particles) int64 values) and make random-access O(1) in
    metadata reads per event.

    Parameters
    ----------
    path : str
        Path to HDF5 file written by :func:`write_events`.
    """

    def __init__(self, path: str) -> None:
        import h5py
        self._path = path
        self._f = h5py.File(path, "r")
        self._n_events: int = int(self._f["n_events"][()])

        # Cache both offset arrays — O(n_events) + O(n_particles) ints
        self._ev_offsets: np.ndarray = self._f["events/offsets"][:]
        self._pc_offsets: np.ndarray = self._f["particles/pc_offsets"][:]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        if self._f.id.valid:
            self._f.close()

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_events

    def __getitem__(self, index: int) -> List:
        """
        Return the list of :class:`~pysupera.data.Particle` objects
        for event *index*.

        Parameters
        ----------
        index : int
            Event index in [0, len(store)).

        Returns
        -------
        list of Particle
        """
        from .data import Particle

        if index < 0 or index >= self._n_events:
            raise IndexError(
                f"Event index {index} out of range [0, {self._n_events})"
            )

        # ---- particle range for this event --------------------------------
        p_start = int(self._ev_offsets[index])
        p_end   = int(self._ev_offsets[index + 1])
        n_parts = p_end - p_start

        if n_parts == 0:
            return []

        # ---- read scalar metadata in 6 contiguous array slices ------------
        ids          = self._f["particles/id"          ][p_start:p_end]
        parent_ids   = self._f["particles/parent_id"   ][p_start:p_end]
        ancestor_ids = self._f["particles/ancestor_id" ][p_start:p_end]
        pdgs         = self._f["particles/pdg"         ][p_start:p_end]
        parent_pdgs  = self._f["particles/parent_pdg"  ][p_start:p_end]
        proc_types   = self._f["particles/process_type"][p_start:p_end]

        # ---- read point-cloud data in one contiguous slice ----------------
        pc_bounds = self._pc_offsets[p_start : p_end + 1]  # (n_parts+1,)
        pt_start  = int(pc_bounds[0])
        pt_end    = int(pc_bounds[-1])
        flat_chunk = self._f["points/flat"][pt_start:pt_end]  # (n_pts, _PC_NDIM)

        # ---- reconstruct Particle objects ----------------------------------
        local_offsets = pc_bounds - pt_start  # zero-indexed within flat_chunk
        particles = []
        for k in range(n_parts):
            cloud = flat_chunk[local_offsets[k] : local_offsets[k + 1]]
            p = Particle(
                id           = int(ids[k]),
                parent_id    = int(parent_ids[k]),
                ancestor_id  = int(ancestor_ids[k]),
                pdg          = int(pdgs[k]),
                parent_pdg   = int(parent_pdgs[k]),
                process_type = int(proc_types[k]),
                point_cloud  = cloud,
            )
            particles.append(p)

        return particles

    def __repr__(self) -> str:
        status = "open" if self._f.id.valid else "closed"
        return (f"EventStore(path={self._path!r}, "
                f"n_events={self._n_events}, {status})")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def iter_events(self):
        """
        Iterate over all events in file order.

        Yields
        ------
        list of Particle
        """
        for i in range(self._n_events):
            yield self[i]

    @property
    def n_events(self) -> int:
        """Number of events in the file."""
        return self._n_events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_cloud(pc) -> np.ndarray:
    """
    Return *pc* as a 2-D float32 array with shape ``(N, _PC_NDIM)``.

    Handles the two common empty-cloud representations:

    * A 1-D array of length 0 (e.g. ``np.array([])``) — converted to
      ``np.empty((0, _PC_NDIM), float32)``.
    * A properly-shaped 2-D array with 0 rows — returned as-is (after
      dtype cast).

    Non-empty 2-D arrays are returned unchanged (modulo dtype cast).

    Raises
    ------
    ValueError
        If *pc* is 1-D **and** non-empty (ambiguous dimensionality) or has
        more than 2 dimensions.
    """
    pc = np.asarray(pc)
    if pc.ndim == 1:
        if pc.shape[0] != 0:
            raise ValueError(
                f"point_cloud has unexpected 1-D shape {pc.shape}; "
                "expected a 2-D array (N, columns) or an empty 1-D array."
            )
        return np.empty((0, _PC_NDIM), dtype=np.float32)
    if pc.ndim != 2:
        raise ValueError(
            f"point_cloud must be a 2-D array, got shape {pc.shape}."
        )
    return pc.astype(np.float32)


def _compress_kwargs(compression: Optional[str],
                     compression_opts: Optional[int]) -> dict:
    """
    Build h5py dataset keyword arguments for the requested filter.

    Falls back gracefully if ``hdf5plugin`` is unavailable (required for
    lz4/blosc) and the user requested lz4.
    """
    if compression is None:
        return {}

    if compression == "lz4":
        try:
            import hdf5plugin  # noqa: F401 — registers the filter
            return {"compression": "lz4"}
        except ImportError:
            import warnings
            warnings.warn(
                "hdf5plugin not installed; falling back to gzip compression. "
                "Install with: pip install hdf5plugin",
                stacklevel=3,
            )
            compression = "gzip"

    kwargs: dict = {"compression": compression}
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    return kwargs
