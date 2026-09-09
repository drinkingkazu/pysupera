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
        root_id            (n_total_particles,)   int32
        pdg                (n_total_particles,)   int32
        parent_pdg         (n_total_particles,)   int32
        interaction_type   (n_total_particles,)   int32
            Raw InteractionType integer value (1-based enum value).
        sem_type           (n_total_particles,)   int8
            SemanticType integer value stored for fast filtered reads
            without re-deriving at load time.
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

import os
import numpy as np
from typing import List, Optional

# Register HDF5 compression filters (LZ4, Blosc, etc.) if available.
# Importing hdf5plugin is enough — it auto-registers all supported filters
# with the HDF5 library so h5py can transparently read/write them.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

FORMAT_VERSION = "2.2.0"

_PC_NDIM = 5                # number of columns in the flat point array (x,y,z,t,e)
_CLOUD_NDIM = 9             # columns in event-cloud datasets (x,y,z,t,e,interaction_id,root_id,frag_id,inst_id)

# Chunk sizing for compressed HDF5 datasets.
#
# Chunk shapes are derived from a *byte budget* rather than a fixed row count,
# because the right row count depends on how wide the row is.  HDF5 reads and
# decompresses a whole chunk to satisfy any read that touches it, so the budget
# is what actually governs read cost.  The HDF5 docs recommend 10 KiB - 1 MiB.
#
# Two budgets, because the two dataset families have different shapes:
#   * point/cloud arrays — wide rows, millions of them; a larger budget keeps
#     the chunk count down without making single-event reads expensive.
#   * metadata columns   — 1-D, 4 bytes per row, and there are ~20 of them per
#     file.  A smaller budget bounds the padding waste on short runs, since the
#     streaming writer creates these datasets empty and cannot know the final
#     length (a chunk longer than the dataset is stored as padding).
_CHUNK_TARGET_BYTES      = 256 * 1024   # point / cloud arrays
_CHUNK_TARGET_BYTES_META =  64 * 1024   # 1-D metadata columns


def _chunk_rows(ncols: int, itemsize: int = 4,
                target_bytes: int = _CHUNK_TARGET_BYTES) -> int:
    """Rows per chunk so that one chunk is about *target_bytes*."""
    return max(1, target_bytes // max(1, ncols * itemsize))


_CHUNK_PARTICLES = _chunk_rows(1, target_bytes=_CHUNK_TARGET_BYTES_META)  # 16384
_CHUNK_POINTS    = _chunk_rows(_PC_NDIM)                                 # 13107
_CHUNK_CLOUD     = _chunk_rows(_CLOUD_NDIM)                              #  7281

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def write_events(path: str,
                 events: List[List],
                 compression: str = "lzf",
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
    Both ``interaction_type`` (raw ``InteractionType`` int32) and
    ``sem_type`` (``SemanticType`` int8) are stored.  ``sem_type`` is
    written directly from the in-memory value so filtered reads do not
    need to re-derive it; ``interaction_type`` is the ground-truth
    physics process integer (1-based enum value).

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
    p_root_id     = np.empty(n_total_particles, dtype=np.int32)
    p_pdg         = np.empty(n_total_particles, dtype=np.int32)
    p_parent_pdg  = np.empty(n_total_particles, dtype=np.int32)
    p_itype       = np.empty(n_total_particles, dtype=np.int32)
    p_sem_type    = np.empty(n_total_particles, dtype=np.int8)

    # Particle → point offsets  (fencepost: n_total_particles + 1 entries)
    pc_offsets = np.zeros(n_total_particles + 1, dtype=np.int64)

    j = 0
    for ev in events:
        for p in ev:
            p_id[j]          = p.id
            p_parent_id[j]   = p.parent_id
            p_root_id[j]     = p.root_id
            p_pdg[j]         = p.pdg
            p_parent_pdg[j]  = p.parent_pdg
            p_itype[j]       = int(p._interaction_type)
            p_sem_type[j]    = int(p.sem_type.value)
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
            _mk("id",               p_id)
            _mk("parent_id",        p_parent_id)
            _mk("root_id",          p_root_id)
            _mk("pdg",              p_pdg)
            _mk("parent_pdg",       p_parent_pdg)
            _mk("interaction_type", p_itype)
            pg.create_dataset(
                "sem_type", data=p_sem_type,
                chunks=(min(_CHUNK_PARTICLES, max(1, n_total_particles)),),
                **ckw,
            )
        else:
            for name in ("id", "parent_id", "root_id",
                         "pdg", "parent_pdg", "interaction_type"):
                pg.create_dataset(name, data=np.empty(0, dtype=np.int32))
            pg.create_dataset("sem_type", data=np.empty(0, dtype=np.int8))
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
                compression: str = "lzf",
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

    # Names of the int32 particle scalar fields (sem_type handled separately
    # because it uses dtype=int8)
    _SCALAR_FIELDS = (
        "id", "geant4_id", "parent_id", "root_id", "pdg", "parent_pdg",
        "interaction_id", "interaction_type"
    )

    def __init__(self, path: str, mode: str = "w",
                 compression: str = "lzf",
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
            # Fragment / instance counters (0 if the groups don't yet exist)
            if "particle_fragments/pc_offsets" in self._f:
                self._n_fragments    = int(self._f["frag_events/offsets"][-1])
                self._n_frag_points  = int(self._f["particle_fragments/pc_offsets"][-1])
                self._n_frag_members = int(self._f["particle_fragments/member_offsets"][-1])
            else:
                self._n_fragments = self._n_frag_points = self._n_frag_members = 0
            if "particle_instances/pc_offsets" in self._f:
                self._n_instances    = int(self._f["inst_events/offsets"][-1])
                self._n_inst_points  = int(self._f["particle_instances/pc_offsets"][-1])
                self._n_inst_members = int(self._f["particle_instances/member_offsets"][-1])
            else:
                self._n_instances = self._n_inst_points = self._n_inst_members = 0
            # Event-cloud counters
            self._n_event_cloud_pts   = int(self._f["non_le_cloud/offsets"][-1])     if "non_le_cloud/offsets"    in self._f else 0
            self._n_le_cloud_pts      = int(self._f["le_scatter_cloud/offsets"][-1]) if "le_scatter_cloud/offsets" in self._f else 0
        else:
            self._f = h5py.File(path, "w")
            self._n_events    = 0
            self._n_particles = 0
            self._n_points    = 0
            self._n_fragments = self._n_frag_points = self._n_frag_members = 0
            self._n_instances = self._n_inst_points = self._n_inst_members = 0
            self._n_event_cloud_pts = 0
            self._n_le_cloud_pts    = 0
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
            self._f["particles/sem_type"].resize(new_p_end, axis=0)

            # Build scalar arrays for this event
            p_id             = np.empty(n_p, dtype=np.int32)
            p_geant4_id      = np.empty(n_p, dtype=np.int32)
            p_parent_id      = np.empty(n_p, dtype=np.int32)
            p_root_id        = np.empty(n_p, dtype=np.int32)
            p_pdg            = np.empty(n_p, dtype=np.int32)
            p_parent_pdg     = np.empty(n_p, dtype=np.int32)
            p_int_id         = np.empty(n_p, dtype=np.int32)
            p_itype          = np.empty(n_p, dtype=np.int32)
            p_sem_type       = np.empty(n_p, dtype=np.int8)
            pc_lengths       = np.empty(n_p, dtype=np.int64)

            for k, p in enumerate(particles):
                p_id[k]          = p.id
                p_geant4_id[k]   = getattr(p, "geant4_id", p.id)
                p_parent_id[k]   = p.parent_id
                p_root_id[k]     = p.root_id
                p_pdg[k]         = p.pdg
                p_parent_pdg[k]  = p.parent_pdg
                p_int_id[k]      = int(p._interaction_id)
                p_itype[k]       = int(p._interaction_type)
                p_sem_type[k]    = int(p.sem_type.value)
                pc_lengths[k]    = len(p.point_cloud)

            self._f["particles/id"               ][new_p_start:new_p_end] = p_id
            self._f["particles/geant4_id"        ][new_p_start:new_p_end] = p_geant4_id
            self._f["particles/parent_id"        ][new_p_start:new_p_end] = p_parent_id
            self._f["particles/root_id"          ][new_p_start:new_p_end] = p_root_id
            self._f["particles/pdg"              ][new_p_start:new_p_end] = p_pdg
            self._f["particles/parent_pdg"       ][new_p_start:new_p_end] = p_parent_pdg
            self._f["particles/interaction_id"   ][new_p_start:new_p_end] = p_int_id
            self._f["particles/interaction_type" ][new_p_start:new_p_end] = p_itype
            self._f["particles/sem_type"         ][new_p_start:new_p_end] = p_sem_type

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

            # ---- extend flat point array (single bulk write) --------------
            if n_pt > 0:
                pt_ds = self._f["points/flat"]
                new_pt_start = self._n_points
                pt_ds.resize(new_pt_start + n_pt, axis=0)
                bulk = np.empty((n_pt, _PC_NDIM), dtype=np.float32)
                cursor = 0
                for p in particles:
                    n = len(p.point_cloud)
                    if n > 0:
                        bulk[cursor : cursor + n] = _normalize_cloud(p.point_cloud)
                    cursor += n
                pt_ds[new_pt_start : new_pt_start + n_pt] = bulk

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

        # particle scalar datasets — start empty (int32)
        pg = f.create_group("particles")
        for name in self._SCALAR_FIELDS:
            pg.create_dataset(
                name,
                shape=(0,), maxshape=(None,), dtype=np.int32,
                chunks=(_CHUNK_PARTICLES,),
                **ckw,
            )
        # sem_type — int8
        pg.create_dataset(
            "sem_type",
            shape=(0,), maxshape=(None,), dtype=np.int8,
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

        # Representative-level groups: particle_fragments and particle_instances
        self._init_rep_group("particle_fragments")
        self._init_rep_group("particle_instances")

        # Event-level union point clouds
        # Layout: (x, y, z, t, energy, interaction_id, root_id, frag_id, inst_id)
        #          — _CLOUD_NDIM = 9 columns (all stored as float32).
        # non_le_cloud  : union of point clouds from particles with sem_type != kLEScatter.
        # le_scatter_cloud : union of point clouds from kLEScatter particles only.
        for grp_name in ("non_le_cloud", "le_scatter_cloud"):
            g = f.create_group(grp_name)
            g.create_dataset(
                "offsets",
                data=np.zeros(1, dtype=np.int64),
                maxshape=(None,),
                chunks=(_CHUNK_PARTICLES,),
            )
            g.create_dataset(
                "flat",
                shape=(0, _CLOUD_NDIM), maxshape=(None, _CLOUD_NDIM), dtype=np.float32,
                chunks=(_CHUNK_CLOUD, _CLOUD_NDIM),
                **ckw,
            )

    def _init_rep_group(self, group_prefix: str) -> None:
        """Create resizable datasets for one representative level.

        Layout (example for ``group_prefix="particle_fragments"``)::

            /frag_events/offsets           (n_events+1,) int64   — per-event fencepost
            /particle_fragments/id         (n_reps,)     int32
            /particle_fragments/parent_id  (n_reps,)     int32
            /particle_fragments/root_id    (n_reps,)     int32
            /particle_fragments/pdg        (n_reps,)     int32
            /particle_fragments/parent_pdg (n_reps,)     int32
            /particle_fragments/interaction_type (n_reps,) int32
            /particle_fragments/sem_type   (n_reps,)     int8
            /particle_fragments/pc_offsets (n_reps+1,)   int64  — fencepost into *_points/flat
            /particle_fragments/member_offsets (n_reps+1,) int64 — fencepost into *_members/flat
            /particle_fragments_points/flat (total_pts, dims) float32
            /particle_fragments_members/flat (total_members,) int32
        """
        f   = self._f
        ckw = self._ckw

        # Derive event-offset group name from prefix
        # "particle_fragments" → "frag_events", "particle_instances" → "inst_events"
        short = group_prefix.split("_", 1)[1]  # "fragments" or "instances"
        ev_group = f"{short[:4]}_events"        # "frag_events" or "inst_events"

        eg = f.create_group(ev_group)
        eg.create_dataset(
            "offsets",
            data=np.zeros(1, dtype=np.int64),
            maxshape=(None,),
            chunks=(_CHUNK_PARTICLES,),
        )

        pg = f.create_group(group_prefix)
        for name in self._SCALAR_FIELDS:
            pg.create_dataset(
                name,
                shape=(0,), maxshape=(None,), dtype=np.int32,
                chunks=(_CHUNK_PARTICLES,),
                **ckw,
            )
        pg.create_dataset(
            "sem_type",
            shape=(0,), maxshape=(None,), dtype=np.int8,
            chunks=(_CHUNK_PARTICLES,),
            **ckw,
        )
        for _name in ("parent_frag_id", "parent_inst_id"):
            pg.create_dataset(
                _name,
                shape=(0,), maxshape=(None,), dtype=np.int32,
                chunks=(_CHUNK_PARTICLES,),
                **ckw,
            )
        pg.create_dataset(
            "pc_offsets",
            data=np.zeros(1, dtype=np.int64),
            maxshape=(None,),
            chunks=(_CHUNK_PARTICLES,),
        )
        pg.create_dataset(
            "member_offsets",
            data=np.zeros(1, dtype=np.int64),
            maxshape=(None,),
            chunks=(_CHUNK_PARTICLES,),
        )

        pts_g = f.create_group(f"{group_prefix}_points")
        pts_g.create_dataset(
            "flat",
            shape=(0, _PC_NDIM), maxshape=(None, _PC_NDIM), dtype=np.float32,
            chunks=(_CHUNK_POINTS, _PC_NDIM),
            **ckw,
        )

        mem_g = f.create_group(f"{group_prefix}_members")
        mem_g.create_dataset(
            "flat",
            shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(_CHUNK_PARTICLES,),
            **ckw,
        )

    # ------------------------------------------------------------------
    # Public API: representative levels
    # ------------------------------------------------------------------

    def append_fragments(self, reps: List) -> None:
        """Append step-1 fragment representatives for the current event.

        Must be called **before** :meth:`append_event` for the same event
        (both methods share the event counter; only :meth:`append_event`
        increments it).

        Parameters
        ----------
        reps : list of Particle
            Step-1 representative particles with ``member_ids`` populated.
        """
        self._append_rep_level(
            "particle_fragments",
            reps,
            "_n_fragments", "_n_frag_points", "_n_frag_members",
        )

    def append_instances(self, reps: List) -> None:
        """Append step-2 shower-instance representatives for the current event.

        Must be called **before** :meth:`append_event` for the same event.

        Parameters
        ----------
        reps : list of Particle
            Step-2 instance representatives with ``member_ids`` populated.
        """
        self._append_rep_level(
            "particle_instances",
            reps,
            "_n_instances", "_n_inst_points", "_n_inst_members",
        )

    def append_event_clouds(
        self,
        non_le_cloud: "np.ndarray",
        le_cloud:     "np.ndarray",
    ) -> None:
        """Append event-level union point clouds for the current event.

        Must be called **before** :meth:`append_event` for the same event.

        Each cloud has 9 columns:
        ``x, y, z, t, energy, interaction_id, root_id, frag_id, inst_id``.
        All values are stored as float32.  Integer-valued columns
        (``interaction_id``, ``root_id``, ``frag_id``, ``inst_id``) should be
        cast by the caller.  Use ``-1`` as a sentinel for absent frag/inst IDs.
        When voxelization is active the spatial columns are voxel-centre
        coordinates; ``interaction_id`` / ``root_id`` are taken from the
        dominant particle across merged points.

        Parameters
        ----------
        non_le_cloud : np.ndarray, shape (N, 9)
            Union of point clouds from particles whose ``sem_type`` is **not**
            ``kLEScatter`` for this event (already voxelized if applicable).
            May be empty (shape ``(0, 9)``).
        le_cloud : np.ndarray, shape (M, 9)
            Union of point clouds from ``kLEScatter`` particles for this event
            (already voxelized if applicable).  May be empty.
        """
        for cloud, grp, n_attr in (
            (non_le_cloud, "non_le_cloud",      "_n_event_cloud_pts"),
            (le_cloud,     "le_scatter_cloud",  "_n_le_cloud_pts"),
        ):
            n_prev = getattr(self, n_attr)
            n_pts  = len(cloud)

            off_ds = self._f[f"{grp}/offsets"]
            off_ds.resize(self._n_events + 2, axis=0)
            off_ds[self._n_events + 1] = np.int64(n_prev + n_pts)

            if n_pts > 0:
                flat_ds = self._f[f"{grp}/flat"]
                flat_ds.resize(n_prev + n_pts, axis=0)
                # Ensure float32 and exactly _CLOUD_NDIM columns.
                c = np.asarray(cloud, dtype=np.float32)
                if c.shape[1] < _CLOUD_NDIM:
                    pad = np.zeros((len(c), _CLOUD_NDIM - c.shape[1]), dtype=np.float32)
                    c = np.concatenate([c, pad], axis=1)
                elif c.shape[1] > _CLOUD_NDIM:
                    c = c[:, :_CLOUD_NDIM]
                flat_ds[n_prev : n_prev + n_pts] = c

            setattr(self, n_attr, n_prev + n_pts)

    def _append_rep_level(
        self,
        group_prefix: str,
        reps: List,
        n_attr: str,
        npt_attr: str,
        nmem_attr: str,
    ) -> None:
        """Generic incremental writer for a representative level.

        Parameters
        ----------
        group_prefix : str
            HDF5 group path prefix (``"particle_fragments"`` or
            ``"particle_instances"``).
        reps : list of Particle
            Representative particles for this event.
        n_attr, npt_attr, nmem_attr : str
            Names of the instance-level running counters on ``self``
            (number of reps, total point-cloud entries, total member IDs).
        """
        n_r   = len(reps)
        n_pt  = sum(len(r.point_cloud) for r in reps)
        n_mem = sum(
            len(r.member_ids) if r.member_ids is not None else 1
            for r in reps
        )

        n_r_prev   = getattr(self, n_attr)
        n_pt_prev  = getattr(self, npt_attr)
        n_mem_prev = getattr(self, nmem_attr)

        # Derive event-offset group name (mirrors _init_rep_group logic)
        short    = group_prefix.split("_", 1)[1]   # "fragments" / "instances"
        ev_group = f"{short[:4]}_events"            # "frag_events" / "inst_events"

        # ---- extend event fencepost ----------------------------------------
        ev_ds = self._f[f"{ev_group}/offsets"]
        ev_ds.resize(self._n_events + 2, axis=0)
        ev_ds[self._n_events + 1] = np.int64(n_r_prev + n_r)

        if n_r > 0:
            new_start = n_r_prev
            new_end   = n_r_prev + n_r

            # ---- extend scalar datasets -----------------------------------
            for name in self._SCALAR_FIELDS:
                self._f[f"{group_prefix}/{name}"].resize(new_end, axis=0)
            self._f[f"{group_prefix}/sem_type"].resize(new_end, axis=0)
            self._f[f"{group_prefix}/parent_frag_id"].resize(new_end, axis=0)
            self._f[f"{group_prefix}/parent_inst_id"].resize(new_end, axis=0)

            # Build scalar arrays
            r_id             = np.empty(n_r, dtype=np.int32)
            r_geant4_id      = np.empty(n_r, dtype=np.int32)
            r_parent_id      = np.empty(n_r, dtype=np.int32)
            r_root_id        = np.empty(n_r, dtype=np.int32)
            r_pdg            = np.empty(n_r, dtype=np.int32)
            r_parent_pdg     = np.empty(n_r, dtype=np.int32)
            r_int_id         = np.empty(n_r, dtype=np.int32)
            r_itype          = np.empty(n_r, dtype=np.int32)
            r_sem_type       = np.empty(n_r, dtype=np.int8)
            r_parent_frag_id = np.empty(n_r, dtype=np.int32)
            r_parent_inst_id = np.empty(n_r, dtype=np.int32)
            pc_lengths       = np.empty(n_r, dtype=np.int64)
            mem_lengths      = np.empty(n_r, dtype=np.int64)

            for k, r in enumerate(reps):
                r_id[k]             = r.id
                r_geant4_id[k]      = getattr(r, "geant4_id", r.id)
                r_parent_id[k]      = r.parent_id
                r_root_id[k]        = r.root_id
                r_pdg[k]            = r.pdg
                r_parent_pdg[k]     = r.parent_pdg
                r_int_id[k]         = int(r._interaction_id)
                r_itype[k]          = int(r._interaction_type)
                r_sem_type[k]       = int(r.sem_type.value)
                _pfid = getattr(r, "parent_frag_id", None)
                _piid = getattr(r, "parent_inst_id", None)
                r_parent_frag_id[k] = int(_pfid) if _pfid is not None else -1
                r_parent_inst_id[k] = int(_piid) if _piid is not None else -1
                pc_lengths[k]       = len(r.point_cloud)
                mem_lengths[k]      = (len(r.member_ids)
                                       if r.member_ids is not None else 1)

            self._f[f"{group_prefix}/id"               ][new_start:new_end] = r_id
            self._f[f"{group_prefix}/geant4_id"        ][new_start:new_end] = r_geant4_id
            self._f[f"{group_prefix}/parent_id"        ][new_start:new_end] = r_parent_id
            self._f[f"{group_prefix}/root_id"          ][new_start:new_end] = r_root_id
            self._f[f"{group_prefix}/pdg"              ][new_start:new_end] = r_pdg
            self._f[f"{group_prefix}/parent_pdg"       ][new_start:new_end] = r_parent_pdg
            self._f[f"{group_prefix}/interaction_id"   ][new_start:new_end] = r_int_id
            self._f[f"{group_prefix}/interaction_type" ][new_start:new_end] = r_itype
            self._f[f"{group_prefix}/sem_type"         ][new_start:new_end] = r_sem_type
            self._f[f"{group_prefix}/parent_frag_id"   ][new_start:new_end] = r_parent_frag_id
            self._f[f"{group_prefix}/parent_inst_id"   ][new_start:new_end] = r_parent_inst_id

            # ---- extend pc_offsets ----------------------------------------
            pc_ds = self._f[f"{group_prefix}/pc_offsets"]
            pc_ds.resize(new_end + 1, axis=0)
            new_pc_offsets = np.empty(n_r, dtype=np.int64)
            running = n_pt_prev
            for k in range(n_r):
                running += pc_lengths[k]
                new_pc_offsets[k] = running
            pc_ds[new_start + 1 : new_end + 1] = new_pc_offsets

            # ---- extend member_offsets ------------------------------------
            mem_ds = self._f[f"{group_prefix}/member_offsets"]
            mem_ds.resize(new_end + 1, axis=0)
            new_mem_offsets = np.empty(n_r, dtype=np.int64)
            running = n_mem_prev
            for k in range(n_r):
                running += mem_lengths[k]
                new_mem_offsets[k] = running
            mem_ds[new_start + 1 : new_end + 1] = new_mem_offsets

            # ---- extend flat point array (single bulk write) --------------
            if n_pt > 0:
                pt_flat = self._f[f"{group_prefix}_points/flat"]
                pt_flat.resize(n_pt_prev + n_pt, axis=0)
                bulk_pts = np.empty((n_pt, _PC_NDIM), dtype=np.float32)
                cursor = 0
                for r in reps:
                    n = len(r.point_cloud)
                    if n > 0:
                        bulk_pts[cursor : cursor + n] = _normalize_cloud(r.point_cloud)
                    cursor += n
                pt_flat[n_pt_prev : n_pt_prev + n_pt] = bulk_pts

            # ---- extend flat member array (single bulk write) -------------
            mem_flat = self._f[f"{group_prefix}_members/flat"]
            mem_flat.resize(n_mem_prev + n_mem, axis=0)
            if n_mem > 0:
                bulk_mem = np.empty(n_mem, dtype=np.int32)
                cursor = 0
                for r in reps:
                    ids = r.member_ids if r.member_ids is not None else [r.id]
                    n = len(ids)
                    bulk_mem[cursor : cursor + n] = ids
                    cursor += n
                mem_flat[n_mem_prev : n_mem_prev + n_mem] = bulk_mem

        setattr(self, n_attr,   n_r_prev   + n_r)
        setattr(self, npt_attr, n_pt_prev  + n_pt)
        setattr(self, nmem_attr, n_mem_prev + n_mem)


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
    chunk_cache_mb : int, optional
        Size of the HDF5 chunk cache in **megabytes**.  Larger values
        speed up sequential or repeated reads by keeping more decompressed
        chunks in RAM.  Default ``256`` (256 MB).
    """

    def __init__(self, path: str, chunk_cache_mb: int = 256) -> None:
        import h5py
        self._path = path
        rdcc_nbytes = chunk_cache_mb * 1024 * 1024
        # rdcc_nslots: comfortably larger than (cache_bytes / chunk_bytes), so
        # the slot table does not thrash before the byte budget is reached.
        # Derived from _CHUNK_POINTS so it tracks the chunk byte budget: with
        # 256 KiB chunks and a 256 MB cache that is ~1024 chunks -> 3073 slots.
        rdcc_nslots = max(127, rdcc_nbytes // (_CHUNK_POINTS * _PC_NDIM * 4) * 3 + 1)
        self._f = h5py.File(
            path, "r",
            rdcc_nbytes=rdcc_nbytes,
            rdcc_nslots=int(rdcc_nslots),
        )
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

        # ---- read scalar metadata in 7 contiguous array slices ------------
        ids             = self._f["particles/id"               ][p_start:p_end]
        # geant4_id arrived in format 2.2.0; older files fall back to `id`,
        # where the two coincided because track IDs were used as the key.
        geant4_ids      = (self._f["particles/geant4_id"       ][p_start:p_end]
                           if "particles/geant4_id" in self._f else ids)
        parent_ids      = self._f["particles/parent_id"        ][p_start:p_end]
        root_ids        = self._f["particles/root_id"          ][p_start:p_end]
        pdgs            = self._f["particles/pdg"              ][p_start:p_end]
        parent_pdgs     = self._f["particles/parent_pdg"       ][p_start:p_end]
        interaction_ids = self._f["particles/interaction_id"   ][p_start:p_end]
        itypes          = self._f["particles/interaction_type" ][p_start:p_end]
        sem_types       = self._f["particles/sem_type"         ][p_start:p_end]

        # ---- read point-cloud data in one contiguous slice ----------------
        pc_bounds = self._pc_offsets[p_start : p_end + 1]  # (n_parts+1,)
        pt_start  = int(pc_bounds[0])
        pt_end    = int(pc_bounds[-1])
        flat_chunk = self._f["points/flat"][pt_start:pt_end]  # (n_pts, _PC_NDIM)

        # ---- reconstruct Particle objects ----------------------------------
        from .utils import SemanticType
        local_offsets = pc_bounds - pt_start  # zero-indexed within flat_chunk
        particles = []
        for k in range(n_parts):
            cloud = flat_chunk[local_offsets[k] : local_offsets[k + 1]]
            p = Particle(
                id               = int(ids[k]),
                geant4_id        = int(geant4_ids[k]),
                parent_id        = int(parent_ids[k]),
                root_id          = int(root_ids[k]),
                pdg              = int(pdgs[k]),
                parent_pdg       = int(parent_pdgs[k]),
                interaction_id   = int(interaction_ids[k]),
                interaction_type = int(itypes[k]),
                point_cloud      = cloud,
            )
            # Override the re-derived sem_type with the stored value so that
            # the on-disk label is authoritative (avoids re-running rules).
            p.sem_type = SemanticType(int(sem_types[k]))
            particles.append(p)

        return particles

    def __repr__(self) -> str:
        status = "open" if self._f.id.valid else "closed"
        return (f"EventStore(path={self._path!r}, "
                f"n_events={self._n_events}, {status})")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def read_bulk(self, start: int = 0, stop: int = -1,
                  indices=None) -> List[List]:
        """
        Read a batch of events with a fixed number of HDF5 reads.

        All scalar datasets and the flat point array are loaded in **9 HDF5
        reads total**, then split into per-event lists in NumPy/Python.

        Parameters
        ----------
        start : int, optional
            First event index (inclusive).  Ignored when *indices* is given.
            Default ``0``.
        stop : int, optional
            Last event index (exclusive).  ``-1`` means ``len(store)``.
            Ignored when *indices* is given.  Default ``-1``.
        indices : sequence of int, optional
            Explicit list (or array) of event indices to fetch, in any order
            and with duplicates allowed.  The returned list preserves the
            order of *indices*.  When given, *start* and *stop* are ignored.

        Returns
        -------
        list of list of Particle
            One entry per requested event, in the same order as *indices*
            (or sequential order when using *start*/*stop*).
        """
        from .data import Particle
        from .utils import SemanticType

        # ---- resolve the set of event indices to load ---------------------
        if indices is not None:
            req = list(indices)
            if not req:
                return []
            unique_sorted = sorted(set(req))
            ev_min, ev_max = unique_sorted[0], unique_sorted[-1]
            if ev_min < 0 or ev_max >= self._n_events:
                raise IndexError(
                    f"Index out of range [0, {self._n_events}): "
                    f"got min={ev_min}, max={ev_max}"
                )
            span_start, span_stop = ev_min, ev_max + 1
        else:
            if stop < 0:
                stop = self._n_events
            stop = min(stop, self._n_events)
            if start >= stop:
                return []
            req = list(range(start, stop))
            span_start, span_stop = start, stop

        # ---- one contiguous particle range covers the whole span ----------
        p_start = int(self._ev_offsets[span_start])
        p_end   = int(self._ev_offsets[span_stop])
        n_parts = p_end - p_start

        if n_parts == 0:
            return [[] for _ in req]

        # 8 scalar reads + 1 pc_offsets slice + 1 flat-points read = 10 HDF5 ops
        ids             = self._f["particles/id"               ][p_start:p_end]
        # geant4_id arrived in format 2.2.0; older files fall back to `id`,
        # where the two coincided because track IDs were used as the key.
        geant4_ids      = (self._f["particles/geant4_id"       ][p_start:p_end]
                           if "particles/geant4_id" in self._f else ids)
        parent_ids      = self._f["particles/parent_id"        ][p_start:p_end]
        root_ids        = self._f["particles/root_id"          ][p_start:p_end]
        pdgs            = self._f["particles/pdg"              ][p_start:p_end]
        parent_pdgs     = self._f["particles/parent_pdg"       ][p_start:p_end]
        interaction_ids = self._f["particles/interaction_id"   ][p_start:p_end]
        itypes          = self._f["particles/interaction_type" ][p_start:p_end]
        sem_types       = self._f["particles/sem_type"         ][p_start:p_end]

        pc_bounds = self._pc_offsets[p_start : p_end + 1]   # (n_parts+1,)
        pt_start  = int(pc_bounds[0])
        pt_end    = int(pc_bounds[-1])
        flat_all  = self._f["points/flat"][pt_start:pt_end]  # single read
        local_pc  = pc_bounds - pt_start                     # zero-indexed

        # ---- helper: build particle list for one event index --------------
        def _build_event(ev):
            ep_start = int(self._ev_offsets[ev])     - p_start
            ep_end   = int(self._ev_offsets[ev + 1]) - p_start
            particles = []
            for k in range(ep_start, ep_end):
                cloud = flat_all[local_pc[k] : local_pc[k + 1]]
                p = Particle(
                    id               = int(ids[k]),
                    geant4_id        = int(geant4_ids[k]),
                    parent_id        = int(parent_ids[k]),
                    root_id          = int(root_ids[k]),
                    pdg              = int(pdgs[k]),
                    parent_pdg       = int(parent_pdgs[k]),
                    interaction_id   = int(interaction_ids[k]),
                    interaction_type = int(itypes[k]),
                    point_cloud      = cloud,
                )
                p.sem_type = SemanticType(int(sem_types[k]))
                particles.append(p)
            return particles

        return [_build_event(ev) for ev in req]

    def iter_events(self, batch_size: int = 64):
        """
        Iterate over all events in file order.

        Reads events in batches of *batch_size* to amortise HDF5 overhead
        while keeping memory bounded.  Each ``yield`` still returns a single
        event's particle list.

        Parameters
        ----------
        batch_size : int, optional
            Number of events to pre-fetch per HDF5 batch.  Default ``64``.
            Set to ``1`` to reproduce the old one-at-a-time behaviour.

        Yields
        ------
        list of Particle
        """
        for batch_start in range(0, self._n_events, batch_size):
            batch_stop = min(batch_start + batch_size, self._n_events)
            for ev in self.read_bulk(batch_start, batch_stop):
                yield ev

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
      dtype cast and column adjustment).

    Non-empty 2-D arrays are trimmed to ``_PC_NDIM`` columns when they have
    more, or zero-padded on the right when they have fewer.

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
    pc = pc.astype(np.float32)
    n_cols = pc.shape[1]
    if n_cols == _PC_NDIM:
        return pc
    if n_cols > _PC_NDIM:
        return pc[:, :_PC_NDIM]
    # Fewer columns than _PC_NDIM — zero-pad on the right
    padded = np.zeros((pc.shape[0], _PC_NDIM), dtype=np.float32)
    padded[:, :n_cols] = pc
    return padded


def _compress_kwargs(compression: Optional[str],
                     compression_opts: Optional[int]) -> dict:
    """
    Build h5py dataset keyword arguments for the requested filter.

    Supported values for *compression*:

    * ``None``        — no compression; fastest reads, largest files.
    * ``"lzf"``       — LZF filter built into h5py; always available;
                        fast decompression, moderate ratio.  **Default.**
    * ``"lz4"``       — LZ4 via ``hdf5plugin``; faster than LZF for large
                        arrays with byte-shuffle.  Falls back to LZF if
                        ``hdf5plugin`` is not installed.
    * ``"blosc_lz4"`` — Blosc+LZ4 via ``hdf5plugin``; supports parallel
                        decompression (set ``HDF5_PLUGIN_PATH`` or install
                        the Blosc plugin).  Falls back to LZF.
    * ``"gzip"``      — Standard DEFLATE; good ratio, slowest reads.
    """
    if compression is None:
        return {}

    if compression in ("lz4", "blosc_lz4"):
        try:
            import hdf5plugin
            if compression == "blosc_lz4":
                return dict(hdf5plugin.Blosc(cname="lz4", shuffle=hdf5plugin.Blosc.BYTE_SHUFFLE))
            else:
                return dict(hdf5plugin.LZ4())
        except ImportError:
            import warnings
            warnings.warn(
                f"hdf5plugin not installed; falling back to lzf compression "
                f"(requested {compression!r}).  Install with: pip install hdf5plugin",
                stacklevel=3,
            )
            compression = "lzf"

    kwargs: dict = {"compression": compression}
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    return kwargs


def inspect_compression(path: str) -> dict:
    """
    Report the HDF5 filter and chunk geometry for the key datasets in *path*.

    Useful for diagnosing read performance::

        >>> from pysupera.io import inspect_compression
        >>> inspect_compression("output.h5")
        {'points/flat':    {'compression': 'gzip', 'compression_opts': 1, 'chunks': (65536, 5)},
         'particles/id':   {'compression': 'gzip', 'compression_opts': 1, 'chunks': (256,)}, ...}

    Parameters
    ----------
    path : str
        Path to any HDF5 file written by this library.

    Returns
    -------
    dict
        ``{dataset_path: {"compression": ..., "compression_opts": ..., "chunks": ...}}``
    """
    import h5py
    keys = [
        "points/flat",
        "particles/id",
        "particles/pc_offsets",
    ]
    result = {}
    with h5py.File(path, "r") as f:
        for k in keys:
            if k not in f:
                continue
            ds = f[k]
            result[k] = {
                "compression":      ds.compression,
                "compression_opts": ds.compression_opts,
                "chunks":           ds.chunks,
                "shape":            ds.shape,
            }
    return result


def recompress(src: str, dst: str,
               compression: str = "lzf",
               compression_opts: Optional[int] = None,
               batch_size: int = 256) -> None:
    """
    Rewrite *src* into *dst* with a different compression filter.

    Copies particle events **and** event-level point clouds
    (``non_le_cloud`` / ``le_scatter_cloud``).
    Fragment and instance representative groups are also copied when
    present in the source file.

    Parameters
    ----------
    src : str
        Path to the source HDF5 file (any supported compression).
    dst : str
        Path for the output file (will be overwritten if it exists).
    compression : str, optional
        Compression for the output file.  Default ``"lzf"``.
        Use ``"gzip"`` to produce a file readable by h5wasm in the browser.
    compression_opts : int or None, optional
        Compression level for the output filter.
    batch_size : int, optional
        Events per read batch.  Default ``256``.
    """
    import h5py as _h5
    import numpy as _np

    with EventStore(src) as src_store, \
         _h5.File(src, "r") as src_f, \
         open_writer(dst, compression=compression,
                     compression_opts=compression_opts) as w:

        n_ev = len(src_store)

        # Pre-read cloud offset arrays if present
        _has_nle = "non_le_cloud/offsets" in src_f
        _has_le  = "le_scatter_cloud/offsets" in src_f
        if _has_nle:
            _nle_off = src_f["non_le_cloud/offsets"][:]
        if _has_le:
            _le_off  = src_f["le_scatter_cloud/offsets"][:]

        # Pre-read fragment / instance event-offset arrays if present
        _has_frags = "frag_events/offsets" in src_f
        _has_insts = "inst_events/offsets" in src_f
        if _has_frags:
            _fr_ev_off  = src_f["frag_events/offsets"][:]
        if _has_insts:
            _in_ev_off  = src_f["inst_events/offsets"][:]

        for batch_start in range(0, n_ev, batch_size):
            batch_stop = min(batch_start + batch_size, n_ev)

            for ev_idx, particles in zip(
                range(batch_start, batch_stop),
                src_store.read_bulk(batch_start, batch_stop),
            ):
                # ---- event-level clouds -----------------------------------
                if _has_nle or _has_le:
                    nle_cloud = (
                        src_f["non_le_cloud/flat"][
                            int(_nle_off[ev_idx]) : int(_nle_off[ev_idx + 1])
                        ] if _has_nle
                        else _np.empty((0, _CLOUD_NDIM), dtype=_np.float32)
                    )
                    le_cloud  = (
                        src_f["le_scatter_cloud/flat"][
                            int(_le_off[ev_idx]) : int(_le_off[ev_idx + 1])
                        ] if _has_le
                        else _np.empty((0, _CLOUD_NDIM), dtype=_np.float32)
                    )
                    w.append_event_clouds(nle_cloud, le_cloud)

                # ---- fragment representatives -----------------------------
                if _has_frags:
                    _fr_s = int(_fr_ev_off[ev_idx])
                    _fr_e = int(_fr_ev_off[ev_idx + 1])
                    frags = _read_rep_level(src_f, "particle_fragments", _fr_s, _fr_e)
                    w.append_fragments(frags)

                # ---- instance representatives ----------------------------
                if _has_insts:
                    _in_s = int(_in_ev_off[ev_idx])
                    _in_e = int(_in_ev_off[ev_idx + 1])
                    insts = _read_rep_level(src_f, "particle_instances", _in_s, _in_e)
                    w.append_instances(insts)

                w.append_event(particles)


def _read_rep_level(f, group: str, r_start: int, r_end: int) -> list:
    """Reconstruct minimal Particle-like objects from a stored representative group.

    Used internally by :func:`recompress`.
    """
    from .data import Particle
    from .utils import SemanticType

    n_r = r_end - r_start
    if n_r == 0:
        return []

    ids    = f[f"{group}/id"              ][r_start:r_end]
    g4ids  = (f[f"{group}/geant4_id"      ][r_start:r_end]
              if f"{group}/geant4_id" in f else ids)
    pids   = f[f"{group}/parent_id"       ][r_start:r_end]
    rids   = f[f"{group}/root_id"         ][r_start:r_end]
    pdgs   = f[f"{group}/pdg"             ][r_start:r_end]
    ppdgs  = f[f"{group}/parent_pdg"      ][r_start:r_end]
    intids = f[f"{group}/interaction_id"  ][r_start:r_end]
    itypes = f[f"{group}/interaction_type"][r_start:r_end]
    stypes = f[f"{group}/sem_type"        ][r_start:r_end]
    pfids  = f[f"{group}/parent_frag_id"  ][r_start:r_end] if f"{group}/parent_frag_id" in f else None
    piids  = f[f"{group}/parent_inst_id"  ][r_start:r_end] if f"{group}/parent_inst_id" in f else None

    pc_off   = f[f"{group}/pc_offsets"    ][r_start : r_end + 1]
    mem_off  = f[f"{group}/member_offsets"][r_start : r_end + 1]

    pt_start = int(pc_off[0])
    pt_end   = int(pc_off[-1])
    pts_flat = f[f"{group}_points/flat"][pt_start:pt_end] if pt_end > pt_start else None

    mem_start = int(mem_off[0])
    mem_end   = int(mem_off[-1])
    mem_flat  = f[f"{group}_members/flat"][mem_start:mem_end] if mem_end > mem_start else None

    reps = []
    for k in range(n_r):
        cloud = pts_flat[int(pc_off[k]) - pt_start : int(pc_off[k + 1]) - pt_start] \
                if pts_flat is not None else __import__('numpy').empty((0, _PC_NDIM), dtype='float32')
        ms = int(mem_off[k]) - mem_start
        me = int(mem_off[k + 1]) - mem_start
        mids = list(mem_flat[ms:me].astype(int)) if mem_flat is not None and me > ms else None
        p = Particle(
            id               = int(ids[k]),
            geant4_id        = int(g4ids[k]),
            parent_id        = int(pids[k]),
            root_id          = int(rids[k]),
            pdg              = int(pdgs[k]),
            parent_pdg       = int(ppdgs[k]),
            interaction_id   = int(intids[k]),
            interaction_type = int(itypes[k]),
            point_cloud      = cloud,
        )
        p.sem_type   = SemanticType(int(stypes[k]))
        p.member_ids = mids
        if pfids is not None:
            p.parent_frag_id = int(pfids[k])
        if piids is not None:
            p.parent_inst_id = int(piids[k])
        reps.append(p)
    return reps


def repack(path: str,
           compression: Optional[str] = None,
           compression_opts: Optional[int] = None,
           dst: Optional[str] = None,
           rechunk: bool = False,
           verify: bool = False,
           verbose: bool = True) -> dict:
    """
    Rewrite *path* with chunk shapes sized from the final dataset lengths,
    optionally changing the compression filter at the same time.

    The streaming writer creates every dataset empty and grows it with
    ``resize()``, so at creation time it cannot know how long a dataset will
    end up.  It therefore uses the byte-budget chunk shapes from
    ``_CHUNK_*``, and any dataset that ends up **shorter than one chunk**
    keeps a chunk longer than itself -- stored as padding.  This function is
    the post-pass that removes that: every chunk is clamped to the dataset's
    actual length.

    Note that the byte budget already bounds the waste to at most one chunk per
    dataset, so this is a modest optimisation for long runs.  It matters most
    for short runs (few events), where several datasets can be shorter than a
    single chunk.

    The rewrite is generic -- it walks every group and dataset rather than
    reconstructing events -- so it preserves any layout the writer produced.
    Data is copied in chunk-aligned blocks, so memory use stays bounded
    regardless of file size.

    Parameters
    ----------
    path : str
        Source file.
    compression : str or None, optional
        Filter for the output.  ``None`` (default) keeps whatever filter each
        source dataset already uses.  Pass e.g. ``"gzip"`` to switch -- which is
        how to produce a file the browser viewer can read, since h5wasm handles
        gzip only.
    compression_opts : int or None, optional
        Compression level for *compression* (gzip: 1-9).
    dst : str or None, optional
        Output path.  ``None`` (default) rewrites *path* in place, via a
        temporary file swapped in with :func:`os.replace`, so the original
        survives untouched if anything raises.
    rechunk : bool, optional
        Let h5py choose chunk shapes instead of clamping the source's.  On
        pysupera output this tends to win on both size and row-range read
        speed, because h5py picks smaller column-wise chunks, but it does
        change the layout rather than only the filter.
    verify : bool, optional
        After writing, compare every dataset against the source and raise
        ``ValueError`` on any mismatch.  For an in-place repack the check runs
        *before* the original is replaced.
    verbose : bool, optional
        Print a one-line before/after summary.

    Returns
    -------
    dict
        ``{"size_before", "size_after", "n_datasets", "n_reshaped"}``

    Raises
    ------
    ValueError
        If *verify* is set and any dataset differs from the source.
    """
    import h5py
    import numpy as _np

    if dst is not None and os.path.abspath(dst) == os.path.abspath(path):
        raise ValueError(
            f"dst must differ from path; pass dst=None to repack in place "
            f"({path!r})"
        )

    size_before = os.path.getsize(path)
    in_place = dst is None
    # In place: build beside the original and swap only once it is complete.
    out = f"{path}.repack-tmp" if in_place else dst
    n_datasets = n_reshaped = 0

    try:
        with h5py.File(path, "r") as fin, h5py.File(out, "w") as fout:
            for key, val in fin.attrs.items():
                fout.attrs[key] = val

            def visit(name, obj):
                nonlocal n_datasets, n_reshaped
                if isinstance(obj, h5py.Group):
                    grp = fout.require_group(name)
                    for key, val in obj.attrs.items():
                        grp.attrs[key] = val
                    return

                n_datasets += 1
                src_chunks = obj.chunks
                kwargs: dict = {}
                chunks = None

                if src_chunks is not None:
                    if any(d == 0 for d in obj.shape):
                        chunks = None          # cannot chunk an empty dataset
                    elif rechunk:
                        chunks = True          # let h5py choose
                        n_reshaped += 1
                    else:
                        # Clamp each dimension to the real shape; HDF5 rejects
                        # a chunk larger than a fixed-size dataset.
                        chunks = tuple(max(1, min(c, s))
                                       for c, s in zip(src_chunks, obj.shape))
                        if chunks != src_chunks:
                            n_reshaped += 1

                if chunks is not None:
                    if compression is None:
                        # Preserve the source filter.  Third-party filters
                        # (LZ4, Blosc) must be matched by filter ID: h5py
                        # reports Dataset.compression as 'unknown' for those,
                        # which is not a value it accepts on write.
                        kwargs = _compression_kwargs_from_filters(obj)
                        if not kwargs and obj.compression in ("gzip", "lzf",
                                                              "szip"):
                            kwargs = {"compression":      obj.compression,
                                      "compression_opts": obj.compression_opts}
                    elif str(compression).lower() in ("none", "~", ""):
                        kwargs = {}     # explicitly store uncompressed
                    else:
                        kwargs = _compress_kwargs(compression,
                                                  compression_opts)

                dset = fout.create_dataset(name, shape=obj.shape,
                                           dtype=obj.dtype, chunks=chunks,
                                           **{k: v for k, v in kwargs.items()
                                              if v is not None})
                for key, val in obj.attrs.items():
                    dset.attrs[key] = val

                if obj.ndim == 0:
                    dset[()] = obj[()]
                    return
                n = obj.shape[0]
                if n == 0:
                    return
                # Copy in chunk-aligned blocks so memory stays bounded.
                step = dset.chunks[0] if dset.chunks else min(n, 1 << 16)
                for start in range(0, n, step):
                    stop = min(start + step, n)
                    dset[start:stop] = obj[start:stop]

            fin.visititems(visit)

        if verify:
            _verify_same_datasets(path, out)

        if in_place:
            os.replace(out, path)
    except BaseException:
        # Only ever clean up our own temporary; never an explicit dst.
        if in_place and os.path.exists(out):
            os.remove(out)
        raise

    size_after = os.path.getsize(path if in_place else out)
    if verbose:
        # Signed relative to the original: negative means the file shrank.
        pct = (100.0 * (size_after - size_before) / size_before) \
            if size_before else 0.0
        _where = path if in_place else f"{path} -> {out}"
        print(f"[repack] {_where}: {size_before / 2**20:.2f} MiB -> "
              f"{size_after / 2**20:.2f} MiB ({pct:+.1f}%), "
              f"{n_reshaped}/{n_datasets} dataset(s) re-chunked"
              + ("  [verified]" if verify else ""))
    return {"size_before": size_before, "size_after": size_after,
            "n_datasets": n_datasets, "n_reshaped": n_reshaped}


def _verify_same_datasets(src: str, dst: str) -> None:
    """
    Raise ValueError unless *dst* holds the same datasets, byte for byte, as
    *src*.  Compression and chunking are irrelevant here -- only values.
    """
    import h5py
    import numpy as _np

    problems: list = []

    with h5py.File(src, "r") as fa, h5py.File(dst, "r") as fb:
        def check(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name not in fb:
                problems.append(f"{name}: missing in output")
                return
            other = fb[name]
            if obj.shape != other.shape or obj.dtype != other.dtype:
                problems.append(
                    f"{name}: shape/dtype {obj.shape}/{obj.dtype} != "
                    f"{other.shape}/{other.dtype}"
                )
                return
            if obj.ndim == 0:
                same = _np.array_equal(_np.asarray(obj[()]),
                                       _np.asarray(other[()]))
            elif obj.shape[0] == 0:
                same = True
            else:
                try:
                    same = _np.array_equal(obj[...], other[...], equal_nan=True)
                except TypeError:
                    # equal_nan is rejected for non-float dtypes.
                    same = _np.array_equal(obj[...], other[...])
            if not same:
                problems.append(f"{name}: data differs")

        fa.visititems(check)

    if problems:
        raise ValueError(
            f"repack verification failed for {dst!r}: "
            f"{len(problems)} problem(s); first few: {problems[:5]}"
        )


def _compression_kwargs_from_filters(ds) -> dict:
    """Reproduce a dataset's third-party filter (LZ4 / Blosc) as kwargs."""
    filters = dict(ds._filters or {})
    if "32004" in filters:
        return _compress_kwargs("lz4", None)
    if "32001" in filters:
        return _compress_kwargs("blosc_lz4", None)
    return {}


def repack_cli() -> None:
    """Entry point for the ``pysupera-repack`` shell command.

    Usage::

        # rewrite in place, clamping chunks, keeping each dataset's filter
        pysupera-repack out.h5

        # write a gzip copy the browser viewer can read
        pysupera-repack out.h5 out_vis.h5 --compression gzip

        # smallest and fastest for row-range reads, at the cost of relayout
        pysupera-repack out.h5 out_vis.h5 --compression gzip --rechunk

    Unlike ``pysupera-recompress``, this walks the file generically instead of
    reconstructing events, so it can resize chunks and handles any layout --
    including the voxmap companion file.  Prefer it unless you specifically
    need the event-by-event rewrite.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="pysupera-repack",
        description="Resize HDF5 chunks (and optionally change the "
                    "compression filter) in a pysupera output file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("src", help="Source HDF5 file")
    parser.add_argument("dst", nargs="?", default=None,
                        help="Output file; omit to rewrite src in place")
    parser.add_argument("-c", "--compression", default=None,
                        help="Filter for the output: gzip, lzf, lz4, "
                             "blosc_lz4, none.  Omit to keep each dataset's "
                             "existing filter")
    parser.add_argument("-l", "--level", type=int, default=None,
                        help="Compression level (gzip: 1-9)")
    parser.add_argument("--rechunk", action="store_true",
                        help="Let h5py choose chunk shapes rather than "
                             "clamping the source's; usually smaller and "
                             "faster for row-range reads, but relayouts")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="Skip the post-write dataset equality check")
    args = parser.parse_args()

    if args.level is not None and not 1 <= args.level <= 9:
        parser.error("--level must be between 1 and 9")

    # Note the distinction: omitting --compression preserves each dataset's
    # existing filter, whereas "--compression none" stores them uncompressed.
    repack(args.src,
           compression=args.compression,
           compression_opts=args.level,
           dst=args.dst,
           rechunk=args.rechunk,
           verify=args.verify)


def recompress_cli() -> None:
    """Entry point for the ``pysupera-recompress`` shell command.

    Usage::

        pysupera-recompress src.h5 dst.h5 [--compression gzip] [--level 1]

    Rewrites *src.h5* into *dst.h5* with the chosen compression filter.
    Use ``--compression gzip`` to produce a browser-compatible file for
    the WebGL viewer (h5wasm only supports gzip).
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="pysupera-recompress",
        description="Recompress a pysupera HDF5 file with a different filter.",
    )
    parser.add_argument("src",  help="Source HDF5 file")
    parser.add_argument("dst",  help="Output HDF5 file (will be overwritten)")
    parser.add_argument("-c", "--compression", default="gzip",
                        help="Compression filter: gzip, lzf, lz4, blosc_lz4, none  (default: gzip)")
    parser.add_argument("-l", "--level", type=int, default=None,
                        help="Compression level (gzip: 1-9; default: filter default)")
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Events per read batch (default: 256)")
    args = parser.parse_args()
    comp = None if args.compression in ("none", "~", "") else args.compression
    print(f"[recompress] {args.src} → {args.dst}  compression={comp!r} level={args.level}")
    recompress(args.src, args.dst,
               compression=comp,
               compression_opts=args.level,
               batch_size=args.batch_size)
    print("[recompress] Done.")


# ---------------------------------------------------------------------------
# Voxelization-mapping companion file
# ---------------------------------------------------------------------------

def voxmap_path(main_path: str) -> str:
    """
    Return the companion voxmap file path for *main_path*.

    The companion file is written alongside the main HDF5 file with
    ``_voxmap`` inserted before the extension::

        "run042.h5" → "run042_voxmap.h5"

    Parameters
    ----------
    main_path : str
        Path to the main particle event file.
    """
    stem, ext = os.path.splitext(main_path)
    return f"{stem}_voxmap{ext}"


def write_voxmap(main_path: str,
                 events_voxmaps,
                 compression: str = "lzf",
                 compression_opts: Optional[int] = None) -> str:
    """
    Write a companion voxelization-mapping file alongside *main_path*.

    The companion is a three-level CSR file that encodes, for every
    voxel in every particle in every event, which input point IDs were
    merged into that voxel and their individual energy contributions.

    File layout
    -----------
    .. code-block:: text

        /format_version                   scalar str
        /n_events                         scalar int64

        /events/
            offsets        (n_events + 1,)              int64
                Fencepost into the particle index space.

        /particles/
            vox_offsets    (n_total_particles + 1,)     int64
                Fencepost into the voxel index space.

        /voxels/
            input_offsets  (n_total_voxels + 1,)        int64
                Fencepost into /flat/*.

        /flat/
            input_ids      (n_total_mappings,)           int64
            input_energies (n_total_mappings,)           float32

    Parameters
    ----------
    main_path : str
        Path to the main particle event HDF5 file.  The companion is
        created at :func:`voxmap_path(main_path) <voxmap_path>`.
    events_voxmaps : list of list of VoxelizeRecord
        Outer list indexed by event; inner list is the
        :class:`~pysupera.preproc.VoxelizeRecord` for each particle.
        Records must appear in the same particle order as in *main_path*.
    compression : str, optional
        Same as :func:`write_events`.
    compression_opts : int or None, optional
        Same as :func:`write_events`.

    Returns
    -------
    str
        Path of the companion file that was written.
    """
    import h5py
    ckw  = _compress_kwargs(compression, compression_opts)
    vpath = voxmap_path(main_path)

    n_events = len(events_voxmaps)

    # Event → particle fencepost
    ev_offsets = np.zeros(n_events + 1, dtype=np.int64)
    for i, ev_recs in enumerate(events_voxmaps):
        ev_offsets[i + 1] = ev_offsets[i] + len(ev_recs)
    n_total_particles = int(ev_offsets[-1])

    # Build particle/voxel/flat CSR arrays in one pass
    # particles/vox_offsets : fencepost, length n_total_particles + 1
    vox_offsets      = np.zeros(n_total_particles + 1, dtype=np.int64)
    # voxels/input_offsets collected as list (variable total voxels)
    vox_input_segs   = []   # one entry per particle: rec.voxel_offsets[1:] shifted
    flat_ids_segs    = []
    flat_eng_segs    = []

    p_idx         = 0
    running_vox   = np.int64(0)
    running_flat  = np.int64(0)

    for ev_recs in events_voxmaps:
        for rec in ev_recs:
            n_vox = np.int64(len(rec.voxel_offsets) - 1)
            vox_offsets[p_idx + 1] = running_vox + n_vox

            # Shift this particle's per-voxel fencepost values by running_flat
            vox_input_segs.append(
                rec.voxel_offsets[1:].astype(np.int64) + running_flat
            )

            flat_ids_segs.append(rec.input_ids.astype(np.int64))
            flat_eng_segs.append(rec.input_energies.astype(np.float32))

            running_flat += np.int64(len(rec.input_ids))
            running_vox  += n_vox
            p_idx        += 1

    n_total_voxels   = int(running_vox)
    n_total_mappings = int(running_flat)

    # Assemble voxels/input_offsets: starts with 0, then all shifted segments
    if vox_input_segs:
        vox_input_offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64)] + vox_input_segs
        )
    else:
        vox_input_offsets = np.zeros(1, dtype=np.int64)

    flat_ids      = np.concatenate(flat_ids_segs)   if flat_ids_segs   else np.empty(0, dtype=np.int64)
    flat_energies = np.concatenate(flat_eng_segs)   if flat_eng_segs   else np.empty(0, dtype=np.float32)

    # Write
    _cp = lambda n, size: (min(_CHUNK_PARTICLES, max(1, size)),)
    _cp2 = lambda size: (min(_CHUNK_POINTS, max(1, size)),)

    with h5py.File(vpath, "w") as f:
        f.create_dataset("format_version", data=FORMAT_VERSION)
        f.create_dataset("n_events",       data=np.int64(n_events))

        eg = f.create_group("events")
        eg.create_dataset("offsets", data=ev_offsets)

        pg = f.create_group("particles")
        pg.create_dataset(
            "vox_offsets", data=vox_offsets,
            chunks=_cp("vox", n_total_particles + 1),
        )

        vg = f.create_group("voxels")
        vg.create_dataset(
            "input_offsets", data=vox_input_offsets,
            chunks=_cp("inp", len(vox_input_offsets)),
        )

        fg = f.create_group("flat")
        if n_total_mappings > 0:
            fg.create_dataset(
                "input_ids", data=flat_ids,
                chunks=_cp2(n_total_mappings),
                **ckw,
            )
            fg.create_dataset(
                "input_energies", data=flat_energies,
                chunks=_cp2(n_total_mappings),
                **ckw,
            )
        else:
            fg.create_dataset("input_ids",      data=np.empty(0, dtype=np.int64))
            fg.create_dataset("input_energies", data=np.empty(0, dtype=np.float32))

    return vpath


def open_voxmap_writer(main_path: str,
                       compression: str = "lzf",
                       compression_opts: Optional[int] = None) -> "VoxmapWriter":
    """
    Open a :class:`VoxmapWriter` for incremental voxmap writing.

    Parameters
    ----------
    main_path : str
        Path of the *main* particle event file.  The companion is
        written at :func:`voxmap_path(main_path) <voxmap_path>`.

    Returns
    -------
    VoxmapWriter
    """
    return VoxmapWriter(main_path, compression=compression,
                        compression_opts=compression_opts)


def read_voxmap(main_path: str) -> "VoxmapStore":
    """
    Open the companion voxmap file for *main_path*.

    Parameters
    ----------
    main_path : str
        Path of the main particle event file.

    Returns
    -------
    VoxmapStore
    """
    return VoxmapStore(main_path)


# ---------------------------------------------------------------------------
# VoxmapWriter
# ---------------------------------------------------------------------------

class VoxmapWriter:
    """
    Incremental writer for the companion voxelization-mapping file.

    Mirrors :class:`EventWriter`: append one event's worth of
    :class:`~pysupera.preproc.VoxelizeRecord` objects per call.

    Parameters
    ----------
    main_path : str
        Path of the main particle event HDF5 file.  The companion is
        created at :func:`voxmap_path(main_path) <voxmap_path>`.
    compression : str, optional
        HDF5 filter name.  Same options as :func:`write_events`.
    compression_opts : int or None, optional
        Same as :func:`write_events`.
    """

    def __init__(self, main_path: str,
                 compression: str = "lzf",
                 compression_opts: Optional[int] = None) -> None:
        import h5py
        self._ckw   = _compress_kwargs(compression, compression_opts)
        self._vpath = voxmap_path(main_path)

        self._f = h5py.File(self._vpath, "w")
        self._n_events    = 0
        self._n_particles = 0
        self._n_voxels    = 0
        self._n_flat      = 0
        self._init_datasets()

    # ------------------------------------------------------------------ CM --

    def __enter__(self) -> "VoxmapWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Finalise and close the companion file.  Idempotent."""
        if self._f.id.valid:
            self._f["n_events"][()] = np.int64(self._n_events)
            self._f.close()

    # ------------------------------------------------------------------ API --

    def append_event(self, vox_records) -> None:
        """
        Append one event's voxelization records.

        Parameters
        ----------
        vox_records : list of VoxelizeRecord
            One record per particle in the event, in the same order as
            the corresponding :meth:`EventWriter.append_event` call.
            May be empty.
        """
        n_p  = len(vox_records)
        n_vox_total  = sum(len(r.voxel_offsets) - 1 for r in vox_records)
        n_flat_total = sum(len(r.input_ids)         for r in vox_records)

        # Extend events/offsets
        ev_ds = self._f["events/offsets"]
        ev_ds.resize(self._n_events + 2, axis=0)
        ev_ds[self._n_events + 1] = np.int64(self._n_particles + n_p)

        if n_p > 0:
            new_p_start  = self._n_particles
            new_p_end    = self._n_particles + n_p
            new_vox_end  = self._n_voxels  + n_vox_total
            new_flat_end = self._n_flat    + n_flat_total

            # ---- particles/id and particles/vox_offsets (one write each) ----
            pid_ds     = self._f["particles/id"]
            vox_off_ds = self._f["particles/vox_offsets"]
            pid_ds.resize(new_p_end, axis=0)
            vox_off_ds.resize(new_p_end + 1, axis=0)

            bulk_pids     = np.array([r.particle_id for r in vox_records], dtype=np.int32)
            bulk_vox_ends = np.empty(n_p, dtype=np.int64)   # absolute voxel end per particle
            running = np.int64(self._n_voxels)
            for k, rec in enumerate(vox_records):
                running += np.int64(len(rec.voxel_offsets) - 1)
                bulk_vox_ends[k] = running

            pid_ds    [new_p_start : new_p_end]     = bulk_pids
            vox_off_ds[new_p_start + 1 : new_p_end + 1] = bulk_vox_ends

            # ---- voxels/input_offsets (one write) ----------------------------
            vox_inp_ds = self._f["voxels/input_offsets"]
            vox_inp_ds.resize(new_vox_end + 1, axis=0)

            bulk_inp_offs = np.empty(n_vox_total, dtype=np.int64)
            cursor_vox = 0
            running_flat = np.int64(self._n_flat)
            for rec in vox_records:
                n_vox = len(rec.voxel_offsets) - 1
                bulk_inp_offs[cursor_vox : cursor_vox + n_vox] = (
                    rec.voxel_offsets[1:].astype(np.int64) + running_flat
                )
                running_flat += np.int64(len(rec.input_ids))
                cursor_vox   += n_vox
            vox_inp_ds[self._n_voxels + 1 : new_vox_end + 1] = bulk_inp_offs

            # ---- flat input_ids and input_energies (one write each) ----------
            flat_ids_ds = self._f["flat/input_ids"]
            flat_eng_ds = self._f["flat/input_energies"]
            flat_ids_ds.resize(new_flat_end, axis=0)
            flat_eng_ds.resize(new_flat_end, axis=0)

            bulk_ids = np.empty(n_flat_total, dtype=np.int64)
            bulk_eng = np.empty(n_flat_total, dtype=np.float32)
            cursor = 0
            for rec in vox_records:
                n = len(rec.input_ids)
                if n > 0:
                    bulk_ids[cursor : cursor + n] = rec.input_ids.astype(np.int64)
                    bulk_eng[cursor : cursor + n] = rec.input_energies.astype(np.float32)
                cursor += n
            flat_ids_ds[self._n_flat : new_flat_end] = bulk_ids
            flat_eng_ds[self._n_flat : new_flat_end] = bulk_eng

        self._n_events    += 1
        self._n_particles += n_p
        self._n_voxels    += n_vox_total
        self._n_flat      += n_flat_total

    @property
    def n_events(self) -> int:
        return self._n_events

    def __repr__(self) -> str:
        status = "open" if self._f.id.valid else "closed"
        return (f"VoxmapWriter(path={self._vpath!r}, "
                f"n_events={self._n_events}, {status})")

    # ---------------------------------------------------------- private -----

    def _init_datasets(self) -> None:
        f   = self._f
        ckw = self._ckw

        f.create_dataset("format_version", data=FORMAT_VERSION)
        f.create_dataset("n_events",       data=np.int64(0))

        eg = f.create_group("events")
        eg.create_dataset("offsets", data=np.zeros(1, dtype=np.int64),
                          maxshape=(None,), chunks=(_CHUNK_PARTICLES,))

        pg = f.create_group("particles")
        pg.create_dataset("vox_offsets", data=np.zeros(1, dtype=np.int64),
                          maxshape=(None,), chunks=(_CHUNK_PARTICLES,))
        pg.create_dataset("id",
                          shape=(0,), maxshape=(None,), dtype=np.int32,
                          chunks=(_CHUNK_PARTICLES,))

        vg = f.create_group("voxels")
        vg.create_dataset("input_offsets", data=np.zeros(1, dtype=np.int64),
                          maxshape=(None,), chunks=(_CHUNK_PARTICLES,))

        fg = f.create_group("flat")
        fg.create_dataset("input_ids",
                          shape=(0,), maxshape=(None,), dtype=np.int64,
                          chunks=(_CHUNK_POINTS,), **ckw)
        fg.create_dataset("input_energies",
                          shape=(0,), maxshape=(None,), dtype=np.float32,
                          chunks=(_CHUNK_POINTS,), **ckw)


# ---------------------------------------------------------------------------
# VoxmapStore
# ---------------------------------------------------------------------------

class VoxmapStore:
    """
    Random-access reader for a companion voxmap file.

    Supports ``len()`` and ``store[i]`` (returns per-particle mapping
    data for event *i*).

    Parameters
    ----------
    main_path : str
        Path to the main particle event HDF5 file.
    """

    def __init__(self, main_path: str) -> None:
        import h5py
        self._vpath = voxmap_path(main_path)
        self._f = h5py.File(self._vpath, "r")
        self._n_events: int = int(self._f["n_events"][()])

        self._ev_offsets  = self._f["events/offsets"][:]
        self._vox_offsets = self._f["particles/vox_offsets"][:]
        # particle IDs — load fully into RAM (same size as _vox_offsets)
        self._particle_ids = (
            self._f["particles/id"][:]
            if "particles/id" in self._f
            else None
        )
        # voxels/input_offsets is potentially large — load lazily
        # (accessed via HDF5 slice in __getitem__)

    def __enter__(self) -> "VoxmapStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._f.id.valid:
            self._f.close()

    def __len__(self) -> int:
        return self._n_events

    def __getitem__(self, index: int):
        """
        Return per-particle mapping data for event *index*.

        Returns
        -------
        list of dict
            One dict per particle with keys:

            ``"vox_input_offsets"`` : np.ndarray, int64, shape (n_vox+1,)
                Per-voxel fencepost into the flat arrays (zero-indexed
                within this particle — i.e. shifted to start from 0).
            ``"input_ids"`` : np.ndarray, int64, shape (n_total_inputs,)
                Input point IDs for all voxels of this particle.
            ``"input_energies"`` : np.ndarray, float32, shape (n_total_inputs,)
                Energy contributions corresponding to each input_id.
        """
        if index < 0 or index >= self._n_events:
            raise IndexError(
                f"Event index {index} out of range [0, {self._n_events})"
            )

        p_start = int(self._ev_offsets[index])
        p_end   = int(self._ev_offsets[index + 1])
        n_parts = p_end - p_start

        if n_parts == 0:
            return []

        vox_start = int(self._vox_offsets[p_start])
        vox_end   = int(self._vox_offsets[p_end])
        n_voxels  = vox_end - vox_start

        vox_input_bounds = self._f["voxels/input_offsets"][vox_start : vox_end + 1]
        flat_start = int(vox_input_bounds[0])
        flat_end   = int(vox_input_bounds[-1])

        all_ids = self._f["flat/input_ids"     ][flat_start:flat_end]
        all_eng = self._f["flat/input_energies"][flat_start:flat_end]

        result = []
        for k in range(n_parts):
            v_start = int(self._vox_offsets[p_start + k])     - vox_start
            v_end   = int(self._vox_offsets[p_start + k + 1]) - vox_start
            # local fencepost into all_ids/all_eng (zero-indexed)
            local_vox_offsets = vox_input_bounds[v_start : v_end + 1] - flat_start
            f_s = int(local_vox_offsets[0])
            f_e = int(local_vox_offsets[-1])
            result.append({
                "particle_id":        int(self._particle_ids[p_start + k])
                                      if self._particle_ids is not None else None,
                "vox_input_offsets": local_vox_offsets - f_s,
                "input_ids":         all_ids[f_s:f_e],
                "input_energies":    all_eng[f_s:f_e],
            })

        return result

    def iter_events(self):
        """Iterate over all events in file order."""
        for i in range(self._n_events):
            yield self[i]

    @property
    def n_events(self) -> int:
        return self._n_events

    def __repr__(self) -> str:
        status = "open" if self._f.id.valid else "closed"
        return (f"VoxmapStore(path={self._vpath!r}, "
                f"n_events={self._n_events}, {status})")
