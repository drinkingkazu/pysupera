"""
Writer and reader for the 3.0.0 on-disk format.

3.0.0 collapses the three parallel particle tables of 2.x into one, and stores
each point exactly once.  See :mod:`pysupera.layout` for how a single event's
ordering, row selection and group ranges are decided; this module only puts
the result on disk and reads it back.

Layout
------
::

    format_version   scalar str  "3.0.0"
    n_events         scalar int64
    events/offsets   (n_events+1,) int64   particle rows per event
    points/offsets   (n_events+1,) int64   point rows per event
    points/flat      (N, 6) float32        x, y, z, time, dE, dX
    points/true_x_shift (N,) float32       true_x = flat[:, 0] + this
    particles/<col>  (M,)                  see _INT32 / _INT8 / _INT64
    inter/offsets    (n_events+1,) int64   interaction rows per event
    inter/<col>      (K,)                  see _INTER

Index conventions
-----------------
``id``, ``parent_id``, ``ancestor_id``, ``frag_id`` and ``inst_id`` are
**per-event** particle indices: row ``events/offsets[ev] + id``.

Point ranges (``pc_*``, ``frag_pc_*``, ``inst_pc_*``) are stored **absolute**
into ``points/flat``, so a whole-file reader can slice them with no arithmetic.
:class:`EventView` rebases them to its own ``points`` array on read, so that
within a view *everything* is event-local and consistent -- ``v["pc_start"]``
indexes ``v.points`` directly.  Mixing the two bases is the easiest mistake
this format allows, so the view does not expose the mixture.

``frag_id == id`` marks a fragment representative and ``inst_id == id`` an
instance representative; ``-1`` means the particle heads no group there.

``interaction_id`` is a row index into this event's slice of the interactions
table, so the vertex a particle came from is one lookup away.  Interactions no
stored particle references are dropped and the rest renumbered, which is why
the original vertex row survives as ``vertex_id``.
"""

from __future__ import annotations

import os
import numpy as np

from .io import _compress_kwargs, _chunk_rows, _CHUNK_TARGET_BYTES_META

FORMAT_VERSION_V3 = "3.1.0"

_PT_NDIM = 6                      # x, y, z, time, dE, dX

_INT32 = ("id", "geant4_trackid", "parent_id", "ancestor_id",
          "pdg", "parent_pdg",
          # The true Geant4 parent.  parent_id points at the nearest *stored*
          # ancestor, which is usually not the real one -- only ~17% of
          # particles are written -- so the real link is kept separately.
          "geant4_parent_trackid", "geant4_parent_pdg",
          "interaction_id", "interaction_type",
          "frag_id", "frag_merge_count", "frag_parent_id",
          "inst_id", "inst_merge_count", "inst_parent_id")
_INT8 = ("sem_type", "frag_sem_type", "inst_sem_type")
# Both levels describe two runs the same way: a non-LE range and an LE range.
# At instance level the two are adjacent, so inst_pc_end == inst_pc_le_start;
# at fragment level other fragments' points lie between them.
_INT64 = ("pc_start", "pc_end",
          "frag_pc_start", "frag_pc_end", "frag_pc_le_start", "frag_pc_le_end",
          "inst_pc_start", "inst_pc_end", "inst_pc_le_start", "inst_pc_le_end")
COLUMNS = _INT32 + _INT8 + _INT64

#: 3.0.0 column names that were renamed in 3.1.0, old -> new.
_RENAMED_3_0 = {
    "geant4_id":         "geant4_trackid",
    "sem_type_frag":     "frag_sem_type",
    "sem_type_inst":     "inst_sem_type",
    "parent_frag_id":    "frag_parent_id",
    "parent_inst_id":    "inst_parent_id",
    "frag_non_le_start": "frag_pc_start",
    "frag_non_le_end":   "frag_pc_end",
    "frag_le_start":     "frag_pc_le_start",
    "frag_le_end":       "frag_pc_le_end",
}

_DTYPE = {**{c: np.int32 for c in _INT32},
          **{c: np.int8 for c in _INT8},
          **{c: np.int64 for c in _INT64}}

# Ranges are stored absolute, so every one of these needs the event's point
# base added as events are appended.
_POINT_RANGE_COLS = ("pc_start", "pc_end",
                     "frag_pc_start", "frag_pc_end",
                     "frag_pc_le_start", "frag_pc_le_end",
                     "inst_pc_start", "inst_pc_end",
                     "inst_pc_le_start", "inst_pc_le_end")

# Interactions table.  part_* index this event's particle rows, pc_* the point
# array; both are rebased by the reader exactly as the particle columns are.
# interaction_id is this table's own dense index -- the value particles carry
# in their own interaction_id column.  vertex_id is the link back to the
# EDepSim vertex row, and survives the renumbering that drops dead vertices.
_INTER_I32 = ("interaction_id", "vertex_id", "part_start", "part_end")
_INTER_I64 = ("pc_start", "pc_end")
_INTER_F32 = ("x", "y", "z", "time", "energy_sum", "ke_sum")
_INTER_STR = ("reaction",)
INTER_COLUMNS = _INTER_I32 + _INTER_I64 + _INTER_F32 + _INTER_STR
_INTER_DTYPE = {**{c: np.int32 for c in _INTER_I32},
                **{c: np.int64 for c in _INTER_I64},
                **{c: np.float32 for c in _INTER_F32},
                **{c: object for c in _INTER_STR}}

#: 3.0.0 interaction columns that were renamed in 3.1.0.
_INTER_RENAMED_3_0 = {"t": "time"}
_INTER_POINT_COLS = ("pc_start", "pc_end")
_INTER_PART_COLS = ("part_start", "part_end")


class EventWriterV3:
    """Incremental writer.  One :meth:`append_event` call per event."""

    def __init__(self, path, compression="lz4", compression_opts=None):
        import h5py
        self._path = path
        self._ckw = _compress_kwargs(compression, compression_opts)
        self._f = h5py.File(path, "w")
        self._n_events = 0
        self._n_rows = 0
        self._n_points = 0
        self._n_inter = 0
        self._create()

    def _create(self):
        import h5py
        f = self._f
        f.create_dataset("format_version", data=FORMAT_VERSION_V3)
        f.create_dataset("n_events", data=np.int64(0))
        rows = _chunk_rows(1, target_bytes=_CHUNK_TARGET_BYTES_META)
        for grp in ("events", "points"):
            g = f.create_group(grp)
            g.create_dataset("offsets", data=np.zeros(1, dtype=np.int64),
                             maxshape=(None,), chunks=(rows,))
        f["points"].create_dataset(
            "flat", shape=(0, _PT_NDIM), maxshape=(None, _PT_NDIM),
            dtype=np.float32, chunks=(_chunk_rows(_PT_NDIM), _PT_NDIM),
            **self._ckw)
        # true_x = flat[:, 0] + true_x_shift.  Only pixel hit mode with a
        # nominal t0 puts anything but zero here, and it takes as many
        # distinct values as there are (interaction, volume) pairs -- 28 in a
        # test event -- so it costs about 1.5% of the file under LZ4.  A
        # separate dataset rather than a seventh column so that every reader
        # of the (N, 6) layout keeps working untouched.
        f["points"].create_dataset(
            "true_x_shift", shape=(0,), maxshape=(None,),
            dtype=np.float32, chunks=(_chunk_rows(1),), **self._ckw)
        g = f.create_group("inter")
        g.create_dataset("offsets", data=np.zeros(1, dtype=np.int64),
                         maxshape=(None,), chunks=(rows,))
        for c in INTER_COLUMNS:
            dt = (h5py.string_dtype() if c in _INTER_STR
                  else _INTER_DTYPE[c])
            g.create_dataset(c, shape=(0,), maxshape=(None,), dtype=dt,
                             chunks=(rows,), **self._ckw)

        # groups/: JAXTPC group -> owning particle id, one int32 per group,
        # -1 where no stored particle claims it.  Written only in JAXTPC
        # mode; absent otherwise.  volume_offsets records where each volume's
        # groups start so (volume, local group) is recoverable.
        gg = f.create_group("groups")
        gg.create_dataset("offsets", data=np.zeros(1, dtype=np.int64),
                          maxshape=(None,), chunks=(rows,))
        # Only the fragment is stored.  The instance is recoverable from it
        # positionally -- a fragment's points lie inside exactly one
        # instance's block -- so storing it too would be duplication.  See
        # pysupera.provenance.instance_of_groups.
        gg.create_dataset("fragment_id", shape=(0,), maxshape=(None,),
                          dtype=np.int32, chunks=(_chunk_rows(1, 4),),
                          **self._ckw)
        # is_le is exact at group level: a group's deposits belong to one
        # particle, so the flag is that particle's, not its instance's.
        gg.create_dataset("is_le", shape=(0,), maxshape=(None,),
                          dtype=np.int8, chunks=(_chunk_rows(1, 1),),
                          **self._ckw)
        gg.create_dataset("volume_offsets", shape=(0,), maxshape=(None,),
                          dtype=np.int64, chunks=(rows,))
        gg.create_dataset("volume_offsets_index",
                          data=np.zeros(1, dtype=np.int64),
                          maxshape=(None,), chunks=(rows,))
        self._n_groups = 0
        self._n_voloff = 0

        pg = f.create_group("particles")
        for c in COLUMNS:
            dt = _DTYPE[c]
            pg.create_dataset(c, shape=(0,), maxshape=(None,), dtype=dt,
                              chunks=(_chunk_rows(1, np.dtype(dt).itemsize,
                                                  _CHUNK_TARGET_BYTES_META),),
                              **self._ckw)

    def append_group_owners(self, frag_owner, is_le=None, volume_offsets=()):
        """
        Append one event's group-ownership table.

        Must be called once per event, in the same order as
        :meth:`append_event`.  Pass empty arrays for events with no group
        information, so the per-event offsets stay aligned.
        """
        g = self._f["groups"]
        frag_owner = np.asarray(frag_owner, dtype=np.int32)
        n = len(frag_owner)
        if is_le is None:
            is_le = np.zeros(n, dtype=np.int8)
        is_le = np.asarray(is_le, dtype=np.int8)
        if len(is_le) != n:
            raise ValueError(
                f"is_le must match the owner array: {len(is_le)} vs {n}")
        for name, arr in (("fragment_id", frag_owner), ("is_le", is_le)):
            ds = g[name]
            ds.resize(self._n_groups + n, axis=0)
            if n:
                ds[self._n_groups:self._n_groups + n] = arr
        self._n_groups += n
        off = g["offsets"]
        off.resize(off.shape[0] + 1, axis=0)
        off[-1] = self._n_groups

        vo = np.asarray(volume_offsets, dtype=np.int64)
        vds = g["volume_offsets"]
        vds.resize(self._n_voloff + len(vo), axis=0)
        if len(vo):
            vds[self._n_voloff:self._n_voloff + len(vo)] = vo
        self._n_voloff += len(vo)
        vidx = g["volume_offsets_index"]
        vidx.resize(vidx.shape[0] + 1, axis=0)
        vidx[-1] = self._n_voloff

    def set_meta(self, meta: dict) -> None:
        """
        Record how this file was produced, as root attributes.

        Only a handful of short strings, but they remove a guess: nothing in
        the point columns says whether the energy column holds true dE or
        measured charge, and a viewer that has to infer it from the presence
        of negative values gets it wrong on a quiet event.
        """
        for k, v in meta.items():
            if v is not None:
                self._f.attrs[str(k)] = str(v)

    def append_event(self, layout):
        """Append one event described by an :class:`~pysupera.layout.EventLayout`."""
        cols = layout.columns
        n_rows = len(layout.rows)
        n_pts = layout.n_points

        # ---- points -------------------------------------------------------
        pts = self._f["points/flat"]
        pts.resize(self._n_points + n_pts, axis=0)
        if n_pts:
            block = np.zeros((n_pts, _PT_NDIM), dtype=np.float32)
            cursor = 0
            for p in layout.order:
                pc = np.asarray(p.point_cloud, dtype=np.float32)
                k = len(pc)
                if k:
                    take = min(pc.shape[1], _PT_NDIM)
                    block[cursor:cursor + k, :take] = pc[:, :take]
                cursor += k
            pts[self._n_points:self._n_points + n_pts] = block

        shift = self._f["points"].get("true_x_shift")
        if shift is not None:
            shift.resize(self._n_points + n_pts, axis=0)
            if n_pts:
                sblock = np.zeros(n_pts, dtype=np.float32)
                cursor = 0
                for p in layout.order:
                    k = len(p.point_cloud)
                    v = getattr(p, "true_x_shift", None)
                    if k and v is not None and len(v) == k:
                        sblock[cursor:cursor + k] = v
                    cursor += k
                shift[self._n_points:self._n_points + n_pts] = sblock

        # ---- particle columns ---------------------------------------------
        base = self._n_points
        for c in COLUMNS:
            ds = self._f[f"particles/{c}"]
            ds.resize(self._n_rows + n_rows, axis=0)
            if n_rows:
                vals = np.asarray(cols[c], dtype=_DTYPE[c])
                if c in _POINT_RANGE_COLS:
                    # Shift into absolute file coordinates, leaving the -1
                    # sentinel (particle heads no group at this level) alone.
                    vals = np.where(vals >= 0, vals + base, -1).astype(_DTYPE[c])
                ds[self._n_rows:self._n_rows + n_rows] = vals

        # ---- interactions ---------------------------------------------------
        inter = layout.interactions or {}
        n_int = len(inter.get("vertex_id", ()))
        for c in INTER_COLUMNS:
            ds = self._f[f"inter/{c}"]
            ds.resize(self._n_inter + n_int, axis=0)
            if n_int:
                if c in _INTER_STR:
                    ds[self._n_inter:self._n_inter + n_int] = [
                        (x.decode() if isinstance(x, bytes) else str(x))
                        for x in inter[c]]
                    continue
                vals = np.asarray(inter[c], dtype=_INTER_DTYPE[c])
                if c in _INTER_POINT_COLS:
                    vals = np.where(vals >= 0, vals + base, -1).astype(_INTER_DTYPE[c])
                elif c in _INTER_PART_COLS:
                    vals = np.where(vals >= 0, vals + self._n_rows, -1
                                    ).astype(_INTER_DTYPE[c])
                ds[self._n_inter:self._n_inter + n_int] = vals
        self._n_inter += n_int

        # ---- fenceposts ----------------------------------------------------
        self._n_rows += n_rows
        self._n_points += n_pts
        self._n_events += 1
        for grp, total in (("events", self._n_rows), ("points", self._n_points),
                           ("inter", self._n_inter)):
            ds = self._f[f"{grp}/offsets"]
            ds.resize(self._n_events + 1, axis=0)
            ds[self._n_events] = total
        self._f["n_events"][()] = np.int64(self._n_events)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class EventView:
    """One event's rows and points, as plain arrays."""

    __slots__ = ("columns", "points", "event", "interactions")

    def __init__(self, columns, points, event, interactions=None):
        self.columns = columns
        self.points = points
        self.event = event
        #: per-interaction arrays; ``columns["interaction_id"]`` indexes these
        self.interactions = interactions or {}

    def __len__(self):
        return len(self.columns["id"])

    def __getitem__(self, name):
        return self.columns[name]

    @property
    def is_fragment(self):
        """Boolean mask: this row represents a fragment."""
        return self.columns["frag_id"] == self.columns["id"]

    @property
    def is_instance(self):
        """Boolean mask: this row represents an instance."""
        return self.columns["inst_id"] == self.columns["id"]

    def interaction_of(self, row):
        """The interactions-table row this particle belongs to, or ``None``."""
        k = int(self.columns["interaction_id"][row])
        if k < 0 or not self.interactions:
            return None
        return {c: self.interactions[c][k] for c in self.interactions}

    def points_of(self, row, level=None, le=None):
        """
        Points of *row*.

        *level* selects whose points: ``None`` for the particle's own,
        ``'frag'`` or ``'inst'`` for its group's.  *le* selects which kind:
        ``None`` for all, ``False`` for non-LE only, ``True`` for LE only.

        Every combination is one contiguous slice.  At instance level the
        non-LE and LE runs are adjacent (``inst_pc_end == inst_pc_le_start``)
        so "all" is still a single slice; at
        fragment level the two sides are stored as separate ranges because
        other fragments' points lie between them.
        """
        c = self.columns
        if level == "inst":
            s, e = int(c["inst_pc_start"][row]), int(c["inst_pc_end"][row])
            ls, le_ = (int(c["inst_pc_le_start"][row]),
                       int(c["inst_pc_le_end"][row]))
            if le is True:
                return self.points[:0] if ls < 0 else self.points[ls:le_]
            if le is False:
                return self.points[:0] if s < 0 else self.points[s:e]
            # Both: the runs are adjacent at instance level, so one slice.
            lo = s if s >= 0 else ls
            hi = le_ if le_ >= 0 else e
            return self.points[:0] if lo < 0 else self.points[lo:hi]
        if level == "frag":
            if le is None:
                # The two sides are not adjacent, so "all" needs both.
                a = self.points_of(row, "frag", False)
                b = self.points_of(row, "frag", True)
                if not len(a): return b
                if not len(b): return a
                return np.concatenate([a, b])
            pre = "frag_pc_le" if le else "frag_pc"
            s, e = int(c[f"{pre}_start"][row]), int(c[f"{pre}_end"][row])
            return self.points[:0] if s < 0 else self.points[s:e]
        s, e = int(c["pc_start"][row]), int(c["pc_end"][row])
        return self.points[:0] if s < 0 else self.points[s:e]

    def __repr__(self):
        return (f"EventView(event={self.event}, rows={len(self)}, "
                f"points={len(self.points)})")


class EventStoreV3:
    """Reader for a 3.0.0 file."""

    def __init__(self, path):
        import h5py
        self._f = h5py.File(path, "r")
        ver = self._f["format_version"][()]
        ver = ver.decode() if isinstance(ver, bytes) else str(ver)
        if not ver.startswith("3."):
            raise ValueError(
                f"{path!r} is format {ver}, not 3.x.  Use pysupera.io."
                f"EventStore for 2.x files."
            )
        self.format_version = ver
        #: 3.0.0 files use the pre-rename column names and encode the
        #: instance LE boundary as a single split index.
        self._legacy = ver.startswith("3.0")
        self._ev = self._f["events/offsets"][:]
        self._po = self._f["points/offsets"][:]
        self._io = (self._f["inter/offsets"][:]
                    if "inter/offsets" in self._f else None)

    def __len__(self):
        return int(self._f["n_events"][()])

    def __getitem__(self, ev):
        n = len(self)
        if ev < -n or ev >= n:
            raise IndexError(f"event {ev} out of range for {n} events")
        if ev < 0:
            ev += n
        a, b = int(self._ev[ev]), int(self._ev[ev + 1])
        pa, pb = int(self._po[ev]), int(self._po[ev + 1])
        cols = (self._read_legacy_particles(a, b) if self._legacy
                else {c: self._f[f"particles/{c}"][a:b] for c in COLUMNS})
        # Rebase the absolute point ranges onto this event's slice, leaving
        # the -1 sentinel alone, so every index in the view shares one origin.
        for c in _POINT_RANGE_COLS:
            v = cols[c]
            cols[c] = np.where(v >= 0, v - pa, -1).astype(v.dtype)
        inter: dict = {}
        if self._io is not None:
            ia, ib = int(self._io[ev]), int(self._io[ev + 1])
            for c in INTER_COLUMNS:
                name = c
                if self._legacy:
                    old = {v: k for k, v in _INTER_RENAMED_3_0.items()}.get(c, c)
                    if f"inter/{old}" not in self._f:
                        # 3.0.0 has no interaction_id / sums / reaction
                        n = ib - ia
                        inter[c] = (np.arange(n, dtype=np.int32)
                                    if c == "interaction_id" else
                                    np.array([""] * n, dtype=object)
                                    if c in _INTER_STR else
                                    np.zeros(n, dtype=_INTER_DTYPE[c]))
                        continue
                    name = old
                v = self._f[f"inter/{name}"][ia:ib]
                if c in _INTER_STR:
                    inter[c] = np.array(
                        [x.decode() if isinstance(x, bytes) else str(x)
                         for x in v], dtype=object)
                    continue
                if c in _INTER_POINT_COLS:
                    v = np.where(v >= 0, v - pa, -1).astype(v.dtype)
                elif c in _INTER_PART_COLS:
                    v = np.where(v >= 0, v - a, -1).astype(v.dtype)
                inter[c] = v
        return EventView(cols, self._f["points/flat"][pa:pb], ev, inter)

    def true_x_shift(self, ev):
        """
        Per point, what to add to ``x`` to recover the true drift position.

        ``true_x = view.points[:, 0] + store.true_x_shift(ev)``.  Zero
        everywhere except a pixel hit-mode run read with a nominal t0, and
        absent from files written before the column existed.
        """
        ds = self._f["points"].get("true_x_shift")
        if ds is None:
            return None
        po = self._f["points/offsets"]
        a, b = int(po[ev]), int(po[ev + 1])
        return ds[a:b]

    def _read_legacy_particles(self, a, b):
        """
        Read a 3.0.0 particle block under 3.1.0 names.

        Two shapes differ, not just names: the Geant4 parent columns did not
        exist (filled with -1, since the value is unrecoverable from the file
        alone), and the instance LE boundary was one split index rather than
        two adjacent ranges.
        """
        f, old_of = self._f, {v: k for k, v in _RENAMED_3_0.items()}
        cols = {}
        for c in COLUMNS:
            if c in ("geant4_parent_trackid", "geant4_parent_pdg"):
                continue                      # filled below
            if c.startswith("inst_pc_"):
                continue                      # derived below
            cols[c] = f[f"particles/{old_of.get(c, c)}"][a:b]

        n = b - a
        for c in ("geant4_parent_trackid", "geant4_parent_pdg"):
            cols[c] = np.full(n, -1, dtype=_DTYPE[c])

        st = f["particles/inst_pc_start"][a:b]
        sp = f["particles/inst_pc_split"][a:b]
        en = f["particles/inst_pc_end"][a:b]
        cols["inst_pc_start"] = st
        cols["inst_pc_end"] = sp            # non-LE run ends at the split
        cols["inst_pc_le_start"] = np.where(st >= 0, sp, -1).astype(sp.dtype)
        cols["inst_pc_le_end"] = en
        return cols

    def group_owners(self, ev):
        """
        ``(instance_id, fragment_id, volume_offsets)`` for event *ev*, or
        ``(None, None, None)`` when the file carries no group table
        (non-JAXTPC runs).

        Indexed by event-global group number; ``fragment_id`` is -1 where no
        stored fragment claims the group.  The owning *instance* is not
        stored -- derive it with
        :func:`pysupera.provenance.instance_of_groups`.  *volume_offsets* says where
        each volume's groups begin, so a plane's local group number can be
        shifted into this space.  See :mod:`pysupera.provenance`.
        """
        if "groups/fragment_id" not in self._f:
            return None, None, None
        go = self._f["groups/offsets"]
        if ev + 1 >= go.shape[0]:
            return None, None, None
        a, b = int(go[ev]), int(go[ev + 1])
        vi = self._f["groups/volume_offsets_index"]
        va, vb = int(vi[ev]), int(vi[ev + 1])
        le = (self._f["groups/is_le"][a:b] if "groups/is_le" in self._f
              else np.zeros(b - a, dtype=np.int8))
        return (self._f["groups/fragment_id"][a:b], le,
                self._f["groups/volume_offsets"][va:vb])

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_writer_v3(path, compression="lz4", compression_opts=None):
    """Open an :class:`EventWriterV3`."""
    return EventWriterV3(path, compression, compression_opts)


def read_events_v3(path):
    """Open an :class:`EventStoreV3`."""
    return EventStoreV3(path)
