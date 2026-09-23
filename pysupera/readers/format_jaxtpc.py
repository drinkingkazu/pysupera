"""
Reader for JAXTPC simulation output — visibility-filtered EDepSim data.

JAXTPC is a JAX-based TPC simulation that takes EDepSim truth deposits,
drifts them through an electric field, and simulates wire-plane or pixel
readout with realistic noise and thresholds.  The simulation produces three
file types that together fully describe one batch of events:

=====================  ======================================================
File                   Description
=====================  ======================================================
*seg* (seg HDF5)       3-D truth deposits per volume: positions, dE/dx, charge,
                       etc.  One entry per Geant4 step (segment).
*inst* (inst HDF5)     Per-readout-plane correspondence data.  Each plane
                       sub-group stores the ``group_ids`` of segments that
                       survived the signal threshold and were reconstructed.
                       Also stores the lookup tables ``segment_to_group``
                       (segment index → group index; newer JAXTPC output names
                       this ``deposit_to_group``) and ``group_to_track``
                       (group index → Geant4 track ID) for each volume.
*edepsim* (H5)         Original EDepSim HDF5 file with particle-level metadata
                       (PDG, parent/root track IDs, start vertex, interaction
                       type, etc.) and the full flat step array.
=====================  ======================================================

This reader combines all three files to return
:class:`~pysupera.data.Particle` objects whose point clouds contain **only
the energy-deposit segments that were visible** in the JAXTPC readout — i.e.
whose group_id appeared in at least one readout-plane's ``group_ids`` array.

Wire and pixel readout
----------------------
JAXTPC simulates two LArTPC readout geometries, and this reader handles both
without being told which it is looking at:

=========  ==============  =============================================
readout    plane subgroup  hit centre stored per group
=========  ==============  =============================================
``wire``   ``U/V/Y``       ``center_wires``, ``center_times`` — three 2-D
                           projections of the same ionisation
``pixel``  ``Pixel``       ``center_py``, ``center_pz``, ``center_times``
                           — one natively 3-D image
=========  ==============  =============================================

Visibility is a question about *groups*, and a group is above threshold or it
is not regardless of how the readout is arranged, so the filtering below reads
only ``group_ids`` and never the hit centres.  That is why the two geometries
share one code path: the reader discovers plane subgroups structurally rather
than by name.  Downstream consumers that *do* draw hits (``vis_hits.html``,
``pysupera-hits-subset``) need the distinction, and get it from
:func:`read_readout_type`.

Visibility logic (mirrors ``get_visible_segments_by_track`` in
``seg_visibility.ipynb``):
    1. Open the inst file and, for each volume, scan every readout-plane
       sub-group to collect the set of *active_group_ids* (groups above
       threshold).
    2. Map every segment in that volume to its group via
       ``segment_to_group`` / ``deposit_to_group``.  A segment is *visible* if
       its group is in ``active_group_ids``.
    3. Map visible group IDs to Geant4 track IDs via ``group_to_track``.
    4. Collect visible segments per track across all volumes, then build a
       flat point cloud and per-particle offset array for
       :meth:`~pysupera.data.Particle.from_flat_arrays`.

Usage
-----
::

    from pysupera.readers import JaxtpcHDF5Reader

    with JaxtpcHDF5Reader(
        edepsim_path="edepsim.h5",
        seg_path="sim_seg_0000.h5",
        inst_path="sim_inst_0000.h5",
    ) as reader:
        print(len(reader), "events")
        particles = reader[0]          # list[Particle], visible segments only
        for particles in reader:
            ...
"""

from __future__ import annotations

import numpy as np

# Register HDF5 compression filters (LZ4, Blosc, etc.) if available.
# JAXTPC seg/inst files are written with Blosc (HDF5 filter 32001), which
# h5py cannot decode unless these filters have been registered first.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from .base import EventReaderBase
from .format_edepsim_h5 import (
    _DEFAULT_VERTEX_KEY,
    _DEFAULT_PARTICLE_KEY,
    _DEFAULT_ELECTRON_ENERGY_THRESHOLD,
    _get_parent_pdg,
    _get_interaction_id,
    _get_interaction_type,
    _get_ancestor_id,
    _to_index_space,
)
from ..data import Particle
from ..utils import PointFeature
from . import pixel_hits as _px


#: What JAXTPC calls the geometry, on the inst file's ``config`` group.
#: ``wire`` is the fallback rather than an error because JAXTPC wrote hits
#: files before the attribute existed, and its own loader assumes wire in
#: exactly the same way (``production/load.py``).
_DEFAULT_READOUT_TYPE = 'wire'

#: Plane subgroup names JAXTPC writes for each readout type.  Used only for
#: error messages and by consumers that draw hits — the visibility filter
#: below discovers planes structurally, so an unlisted name still works.
READOUT_PLANE_NAMES = {'wire': ('U', 'V', 'Y'), 'pixel': ('Pixel',)}


def read_readout_type(inst_file) -> str:
    """
    ``'wire'`` or ``'pixel'`` for an open JAXTPC inst/hits file.

    JAXTPC records this as ``config.attrs['readout_type']``.  A file without
    it predates the attribute and is wire.
    """
    cfg = inst_file.get('config') if hasattr(inst_file, 'get') else None
    raw = cfg.attrs.get('readout_type', _DEFAULT_READOUT_TYPE) if cfg is not None \
        else _DEFAULT_READOUT_TYPE
    return raw.decode() if isinstance(raw, bytes) else str(raw)


# Per-volume table mapping segment (deposit) index → group index.  Older
# JAXTPC output names this ``segment_to_group``; newer output writes
# ``deposit_to_group``.  Both spellings are accepted, in this order.
_SEG_TO_GROUP_KEYS = ('segment_to_group', 'deposit_to_group')

# Datasets every non-empty seg volume must provide.
_SEG_REQUIRED_DATASETS = ('positions', 'de', 'dx', 't0_us', 'charge')

# Attributes every non-empty seg volume must provide.
_SEG_REQUIRED_ATTRS = ('pos_step_mm', 'pos_origin_x', 'pos_origin_y', 'pos_origin_z')


# ---------------------------------------------------------------------------
# Visibility filtering (ported from seg_visibility.ipynb)
# ---------------------------------------------------------------------------

def get_visible_segment_indices_by_track(
    seg_volumes: list[dict],
    inst_file,          # open h5py.File handle for the inst file
    event_key: str,
) -> list[dict[int, np.ndarray]]:
    """
    For each volume, return a mapping from Geant4 track ID to the indices
    of its *visible* segments (i.e. segments whose group appeared in the
    readout above threshold).

    Parameters
    ----------
    seg_volumes : list of dict
        Per-volume segment dicts as returned by ``load_event_seg``.  Only
        ``n_actual`` is used here (to bound the ``segment_to_group`` slice).
    inst_file : h5py.File
        Open read-only handle to the inst HDF5 file.
    event_key : str
        HDF5 group name for this event, e.g. ``"event_000"``.

    Returns
    -------
    list of dict
        One dict per volume.  Each maps ``track_id (int)`` →
        ``seg_indices (np.ndarray of int32)``, the row indices of that
        track's visible segments in the seg volume array.  Tracks with no
        visible segments are absent from the dict.
    """
    result: list[dict[int, np.ndarray]] = []

    if event_key not in inst_file:
        raise KeyError(
            f"Event group {event_key!r} not found in the JAXTPC inst file "
            f"{inst_file.filename!r}.  Top-level groups present: "
            f"{sorted(inst_file.keys())[:6]}.  The inst file must cover the "
            f"same events as the EDepSim file given by io.input_path."
        )

    event_group = inst_file[event_key]

    for v, seg in enumerate(seg_volumes):
        track_to_indices: dict[int, np.ndarray] = {}
        vol_key = f'volume_{v}'

        # A missing volume group means the inst file does not describe the same
        # volumes as the seg file — always a configuration error, never data.
        if vol_key not in event_group:
            raise KeyError(
                f"{event_key}/{vol_key} is present in the JAXTPC seg file but "
                f"missing from the inst file {inst_file.filename!r} "
                f"(volumes present: {sorted(event_group.keys())}).  The seg and "
                f"inst files must come from the same JAXTPC run."
            )

        # A volume that genuinely holds zero segments is legitimately empty.
        if seg.get('n_actual', 0) == 0:
            result.append(track_to_indices)
            continue

        vol_group = event_group[vol_key]

        # look-up tables stored in the inst file for this volume
        # group_to_track : group index → Geant4 track ID
        # seg_to_group   : segment index → group index (see _SEG_TO_GROUP_KEYS)
        seg_to_group_key = next(
            (k for k in _SEG_TO_GROUP_KEYS if k in vol_group), None
        )
        if 'group_to_track' not in vol_group or seg_to_group_key is None:
            raise KeyError(
                f"{event_key}/{vol_key} in the JAXTPC inst file "
                f"{inst_file.filename!r} is missing the visibility lookup "
                f"tables.  Required: 'group_to_track' plus one of "
                f"{list(_SEG_TO_GROUP_KEYS)}; found {sorted(vol_group.keys())}.  "
                f"Check that reader.jaxtpc_inst_path points at the JAXTPC "
                f"hits/inst file (not the seg or sensor file)."
            )

        group_to_track = vol_group['group_to_track'][:]           # (G,)  int32
        n_segs         = seg['n_actual']
        seg_to_group   = vol_group[seg_to_group_key][:n_segs]     # (N,)  int32

        # Collect the group IDs that produced a readout signal above threshold
        # by scanning every readout-plane sub-group stored under this volume.
        active_group_ids: set[int] = set()
        import h5py
        n_planes = 0
        for plane_key in vol_group:
            plane_group = vol_group[plane_key]
            if isinstance(plane_group, h5py.Group) and 'group_ids' in plane_group:
                n_planes += 1
                active_group_ids.update(plane_group['group_ids'][:].tolist())

        # No readout-plane sub-groups at all is a schema problem; planes that
        # exist but recorded nothing above threshold are legitimately empty.
        if n_planes == 0:
            raise KeyError(
                f"{event_key}/{vol_key} in the JAXTPC inst file "
                f"{inst_file.filename!r} contains no readout-plane sub-group "
                f"with a 'group_ids' dataset (expected U/V/Y for wire readout "
                f"or Pixel for pixel readout); found "
                f"{sorted(vol_group.keys())}.  Without these, segment "
                f"visibility cannot be determined."
            )

        if not active_group_ids:
            result.append(track_to_indices)
            continue

        # Index arrays for all segments in this volume.
        seg_indices_all = np.arange(n_segs, dtype=np.int32)
        group_ids_all   = seg_to_group  # group ID assigned to each segment

        # Boolean mask: True where the segment belongs to an active group.
        is_visible = np.isin(group_ids_all, list(active_group_ids))

        # Restrict to visible segments only.
        visible_seg_indices = seg_indices_all[is_visible]
        visible_group_ids   = group_ids_all[is_visible]

        # Map visible group IDs → Geant4 track IDs.
        # Groups whose index is out of range receive track ID -1.
        track_ids_for_visible = np.where(
            visible_group_ids < len(group_to_track),
            group_to_track[visible_group_ids],
            -1,
        )

        # Group visible segment indices by track ID.
        #
        # Sort once and split at the boundaries: O(N log N).  Comparing the
        # whole array against each unique track instead is O(T*N), and with
        # ~5k tracks over ~250k visible segments that single loop was the
        # largest self-time cost in the whole pipeline.
        if len(track_ids_for_visible):
            order      = np.argsort(track_ids_for_visible, kind='stable')
            tid_sorted = track_ids_for_visible[order]
            seg_sorted = visible_seg_indices[order]
            edges = np.flatnonzero(
                np.r_[True, tid_sorted[1:] != tid_sorted[:-1], True]
            )
            for a, b in zip(edges[:-1], edges[1:]):
                track_to_indices[int(tid_sorted[a])] = seg_sorted[a:b]

        result.append(track_to_indices)

    return result


# ---------------------------------------------------------------------------
# Point-cloud builder
# ---------------------------------------------------------------------------

def _build_visible_point_cloud(
    seg_volumes: list[dict],
    visible_by_track: list[dict[int, np.ndarray]],
    track_ids_edepsim: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a flat point-cloud array and per-particle offset pairs from the
    visibility-filtered JAXTPC seg data.

    The output columns follow :class:`~pysupera.utils.PointFeature` order:
    ``x=0, y=1, z=2, time=3, energy=4, dx=5, id=6``.

    Segment positions are taken from each volume's ``positions_mm``
    (x, y, z in mm).  Time comes from ``t0_us``, energy from ``de``,
    and path length from ``dx``.  Column 6 (``PointFeature.id``) is
    assigned as a global 0-based integer across all visible segments in
    the event.

    The returned ``deposit_id`` is the *input* deposit index -- the row in
    the seg file that produced each point -- offset per volume so one
    integer identifies a deposit across the whole event.  It is carried as
    int64 rather than in the float32 cloud, because the cloud is exact only
    to 2**24 and a future hundredfold increase in deposits would reach that.

    Parameters
    ----------
    seg_volumes : list of dict
        Per-volume segment dicts from ``load_event_seg``.
    visible_by_track : list of dict
        Per-volume ``{track_id: seg_indices}`` as returned by
        :func:`get_visible_segment_indices_by_track`.
    track_ids_edepsim : np.ndarray of int
        Track IDs in the same order as the edepsim particle array.  The
        output offsets are aligned to this ordering.

    Returns
    -------
    point_cloud_flat : np.ndarray, shape (M, 7), dtype float32
        Concatenated visible segments for all tracks, in the order of
        ``track_ids_edepsim``.
    offsets : np.ndarray, shape (N, 2), dtype int64
        ``offsets[i] = [start, end)`` into ``point_cloud_flat`` for
        particle *i*.  Particles with no visible segments get a zero-length
        slice (``start == end``).
    deposit_id : np.ndarray, shape (M,), dtype int64
        Event-global input deposit index for each row of
        *point_cloud_flat*.
    volume_offsets : list of int
        Where each volume's deposits start in that global index, so
        ``(volume, local index)`` can be recovered.
    """
    # Merge per-volume dicts into a single {track_id: list_of_segment_rows}
    # where each segment row is a 1-D float32 array of length 7.
    track_segs: dict[int, list[np.ndarray]] = {}
    track_deps: dict[int, list[np.ndarray]] = {}

    # Deposit indices are per volume in the seg file; offset them so one
    # integer names a deposit across the event.
    volume_offsets: list[int] = []
    _cum = 0
    for sv in seg_volumes:
        volume_offsets.append(_cum)
        _cum += int(sv.get('n_actual', 0) or 0)

    for _vi, (seg_vol, vol_visible) in enumerate(
            zip(seg_volumes, visible_by_track)):
        if seg_vol.get('n_actual', 0) == 0:
            continue
        pos   = seg_vol['positions_mm']   # (N, 3)  x, y, z in mm
        t0    = seg_vol['t0_us']          # (N,)    drift start time in μs
        de    = seg_vol['de']             # (N,)    energy deposit in MeV
        dx    = seg_vol['dx']             # (N,)    path length in mm

        for track_id, seg_idx in vol_visible.items():
            if len(seg_idx) == 0:
                continue
            # Build a (M, 7) float32 array for this track's visible segments
            # in this volume; PointFeature.id column is filled globally later.
            chunk = np.zeros((len(seg_idx), 7), dtype=np.float32)
            chunk[:, PointFeature.x]      = pos[seg_idx, 0]
            chunk[:, PointFeature.y]      = pos[seg_idx, 1]
            chunk[:, PointFeature.z]      = pos[seg_idx, 2]
            chunk[:, PointFeature.time]   = t0[seg_idx]
            chunk[:, PointFeature.energy] = de[seg_idx]
            chunk[:, PointFeature.dx]     = dx[seg_idx]
            # id column left as zero; assigned globally below.
            track_segs.setdefault(track_id, []).append(chunk)
            track_deps.setdefault(track_id, []).append(
                volume_offsets[_vi] + np.asarray(seg_idx, dtype=np.int64))

    # Concatenate chunks per track and build the flat array + offsets in the
    # same order as track_ids_edepsim so that Particle.from_flat_arrays
    # correctly pairs metadata rows with point-cloud slices.
    n_particles = len(track_ids_edepsim)
    offsets = np.zeros((n_particles, 2), dtype=np.int64)
    flat_chunks: list[np.ndarray] = []
    dep_chunks: list[np.ndarray] = []
    cursor = 0

    for i, tid in enumerate(track_ids_edepsim):
        chunks = track_segs.get(int(tid))
        if chunks is None:
            # Track has no visible segments — leave an empty slice.
            offsets[i] = [cursor, cursor]
            continue
        combined = np.concatenate(chunks, axis=0)  # (M_i, 7)
        offsets[i] = [cursor, cursor + len(combined)]
        flat_chunks.append(combined)
        dep_chunks.append(np.concatenate(track_deps[int(tid)], axis=0))
        cursor += len(combined)

    if flat_chunks:
        point_cloud_flat = np.concatenate(flat_chunks, axis=0)
        deposit_id = np.concatenate(dep_chunks, axis=0)
    else:
        point_cloud_flat = np.zeros((0, 7), dtype=np.float32)
        deposit_id = np.zeros(0, dtype=np.int64)

    # Assign global, event-wide, 0-based point IDs.
    point_cloud_flat[:, PointFeature.id] = np.arange(
        len(point_cloud_flat), dtype=np.float32
    )

    return point_cloud_flat, offsets, deposit_id, volume_offsets


# ---------------------------------------------------------------------------
# Reader class
# ---------------------------------------------------------------------------

class JaxtpcHDF5Reader(EventReaderBase):
    """
    Produce visibility-filtered pysupera :class:`~pysupera.data.Particle`
    objects by combining an EDepSim HDF5 file, a JAXTPC *seg* file, and a
    JAXTPC *inst* file.

    Each particle's point cloud contains **only** the energy-deposit
    segments that reached the readout above the signal threshold in the
    JAXTPC simulation (i.e. segments that are "visible" to the detector).
    Particles whose every segment was below threshold will have an empty
    (or sub-threshold) point cloud in the output.

    Parameters
    ----------
    edepsim_path : str
        Path to the EDepSim HDF5 file (provides particle-level metadata:
        PDG, track IDs, start vertices, etc.).
    seg_path : str
        Path to the JAXTPC *seg* HDF5 file (provides per-segment truth
        quantities: positions in mm, dE, dx, t0, charge, …).
    inst_path : str
        Path to the JAXTPC *inst* HDF5 file (provides ``segment_to_group`` or
        ``deposit_to_group``, ``group_to_track``, and per-plane ``group_ids``
        used to determine which segments were detected).
    vertex_key : str, optional
        HDF5 dataset key for per-event vertex arrays in *edepsim_path*.
        Default: ``"vertex/geant4"``.
    particle_key : str, optional
        HDF5 dataset key for per-event particle arrays in *edepsim_path*.
        Default: ``"particle/geant4"``.
    electron_energy_threshold : float, optional
        Kinetic-energy threshold (MeV) for the interaction-type classifier
        (forwarded to :func:`~.format_edepsim_h5._get_interaction_type`).
        Default: ``0.05``.
    min_pc_size : int or None, optional
        Forwarded to :meth:`~pysupera.data.Particle.from_flat_arrays`.
        ``None`` (default) uses the module-level default.

    Examples
    --------
    ::

        from pysupera.readers import JaxtpcHDF5Reader

        with JaxtpcHDF5Reader(
            edepsim_path="out_edepsim.h5",
            seg_path="sim_seg_0000.h5",
            inst_path="sim_inst_0000.h5",
        ) as reader:
            print(len(reader), "events")
            for particles in reader:
                ...
    """

    def __init__(
        self,
        edepsim_path: str,
        seg_path: str,
        inst_path: str,
        *,
        vertex_key: str   = _DEFAULT_VERTEX_KEY,
        particle_key: str = _DEFAULT_PARTICLE_KEY,
        electron_energy_threshold: float = _DEFAULT_ELECTRON_ENERGY_THRESHOLD,
        min_pc_size: int | None = None,
        voxel_size: float | None = None,
        point_source: str = 'deposits',
        hit_x_from: str = 'nominal',
        hit_energy: str = 'charge',
        hit_reference_tick: float = 0.0,
        hit_charge_threshold: float = 0.0,
        sensor_path: str | None = None,
        pixel_pitch_mm: float | None = None,
        pixel_drift_direction=None,
        drift_velocity_mm_us: float | None = None,
        readout_time_step_us: float | None = None,
        pixel_geometry_from_fit: bool = False,
        pixel_geometry: list | None = None,
    ) -> None:
        import h5py

        if point_source not in ('deposits', 'hits'):
            raise ValueError(
                f"point_source must be 'deposits' or 'hits', "
                f"got {point_source!r}")
        self._point_source = point_source
        self._hit_x_from = hit_x_from
        self._hit_energy = hit_energy
        self._hit_reference_tick = float(hit_reference_tick or 0.0)
        # Drop hits below this many induced electrons.  JAXTPC applies
        # essentially none, so this is where a readout threshold enters.
        self._hit_charge_threshold = float(hit_charge_threshold or 0.0)
        # Stated pixel geometry.  Preferred over anything fitted: see
        # pysupera.readers.pixel_hits.verify_volume_geometry for why a
        # constant derived from the truth it is later checked against cannot
        # catch a geometry that is wrong but linear.
        self._sensor_path = sensor_path
        self._pixel_pitch_mm = pixel_pitch_mm
        self._pixel_drift_direction = pixel_drift_direction
        self._drift_velocity_mm_us = drift_velocity_mm_us
        self._readout_time_step_us = readout_time_step_us
        self._geometry_from_fit = bool(pixel_geometry_from_fit)
        # Calibrated on first use and reused; the geometry is a property of
        # the run, not of the event.  An explicit list skips calibration.
        self._pixel_geoms = pixel_geometry
        self._pixel_report = None

        #: Hit counts for the most recently read event in hit mode.
        self.last_hit_stats: dict | None = None

        #: Groups in the most recently read event, spanning every volume.
        self.last_n_groups: int | None = None

        #: Per input hit, what to add to x to recover the true drift
        #: position.  Hit mode only; None otherwise.
        self.last_true_x_shift = None

        self._edepsim_path = edepsim_path
        self._seg_path     = seg_path
        self._inst_path    = inst_path
        self._vertex_key   = vertex_key
        self._part_key     = particle_key
        self._e_thresh     = electron_energy_threshold
        self._min_pc_size  = min_pc_size
        # Cell size classification measures a cloud's extent in; see
        # pysupera.utils.count_extent for why rows will not do.
        self._voxel_size   = voxel_size

        # Visibility counts for the most recently read event; populated by
        # __getitem__.  None until the first event has been read.
        self.last_mask_stats: dict | None = None

        # Open all three files for the lifetime of the reader.
        self._edepsim_file = h5py.File(edepsim_path, 'r')
        self._seg_file     = h5py.File(seg_path, 'r')
        self._inst_file    = h5py.File(inst_path, 'r')
        self._sensor_file  = (h5py.File(sensor_path, 'r')
                              if sensor_path else None)

        self._n_events = len(self._edepsim_file[self._part_key])

    # ------------------------------------------------------------------ #
    # EventReaderBase interface                                            #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._n_events

    def __getitem__(self, index: int) -> list[Particle]:
        """
        Return visibility-filtered :class:`~pysupera.data.Particle` objects
        for event *index*.

        Parameters
        ----------
        index : int
            0-based event index.  Negative indices are supported.

        Returns
        -------
        list of Particle

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        n = self._n_events
        if index < -n or index >= n:
            raise IndexError(
                f"Event index {index} out of range for file with {n} events."
            )
        if index < 0:
            index += n

        event_key = f'event_{index:03d}'

        # ── Step 1: load particle metadata from the EDepSim file ──────────
        # These provide PDG code, track IDs, interaction type, etc.
        verts = self._edepsim_file[self._vertex_key][index]
        #: vertices of the most recently read event, for the interactions table
        self.last_vertices = verts
        parts = self._edepsim_file[self._part_key][index]

        # ── Step 2: load per-volume segment data from the JAXTPC seg file ─
        # Each volume dict contains positions_mm (N,3), de, dx, t0_us, …
        seg_volumes = self._load_seg_volumes(event_key)

        # ── Steps 3 and 4: build the point cloud ──────────────────────────
        # Two sources, and they differ in what a point *is*.  In deposit mode
        # a point is a Geant4 energy deposit the readout happened to detect,
        # so the geometry is truth geometry and the readout only selects.  In
        # hit mode a point is a fired pixel: the detected image itself, with
        # pixelation, diffusion, threshold and -- unless hit_x_from removes
        # it -- the drift-time ambiguity baked into the coordinates.
        _hit_mode = self._point_source == 'hits'
        if _hit_mode:
            (point_cloud_flat, offsets, _group_of_point, _x_shift,
             _g_offs, _hstats) = self._build_hit_event(
                event_key, seg_volumes, parts['track_id'])
            #: true_x = x + this, per input hit.  See pixel_hits.
            self.last_true_x_shift = _x_shift
            self.last_hit_stats = _hstats
            # No visibility pass is run here: hit mode never asks which
            # deposits were detected, it reads what the readout recorded.
            self.last_mask_stats = None
            # A point *is* a hit, so its group is read rather than resolved
            # through a voxel map.  Handing it over under the same name lets
            # the group-provenance machinery downstream stay as it is: it
            # only ever asks "which group produced the row at this index".
            self.last_deposit_to_group = _group_of_point
            self.last_group_volume_offsets = _g_offs
            self.last_n_groups = _hstats['n_groups']
            self.last_deposit_volume_offsets = []
            deposit_id = np.arange(len(point_cloud_flat), dtype=np.int64)
        else:
            # A segment is visible if its group_id appears in a readout-plane's
            # group_ids array in the inst file (i.e. it survived thresholding).
            visible_by_track = get_visible_segment_indices_by_track(
                seg_volumes, self._inst_file, event_key
            )

            # Segments are assembled in the same track order as `parts` so
            # that Particle.from_flat_arrays matches metadata to point clouds.
            (point_cloud_flat, offsets,
             deposit_id, volume_offsets) = _build_visible_point_cloud(
                seg_volumes, visible_by_track, parts['track_id']
            )

            # ── Visibility bookkeeping (free: no extra file reads) ─────────
            # n_total comes from the per-volume 'n_actual' attributes already
            # read above; n_visible sums the index arrays the visibility pass
            # built; n_attached is the length of the cloud just produced.
            _n_total = sum(int(sv.get('n_actual', 0)) for sv in seg_volumes)
            _n_visible = sum(len(idx)
                             for vol in visible_by_track
                             for idx in vol.values())
            _n_attached = len(point_cloud_flat)
            self.last_true_x_shift = None
            self.last_mask_stats = {
                'n_total':    _n_total,     # deposits in the seg file
                'n_visible':  _n_visible,   # survived the readout threshold
                'n_masked':   _n_total - _n_visible,
                # Visible but not attached to any EDepSim particle -- a
                # visible segment whose group_to_track ID is absent from the
                # particle list (including the -1 sentinel).
                'n_attached': _n_attached,
                'n_unmatched': _n_visible - _n_attached,
            }
            self.last_hit_stats = None

        # ── Step 5: derive per-particle scalar labels from EDepSim data ───
        itype   = _get_interaction_type(parts, self._e_thresh)
        int_ids = _get_interaction_id(parts, verts)


        _ids, _par, _root, _g4 = _to_index_space(parts, _get_ancestor_id(parts))

        # ── Deposit provenance for this event ────────────────────────────
        # deposit_to_group is per volume; concatenate it onto the same
        # event-global index the point clouds now carry, and do the same for
        # group_to_track so a group number is unambiguous event-wide.
        # Hit mode filled these in above, from the hits themselves.
        if not _hit_mode:
            self.last_deposit_to_group, self.last_group_volume_offsets = \
                self._load_deposit_groups(event_key, seg_volumes)
            self.last_deposit_volume_offsets = volume_offsets

        _particles = Particle.from_flat_arrays(
            ids                 = _ids,
            parent_ids          = _par,
            ancestor_ids            = _root,
            pdgs                = parts['pdg'],
            parent_pdgs         = _get_parent_pdg(parts),
            interaction_ids     = int_ids,
            interaction_types   = itype,
            point_cloud_flat    = point_cloud_flat,
            point_cloud_offsets = offsets,
            min_pc_size         = self._min_pc_size,
            voxel_size          = self._voxel_size,
            geant4_ids          = _g4,
        )

        # Attach each particle's slice of the deposit index -- the input row
        # behind every point, which in hit mode is the hit itself.  Kept
        # beside the cloud rather than inside it: the cloud is float32 and
        # exact only to 2**24, which millions of hits per event would reach.
        for _i, _p in enumerate(_particles):
            _a, _b = int(offsets[_i][0]), int(offsets[_i][1])
            _p.deposit_id = deposit_id[_a:_b]
        return _particles

    # ------------------------------------------------------------------ #
    # Hit mode                                                            #
    # ------------------------------------------------------------------ #

    def _volume_group_offsets(self, ev):
        """Where each volume's groups start in the event-global numbering."""
        offs, cum = [], 0
        for vol_name in self._volume_names(ev):
            vg = ev[vol_name]
            if 'group_to_track' not in vg:
                continue
            offs.append(cum)
            cum += len(vg['group_to_track'])
        return offs, cum

    @staticmethod
    def _pixel_plane(vol_group):
        """The pixel plane of one volume, or None if it has no hit centres."""
        import h5py
        for k in vol_group:
            g = vol_group[k]
            if isinstance(g, h5py.Group) and 'center_py' in g:
                return g
        return None

    def _group_truth(self, seg_vol, vol_group, n_groups, want_de):
        """Per-group mean t0 and summed dE, from the deposits behind them."""
        n = int(seg_vol.get('n_actual', 0) or 0)
        key = next((k for k in _SEG_TO_GROUP_KEYS if k in vol_group), None)
        if not n or key is None:
            z = np.zeros(n_groups, dtype=np.float32)
            return z, z.copy(), np.zeros(n_groups, dtype=np.int64)
        d2g = vol_group[key][:n].astype(np.int64)
        d2g = np.clip(d2g, 0, max(0, n_groups - 1))
        red = _px.group_reductions(
            d2g, n_groups,
            t0_us=seg_vol['t0_us'][:n].astype(np.float64),
            de=seg_vol['de'][:n].astype(np.float64) if want_de else None)
        counts = np.bincount(d2g, minlength=n_groups)
        return red['t0'], red.get('de'), counts

    def _volume_ranges(self):
        """Per-volume (3, 2) x/y/z extent in mm, from the inst file."""
        try:
            return self._inst_file['config/volume_ranges'][:]
        except Exception:
            return None

    def _volume_x_ranges(self):
        """Per-volume x extent in mm, from the inst file's config group."""
        vr = self._volume_ranges()
        if vr is None:
            return None
        # (n_volumes, 3, 2), x first.  Without it the anode cannot be told
        # apart from the reference tick, so the caller falls back to 0.
        return [(float(v[0][0]), float(v[0][1])) for v in vr]

    def _num_time_steps(self):
        """Length of the readout window in ticks, or None if unstated."""
        cfg = self._inst_file.get('config')
        if cfg is None:
            return None
        n = cfg.attrs.get('num_time_steps')
        return int(n) if n is not None else None

    def _sensor_attr(self, name):
        """One config attribute of the sensor file, or None."""
        f = self._sensor_file
        if f is None:
            return None
        cfg = f.get('config')
        if cfg is None:
            return None
        v = cfg.attrs.get(name)
        return None if v is None else float(v)

    def _stated_drift_terms(self):
        """
        ``(velocity_mm_us, time_step_us, why)`` from config or the sensor
        file, or ``(None, None, why)`` when neither supplies them.
        """
        v = self._drift_velocity_mm_us
        dt = self._readout_time_step_us
        src = []
        if v is None:
            cm = self._sensor_attr('velocity_cm_us')
            if cm is not None:
                v = cm * 10.0            # JAXTPC states it in cm/us
                src.append('velocity from the sensor file')
        else:
            src.append('velocity from config')
        if dt is None:
            dt = self._sensor_attr('time_step_us')
            if dt is not None:
                src.append('time step from the sensor file')
        else:
            src.append('time step from config')
        return v, dt, ', '.join(src)

    def _stated_drift_direction(self, v):
        """Drift direction for volume *v*, from config; int or per-volume list."""
        d = self._pixel_drift_direction
        if d is None:
            return None
        if isinstance(d, (int, float)):
            return int(d)
        try:
            return int(d[v])
        except (TypeError, IndexError, ValueError):
            return None

    def _resolve_pixel_geometry(self, seg_volumes):
        """
        The stated geometry for every volume, or None if it is incomplete.

        Returns ``(geoms, note)``; *geoms* is None when something no file
        records -- the pixel pitch and the drift direction -- was not
        configured, in which case the caller falls back to fitting and says
        so.
        """
        ranges = self._volume_ranges()
        if ranges is None:
            return None, "the hits file has no config/volume_ranges"
        pitch = self._pixel_pitch_mm
        if pitch is None:
            return None, "reader.pixel_pitch_mm is not set"
        vel, dt, src = self._stated_drift_terms()
        if vel is None or dt is None:
            return None, ("no drift velocity / sampling period: set "
                          "reader.jaxtpc_sensor_path, or "
                          "reader.drift_velocity_mm_us and "
                          "reader.readout_time_step_us")
        geoms = []
        for v in range(len(seg_volumes)):
            ddir = self._stated_drift_direction(v)
            if ddir is None:
                return None, ("reader.pixel_drift_direction is not set for "
                              f"volume {v}")
            if v >= len(ranges):
                return None, f"config/volume_ranges has no volume {v}"
            geoms.append(_px.resolve_volume_geometry(
                ranges[v], pitch_mm=pitch, drift_direction=ddir,
                drift_velocity_mm_us=vel, time_step_us=dt,
                reference_tick=self._hit_reference_tick, volume_id=v))
        return geoms, src

    def _group_centre_truth(self, event_key, seg_vol, v):
        """
        For one volume: each group's hit centre beside its deposits' centroid.

        The pairing every geometry question is answered from -- a group has
        both a place in the readout and a place in the detector, and the two
        are the same place.
        """
        ev = self._inst_file[event_key]
        vol_key = f'volume_{v}'
        plane = self._pixel_plane(ev[vol_key]) if vol_key in ev else None
        if plane is None:
            raise _px.PixelGeometryError(
                f"{event_key}/{vol_key} has no pixel plane (no subgroup "
                f"with 'center_py').  point_source='hits' reads the "
                f"detected pixel image, which only a pixel readout "
                f"produces; this file is "
                f"{read_readout_type(self._inst_file)!r} readout.")
        vg = ev[vol_key]
        n_groups = len(vg['group_to_track'])
        t0, _de, counts = self._group_truth(seg_vol, vg, n_groups,
                                            want_de=False)
        n = int(seg_vol.get('n_actual', 0) or 0)
        key = next((k for k in _SEG_TO_GROUP_KEYS if k in vg), None)
        d2g = vg[key][:n].astype(np.int64)
        pos = seg_vol.get('positions_mm')
        cent = np.zeros((n_groups, 3), dtype=np.float64)
        safe = np.where(counts > 0, counts, 1).astype(np.float64)
        for i in range(3):
            cent[:, i] = np.bincount(d2g, weights=pos[:n, i],
                                     minlength=n_groups) / safe

        gid = plane['group_ids'][:].astype(np.int64)
        keep = counts[gid] > 0          # groups whose deposits we can see
        return (plane['center_py'][:][keep], plane['center_pz'][:][keep],
                plane['center_times'][:][keep], t0[gid][keep],
                cent[gid][keep])

    def _ensure_pixel_geometry(self, event_key, seg_volumes):
        """
        Settle the pixel geometry once, from configuration, and check it.

        Configuration is the source and the fit is the audit -- never the
        other way round.  A constant derived from the truth it is then
        compared against cannot fail for a geometry that is wrong but
        linear: the fit absorbs the error, and a detector effect this mode
        exists to measure gets calibrated away instead of measured.  So the
        stated numbers are applied as given and the residual is allowed to
        object.
        """
        if self._pixel_geoms is not None:
            return self._pixel_geoms

        geoms, note = self._resolve_pixel_geometry(seg_volumes)
        pairs = [self._group_centre_truth(event_key, sv, v)
                 for v, sv in enumerate(seg_volumes)]

        if geoms is not None:
            reports = []
            for v, (geom, (py, pz, tk, t0, cent)) in enumerate(
                    zip(geoms, pairs)):
                rep = _px.verify_volume_geometry(geom, py, pz, tk, t0, cent,
                                                 volume_id=v)
                rep['source_note'] = note
                reports.append(rep)
            self._pixel_geoms, self._pixel_report = geoms, reports
            return geoms

        if not self._geometry_from_fit:
            # Say what is missing *and* what the data says it should be, so
            # the config can be written from the error rather than guessed.
            say = []
            for v, (py, pz, tk, t0, cent) in enumerate(pairs):
                m = _px._measure_volume(py, pz, tk, t0,
                                        cent[:, 0], cent[:, 1], cent[:, 2])
                say.append(
                    f"    volume {v}: pixel_pitch_mm ~ {m['pitch_mm']:.4f}, "
                    f"pixel_drift_direction ~ {m['drift_direction']:+d}, "
                    f"drift_velocity_mm_us ~ {m['drift_velocity_mm_us']:.4f}, "
                    f"readout_time_step_us ~ {m['time_step_us']:.4f}")
            raise _px.PixelGeometryError(
                f"point_source='hits' needs the pixel geometry, and "
                f"{note}.\n"
                f"These are detector constants, so they are configured "
                f"rather than inferred -- a value fitted to the truth it is "
                f"later checked against cannot be caught when it is wrong.\n"
                f"The drift velocity and sampling period are in the JAXTPC "
                f"sensor file: set reader.jaxtpc_sensor_path.  The pitch and "
                f"the drift direction are in no output file; take them from "
                f"the detector YAML.\n"
                f"For this batch the data is consistent with:\n"
                + "\n".join(say) +
                f"\nSet those, or reader.pixel_geometry_from_fit=true to "
                f"measure them instead and accept that the check becomes "
                f"circular.")

        import warnings
        warnings.warn(
            f"pixel geometry fitted from the truth deposits because {note}. "
            f"The residual check is then circular -- it cannot catch a "
            f"geometry that is wrong but linear.  Prefer configuring "
            f"reader.pixel_pitch_mm, reader.pixel_drift_direction and "
            f"reader.jaxtpc_sensor_path.", RuntimeWarning, stacklevel=2)
        geoms, reports = [], []
        xr = self._volume_x_ranges()
        for v, (py, pz, tk, t0, cent) in enumerate(pairs):
            geom, rep = _px.calibrate_volume(
                py, pz, tk, t0, cent, volume_id=v,
                x_range=xr[v] if xr and v < len(xr) else None,
                reference_tick=self._hit_reference_tick)
            rep['source'] = 'fitted'
            rep['source_note'] = note
            geoms.append(geom)
            reports.append(rep)
        self._pixel_geoms, self._pixel_report = geoms, reports
        return geoms

    def _build_hit_event(self, event_key, seg_volumes, track_ids):
        """Point cloud, offsets and per-point group for one event."""
        ev = self._inst_file[event_key]
        self._ensure_pixel_geometry(event_key, seg_volumes)
        g_offs, n_groups_total = self._volume_group_offsets(ev)

        want_de = self._hit_energy == 'true_de'
        # Needed by both conventions now: 'true_t0' to move x, 'nominal' to
        # record how far x would have to move -- the true_x_shift column.
        want_t0 = True
        # The window truncation is a truth question -- which deposits would
        # have needed a tick the readout never reached -- so it needs t0 and
        # the true x whichever drift convention is in force.
        n_steps = self._num_time_steps()
        trunc_before = trunc_after = trunc_total = 0
        volumes = []
        for v, seg_vol in enumerate(seg_volumes):
            vol_key = f'volume_{v}'
            if vol_key not in ev:
                continue
            vg = ev[vol_key]
            plane = self._pixel_plane(vg)
            if plane is None:
                continue
            n_groups = len(vg['group_to_track'])
            entry = {
                'hits': _px.decode_plane_hits(plane),
                'geom': self._pixel_geoms[v],
                'group_to_track': vg['group_to_track'][:].astype(np.int64),
                'group_offset': g_offs[v] if v < len(g_offs) else 0,
            }
            if want_de or want_t0:
                t0, de, _ = self._group_truth(seg_vol, vg, n_groups, want_de)
                if want_t0:
                    entry['group_t0'] = t0
                if want_de:
                    entry['group_de'] = de
            volumes.append(entry)

            n_act = int(seg_vol.get('n_actual', 0) or 0)
            if n_steps and n_act:
                b, a, tot = _px.window_truncation(
                    seg_vol['positions_mm'][:n_act, 0],
                    seg_vol['t0_us'][:n_act],
                    self._pixel_geoms[v], n_steps)
                trunc_before += b
                trunc_after += a
                trunc_total += tot

        # dt is needed only to express the time column and to convert t0 into
        # ticks; the calibration recovered it alongside the geometry.
        dt = 1.0
        if self._pixel_geoms:
            dt = float(self._pixel_geoms[0].time_step_us)
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
        flat, offsets, group_of_point, x_shift, stats = \
            _px.build_hit_point_cloud(
                volumes, track_ids, x_from=self._hit_x_from,
                energy=self._hit_energy, time_step_us=dt,
                charge_threshold=self._hit_charge_threshold)
        stats['n_groups'] = n_groups_total
        stats['n_deposits'] = trunc_total
        stats['n_before_window'] = trunc_before
        stats['n_after_window'] = trunc_after
        return flat, offsets, group_of_point, x_shift, g_offs, stats

    def _load_deposit_groups(self, event_key, seg_volumes):
        """
        ``(deposit_to_group, group_volume_offsets)`` for one event.

        Both the deposit index and the group number are rebased onto an
        event-global space, matching the deposit ids attached to the point
        clouds, so a caller never has to know which volume a deposit came
        from to look up its group.
        """
        import numpy as _np
        ev = (self._inst_file[event_key]
              if event_key in self._inst_file else None)
        if ev is None:
            return None, []
        d2g, g_off, cum_g = [], [], 0
        for vol_name in self._volume_names(ev):
            vg = ev[vol_name]
            key = next((k for k in _SEG_TO_GROUP_KEYS if k in vg), None)
            if key is None or 'group_to_track' not in vg:
                continue
            g_off.append(cum_g)
            d2g.append(vg[key][:].astype(_np.int64) + cum_g)
            cum_g += len(vg['group_to_track'])
        # The group space is what group_to_track spans, not what the deposits
        # happen to reference: a group nothing deposited into still occupies a
        # number, and deriving the count from max()+1 instead would leave the
        # owner table short of the volume offsets that index it.
        self.last_n_groups = cum_g
        if not d2g:
            return None, []
        return _np.concatenate(d2g), g_off

    @staticmethod
    def _volume_names(ev):
        """Volume subgroups of one event, in the order the seg loader uses."""
        return sorted(k for k in ev.keys() if k.startswith('volume_'))

    def close(self) -> None:
        """Close all open HDF5 file handles."""
        for attr in ('_edepsim_file', '_seg_file', '_inst_file',
                     '_sensor_file'):
            fh = getattr(self, attr, None)
            if fh is not None:
                fh.close()
                setattr(self, attr, None)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_seg_volumes(self, event_key: str) -> list[dict]:
        """
        Load per-volume segment data from the JAXTPC seg file for one event.

        Returns a list of dicts analogous to those produced by
        ``production.load.load_event_seg``, each containing at minimum:
        ``positions_mm`` (N,3), ``de`` (N,), ``dx`` (N,), ``t0_us`` (N,),
        ``charge`` (N,), and ``n_actual`` (int).  Volumes with no segments
        return ``{'n_actual': 0}``.
        """
        if event_key not in self._seg_file:
            raise KeyError(
                f"Event group {event_key!r} not found in the JAXTPC seg file "
                f"{self._seg_path!r}.  Top-level groups present: "
                f"{sorted(self._seg_file.keys())[:6]}.  The seg file must cover "
                f"the same events as the EDepSim file given by io.input_path."
            )

        evt = self._seg_file[event_key]
        n_volumes = int(evt.attrs.get('n_volumes', 1))
        volumes: list[dict] = []

        for v in range(n_volumes):
            vol_key = f'volume_{v}'
            if vol_key not in evt:
                raise KeyError(
                    f"{event_key}/{vol_key} missing from the JAXTPC seg file "
                    f"{self._seg_path!r}, but {event_key} declares "
                    f"n_volumes={n_volumes} (groups present: "
                    f"{sorted(evt.keys())})."
                )

            vg = evt[vol_key]

            # A missing n_actual attribute means this is not a JAXTPC seg
            # volume at all.  Previously it defaulted to 0, which silently
            # produced empty point clouds for every particle in the event.
            if 'n_actual' not in vg.attrs:
                raise KeyError(
                    f"{event_key}/{vol_key} in {self._seg_path!r} has no "
                    f"'n_actual' attribute, so it is not a JAXTPC seg volume "
                    f"(attributes present: {sorted(vg.attrs.keys())}).  Check "
                    f"that reader.jaxtpc_seg_path points at the JAXTPC seg/step "
                    f"file (not the hits/inst or sensor file)."
                )

            n = int(vg.attrs['n_actual'])

            if n == 0:
                volumes.append({'n_actual': 0})
                continue

            _missing_ds = [k for k in _SEG_REQUIRED_DATASETS if k not in vg]
            _missing_at = [k for k in _SEG_REQUIRED_ATTRS if k not in vg.attrs]
            if _missing_ds or _missing_at:
                raise KeyError(
                    f"{event_key}/{vol_key} in the JAXTPC seg file "
                    f"{self._seg_path!r} declares n_actual={n} but is missing "
                    f"required datasets {_missing_ds} / attributes "
                    f"{_missing_at}."
                )

            # Positions are stored as quantised integers; reconstruct mm coords.
            pos_step = float(vg.attrs['pos_step_mm'])
            origin   = np.array([
                vg.attrs['pos_origin_x'],
                vg.attrs['pos_origin_y'],
                vg.attrs['pos_origin_z'],
            ])
            positions_mm = vg['positions'][:].astype(np.float32) * pos_step + origin

            volumes.append({
                'positions_mm': positions_mm,              # (N, 3)
                'de':           vg['de'][:].astype(np.float32),
                'dx':           vg['dx'][:].astype(np.float32),
                't0_us':        vg['t0_us'][:].astype(np.float32),
                'charge':       vg['charge'][:].astype(np.float32),
                'n_actual':     n,
            })

        return volumes

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, cfg) -> "JaxtpcHDF5Reader":
        """
        Construct a :class:`JaxtpcHDF5Reader` from a Hydra/OmegaConf config.

        The EDepSim file path is read from ``cfg.io.input_path``; the JAXTPC
        seg and inst file paths come from ``cfg.reader.jaxtpc_seg_path`` and
        ``cfg.reader.jaxtpc_inst_path``; all other reader parameters come from
        ``cfg.reader``.

        The config group is ``reader=edepsim_h5`` — JAXTPC filtering is an
        optional overlay activated by setting the two JAXTPC path fields:

        .. code-block:: yaml

            reader:
              jaxtpc_seg_path: /data/sim_seg_0000.h5
              jaxtpc_inst_path: /data/sim_inst_0000.h5

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config as returned by
            :func:`~pysupera.config.load_cfg`.  Must contain at minimum
            ``cfg.io.input_path``, ``cfg.reader.jaxtpc_seg_path``, and
            ``cfg.reader.jaxtpc_inst_path``.

        Returns
        -------
        JaxtpcHDF5Reader

        Examples
        --------
        ::

            from pysupera.config import load_cfg
            from pysupera.readers import JaxtpcHDF5Reader

            cfg = load_cfg([
                "io.input_path=/data/edepsim.h5",
                "reader.jaxtpc_seg_path=/data/sim_seg_0000.h5",
                "reader.jaxtpc_inst_path=/data/sim_inst_0000.h5",
            ])
            with JaxtpcHDF5Reader.from_config(cfg) as reader:
                particles = reader[0]
        """
        r = cfg.reader
        return cls(
            edepsim_path              = str(cfg.io.input_path),
            seg_path                  = str(r.jaxtpc_seg_path),
            inst_path                 = str(r.jaxtpc_inst_path),
            vertex_key                = str(r.get("vertex_key",   _DEFAULT_VERTEX_KEY)),
            particle_key              = str(r.get("particle_key", _DEFAULT_PARTICLE_KEY)),
            electron_energy_threshold = float(r.get(
                                            "electron_energy_threshold",
                                            _DEFAULT_ELECTRON_ENERGY_THRESHOLD)),
            min_pc_size               = int(cfg.particle.get("min_pc_size", -1))
                                        if hasattr(cfg, "particle") else None,
        )

    @classmethod
    def from_production_dir(
        cls,
        production_dir: str,
        dataset: str = 'sim',
        file_index: int = 0,
        edepsim_path: str | None = None,
        **kwargs,
    ) -> "JaxtpcHDF5Reader":
        """
        Construct a :class:`JaxtpcHDF5Reader` from a JAXTPC production
        directory that follows the standard naming convention::

            <production_dir>/sensor/sim_sensor_<NNNN>.h5
            <production_dir>/seg/sim_seg_<NNNN>.h5
            <production_dir>/inst/sim_inst_<NNNN>.h5

        Parameters
        ----------
        production_dir : str
            Root directory of the JAXTPC output batch.
        dataset : str, optional
            Dataset prefix (default: ``"sim"``).
        file_index : int, optional
            Batch file index (default: ``0``).
        edepsim_path : str
            Path to the matching EDepSim HDF5 file (required for particle
            metadata; not stored in the production directory).
        **kwargs
            Forwarded to :class:`JaxtpcHDF5Reader.__init__`.

        Returns
        -------
        JaxtpcHDF5Reader
        """
        import os
        if edepsim_path is None:
            raise ValueError(
                "edepsim_path is required — the production directory does not "
                "contain an EDepSim file.  Pass it explicitly via edepsim_path."
            )
        seg_path  = os.path.join(production_dir, 'seg',  f'{dataset}_seg_{file_index:04d}.h5')
        inst_path = os.path.join(production_dir, 'inst', f'{dataset}_inst_{file_index:04d}.h5')
        return cls(edepsim_path, seg_path, inst_path, **kwargs)

    @property
    def edepsim_path(self) -> str:
        """Path to the EDepSim HDF5 file."""
        return self._edepsim_path

    @property
    def seg_path(self) -> str:
        """Path to the JAXTPC seg HDF5 file."""
        return self._seg_path

    @property
    def inst_path(self) -> str:
        """Path to the JAXTPC inst HDF5 file."""
        return self._inst_path

    @property
    def point_source(self) -> str:
        """``'deposits'`` (truth geometry) or ``'hits'`` (detected image)."""
        return self._point_source

    @property
    def pixel_geometry(self) -> list | None:
        """Per-volume :class:`~.pixel_hits.VolumePixelGeometry`, once known."""
        return self._pixel_geoms

    @property
    def pixel_geometry_report(self) -> list | None:
        """What the geometry calibration found, including its residuals."""
        return self._pixel_report

    @property
    def readout_type(self) -> str:
        """
        ``'wire'`` or ``'pixel'`` — which readout the inst file describes.

        Nothing in this reader branches on it: visibility is a property of a
        group, not of the geometry that recorded it.  It is surfaced for the
        run report and for the tools that draw hits, which do differ.
        """
        return read_readout_type(self._inst_file)

    def __repr__(self) -> str:
        return (
            f"JaxtpcHDF5Reader("
            f"edepsim={self._edepsim_path!r}, "
            f"seg={self._seg_path!r}, "
            f"inst={self._inst_path!r}, "
            f"n_events={self._n_events})"
        )
