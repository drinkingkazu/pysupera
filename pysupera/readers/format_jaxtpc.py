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
    _get_root_id,
    _to_index_space,
)
from ..data import Particle
from ..utils import PointFeature


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
                f"with a 'group_ids' dataset (expected e.g. U/V/Y); found "
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
        for track_id in np.unique(track_ids_for_visible):
            mask = track_ids_for_visible == track_id
            track_to_indices[int(track_id)] = visible_seg_indices[mask]

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
    """
    # Merge per-volume dicts into a single {track_id: list_of_segment_rows}
    # where each segment row is a 1-D float32 array of length 7.
    track_segs: dict[int, list[np.ndarray]] = {}

    for seg_vol, vol_visible in zip(seg_volumes, visible_by_track):
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

    # Concatenate chunks per track and build the flat array + offsets in the
    # same order as track_ids_edepsim so that Particle.from_flat_arrays
    # correctly pairs metadata rows with point-cloud slices.
    n_particles = len(track_ids_edepsim)
    offsets = np.zeros((n_particles, 2), dtype=np.int64)
    flat_chunks: list[np.ndarray] = []
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
        cursor += len(combined)

    if flat_chunks:
        point_cloud_flat = np.concatenate(flat_chunks, axis=0)
    else:
        point_cloud_flat = np.zeros((0, 7), dtype=np.float32)

    # Assign global, event-wide, 0-based point IDs.
    point_cloud_flat[:, PointFeature.id] = np.arange(
        len(point_cloud_flat), dtype=np.float32
    )

    return point_cloud_flat, offsets


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
    ) -> None:
        import h5py

        self._edepsim_path = edepsim_path
        self._seg_path     = seg_path
        self._inst_path    = inst_path
        self._vertex_key   = vertex_key
        self._part_key     = particle_key
        self._e_thresh     = electron_energy_threshold
        self._min_pc_size  = min_pc_size

        # Visibility counts for the most recently read event; populated by
        # __getitem__.  None until the first event has been read.
        self.last_mask_stats: dict | None = None

        # Open all three files for the lifetime of the reader.
        self._edepsim_file = h5py.File(edepsim_path, 'r')
        self._seg_file     = h5py.File(seg_path, 'r')
        self._inst_file    = h5py.File(inst_path, 'r')

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
        parts = self._edepsim_file[self._part_key][index]

        # ── Step 2: load per-volume segment data from the JAXTPC seg file ─
        # Each volume dict contains positions_mm (N,3), de, dx, t0_us, …
        seg_volumes = self._load_seg_volumes(event_key)

        # ── Step 3: determine which segments are visible in the readout ───
        # A segment is visible if its group_id appears in a readout-plane's
        # group_ids array in the inst file (i.e. it survived thresholding).
        visible_by_track = get_visible_segment_indices_by_track(
            seg_volumes, self._inst_file, event_key
        )

        # ── Step 4: build the flat point cloud (visible segments only) ────
        # Segments are assembled in the same track order as `parts` so that
        # Particle.from_flat_arrays correctly matches metadata to point clouds.
        point_cloud_flat, offsets = _build_visible_point_cloud(
            seg_volumes, visible_by_track, parts['track_id']
        )

        # ── Visibility bookkeeping (free: no extra file reads) ────────────
        # n_total comes from the per-volume 'n_actual' attributes already read
        # in step 2; n_visible sums the index arrays step 3 already built; and
        # n_attached is just the length of the cloud step 4 just produced.
        # Nothing here touches the EDepSim step array.
        _n_total = sum(int(sv.get('n_actual', 0)) for sv in seg_volumes)
        _n_visible = sum(len(idx)
                         for vol in visible_by_track
                         for idx in vol.values())
        _n_attached = len(point_cloud_flat)
        self.last_mask_stats = {
            'n_total':    _n_total,     # deposits in the seg file
            'n_visible':  _n_visible,   # survived the readout threshold
            'n_masked':   _n_total - _n_visible,
            # Visible but not attached to any EDepSim particle -- a visible
            # segment whose group_to_track ID is absent from the particle list
            # (including the -1 sentinel for out-of-range groups).
            'n_attached': _n_attached,
            'n_unmatched': _n_visible - _n_attached,
        }

        # ── Step 5: derive per-particle scalar labels from EDepSim data ───
        itype   = _get_interaction_type(parts, self._e_thresh)
        int_ids = _get_interaction_id(parts, verts)


        _ids, _par, _root, _g4 = _to_index_space(parts, _get_root_id(parts))

        return Particle.from_flat_arrays(
            ids                 = _ids,
            parent_ids          = _par,
            root_ids            = _root,
            pdgs                = parts['pdg'],
            parent_pdgs         = _get_parent_pdg(parts),
            interaction_ids     = int_ids,
            interaction_types   = itype,
            point_cloud_flat    = point_cloud_flat,
            point_cloud_offsets = offsets,
            min_pc_size         = self._min_pc_size,
            geant4_ids          = _g4,
        )

    def close(self) -> None:
        """Close all open HDF5 file handles."""
        for attr in ('_edepsim_file', '_seg_file', '_inst_file'):
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

    def __repr__(self) -> str:
        return (
            f"JaxtpcHDF5Reader("
            f"edepsim={self._edepsim_path!r}, "
            f"seg={self._seg_path!r}, "
            f"inst={self._inst_path!r}, "
            f"n_events={self._n_events})"
        )
