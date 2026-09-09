"""
Reader for EDepSim HDF5 output files.

EDepSim (https://github.com/ClarkMcGrew/edep-sim) stores Geant4 simulation
output as structured arrays inside an HDF5 file.  This module translates
that layout into lists of pysupera :class:`~pysupera.data.Particle` objects.

Expected HDF5 layout
--------------------
``particle/geant4``  — structured array, one entry per event.  Each entry is
    a particle-level array with fields:

    ============= ======================================================
    Field         Description
    ============= ======================================================
    track_id      Geant4 track ID.  Kept as pysupera *geant4_id*; it is NOT
                  used as the pysupera *id*, because track IDs are not
                  guaranteed contiguous.  pysupera assigns its own 0-based
                  per-event particle index instead (see _to_index_space).
    parent_track_id  Immediate parent track ID
    root_track_id     Root ancestor track ID.  Unreliable in some files -- see
                  _get_root_id, which re-derives it from the parent chain.
    pdg           PDG Monte Carlo particle code
    proc_start    G4 process type integer at particle start vertex
    subproc_start G4 process subtype integer at particle start vertex
    ke            Kinetic energy at start vertex (MeV)
    x, y, z       Particle start position components
    ============= ======================================================

``pstep/lar_vol`` — flat array of all energy-deposit steps (point cloud) for
    all particles in each event.  Columns 0-2 must be x, y, z.

``ass/particle_pstep_lar_vol`` — association array, one entry per event.
    Fields ``start`` and ``end`` store the slice ``[start[i], end[i])``
    into the step array that belongs to particle *i*.

Usage
-----
::

    from pysupera.readers import EDepSimHDF5Reader

    with EDepSimHDF5Reader("out_0100.h5") as store:
        print(len(store), "events")
        particles = store[0]           # first event
        for particles in store:        # iterate all events
            ...
"""

from __future__ import annotations

import warnings
import numpy as np
from enum import IntEnum

from .base import EventReaderBase
from ..data import Particle
from ..utils import PointFeature


# ---------------------------------------------------------------------------
# Step-array conversion
# ---------------------------------------------------------------------------
# The EDepSim HDF5 step datasets are structured arrays with named fields.
# pysupera's Particle expects a plain float32 2-D point cloud with columns
# ordered by PointFeature (x=0, y=1, z=2, time=3, energy=4, dx=5, id=6).
# This mapping is EDepSim-format-specific and lives here rather than in
# Particle.from_flat_arrays so that the core data class stays format-agnostic.
#
# PointFeature.id (col 6) is NOT read from the HDF5 file — it is assigned
# here as a global, event-wide, 0-based increasing integer so that every
# input point can be uniquely identified across the entire event.

# Maps PointFeature column → candidate HDF5 field name(s) to try, in order.
# PointFeature.id is handled separately (generated, not read from file).
_warned_cols: set[int] = set()   # columns for which a missing-field warning has already been shown
_STEP_FIELD_MAP: list[tuple[int, tuple[str, ...]]] = [
    (PointFeature.x,      ("x",)),
    (PointFeature.y,      ("y",)),
    (PointFeature.z,      ("z",)),
    (PointFeature.time,   ("t", "time")),
    (PointFeature.energy, ("energy", "e", "de")),
    (PointFeature.dx,     ("dx")),
]


def _steps_to_plain_array(steps: np.ndarray) -> np.ndarray:
    """
    Convert a structured EDepSim step array to a plain float32 2-D array.

    The output columns follow :class:`~pysupera.utils.PointFeature` order:
    ``x=0, y=1, z=2, time=3, energy=4, dx=5, id=6``.
    x, y, z are mandatory; time, energy, dx default to zero if absent.

    Column 6 (``PointFeature.id``) is **not** read from the HDF5 file.
    It is assigned here as a global, event-wide, 0-based integer
    ``0, 1, 2, …, N-1`` so that every input step can be uniquely
    identified across the whole event (the caller passes all steps for
    one event at once).

    Parameters
    ----------
    steps : numpy.ndarray
        1-D structured array read from the HDF5 step dataset.

    Returns
    -------
    numpy.ndarray, shape (N, 7), dtype float32
    """
    if steps.dtype.names is None:
        # Already a plain (non-structured) array.
        # Ensure it has at least 7 columns, padding with zeros if needed.
        arr = steps.astype(np.float32, copy=False)
        n_cols = arr.shape[1] if arr.ndim == 2 else 0
        n_extra = max(0, 7 - n_cols)
        if n_extra:
            arr = np.concatenate(
                [arr, np.zeros((len(arr), n_extra), dtype=np.float32)], axis=1
            )
        # Assign global point IDs if the id column is all-zero
        if arr.shape[1] > PointFeature.id and (arr[:, PointFeature.id] == 0).all():
            arr[:, PointFeature.id] = np.arange(len(arr), dtype=np.float32)
        return arr

    n = len(steps)
    # 7 columns: the 6 from _STEP_FIELD_MAP plus PointFeature.id
    out = np.zeros((n, 7), dtype=np.float32)
    available = set(steps.dtype.names)

    for col, candidates in _STEP_FIELD_MAP:
        for fname in candidates:
            if fname in available:
                out[:, col] = steps[fname]
                break
        else:
            if col < 3:   # x, y, z are required
                raise ValueError(
                    f"EDepSim step array is missing a required coordinate field. "
                    f"Tried {candidates}; available fields: {sorted(available)}"
                )
            # optional columns default to zero — already set
            if col not in _warned_cols:
                _warned_cols.add(col)
                col_name = next(
                    name for name, val in PointFeature.__dict__.items()
                    if isinstance(val, int) and val == col
                )
                warnings.warn(
                    f"EDepSim step array has no field for '{col_name}' "
                    f"(tried {candidates}); available fields: {sorted(available)}. "
                    f"Column {col} will be zero.",
                    UserWarning,
                    stacklevel=2,
                )

    # Assign global, event-wide, 0-based point IDs.
    out[:, PointFeature.id] = np.arange(n, dtype=np.float32)

    return out


# ---------------------------------------------------------------------------
# Geant4 process / subprocess enumerations
# ---------------------------------------------------------------------------
# IntEnum is used deliberately so that comparisons with numpy integer arrays
# work element-wise:  ``proc_start_array == G4ProcessType.kProcessDecay``
# correctly produces a boolean numpy array.

class G4ProcessType(IntEnum):
    kProcessNotDefined         = 0
    kProcessTransportation     = 1
    kProcessElectromagnetic    = 2
    kProcessOptical            = 3
    kProcessHadronic           = 4
    kProcessPhotoLeptonHadron  = 5
    kProcessDecay              = 6
    kProcessGeneral            = 7
    kProcessParameterization   = 8
    kProcessUserDefined        = 9


class G4ProcessSubtype(IntEnum):
    kSubtypeEMCoulombScattering  = 1
    kSubtypeEMIonization         = 2
    kSubtypeEMBremsstrahlung     = 3
    kSubtypeEMPairProdByCharged  = 4
    kSubtypeEMNuclearStopping    = 8
    kSubtypeEMMultipleScattering = 10
    kSubtypeEMPhotoelectric      = 12
    kSubtypeEMComptonScattering  = 13
    kSubtypeEMGammaConversion    = 14
    kSubtypeHadronElastic        = 111
    kSubtypeHadronInelastic      = 121
    kSubtypeHadronCapture        = 131
    kSubtypeHadronChargeExchange = 161
    kSubtypeGeneralStepLimit     = 401


# ---------------------------------------------------------------------------
# Default HDF5 dataset keys
# ---------------------------------------------------------------------------
_DEFAULT_VERTEX_KEY   = "vertex/geant4"
_DEFAULT_PARTICLE_KEY = "particle/geant4"
_DEFAULT_STEP_KEY     = "pstep/lar_vol"
_DEFAULT_ASS_KEY      = "ass/particle_pstep_lar_vol"
_DEFAULT_ELECTRON_ENERGY_THRESHOLD = 0.05   # MeV


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _search_parents(track_ids: np.ndarray,
                    parent_track_ids: np.ndarray) -> np.ndarray:
    """
    For each element in *parent_track_ids*, return the index of the matching
    entry in *track_ids*, or ``-1`` if not found.

    Parameters
    ----------
    track_ids : ndarray of int, shape (N,)
        The full set of track IDs in this event.
    parent_track_ids : ndarray of int, shape (N,)
        The parent track ID of each particle.

    Returns
    -------
    ndarray of int, shape (N,)
        ``result[i]`` is the index *j* such that
        ``track_ids[j] == parent_track_ids[i]``, or ``-1`` if absent.
    """
    sorter       = np.argsort(track_ids)
    idx          = np.searchsorted(track_ids, parent_track_ids, sorter=sorter)
    idx_clipped  = np.clip(idx, 0, len(track_ids) - 1)
    valid        = track_ids[sorter[idx_clipped]] == parent_track_ids
    return np.where(valid, sorter[idx_clipped], -1)


def _get_parent_pdg(parts: np.ndarray) -> np.ndarray:
    """
    Look up the PDG code of each particle's direct parent.

    Returns 0 for particles whose parent is not present in *parts*
    (e.g. primary particles).

    Parameters
    ----------
    parts : structured ndarray
        Must contain fields ``track_id``, ``parent_track_id``, ``pdg``.

    Returns
    -------
    ndarray of int, shape (N,)
    """
    parent_locs  = _search_parents(parts["track_id"], parts["parent_track_id"])
    parent_valid = parent_locs != -1
    parent_zfill = np.where(parent_valid, parent_locs, 0)
    return np.where(parent_valid, parts["pdg"][parent_zfill], 0)


def _to_index_space(parts: np.ndarray, root_track_ids: np.ndarray):
    """
    Translate Geant4 track IDs into contiguous per-event particle indices.

    pysupera addresses particles by their position in the event's particle
    list, not by Geant4 track ID: track IDs are not guaranteed contiguous --
    an upstream stage may drop particles before pysupera ever sees them -- so
    they cannot be used as direct array indices.  The track IDs are preserved
    separately as provenance.

    Returns
    -------
    ids : ndarray of int32
        ``arange(len(parts))``.
    parent_idx : ndarray of int32
        Index of each particle's parent.  A particle whose parent is absent
        from this list (a primary, or one whose parent was dropped upstream)
        becomes its own parent, which is the convention every genealogy walk
        in pysupera terminates on.
    root_idx : ndarray of int32
        Index of each particle's primary ancestor.
    geant4_ids : ndarray of int32
        The original ``track_id`` values, unchanged.
    """
    n = len(parts)
    track_ids = parts['track_id']
    index_of = {int(t): i for i, t in enumerate(track_ids)}

    ids = np.arange(n, dtype=np.int32)
    parent_idx = np.empty(n, dtype=np.int32)
    root_idx = np.empty(n, dtype=np.int32)
    for i in range(n):
        parent_idx[i] = index_of.get(int(parts['parent_track_id'][i]), i)
        root_idx[i] = index_of.get(int(root_track_ids[i]), i)
    return ids, parent_idx, root_idx, track_ids.astype(np.int32)


def _get_root_id(parts: np.ndarray) -> np.ndarray:
    """
    Return the true primary ancestor track ID of every particle.

    The stored ancestor field (``root_track_id`` / ``ancestor_track_id``) is not
    reliable in every EDepSim file: a secondary is sometimes written with
    ``ancestor_track_id == its own track_id`` despite having a real parent.
    Every descendant of such a particle then inherits that wrong ancestor, so a
    single bad entry strands a whole subtree.  In one 13.5k-particle event, 15
    such entries put 4,423 particles (33%) on the wrong root.

    ``parent_track_id`` is trustworthy in those files, so the primary is found
    by walking it upwards until reaching a particle that has no parent, is its
    own parent, or whose parent is absent from this event's list.  Results are
    memoised along each chain, keeping the pass O(n).

    Parameters
    ----------
    parts : structured ndarray
        Must contain fields ``track_id`` and ``parent_track_id``.

    Returns
    -------
    ndarray of int32, shape (N,)
        Primary ancestor track ID per particle.  A particle caught in a
        ``parent_track_id`` cycle is reported as its own root, since no primary
        is reachable.
    """
    n = len(parts)
    roots = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return roots

    track_ids = parts['track_id']
    parent_ids = parts['parent_track_id']
    index_of = {int(t): i for i, t in enumerate(track_ids)}

    for start in range(n):
        if roots[start] >= 0:
            continue
        chain = []
        i = start
        seen = set()
        answer = -1
        while True:
            if i in seen:
                # parent_track_id cycle: no primary is reachable, so every
                # particle in the loop becomes its own root.
                answer = int(track_ids[i])
                break
            if roots[i] >= 0:
                answer = int(roots[i])
                break
            seen.add(i)
            chain.append(i)
            p = int(parent_ids[i])
            if p < 0 or p == int(track_ids[i]) or p not in index_of:
                answer = int(track_ids[i])   # reached a primary
                break
            i = index_of[p]
        for j in chain:
            roots[j] = answer

    return roots


def _get_interaction_id(parts: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """
    Look up the interaction ID of every particle

    Returns -1 for particles whose root particle is not found or the root particle does not match with any vertex in "verts"

    Parameters
    ----------
    parts : structured ndarray
        Must contain fields ``x``, ``y``, ``z``, ``t``, ``track_id``, and either
        ``root_track_id`` or ``ancestor_track_id``.
    verts : structured ndarray
        Must contain fields ``x``, ``y``, ``z``, ``t``, ``interaction_id``.

    Returns
    -------
    ndarray of int, shape (N,)
    """

    interaction_ids = np.full(len(parts), -1, dtype=np.int32)
    if len(parts) == 0:
        return interaction_ids

    track_ids = parts['track_id']
    parent_ids = parts['parent_track_id']
    # Use the walked primary rather than the stored ancestor field, which is
    # unreliable -- see _get_root_id.
    root_refs = _get_root_id(parts)
    index_of = {int(t): i for i, t in enumerate(track_ids)}

    # Step 0: loop over root particles and match to one of vertices by (x,y,z,t) proximity.
    root_indices = np.where(track_ids == root_refs)[0]

    for index in root_indices:
        p = parts[index]
        for vtx in verts:
            if np.isclose(p['x'], vtx['x'], atol=1e-4) and \
               np.isclose(p['y'], vtx['y'], atol=1e-4) and \
               np.isclose(p['z'], vtx['z'], atol=1e-4) and \
               np.isclose(p['t'], vtx['t'], atol=1e-4):
                # Match found, do something
                interaction_ids[index] = vtx['interaction_id']
                break

    # Step 1: for non-root particles, assign the same interaction ID as their
    # root particle.  A root reference that is absent from this event list is
    # left for step 2 rather than raising.
    for i in range(len(parts)):
        if track_ids[i] == root_refs[i]:
            continue
        root_index = index_of.get(int(root_refs[i]))
        if root_index is not None:
            interaction_ids[i] = interaction_ids[root_index]

    # Step 2: fall back to the parent chain for anything still unassigned.
    #
    # Steps 0-1 trust the root/ancestor field, which is not always the true
    # primary: EDepSim files are seen in which a secondary carries
    # ancestor_track_id == its own track_id, or points at another secondary that
    # never started at a vertex.  Either way the particle is treated as a root,
    # fails the vertex match because it was born mid-event, and inherits -1 --
    # as do all of its descendants.
    #
    # parent_track_id is reliable in those files, so walk it upwards until an
    # already-assigned ancestor is found.  Results are memoised onto every
    # particle visited along the way, which keeps the whole pass O(n) even for
    # long decay chains.
    for i in np.where(interaction_ids < 0)[0]:
        chain = [i]
        found = -1
        node = index_of.get(int(parent_ids[i]))
        seen = {i}
        while node is not None and node not in seen:
            seen.add(node)
            if interaction_ids[node] >= 0:
                found = interaction_ids[node]
                break
            chain.append(node)
            node = index_of.get(int(parent_ids[node]))
        if found >= 0:
            for j in chain:
                interaction_ids[j] = found

    return interaction_ids


def _get_interaction_type(parts: np.ndarray,
                          electron_energy_threshold: float = 0.05
                          ) -> np.ndarray:
    """
    Derive the pysupera :class:`~pysupera.utils.InteractionType` integer
    for each particle using Geant4 process information.

    The logic mirrors the ``get_interaction_type`` function in the example
    notebook.  ``np.select`` is used instead of a Python loop for
    performance.

    Parameters
    ----------
    parts : structured ndarray
        Must contain fields: ``track_id``, ``parent_track_id``, ``pdg``,
        ``proc_start``, ``subproc_start``, ``ke``, ``x``, ``y``, ``z``.
    electron_energy_threshold : float, optional
        Kinetic-energy threshold (MeV) separating kCompton-like low-energy
        electrons from kOtherShower-like higher-energy electrons when no
        cleaner process classification is available.

    Returns
    -------
    ndarray of int32, shape (N,)
        Raw ``InteractionType.value`` integers, ready to be passed as
        ``interaction_types`` to :meth:`~pysupera.data.Particle.from_flat_arrays`.
    """
    from ..utils import InteractionType

    track_id      = parts["track_id"]
    parent_tid    = parts["parent_track_id"]
    pdg           = parts["pdg"]
    proc_start    = parts["proc_start"]
    subproc_start = parts["subproc_start"]
    ke            = parts["ke"]
    parent_pdg    = _get_parent_pdg(parts)

    parent_locs  = _search_parents(track_id, parent_tid)
    parent_valid = parent_locs != -1
    parent_zfill = np.where(parent_valid, parent_locs, 0)

    xs, ys, zs = parts["x"], parts["y"], parts["z"]
    dx = xs[parent_zfill] - xs
    dy = ys[parent_zfill] - ys
    dz = zs[parent_zfill] - zs
    dr = np.where(parent_valid, np.sqrt(dx**2 + dy**2 + dz**2), -1.0)

    # Shorthands
    EM     = G4ProcessType.kProcessElectromagnetic
    DK     = G4ProcessType.kProcessDecay
    HAD    = G4ProcessType.kProcessHadronic
    PE     = G4ProcessSubtype.kSubtypeEMPhotoelectric
    CO     = G4ProcessSubtype.kSubtypeEMComptonScattering
    GC     = G4ProcessSubtype.kSubtypeEMGammaConversion
    PP     = G4ProcessSubtype.kSubtypeEMPairProdByCharged
    ION    = G4ProcessSubtype.kSubtypeEMIonization
    _HAD151 = 151   # hadronic sub-process 151 (nuclear de-excitation / gammas)

    is_electron = abs(pdg) == 11

    conditions = [
        pdg == 2112,
        pdg > 1_000_000_000,
        track_id == parent_tid,
        pdg == 22,
        # EM electrons — ordered from most specific to least specific
        is_electron & (proc_start == EM) & (subproc_start == PE),
        is_electron & (proc_start == EM) & (subproc_start == CO),
        is_electron & (proc_start == EM) & ((subproc_start == GC) |
                                             (subproc_start == PP)),
        is_electron & (proc_start == EM) & (subproc_start == ION) & (abs(parent_pdg) == 22),
        is_electron & (proc_start == EM) & (subproc_start == ION) & np.isin(abs(parent_pdg), [211, 13, 2212, 321]),
        is_electron & (proc_start == EM) & (subproc_start == ION),
        is_electron & (proc_start == EM),
        is_electron & (proc_start == DK),
        is_electron & (proc_start == HAD) & (subproc_start == _HAD151) & (dr < 1e-4) & (ke < electron_energy_threshold),
        is_electron & (proc_start == HAD) & (subproc_start == _HAD151) & (dr < 1e-4),
        is_electron & (ke < electron_energy_threshold),
        is_electron,
        ~is_electron,
    ]

    choices = [
        InteractionType.kNeutron,
        InteractionType.kNucleus,
        InteractionType.kPrimary,
        InteractionType.kPhoton,
        InteractionType.kPhotoElectron,
        InteractionType.kCompton,
        InteractionType.kConversion,
        InteractionType.kIonization,
        InteractionType.kDelta,
        InteractionType.kIonization,
        InteractionType.kInvalidProcess,
        InteractionType.kDecay,
        InteractionType.kIonization,
        InteractionType.kDecay,
        InteractionType.kCompton,
        InteractionType.kOtherShower,
        InteractionType.kTrack,
    ]

    # np.select returns the first matching condition; convert enum → int32
    result = np.select(conditions, choices,
                       default=InteractionType.kInvalidProcess)
    return np.array([v.value for v in result], dtype=np.int32)


# ---------------------------------------------------------------------------
# Reader class
# ---------------------------------------------------------------------------

class EDepSimHDF5Reader(EventReaderBase):
    """
    Read EDepSim HDF5 output and produce lists of pysupera
    :class:`~pysupera.data.Particle` objects.

    Parameters
    ----------
    path : str
        Path to the HDF5 file produced by EDepSim.
    particle_key : str, optional
        HDF5 dataset key for the per-event particle arrays.
        Default: ``"particle/geant4"``.
    step_key : str, optional
        HDF5 dataset key for the flat energy-deposit step arrays.
        Default: ``"pstep/lar_vol"``.
    ass_key : str, optional
        HDF5 dataset key for the particle-to-step association arrays.
        Default: ``"ass/particle_pstep_lar_vol"``.
    electron_energy_threshold : float, optional
        Kinetic-energy threshold (MeV) used by the interaction-type
        classifier to separate low-energy from high-energy electrons when
        no cleaner process tag is available.  Default: ``0.05``.
    min_pc_size : int or None, optional
        Forwarded to :class:`~pysupera.data.Particle`.  ``None`` (default)
        uses the module-level ``_DEFAULT_MIN_PC_SIZE``.

    Examples
    --------
    ::

        with EDepSimHDF5Reader("out_0100.h5") as reader:
            print(len(reader), "events")
            particles = reader[0]
            for particles in reader:from_flat_arrays
                ...
    """

    def __init__(
        self,
        path: str,
        *,
        vertex_key:   str = _DEFAULT_VERTEX_KEY,
        particle_key: str = _DEFAULT_PARTICLE_KEY,
        step_key: str     = _DEFAULT_STEP_KEY,
        ass_key: str      = _DEFAULT_ASS_KEY,
        electron_energy_threshold: float = _DEFAULT_ELECTRON_ENERGY_THRESHOLD,
        min_pc_size: int | None = None,
    ) -> None:
        import h5py
        self._path        = path
        self._vertex_key  = vertex_key
        self._part_key    = particle_key
        self._step_key    = step_key
        self._ass_key     = ass_key
        self._e_thresh    = electron_energy_threshold
        self._min_pc_size = min_pc_size
        self._file        = h5py.File(path, "r")
        self._n_events    = len(self._file[self._part_key])

    # ------------------------------------------------------------------ #
    # EventReaderBase interface                                            #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._n_events

    def __getitem__(self, index: int) -> list[Particle]:
        """
        Load and return the particle list for event *index*.

        Parameters
        ----------
        index : int
            0-based event index.  Negative indices are supported
            (e.g. ``reader[-1]`` returns the last event).

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

        verts = self._file[self._vertex_key][index]
        parts = self._file[self._part_key][index]
        steps = self._file[self._step_key][index]
        ass   = self._file[self._ass_key][index]

        num_parts = len(parts)
        itype     = _get_interaction_type(parts, self._e_thresh)
        int_ids   = _get_interaction_id(parts,verts)
        offsets   = np.column_stack([
            ass["start"][:num_parts],
            ass["end"][:num_parts],
        ])


        _ids, _par, _root, _g4 = _to_index_space(parts, _get_root_id(parts))

        return Particle.from_flat_arrays(
            ids                  = _ids,
            parent_ids           = _par,
            root_ids             = _root,
            pdgs                 = parts["pdg"],
            parent_pdgs          = _get_parent_pdg(parts),
            interaction_ids      = int_ids,
            interaction_types    = itype,
            point_cloud_flat     = _steps_to_plain_array(steps),
            point_cloud_offsets  = offsets,
            min_pc_size          = self._min_pc_size,
            geant4_ids           = _g4,
        )

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        if self._file:
            self._file.close()
            self._file = None

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, cfg) -> "EDepSimHDF5Reader":
        """
        Construct an :class:`EDepSimHDF5Reader` from a Hydra/OmegaConf config.

        The file path is read from ``cfg.io.input_path``; all reader-specific
        parameters come from ``cfg.reader``.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config as returned by
            :func:`~pysupera.config.load_cfg`.  Must contain at minimum
            ``cfg.io.input_path`` and ``cfg.reader``.

        Returns
        -------
        EDepSimHDF5Reader

        Examples
        --------
        ::

            from pysupera.config import load_cfg
            from pysupera.readers import EDepSimHDF5Reader

            cfg = load_cfg(["io.input_path=/data/sim.h5",
                            "reader.step_key=pstep/tpc_vol"])
            with EDepSimHDF5Reader.from_config(cfg) as reader:
                particles = reader[0]
        """
        r = cfg.reader
        return cls(
            path                      = str(cfg.io.input_path),
            particle_key              = str(r.get("particle_key",
                                              _DEFAULT_PARTICLE_KEY)),
            step_key                  = str(r.get("step_key",
                                              _DEFAULT_STEP_KEY)),
            ass_key                   = str(r.get("ass_key",
                                              _DEFAULT_ASS_KEY)),
            electron_energy_threshold = float(r.get(
                                              "electron_energy_threshold",
                                              _DEFAULT_ELECTRON_ENERGY_THRESHOLD)),
            min_pc_size               = int(cfg.particle.get("min_pc_size", -1))
                                        if hasattr(cfg, "particle") else None,
        )

    @property
    def path(self) -> str:
        """Path of the open HDF5 file."""
        return self._path

    def __repr__(self) -> str:
        return (
            f"EDepSimHDF5Reader("
            f"path={self._path!r}, "
            f"n_events={self._n_events})"
        )
