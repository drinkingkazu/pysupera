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
    track_id      Geant4 track ID  (used as pysupera *id*)
    parent_track_id  Immediate parent track ID
    root_track_id     Root ancestor track ID
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
# ordered by PointFeature (x=0, y=1, z=2, time=3, energy=4, dedx=5).
# This mapping is EDepSim-format-specific and lives here rather than in
# Particle.from_flat_arrays so that the core data class stays format-agnostic.

# Maps PointFeature column → candidate HDF5 field name(s) to try, in order.
_STEP_FIELD_MAP: list[tuple[int, tuple[str, ...]]] = [
    (PointFeature.x,      ("x",)),
    (PointFeature.y,      ("y",)),
    (PointFeature.z,      ("z",)),
    (PointFeature.time,   ("t", "time")),
    (PointFeature.energy, ("energy", "e")),
    (PointFeature.dedx,   ("dedx", "dEdx", "dE_dx")),
]


def _steps_to_plain_array(steps: np.ndarray) -> np.ndarray:
    """
    Convert a structured EDepSim step array to a plain float32 2-D array.

    The output columns follow :class:`~pysupera.utils.PointFeature` order:
    ``x=0, y=1, z=2, time=3, energy=4, dedx=5``.  x, y, z are mandatory;
    any optional field (time, energy, dedx) that is absent from the dtype
    is left as zero in the output.

    Parameters
    ----------
    steps : numpy.ndarray
        1-D structured array read from the HDF5 step dataset.

    Returns
    -------
    numpy.ndarray, shape (N, 6), dtype float32
    """
    if steps.dtype.names is None:
        # Already a plain (non-structured) array — nothing to do.
        return steps.astype(np.float32, copy=False)

    n = len(steps)
    out = np.zeros((n, len(_STEP_FIELD_MAP)), dtype=np.float32)
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
        ``process_types`` to :meth:`~pysupera.data.Particle.from_flat_arrays`.
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
            for particles in reader:
                ...
    """

    def __init__(
        self,
        path: str,
        *,
        particle_key: str = _DEFAULT_PARTICLE_KEY,
        step_key: str     = _DEFAULT_STEP_KEY,
        ass_key: str      = _DEFAULT_ASS_KEY,
        electron_energy_threshold: float = _DEFAULT_ELECTRON_ENERGY_THRESHOLD,
        min_pc_size: int | None = None,
    ) -> None:
        import h5py
        self._path        = path
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

        parts = self._file[self._part_key][index]
        steps = self._file[self._step_key][index]
        ass   = self._file[self._ass_key][index]

        num_parts = len(parts)
        itype     = _get_interaction_type(parts, self._e_thresh)
        offsets   = np.column_stack([
            ass["start"][:num_parts],
            ass["end"][:num_parts],
        ])

        return Particle.from_flat_arrays(
            ids                  = parts["track_id"],
            parent_ids           = parts["parent_track_id"],
            root_ids             = parts["root_track_id"],
            pdgs                 = parts["pdg"],
            parent_pdgs          = _get_parent_pdg(parts),
            process_types        = itype,
            point_cloud_flat     = _steps_to_plain_array(steps),
            point_cloud_offsets  = offsets,
            min_pc_size          = self._min_pc_size,
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
