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
                  _get_ancestor_id, which re-derives it from the parent chain.
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
    (PointFeature.dx,     ("dx",)),   # 1-tuple: ("dx") is a str, and would match field "x"
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
_DEFAULT_PRIMARY_KEY  = "primary/geant4"
_DEFAULT_ELECTRON_ENERGY_THRESHOLD = 0.05   # MeV


# ---------------------------------------------------------------------------
# Particle kinematics
# ---------------------------------------------------------------------------

#: Speed of light in mm/ns, the units of the step records.
_C_MM_PER_NS = 299.792458

#: Per-particle kinematic arrays, each (N, 4) float64: positions are
#: ``[x, y, z, t]`` (mm, ns) and momenta ``[px, py, pz, E]`` (MeV), E total.
#: ``start_*`` / ``end_*`` describe the Geant4 particle and come from here;
#: ``first_*`` / ``last_*`` describe a reconstructed instance and are made
#: later, from the steps it collected (:mod:`pysupera.kinematics`).
KINEMATICS_FIELDS = ("start_pos", "start_mom", "first_pos", "first_mom",
                     "last_pos", "last_mom", "end_pos", "end_mom")

#: The fields :func:`particle_kinematics` fills.
TRACK_KINEMATICS_FIELDS = ("start_pos", "start_mom", "end_pos", "end_mom")


def particle_kinematics(parts):
    """
    Creation and destruction kinematics of every Geant4 particle.

    ``start_*`` / ``end_*`` come from the particle record, with one
    correction: Geant4 records a photon that converts with zero momentum at
    its end -- the state after it was killed.  Its ``end_mom`` is set to
    its momentum just before converting instead: the start momentum less
    what its Compton electrons carried off, energy the same way.  Compton
    scattering is two-body, so this is exact up to the electron's binding
    energy; Rayleigh scattering, which leaves no secondary, is invisible to
    it.

    Parameters
    ----------
    parts : structured ndarray
        The event's particle records (``particle/geant4``).

    Returns
    -------
    dict
        ``mass`` (N,) float64 and every name in
        :data:`TRACK_KINEMATICS_FIELDS`, (N, 4) float64.
    """
    f8 = lambda name: np.asarray(parts[name], dtype=np.float64)
    mass = f8("mass")
    out = {"mass": mass}
    for pre, key in (("start", ""), ("end", "end_")):
        out[f"{pre}_pos"] = np.stack([f8(key + c) for c in ("x", "y", "z", "t")], 1)
        out[f"{pre}_mom"] = np.stack([f8(key + c) for c in ("px", "py", "pz")]
                                     + [f8(key + "ke") + mass], 1)

    names = parts.dtype.names or ()
    if not {"proc_end", "subproc_end", "proc_start", "subproc_start",
            "track_id", "parent_track_id"} <= set(names):
        return out
    em = int(G4ProcessType.kProcessElectromagnetic)
    pdg = np.asarray(parts["pdg"])
    conv = np.flatnonzero(
        (pdg == 22) & (np.asarray(parts["proc_end"]) == em)
        & (np.asarray(parts["subproc_end"])
           == int(G4ProcessSubtype.kSubtypeEMGammaConversion)))
    if not len(conv):
        return out
    compt = ((np.asarray(parts["proc_start"]) == em)
             & (np.asarray(parts["subproc_start"])
                == int(G4ProcessSubtype.kSubtypeEMComptonScattering)))
    tid = np.asarray(parts["track_id"], dtype=np.int64)
    ptid = np.asarray(parts["parent_track_id"], dtype=np.int64)
    row = {int(t): k for k, t in enumerate(tid)}
    lost = np.zeros((len(parts), 4))
    ke = f8("ke")
    for c in np.flatnonzero(compt):
        k = row.get(int(ptid[c]))
        if k is not None:
            lost[k, :3] += out["start_mom"][c, :3]
            lost[k, 3] += ke[c]
    out["end_mom"][conv] = out["start_mom"][conv] - lost[conv]
    return out


def step_ends(steps, idx, mass, sign):
    """
    Where and with what momentum step(s) *idx* begin (*sign* -1) or end (+1).

    A step record holds its *midpoint* -- measured, a particle's first step
    sits half a ``dx`` from its creation point -- so an end is reached by
    moving ``dx/2`` along the step's direction, as if the step were
    straight, and the time by the matching ``dx/2 / (beta c)``.  The
    momentum is the step's own ``p`` along that direction (theta from the z
    axis), with ``E = sqrt(p^2 + m^2)``; it is not moved.

    Parameters
    ----------
    steps : structured ndarray
        Step records: x, y, z [mm], t [ns], theta, phi [rad], p [MeV/c],
        dx [mm].
    idx : ndarray of int
        Which steps.
    mass : ndarray of float
        The mass of the particle that made each of them [MeV].
    sign : float
        -1 for the start of the step, +1 for its end.

    Returns
    -------
    pos, mom : ndarray, (len(idx), 4) float64
        ``[x, y, z, t]`` and ``[px, py, pz, E]``.
    """
    s = steps[np.asarray(idx, dtype=np.int64)]
    th = np.asarray(s["theta"], dtype=np.float64)
    ph = np.asarray(s["phi"], dtype=np.float64)
    d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                  np.cos(th)], 1)
    p = np.asarray(s["p"], dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    e = np.sqrt(p * p + m * m)
    half = 0.5 * np.asarray(s["dx"], dtype=np.float64)
    beta = np.divide(p, e, out=np.zeros_like(p), where=e > 0)
    dt = np.divide(half, beta * _C_MM_PER_NS, out=np.zeros_like(p),
                   where=beta > 0)
    pos = np.stack([np.asarray(s[c], dtype=np.float64)
                    for c in ("x", "y", "z", "t")], 1)
    pos[:, :3] += sign * half[:, None] * d
    pos[:, 3] += sign * dt
    return pos, np.c_[p[:, None] * d, e]


def step_owner_from_ass(ass, n_parts, n_steps):
    """Per step, the particle whose ``[start, end)`` slice holds it; -1 for none."""
    owner = np.full(n_steps, -1, dtype=np.int64)
    for i in range(min(n_parts, len(ass))):
        a, b = int(ass["start"][i]), int(ass["end"][i])
        owner[a:b] = i
    return owner


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
    ancestor_idx : ndarray of int32
        Index of each particle's primary ancestor.
    geant4_ids : ndarray of int32
        The original ``track_id`` values, unchanged.
    """
    n = len(parts)
    track_ids = parts['track_id']
    index_of = {int(t): i for i, t in enumerate(track_ids)}

    ids = np.arange(n, dtype=np.int32)
    parent_idx = np.empty(n, dtype=np.int32)
    ancestor_idx = np.empty(n, dtype=np.int32)
    for i in range(n):
        parent_idx[i] = index_of.get(int(parts['parent_track_id'][i]), i)
        ancestor_idx[i] = index_of.get(int(root_track_ids[i]), i)
    return ids, parent_idx, ancestor_idx, track_ids.astype(np.int32)


def _get_ancestor_id(parts: np.ndarray) -> np.ndarray:
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
    # unreliable -- see _get_ancestor_id.
    ancestor_refs = _get_ancestor_id(parts)
    index_of = {int(t): i for i, t in enumerate(track_ids)}

    # Step 0: loop over root particles and match to one of vertices by (x,y,z,t) proximity.
    ancestor_indices = np.where(track_ids == ancestor_refs)[0]

    for index in ancestor_indices:
        p = parts[index]
        for vtx_row, vtx in enumerate(verts):
            if np.isclose(p['x'], vtx['x'], atol=1e-4) and \
               np.isclose(p['y'], vtx['y'], atol=1e-4) and \
               np.isclose(p['z'], vtx['z'], atol=1e-4) and \
               np.isclose(p['t'], vtx['t'], atol=1e-4):
                # Match found, do something
                # The *row index* of the vertex, not its interaction_id
                # field: this is what addresses the interactions table.
                interaction_ids[index] = vtx_row
                break

    # Step 1: for non-root particles, assign the same interaction ID as their
    # root particle.  A root reference that is absent from this event list is
    # left for step 2 rather than raising.
    for i in range(len(parts)):
        if track_ids[i] == ancestor_refs[i]:
            continue
        ancestor_index = index_of.get(int(ancestor_refs[i]))
        if ancestor_index is not None:
            interaction_ids[i] = interaction_ids[ancestor_index]

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


def read_primaries(h5file, key: str, index: int):
    """
    Track ids of the particles the generator made in event *index*.

    Returns None when the file has no *key* table, so the caller can fall
    back on :func:`primary_mask`'s parent-less rule.
    """
    if key not in h5file:
        return None
    return np.asarray(h5file[key][index]["track_id"], dtype=np.int64)


def primary_mask(parts: np.ndarray, primary_track_ids=None):
    """
    Which rows of *parts* are generator primaries, and where the two
    descriptions of a primary disagree.

    A primary is a particle of the generator's own list (the primary table,
    *primary_track_ids*).  Geant4 gives exactly those particles
    ``parent_track_id == -1``, so the two are checked against each other.
    Without the table the parent-less rows are taken for the primaries.

    Returns
    -------
    mask : ndarray of bool, shape (N,)
    check : dict or None
        None without the table; otherwise three arrays of track ids, empty
        when everything agrees -- ``parentless_not_primary`` (parent -1 but
        not in the table), ``primary_with_parent`` (in the table but with a
        parent) and ``primary_not_in_particles`` (in the table, absent from
        *parts*).
    """
    tid = np.asarray(parts["track_id"], dtype=np.int64)
    parentless = np.asarray(parts["parent_track_id"]) == -1
    if primary_track_ids is None:
        return parentless, None
    prim = np.asarray(primary_track_ids, dtype=np.int64)
    mask = np.isin(tid, prim)
    return mask, dict(
        parentless_not_primary=tid[parentless & ~mask],
        primary_with_parent=tid[mask & ~parentless],
        primary_not_in_particles=np.setdiff1d(prim, tid),
    )


#: What each kind of :func:`primary_mask` disagreement means, after "track N".
PRIMARY_CHECK_TEXT = {
    "parentless_not_primary": "with parent_track_id -1 but not in the "
                              "primary table",
    "primary_with_parent": "in the primary table but with a parent",
    "primary_not_in_particles": "in the primary table but not among the "
                                "particles",
}


def describe_primary_check(check) -> str:
    """One line naming each disagreement in a :func:`primary_mask` check."""
    parts = []
    for k, text in PRIMARY_CHECK_TEXT.items():
        ids = check[k]
        if len(ids):
            shown = ", ".join(str(int(i)) for i in ids[:5])
            more = f", ... ({len(ids)})" if len(ids) > 5 else ""
            parts.append(f"track {shown}{more} {text}")
    return "; ".join(parts)


def primary_check_failed(check) -> bool:
    """Whether a :func:`primary_mask` check found any disagreement."""
    return check is not None and any(len(v) for v in check.values())


def _get_interaction_type(parts: np.ndarray,
                          electron_energy_threshold: float = 0.05,
                          is_primary=None,
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
    is_primary : ndarray of bool or None, optional
        Which particles are generator primaries (:func:`primary_mask`).
        A primary is ``kPrimary`` -- never LE, whatever its size -- unless
        it is a neutron or a nucleus.  Its ``proc_start`` and
        ``subproc_start`` are not read: Geant4 records no creator process
        for a primary, and the fields hold whatever the converter wrote.
        None takes the particles with ``parent_track_id == -1``.

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
    if is_primary is None:
        is_primary = primary_mask(parts)[0]
    is_primary = np.asarray(is_primary, dtype=bool)

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
        is_primary,
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
    primary_key : str, optional
        HDF5 dataset key for the per-event generator primaries, which decide
        ``kPrimary``.  Default: ``"primary/geant4"``.  A file without it
        falls back on ``parent_track_id == -1``.
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
        primary_key: str  = _DEFAULT_PRIMARY_KEY,
        step_key: str     = _DEFAULT_STEP_KEY,
        ass_key: str      = _DEFAULT_ASS_KEY,
        electron_energy_threshold: float = _DEFAULT_ELECTRON_ENERGY_THRESHOLD,
        min_pc_size: int | None = None,
        voxel_size: float | None = None,
    ) -> None:
        import h5py
        self._path        = path
        self._vertex_key  = vertex_key
        self._part_key    = particle_key
        self._primary_key = primary_key
        #: The primary-table check of the event just read
        #: (:func:`primary_mask`); None when the file has no primary table.
        self.last_primary_check = None
        self._step_key    = step_key
        self._ass_key     = ass_key
        self._e_thresh    = electron_energy_threshold
        self._min_pc_size = min_pc_size
        # Cell size classification measures a cloud's extent in; see
        # pysupera.utils.count_extent for why rows will not do.
        self._voxel_size  = voxel_size
        #: Which steps first_* / last_* are taken from.
        self.kinematics_steps = "all"
        self.last_kinematics = None
        self.last_steps = None
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
        #: vertices of the most recently read event, for the interactions table
        self.last_vertices = verts
        parts = self._file[self._part_key][index]
        steps = self._file[self._step_key][index]
        ass   = self._file[self._ass_key][index]

        num_parts = len(parts)
        is_primary, self.last_primary_check = primary_mask(
            parts, read_primaries(self._file, self._primary_key, index))
        itype     = _get_interaction_type(parts, self._e_thresh, is_primary)
        int_ids   = _get_interaction_id(parts,verts)
        offsets   = np.column_stack([
            ass["start"][:num_parts],
            ass["end"][:num_parts],
        ])


        _ids, _par, _root, _g4 = _to_index_space(parts, _get_ancestor_id(parts))

        #: per-particle kinematics of the event just read, in *parts* order
        #: (see particle_kinematics), plus ``geant4_trackid`` to join on
        self.last_kinematics = dict(particle_kinematics(parts),
                                    geant4_trackid=_g4)
        #: The event's steps, for the instance first/last kinematics
        #: (pysupera.kinematics): the records, the *parts* row that made
        #: each, and which count -- here all of them, there being no readout.
        #: A step goes to the particle whose voxels hold it: a point's id is
        #: its step index.
        self.last_steps = dict(
            steps=steps,
            owner=step_owner_from_ass(ass, num_parts, len(steps)),
            counted=None, group=None)

        return Particle.from_flat_arrays(
            ids                  = _ids,
            parent_ids           = _par,
            ancestor_ids             = _root,
            pdgs                 = parts["pdg"],
            parent_pdgs          = _get_parent_pdg(parts),
            interaction_ids      = int_ids,
            interaction_types    = itype,
            point_cloud_flat     = _steps_to_plain_array(steps),
            point_cloud_offsets  = offsets,
            min_pc_size          = self._min_pc_size,
            voxel_size           = self._voxel_size,
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
            primary_key               = str(r.get("primary_key",
                                              _DEFAULT_PRIMARY_KEY)),
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
