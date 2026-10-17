"""
pysupera.kinematics
===================

The ``first_*`` / ``last_*`` kinematics of reconstructed instances.

``start_*`` and ``end_*`` describe a Geant4 particle and are read off its
record (:func:`~pysupera.readers.format_edepsim_h5.particle_kinematics`).
``first_*`` and ``last_*`` describe what a model is asked to reconstruct --
an instance, as drawn -- so they are made from the energy deposits the
instance actually collected, after the instances exist:

* Each Geant4 step goes to the pysupera particle that holds its deposit
  (:func:`step_particles`), so a step follows its deposit through
  defragmentation, absorption and shower merging, as the points do.
* Of an instance's counted steps, only those of a Geant4 particle with at
  least one non-LE member in it are used -- a photon's conversion pair
  counting as non-LE, being the shower's start.  A low-energy blob absorbed
  from an unrelated particle -- a Compton electron that happened to touch a
  muon -- is part of the picture but not of the trajectory, and it can be
  later than everything else.  An instance with no non-LE member, or whose
  non-LE members left no counted step, uses all of its steps.
* ``first`` is the start of the earliest of those steps, ``last`` the end of
  the latest (:func:`~pysupera.readers.format_edepsim_h5.step_ends`).
* A ``kShower`` instance has no well-defined last point: ``last_*`` is NaN.
  When a photon heads it and the first counted step is its pair's,
  ``first_mom`` is the summed creation momentum of the pair -- the
  shower's own momentum.  When the readout missed the pair, the first
  point is on a later product and ``first_mom`` is that step's.
* ``first_mom`` is the momentum of whichever particle made the step,
  with that particle's mass, so its E can be below the row's own mass:
  an LE electron instance whose earliest step is a photon's local deposit.

Only instance rows carry these values; every other row gets NaN.
"""

from __future__ import annotations

import numpy as np

from .readers.format_edepsim_h5 import step_ends
from .utils import InteractionType, PointFeature, SemanticType

#: Columns made here, each (4,) per instance.
INSTANCE_KINEMATICS_FIELDS = ("first_pos", "first_mom", "last_pos", "last_mom")

_LE = SemanticType.kLEScatter
_SHOWER = SemanticType.kShower.value
_CONV = InteractionType.kConversion.value


def step_particles(n_steps, particles, group=None, group_owner=None):
    """
    Per step, the id of the pysupera particle holding its deposit; -1 if none.

    Parameters
    ----------
    n_steps : int
        Number of steps in the event.
    particles : list of Particle
        The event's particles, after defragmentation and renumbering.
    group, group_owner : ndarray of int or None
        JAXTPC: each step's event-global group, and each group's owning
        particle (:func:`~pysupera.provenance.build_group_owners`).  A step
        then goes where its segment goes in the truth cloud.  Without them
        a step goes to the particle whose voxels hold it: each voxel's
        input ids (``voxmap``), or each point's ``id`` column when nothing
        was voxelized, are step indices.
    """
    out = np.full(n_steps, -1, dtype=np.int64)
    if group is not None:
        # A voxel's input ids name deposits or hits here, not steps.
        if group_owner is None:
            return out
        g = np.asarray(group, dtype=np.int64)
        own = np.asarray(group_owner, dtype=np.int64)
        ok = (g >= 0) & (g < len(own))
        out[ok] = own[g[ok]]
        return out
    for p in particles:
        vm = getattr(p, "voxmap", None)
        if vm is not None and vm[1] is not None:
            ids = np.asarray(vm[1], dtype=np.int64)
        elif len(p.point_cloud) and p.point_cloud.shape[1] > PointFeature.id:
            ids = p.point_cloud[:, PointFeature.id].astype(np.int64)
        else:
            continue
        ids = ids[(ids >= 0) & (ids < n_steps)]
        out[ids] = int(p.id)
    return out


def instance_kinematics(inst_groups, particles, steps, step_owner, step_pid,
                        track_kin, counted=None):
    """
    ``first_*`` / ``last_*`` of every instance in *inst_groups*.

    Parameters
    ----------
    inst_groups : list of Group
        The instances, as :func:`~pysupera.layout.snapshot_groups` took them.
    particles : list of Particle
        The event's particles.
    steps : structured ndarray
        The event's step records.
    step_owner : ndarray of int
        Per step, the row of the reader's particle table that made it.
    step_pid : ndarray of int
        Per step, the pysupera particle holding it (:func:`step_particles`).
    track_kin : dict
        The reader's ``last_kinematics``: ``mass``, ``start_mom`` and
        ``geant4_trackid`` per particle-table row.
    counted : ndarray of bool or None
        Which steps may be used (JAXTPC: the ones the readout saw).  None
        counts all.

    Returns
    -------
    dict
        ``{instance rep id: {name: (4,) float64}}`` for every name in
        :data:`INSTANCE_KINEMATICS_FIELDS`, NaN where undefined.
    """
    by_id = {int(p.id): p for p in particles}
    g4_of_row = np.asarray(track_kin["geant4_trackid"], dtype=np.int64)
    mass_of_row = np.asarray(track_kin["mass"], dtype=np.float64)
    row_of_g4 = {int(t): k for k, t in enumerate(g4_of_row)}
    nan4 = lambda: np.full(4, np.nan)
    out = {int(g.rep_id): {k: nan4() for k in INSTANCE_KINEMATICS_FIELDS}
           for g in inst_groups}
    if not len(steps) or not inst_groups:
        return out

    # Instance of each particle, and which Geant4 particles an instance
    # follows: those with a non-LE member in it.
    max_pid = max(by_id) if by_id else -1
    inst_of_pid = np.full(max_pid + 2, -1, dtype=np.int64)
    span = int(g4_of_row.max()) + 2 if len(g4_of_row) else 1
    allowed, has_track = [], set()
    for g in inst_groups:
        rid = int(g.rep_id)
        # A photon's conversion pair is the shower's start whatever the size
        # of its clouds: a low-energy pair is typed LE, and must not lose
        # the first point to a later product.
        pair = {int(getattr(by_id[m], "geant4_id", m))
                for m in _pair_members(by_id.get(rid), g.member_ids, by_id)}
        for m in g.member_ids:
            m = int(m)
            if 0 <= m <= max_pid:
                inst_of_pid[m] = rid
            p = by_id.get(m)
            if p is None:
                continue
            g4 = int(getattr(p, "geant4_id", m))
            # A delta ray merged into its track is part of the track's
            # picture, not of its trajectory.
            if ((p.sem_type is not _LE
                 and not getattr(p, "buried_in_track", False)) or g4 in pair):
                allowed.append(rid * span + g4 + 1)
                has_track.add(rid)

    pid = np.asarray(step_pid, dtype=np.int64)
    own = np.asarray(step_owner, dtype=np.int64)
    sel = (pid >= 0) & (pid <= max_pid) & (own >= 0) & (own < len(g4_of_row))
    if counted is not None:
        sel &= np.asarray(counted, dtype=bool)
    idx = np.flatnonzero(sel)
    inst = inst_of_pid[pid[idx]]
    keep = inst >= 0
    idx, inst = idx[keep], inst[keep]
    key = inst * span + g4_of_row[own[idx]] + 1
    ok = (~np.isin(inst, np.fromiter(has_track, dtype=np.int64,
                                     count=len(has_track)))
          | np.isin(key, np.asarray(allowed, dtype=np.int64)))
    # An instance whose non-LE particles left no counted step -- the readout
    # missed them -- falls back to all of its counted steps.
    ok |= ~np.isin(inst, np.unique(inst[ok]))
    idx, inst = idx[ok], inst[ok]
    if not len(idx):
        return out

    # Earliest and latest step of each instance, by the recorded time.
    t = np.asarray(steps["t"], dtype=np.float64)[idx]
    order = np.lexsort((t, inst))
    idx, inst = idx[order], inst[order]
    starts = np.flatnonzero(np.r_[True, inst[1:] != inst[:-1]])
    ends = np.r_[starts[1:], len(inst)] - 1
    first, last = idx[starts], idx[ends]
    f_pos, f_mom = step_ends(steps, first, mass_of_row[own[first]], -1.0)
    l_pos, l_mom = step_ends(steps, last, mass_of_row[own[last]], 1.0)

    sem_of = {int(g.rep_id): int(g.sem_type) for g in inst_groups}
    members_of = {int(g.rep_id): g.member_ids for g in inst_groups}
    start_mom = np.asarray(track_kin["start_mom"], dtype=np.float64)
    for k, rid in enumerate(inst[starts]):
        rid = int(rid)
        o = out[rid]
        o["first_pos"], o["first_mom"] = f_pos[k], f_mom[k]
        if sem_of[rid] != _SHOWER:
            o["last_pos"], o["last_mom"] = l_pos[k], l_mom[k]
            continue
        rows = _pair_rows(by_id.get(rid), members_of[rid], by_id, row_of_g4)
        # Only when the shower's first counted step is the pair's: a pair
        # the readout did not see leaves the first point on a later
        # product, and the photon's momentum would not belong to it.
        if rows and int(own[first[k]]) in rows:
            o["first_mom"] = start_mom[rows].sum(axis=0)
    return out


def _pair_members(rep, member_ids, by_id):
    """Ids of the members that are the pair photon *rep* converted into."""
    if rep is None or int(rep.pdg) != 22:
        return []
    return [int(m) for m in member_ids
            if (p := by_id.get(int(m))) is not None
            and int(p.parent_id) == int(rep.id) and abs(int(p.pdg)) == 11
            and int(getattr(p, "_interaction_type", -1)) == _CONV]


def _pair_rows(rep, member_ids, by_id, row_of_g4):
    """Particle-table rows of the pair *rep* converted into; [] if none."""
    rows = []
    for m in _pair_members(rep, member_ids, by_id):
        p = by_id[m]
        r = row_of_g4.get(int(getattr(p, "geant4_id", p.id)))
        if r is not None:
            rows.append(r)
    # A pair split by defragmentation appears more than once; count each
    # Geant4 particle once.
    return sorted(set(rows))
