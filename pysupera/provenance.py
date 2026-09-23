"""
Which JAXTPC hits belong to which reconstructed object.

A hit is the readout projection of a *group* -- JAXTPC's cluster of energy
deposits -- and ``group_to_track`` gives each group exactly one Geant4 track.
That is not enough on its own to regroup hits by reconstructed object, because
defragmentation splits one track into several pysupera particles, so
``track -> particle`` is one-to-many.

What makes this tractable, for truth deposits, is that a group never straddles
a split.  A group is itself a tight spatial cluster and defragmentation
clusters at ``distance_threshold``; measured over eight events, none of
354,959 groups spanned two particles.  So ``group -> pysupera particle`` *is*
a function, and one small table per event answers "which hits belong to
instance X" without going near voxels:

    groups = {g for g, pid in group_particle if pid in particles_of(X)}
    hits   = [h for h in plane if group_ids[h] in groups]

The invariant is checked rather than assumed -- see :func:`build_group_owners`.
Going through voxels instead would be both larger and worse: a group's deposits
land in several voxels 38% of the time, so hit-to-voxel is many-to-many, while
hit-to-group-to-particle is a chain of functions.

It stops being a function under ``reader.point_source=hits``.  There a point
is a fired pixel rather than a deposit, and the readout leaves gaps the
deposit cloud does not have: defragmentation cuts a track at one of them and a
group's hits land on both sides.  That is a property of the detected image,
not a bug, so hit mode resolves the group by majority and reports how often it
had to.
"""

from __future__ import annotations

import numpy as np

#: Written for groups that no stored particle claims.
NO_OWNER = -1


class GroupOwnershipError(ValueError):
    """A group's deposits landed in more than one pysupera particle."""


def build_group_owners(particles, deposit_to_group, n_groups, check=True,
                       resolve="first", report=None):
    """
    Map every JAXTPC group to the pysupera particle that owns it.

    Parameters
    ----------
    particles : list of Particle
        The event's particles, after preprocessing, each carrying the
        ``voxmap`` CSR whose ``input_ids`` are event-global deposit indices.
    deposit_to_group : ndarray of int64
        Event-global deposit index -> event-global group number.
    n_groups : int
        Total groups in the event, so untouched ones can be marked.
    check : bool
        Raise :class:`GroupOwnershipError` when a group spans two particles.
        With *check* false the group is resolved by *resolve* instead, which
        is lossy either way -- enable the deposit-level map if this fires and
        the exact answer matters.
    resolve : {'first', 'majority'}
        How to settle a straddled group when *check* is false.  ``'first'``
        keeps the first owner seen, which is arbitrary but cheap.
        ``'majority'`` gives the group to the particle holding most of its
        rows, which is the only defensible answer when straddling is normal
        rather than exceptional -- see the note below.
    report : dict, optional
        Filled with ``n_claimed`` and ``n_straddled`` so a caller can say how
        often the invariant actually held.  Counting is free here and
        impossible afterwards, the array having collapsed to one owner.

    Returns
    -------
    ndarray of int32, shape (n_groups,)
        Owning particle id per group, :data:`NO_OWNER` where none.

    Notes
    -----
    With truth deposits a group never straddles a split -- measured over
    eight events, none of 354,959 did -- because a group is a tight spatial
    cluster and defragmentation clusters at ``distance_threshold``.  With
    *detected hits* (``reader.point_source=hits``) that no longer holds: the
    readout leaves gaps a deposit cloud does not have, defragmentation cuts a
    track at them, and a group's hits can end up on both sides.  Hit mode
    therefore wants ``resolve='majority'`` and a look at ``report``.
    """
    if resolve not in ("first", "majority"):
        raise ValueError(
            f"resolve must be 'first' or 'majority', got {resolve!r}")

    n_groups = int(n_groups)
    owner = np.full(n_groups, NO_OWNER, dtype=np.int32)
    clash = None
    straddled = set()
    # Rows per (group, particle), kept only when majority resolution needs it.
    tally = {} if resolve == "majority" else None

    for p in particles:
        vm = getattr(p, "voxmap", None)
        if vm is None:
            continue
        _off, ids = vm
        if ids is None or not len(ids):
            continue
        mine = deposit_to_group[np.asarray(ids, dtype=np.int64)]
        mine = mine[(mine >= 0) & (mine < n_groups)]
        if not len(mine):
            continue
        groups, counts = np.unique(mine, return_counts=True)

        taken = owner[groups]
        held = groups[taken != NO_OWNER]
        if len(held):
            straddled.update(int(g) for g in held)
            if clash is None:
                g0 = int(held[0])
                clash = (g0, int(owner[g0]), int(p.id))
        owner[groups[taken == NO_OWNER]] = int(p.id)

        if tally is not None:
            for g, c in zip(groups.tolist(), counts.tolist()):
                cur = tally.get(g)
                if cur is None or c > cur[0]:
                    tally[g] = (c, int(p.id))

    if clash is not None and check:
        g, a, b = clash
        raise GroupOwnershipError(
            f"group {g} has deposits in two pysupera particles ({a} and {b}), "
            f"so hits cannot be attributed to one of them.  This breaks the "
            f"assumption the group table rests on.  Set "
            f"particle.voxelize.store_mapping=true for the exact "
            f"deposit-level mapping, or check_group_ownership=false to "
            f"accept the first owner.  With reader.point_source=hits this is "
            f"expected rather than exceptional -- the readout gaps split "
            f"tracks that the deposits held together -- so run hit mode with "
            f"check_group_ownership=false."
        )

    if tally is not None:
        for g in straddled:
            best = tally.get(g)
            if best is not None:
                owner[g] = np.int32(best[1])

    if report is not None:
        report["n_claimed"] = int((owner != NO_OWNER).sum())
        report["n_straddled"] = len(straddled)
    return owner


def remap_owners(owner, mapping):
    """
    Translate a particle-level owner array into another id space.

    *mapping* is ``{particle_id: other_id}``.  Particles absent from it --
    and unclaimed groups -- come out as :data:`NO_OWNER`.

    The particle id is the primitive the invariant is checked against, but it
    is the wrong key to *store*: only about a sixth of particles get a row in
    the output, so most owners would name something the reader cannot see.
    Instance and fragment ids are always present, which is what makes the
    join possible.
    """
    out = np.full(len(owner), NO_OWNER, dtype=np.int32)
    claimed = np.flatnonzero(owner != NO_OWNER)
    for g in claimed:
        v = mapping.get(int(owner[g]))
        if v is not None:
            out[g] = np.int32(v)
    return out


def instance_of_groups(view, frag_owner):
    """
    The instance owning each group, derived rather than stored.

    A fragment's points lie inside exactly one instance's block, so the
    instance follows from the fragment by containment -- storing it as well
    would be duplication.  Verified over 29,201 groups with no mismatch.

    Note this cannot be done with a column join: ``inst_id`` marks a row as
    *being* an instance representative (it equals ``id`` there and is -1
    everywhere else), it does not record membership.  Membership is
    positional, which is what this function reads.

    Parameters
    ----------
    view : EventView
        The event the groups belong to.
    frag_owner : ndarray
        ``groups/fragment_id`` for that event.

    Returns
    -------
    ndarray of int32
        Owning instance id per group, :data:`NO_OWNER` where none.
    """
    c = view.columns
    row_of = {int(i): k for k, i in enumerate(c["id"])}

    inst = np.flatnonzero(view.is_instance)
    lo = np.asarray(c["inst_pc_start"])[inst]
    hi = np.maximum(np.asarray(c["inst_pc_le_end"])[inst],
                    np.asarray(c["inst_pc_end"])[inst])
    keep = lo >= 0
    inst, lo, hi = inst[keep], lo[keep], hi[keep]
    order = np.argsort(lo)
    inst, lo, hi = inst[order], lo[order], hi[order]
    inst_id = np.asarray(c["id"])[inst]

    out = np.full(len(frag_owner), NO_OWNER, dtype=np.int32)
    for g in np.flatnonzero(np.asarray(frag_owner) >= 0):
        k = row_of.get(int(frag_owner[g]))
        if k is None:
            continue
        p = int(c["frag_pc_start"][k])
        if p < 0:
            p = int(c["frag_pc_le_start"][k])
        if p < 0:
            continue
        j = int(np.searchsorted(lo, p, side="right")) - 1
        if 0 <= j < len(lo) and p < hi[j]:
            out[g] = inst_id[j]
    return out


def le_flags(owner, is_le_of):
    """
    Per-group low-energy flag, from the particle that owns the group.

    A group's deposits all belong to one particle, so LE-ness is exact at
    this level -- unlike the instance, which normally mixes an LE and a
    non-LE side.  Without this the 2D view could only guess, and would have
    to mask by the host instance, which is a different question.

    *is_le_of* is ``{particle_id: bool}``.  Unclaimed groups come out 0.
    """
    out = np.zeros(len(owner), dtype=np.int8)
    for g in np.flatnonzero(owner != NO_OWNER):
        if is_le_of.get(int(owner[g]), False):
            out[g] = 1
    return out


def hits_of_groups(group_ids, wanted):
    """
    Row indices of the hits belonging to *wanted*, on one readout plane.

    *group_ids* is a plane's per-hit group number; *wanted* any collection of
    group numbers.
    """
    wanted = np.asarray(list(wanted), dtype=np.int64)
    if not len(wanted):
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.isin(np.asarray(group_ids), wanted))


def groups_of_particles(group_owner, particle_ids):
    """
    Group numbers owned by any of *particle_ids*.

    :data:`NO_OWNER` is dropped from the request rather than matched: it
    marks a group nothing claims, so asking for it would hand back every
    unclaimed group.  A caller passing an unset ``inst_id`` gets nothing,
    which is the truthful answer.
    """
    want = np.asarray(list(particle_ids), dtype=np.int64)
    want = want[want != NO_OWNER]
    if not len(want):
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.isin(group_owner, want))
