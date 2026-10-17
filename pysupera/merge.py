"""
pysupera.merge
==============

Post-partitioning merge algorithms that operate on the *fragment
representatives* produced by :class:`~pysupera.partitioner.ParticlePartitioner`
to build final *instance*-level groupings.

The algorithms here are **purely genealogical** — they do not require a
second spatial proximity pass.  The main entry point for most workflows is
:func:`merge_em_showers`, which groups step-1 EM fragments into shower
instances by walking the Geant4 parent-ID chain.

Typical call sequence
---------------------
::

    from pysupera.partitioner import ParticlePartitioner
    from pysupera.merge import merge_em_showers

    partitioner = ParticlePartitioner(particles, distance_threshold=5.2, ...)
    fragments   = partitioner.partition_combined(conditions, verbose=False)
    instances   = merge_em_showers(fragments)   # step-2 shower assembly
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .utils import InteractionType, SemanticType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: PDG codes considered "EM" for shower-initiator chain walking.
_EM_PDGS: frozenset = frozenset({11, -11, 22})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _em_like(rep) -> bool:
    """Whether *rep* takes part in the shower walk at all: e±, γ or kShower."""
    return (abs(int(rep.pdg)) in _EM_PDGS
            or rep.sem_type == SemanticType.kShower)


def _can_initiate(rep) -> bool:
    """Whether *rep* may head a shower instance.

    ``kLEScatter`` fragments may *join* a shower -- a Compton blob of a photon
    inside it, say -- but never head one: a shower headed by an LE fragment
    would be dropped whole by the ``drop_le_scatter`` filter downstream, its
    non-LE members with it.
    """
    return _em_like(rep) and rep.sem_type != SemanticType.kLEScatter


#: Interaction types of the electrons a photon makes without converting.
_PHOTON_SECONDARY = frozenset({InteractionType.kCompton.value,
                               InteractionType.kPhotoElectron.value})


def _is_photon_secondary(rep) -> bool:
    """A Compton electron or photoelectron: made by a photon, not a shower."""
    return (int(getattr(rep, "_interaction_type", -1)) in _PHOTON_SECONDARY
            and int(getattr(rep, "parent_pdg", 0)) == 22)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_em_showers(fragments):
    """
    Step-2 EM shower genealogy merger.

    Groups step-1 fragment representatives into *shower instances* by
    walking the parent-ID chain of each EM fragment until a non-EM ancestor
    (or the top of the fragment list) is reached.  The top-most fragment on
    that chain that may head a shower is the **shower initiator**; all EM
    fragments that share the same initiator are merged into it.  Non-EM
    fragments pass through unchanged (each becomes its own instance).

    The merge is **purely genealogical** — no spatial proximity check is
    performed.  Photons (PDG 22) carry no deposits at a conversion and are
    natural shower initiators when no EM parent is found.

    Two rules shape the walk:

    * ``kLEScatter`` fragments are walked through and can be merged in, but
      never head a shower (:func:`_can_initiate`).  An LE fragment whose
      chain reaches no possible initiator stays its own instance.
    * A Compton electron or photoelectron reaches its photon only when that
      photon is itself inside a shower (has an EM parent fragment).  A
      shower begins at the pair production, so the electrons a top-level
      photon scattered out before converting -- or instead of converting --
      are their own instances.  The same electrons made by a photon *within*
      a shower belong to it.

    The parent of a fragment is the fragment holding its parent particle,
    which proximity may have put among the fragment's own descendants.  Where
    the walk loops, the fragments on the loop share one instance.

    Extended parent lookup
    ~~~~~~~~~~~~~~~~~~~~~~
    The walk uses two lookup tables in tandem:

    * ``frag_by_id``        — fragment rep ID → fragment object (fast path).
    * ``frag_by_member_id`` — every original *member* particle ID → fragment
      object.  This is needed because ``rep.parent_id`` is the *original*
      particle's parent ID, which may have been absorbed as a member of a rep
      whose ``rep.id`` differs from that original ID.  Without this extended
      lookup, grandchild fragments whose direct parent was merged into a
      different rep would fail to find their ancestor and would incorrectly
      start a new shower.

    Parameters
    ----------
    fragments : list of Particle
        Step-1 representative particles produced by
        ``ParticlePartitioner.partition_combined`` (or equivalent).  Each
        representative must have its ``member_ids`` attribute populated (list
        of original preprocessed particle IDs).

    Returns
    -------
    list of Particle
        Instance-level representatives.  Each returned particle has
        ``member_ids`` set to the **flat union** of original preprocessed
        particle IDs from all fragment representatives merged into it.
        Non-merged fragments are returned as-is (``member_ids`` unchanged).
    """
    # Fast lookup: fragment representative ID → fragment object
    frag_by_id: dict = {r.id: r for r in fragments}

    # Extended lookup: every original member particle ID → fragment rep.
    frag_by_member_id: dict = {}
    for r in fragments:
        mids = r.member_ids if r.member_ids is not None else [r.id]
        for mid in mids:
            frag_by_member_id[int(mid)] = r

    def parent_frag(rep):
        """The fragment holding *rep*'s parent particle, or None."""
        pid = rep.parent_id
        par = (frag_by_id.get(pid)
               or frag_by_member_id.get(int(pid) if pid is not None else -1))
        return None if par is rep else par

    # Each fragment's next step up the shower chain, or None where the chain
    # stops.  Photons are linked first: whether a photon secondary may climb
    # to its photon depends on whether the photon itself climbs.
    up: dict = {}
    for rep in fragments:
        par = parent_frag(rep)
        up[id(rep)] = (par if par is not None and _em_like(rep)
                       and _em_like(par) else None)
    for rep in fragments:
        par = up[id(rep)]
        if (par is not None and _is_photon_secondary(rep)
                and abs(int(par.pdg)) == 22 and up[id(par)] is None):
            up[id(rep)] = None

    def find_initiator(rep):
        """Walk *rep*'s chain; return the top-most fragment that may head it."""
        if not _em_like(rep):
            return rep
        top = rep if _can_initiate(rep) else None
        seen = {id(rep)}
        cur = up[id(rep)]
        while cur is not None and id(cur) not in seen:   # cycle guard
            seen.add(id(cur))
            if _can_initiate(cur):
                top = cur
            cur = up[id(cur)]
        return rep if top is None else top

    # Chains can loop: an LE particle absorbed by proximity into a fragment
    # of its own descendants -- a stopped muon swallowed by the shower of its
    # Michel's brem photon -- makes that fragment the descendant's parent
    # too.  Each fragment on a loop then names a different initiator, and an
    # initiator outside its own group would be dropped with its members.
    # Follow initiators to a fixed point; a loop of them takes its lowest-id
    # member that may head a shower, so every initiator heads its own group.
    first = {id(rep): find_initiator(rep) for rep in fragments}
    resolved: dict = {}

    def initiator_of(rep):
        path, on_path, cur = [], {}, rep
        while id(cur) not in resolved:
            nxt = first[id(cur)]
            if nxt is cur:
                resolved[id(cur)] = cur
                break
            if id(cur) in on_path:
                loop = path[on_path[id(cur)]:]
                head = min(loop, key=lambda r: (not _can_initiate(r), r.id))
                for r in loop:
                    resolved[id(r)] = head
                break
            on_path[id(cur)] = len(path)
            path.append(cur)
            cur = nxt
        head = resolved[id(cur)]
        for r in path:
            resolved.setdefault(id(r), head)
        return resolved[id(rep)]

    # Group every fragment under its shower initiator.
    groups: dict = defaultdict(list)   # initiator.id → list[fragment rep]
    for rep in fragments:
        initiator = initiator_of(rep)
        groups[initiator.id].append(rep)

    # Build instance representatives by merging point clouds and member ID lists.
    result = []
    for initiator_id, members in groups.items():
        initiator = frag_by_id[initiator_id]

        if len(members) == 1:
            # Singleton — no merging needed; return as-is.
            result.append(initiator)
            continue

        # Concatenate non-empty point clouds from all merged fragments.
        clouds = [m.point_cloud for m in members if len(m.point_cloud) > 0]
        if clouds:
            initiator.point_cloud = np.concatenate(clouds, axis=0)

        # member_ids = flat union of original particle IDs across all merged fragments.
        all_orig_ids: list = []
        for m in members:
            if m.member_ids is not None:
                all_orig_ids.extend(m.member_ids)
            else:
                all_orig_ids.append(m.id)
        initiator.member_ids = all_orig_ids

        result.append(initiator)

    return result


# ---------------------------------------------------------------------------
# Split-off pieces of tracks
# ---------------------------------------------------------------------------

#: Semantic types whose split-off pieces go back to their particle: the ones
#: with a continuous trajectory that a gap can interrupt.  Showers collect
#: their pieces through PhotonDecay and genealogy already.
_TRACK_LIKE = frozenset({SemanticType.kTrack, SemanticType.kMichel,
                         SemanticType.kDelta})

#: Particles whose pieces never go back, whatever their semantic type: a
#: neutron's blobs are separate interactions, not one trajectory.  A primary
#: neutron is ``kTrack``, and its own deposits (sub-cut recoils, in some
#: physics lists) can lie metres apart.
_NOT_A_TRAJECTORY_PDG = frozenset({2112})


def find_split_pieces(fragments, particles):
    """
    Stand-alone fragments that are split-off pieces of a track-like particle.

    Defragmentation cuts a particle's clusters of ``min_pc_size`` voxels or
    fewer off into new ``kLEScatter`` particles that keep its ``geant4_id``
    (a stub past a gap, the last voxels beyond a dead stretch).  When such a
    piece touches nothing it stays a fragment of its own, though it is part
    of its particle's trajectory.  This finds those fragments: every member
    a piece of the same Geant4 particle, that particle ``kTrack``,
    ``kMichel`` or ``kDelta`` and not a neutron, and its own fragment not
    LE.  A piece that
    proximity absorbed into another fragment is left where it is -- it looks
    like part of that fragment, so it is.

    Parameters
    ----------
    fragments : list of Particle
        Step-1 fragment representatives.
    particles : list of Particle
        Every particle of the event.

    Returns
    -------
    list of tuple
        ``(piece_fragment, track_fragment)`` pairs.
    """
    by_id = {int(p.id): p for p in particles}
    g4 = lambda p: int(getattr(p, "geant4_id", p.id))
    frag_of = {}
    for f in fragments:
        for m in (f.member_ids if f.member_ids is not None else [f.id]):
            frag_of[int(m)] = f
    out = []
    for f in fragments:
        if f.sem_type != SemanticType.kLEScatter:
            continue
        piece = by_id.get(int(f.id))
        src = by_id.get(int(piece.parent_id)) if piece is not None else None
        if src is None or src is piece or g4(src) != g4(piece):
            continue
        mids = f.member_ids if f.member_ids is not None else [f.id]
        if any(int(m) not in by_id or g4(by_id[int(m)]) != g4(piece)
               for m in mids):
            continue
        if (src.sem_type not in _TRACK_LIKE
                or abs(int(src.pdg)) in _NOT_A_TRAJECTORY_PDG):
            continue
        home = frag_of.get(int(src.id))
        if home is None or home is f or home.sem_type == SemanticType.kLEScatter:
            continue
        out.append((f, home))
    return out


# ---------------------------------------------------------------------------
# Delta rays buried in their track
# ---------------------------------------------------------------------------

def _cells(point_cloud, voxel_size):
    """The cells *point_cloud* occupies, as (n, 3) coordinates in mm:
    voxel corners, or the exact positions when *voxel_size* is None."""
    xyz = np.asarray(point_cloud[:, :3], dtype=np.float64)
    if voxel_size:
        return np.unique(np.floor(xyz / voxel_size), axis=0) * voxel_size
    return np.unique(xyz, axis=0)


def find_buried_deltas(fragments, particles, min_standoff, voxel_size=None):
    """
    Delta-ray fragments that cannot be told apart from their parent track.

    A delta ray mostly overlaps the track it came from; with a diffused
    readout a low-energy one is invisible, its cells all on or hugging the
    track.  Such a delta ray's *points* belong in the track's fragment.
    What makes a delta ray visible is how far it gets from the track, not
    how many cells it has off it: a halo along the track can hold many.
    This finds the fragments headed by a non-primary ``kDelta`` particle,
    not LE, whose parent sits in a different, track-like fragment, and none
    of whose occupied cells is *min_standoff* or more from the nearest cell
    of that fragment.  Every other member of such a fragment must be LE or
    a piece of the delta ray, so nothing else is moved with it.

    The delta-ray particle is not reclassified: it stays ``kDelta`` with its
    own points (see :func:`merge_buried_deltas`).

    Parameters
    ----------
    fragments : list of Particle
        Step-1 fragment representatives.
    particles : list of Particle
        Every particle of the event.
    min_standoff : float
        Distance in mm, cell to cell, a delta ray must reach from its
        track's fragment to keep its own.  0 or less finds nothing.
    voxel_size : float or None
        Cell size the clouds are counted in; None uses exact positions.

    Returns
    -------
    list of tuple
        ``(delta_fragment, track_fragment, standoff)`` triples, the
        standoff in mm.
    """
    if not min_standoff or min_standoff <= 0:
        return []
    from scipy.spatial import cKDTree
    by_id = {int(p.id): p for p in particles}
    frag_of = {}
    for f in fragments:
        for m in (f.member_ids if f.member_ids is not None else [f.id]):
            frag_of[int(m)] = f
    g4 = lambda p: int(getattr(p, "geant4_id", p.id))
    trees = {}
    out = []
    for f in fragments:
        rep = by_id.get(int(f.id))
        if (rep is None or rep.sem_type != SemanticType.kDelta
                or f.sem_type != SemanticType.kDelta
                or rep.interaction_type == InteractionType.kPrimary):
            continue
        par = by_id.get(int(rep.parent_id)) if rep.parent_id is not None else None
        if par is None or par is rep:
            continue
        home = frag_of.get(int(par.id))
        if home is None or home is f or home.sem_type not in _TRACK_LIKE:
            continue
        mids = f.member_ids if f.member_ids is not None else [f.id]
        if any(int(m) != int(rep.id) and int(m) in by_id
               and by_id[int(m)].sem_type != SemanticType.kLEScatter
               and g4(by_id[int(m)]) != g4(rep) for m in mids):
            continue
        if not len(f.point_cloud) or not len(home.point_cloud):
            continue
        if id(home) not in trees:
            trees[id(home)] = cKDTree(_cells(home.point_cloud, voxel_size))
        standoff = float(trees[id(home)].query(_cells(f.point_cloud, voxel_size))[0].max())
        if standoff < min_standoff:
            out.append((f, home, standoff))
    return out


def merge_buried_deltas(fragments, found, particles):
    """
    Merge each delta-ray fragment of :func:`find_buried_deltas` into its
    track's fragment, in place, before the fragments are snapshotted.

    Only the points move: the delta-ray particle keeps its ``kDelta`` type,
    its row and its own point range, and is marked ``buried_in_track`` so
    that the track instance's first/last points are not drawn from its
    steps.  A delta ray of a buried delta ray follows it into the track.

    Returns
    -------
    list of Particle
        *fragments* without the merged ones.
    """
    by_id = {int(p.id): p for p in particles}
    gone = {}
    root = lambda h: root(gone[id(h)]) if id(h) in gone else h
    for f, home, _ in found:
        h = root(home)
        if h is f:
            continue
        h.member_ids = (list(h.member_ids if h.member_ids is not None else [h.id])
                        + list(f.member_ids if f.member_ids is not None else [f.id]))
        clouds = [c for c in (h.point_cloud, f.point_cloud) if len(c)]
        if len(clouds) > 1:
            h.point_cloud = np.concatenate(clouds, axis=0)
        elif clouds:
            h.point_cloud = clouds[0]
        gone[id(f)] = h
        by_id[int(f.id)].buried_in_track = True
    return [f for f in fragments if id(f) not in gone]


def find_absorbed_deltas(frag_groups, particles):
    """
    Delta rays that the size rule made LE and proximity put in their track.

    A delta ray smaller than ``min_pc_size`` is typed ``kLEScatter`` and,
    touching its track, ``AbsorbLEScatter`` merges it there.  Its points
    are the track's either way; this finds such particles so that, like the
    ones :func:`merge_buried_deltas` moves, they can be written as
    ``kDelta`` rows with their own points.  A delta ray qualifies when it
    has points and sits in the same fragment as its parent, and that
    fragment is track-like.  A piece of the parent (same Geant4 track) is
    not a delta ray of it and is left out.

    Parameters
    ----------
    frag_groups : list of Group
        Step-1 fragments, from :func:`~pysupera.layout.snapshot_groups`.
    particles : list of Particle
        Every particle of the event, carrying the clouds step 1 grouped.

    Returns
    -------
    list of Particle
    """
    by_id = {int(p.id): p for p in particles}
    frag_of, sem_of = {}, {}
    for g in frag_groups:
        sem_of[g.rep_id] = g.sem_type
        for m in g.member_ids:
            frag_of[int(m)] = g.rep_id
    track_like = {t.value for t in _TRACK_LIKE}
    g4 = lambda p: int(getattr(p, "geant4_id", p.id))
    out = []
    for p in particles:
        if (p.interaction_type != InteractionType.kDelta
                or p.sem_type != SemanticType.kLEScatter
                or not len(p.point_cloud)):
            continue
        par = by_id.get(int(p.parent_id)) if p.parent_id is not None else None
        if par is None or par is p or g4(par) == g4(p):
            continue
        f = frag_of.get(int(p.id))
        if (f is None or f == int(p.id) or frag_of.get(int(par.id)) != f
                or sem_of[f] not in track_like):
            continue
        out.append(p)
    return out


def relabel_absorbed_deltas(deltas):
    """
    Type each delta ray of :func:`find_absorbed_deltas` ``kDelta``, in place,
    and mark it ``buried_in_track``.  Call it after step 2, before the
    layout: the fragment decides LE, so the points stay the track's, and
    the mark keeps its steps out of the track instance's first/last points,
    as its LE type did.
    """
    for p in deltas:
        p.sem_type = SemanticType.kDelta
        p.buried_in_track = True


def relabel_split_pieces(pairs):
    """
    Give each piece fragment its track fragment's semantic type, in place.

    Run before the fragments are snapshotted: LE is decided per fragment
    (``frag_sem_type == kLEScatter``), and a piece of a muon is part of the
    muon, not a low-energy blob.  The piece particle's own ``sem_type`` is
    untouched.
    """
    for f, home in pairs:
        f.sem_type = home.sem_type


def attach_split_pieces(instances, pairs):
    """
    Merge each piece's instance into the instance of its track fragment.

    Only a piece that heads an instance of its own is moved.  A ``kDelta``
    or ``kMichel`` piece usually arrives already, since
    :func:`merge_em_showers` walks it up to its electron.

    Returns
    -------
    list of Particle
        *instances* without the ones merged away.
    """
    inst_of = {}
    for inst in instances:
        for m in (inst.member_ids if inst.member_ids is not None else [inst.id]):
            inst_of[int(m)] = inst
    gone = set()
    for f, home in pairs:
        a, b = inst_of.get(int(f.id)), inst_of.get(int(home.id))
        if a is None or b is None or a is b or a.id != f.id:
            continue
        a_ids = list(a.member_ids if a.member_ids is not None else [a.id])
        b.member_ids = list(b.member_ids if b.member_ids is not None
                            else [b.id]) + a_ids
        clouds = [c for c in (b.point_cloud, a.point_cloud) if len(c)]
        if len(clouds) > 1:
            b.point_cloud = np.concatenate(clouds, axis=0)
        elif clouds:
            b.point_cloud = clouds[0]
        for m in a_ids:
            inst_of[int(m)] = b
        gone.add(id(a))
    return [i for i in instances if id(i) not in gone]
