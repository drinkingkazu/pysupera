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

    # Group every fragment under its shower initiator.
    groups: dict = defaultdict(list)   # initiator.id → list[fragment rep]
    for rep in fragments:
        initiator = find_initiator(rep)
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
#: their pieces through PhotonDecay and genealogy already; a neutron's blobs
#: are separate interactions, not one trajectory.
_TRACK_LIKE = frozenset({SemanticType.kTrack, SemanticType.kMichel,
                         SemanticType.kDelta})


def find_split_pieces(fragments, particles):
    """
    Stand-alone fragments that are split-off pieces of a track-like particle.

    Defragmentation cuts a particle's clusters of ``min_pc_size`` voxels or
    fewer off into new ``kLEScatter`` particles that keep its ``geant4_id``
    (a stub past a gap, the last voxels beyond a dead stretch).  When such a
    piece touches nothing it stays a fragment of its own, though it is part
    of its particle's trajectory.  This finds those fragments: every member
    a piece of the same Geant4 particle, that particle ``kTrack``,
    ``kMichel`` or ``kDelta``, and its own fragment not LE.  A piece that
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
        if src.sem_type not in _TRACK_LIKE:
            continue
        home = frag_of.get(int(src.id))
        if home is None or home is f or home.sem_type == SemanticType.kLEScatter:
            continue
        out.append((f, home))
    return out


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
