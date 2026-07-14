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

from .utils import SemanticType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: PDG codes considered "EM" for shower-initiator chain walking.
_EM_PDGS: frozenset = frozenset({11, -11, 22})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_shower_eligible(rep) -> bool:
    """Return ``True`` if *rep* should participate in EM shower genealogy merging.

    A fragment is shower-eligible when it has an EM PDG code (e±, γ) **or**
    carries the ``kShower`` semantic type, **and** is not a ``kLEScatter``
    fragment.

    ``kLEScatter`` fragments are explicitly excluded because they are handled
    by step-1 (``AbsorbLEScatter``) and must never become shower initiators —
    if they did, non-LE fragments merged into them would be silently dropped by
    the ``drop_le_scatter`` filter downstream.
    """
    if rep.sem_type == SemanticType.kLEScatter:
        return False
    return (abs(rep.pdg) in _EM_PDGS
            or rep.sem_type == SemanticType.kShower)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_em_showers(fragments):
    """
    Step-2 EM shower genealogy merger.

    Groups step-1 fragment representatives into *shower instances* by
    walking the parent-ID chain of each EM fragment until a non-EM ancestor
    (or the top of the fragment list) is reached.  That top-most EM fragment
    is the **shower initiator**; all EM fragments that share the same
    initiator are merged into it.  Non-EM fragments pass through unchanged
    (each becomes its own instance).

    The merge is **purely genealogical** — no spatial proximity check is
    performed.  Photons (PDG 22) always have empty point clouds and are
    natural shower initiators when no EM parent is found.

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

    def find_initiator(rep):
        """Iteratively walk the parent chain; return the shower-initiator fragment."""
        visited: set = set()
        current = rep
        while True:
            if id(current) in visited:
                # Cycle guard (should never occur with well-formed genealogy)
                return current
            visited.add(id(current))

            # Non-shower-eligible particles are their own initiator.
            if not _is_shower_eligible(current):
                return current

            # Walk to parent fragment — try rep.id first, then any member id.
            parent_pid = current.parent_id
            parent = (
                frag_by_id.get(parent_pid)
                or frag_by_member_id.get(int(parent_pid) if parent_pid is not None else -1)
            )
            if parent is None or parent is current:
                # No shower-eligible parent in fragment list → current is the initiator.
                return current
            if not _is_shower_eligible(parent):
                # Parent is non-shower → current is the shower initiator.
                return current
            # Parent is also shower-eligible → keep walking up.
            current = parent

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
