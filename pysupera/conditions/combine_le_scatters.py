"""
Condition: CombineLEScatters
=============================
Collapse all ``kLEScatter`` particles that are transitively connected through
spatial proximity into a single consolidated ``kLEScatter`` particle before the
main ``AbsorbLEScatter`` stage.

Motivation
----------
``AbsorbLEScatter`` makes *pairwise* merge decisions: it absorbs a
``kLEScatter`` particle into the first touching non-``kLEScatter`` neighbour it
finds.  When a chain exists — e.g. Shower A touches LEScatter B which touches
LEScatter C, but A does not directly touch C — the transitively connected
particles B and C are not guaranteed to end up in the same partition.

Running this condition **before** ``AbsorbLEScatter`` resolves the chain by
first grouping B and C into one combined ``kLEScatter``, which then touches A
and is absorbed as a whole.

Algorithm
---------
Proximity graph is built over ``kLEScatter`` particles only.  Connected
components (found iteratively by the partition-level incremental engine)
define the merged groups.

Representative selection
------------------------
Within each LEScatter group the particle with the **largest point cloud** is
chosen as the representative (i.e. its scalar attributes — id, pdg, sem_type,
ancestry — survive).  Point clouds of all group members are concatenated onto
the representative.

Tie-breaking rule
-----------------
If two candidates have equal point-cloud sizes, the one with the higher
**energy sum** (column ``PointFeature.energy = 4``) is kept as representative.
If the column is absent the energy is treated as zero.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from .base import PartitionConditionBase
from ..utils import SemanticType, PointFeature

if TYPE_CHECKING:
    from ..partitioner import ParticlePartitioner
    from ..data import Particle


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _le_direction(a: 'Particle', b: 'Particle') -> Tuple['Particle', 'Particle']:
    """
    Return ``(child, parent)`` for a pair of ``kLEScatter`` particles.

    The *parent* is the one whose scalar attributes (id, sem_type, etc.)
    will survive the merge.  Selection priority:

    1. Larger point cloud is parent.
    2. On equal size: higher energy sum (``PointFeature.energy`` column) is parent.
    3. On equal energy: ``a`` is parent (stable, arbitrary).

    Parameters
    ----------
    a, b : Particle
        Two ``kLEScatter`` representative particles to compare.

    Returns
    -------
    tuple[Particle, Particle]
        ``(child, parent)`` — child's cloud is appended to parent's cloud
        and child's scalar attributes are discarded.
    """
    n_a = len(a.point_cloud)
    n_b = len(b.point_cloud)

    if n_a != n_b:
        # Larger point cloud becomes the parent
        return (b, a) if n_a > n_b else (a, b)

    # Tie-break: energy sum
    col = int(PointFeature.energy)

    def _esum(p: 'Particle') -> float:
        return float(p.point_cloud[:, col].sum()) if p.point_cloud.shape[1] > col else 0.0

    e_a = _esum(a)
    e_b = _esum(b)

    # Higher energy → parent; tie → a is parent (stable)
    if e_a >= e_b:
        return (b, a)   # a is parent
    return (a, b)       # b is parent


# ---------------------------------------------------------------------------
# Condition class
# ---------------------------------------------------------------------------

class CombineLEScatters(PartitionConditionBase):
    """
    Merge all mutually touching ``kLEScatter`` particles into groups before
    the ``AbsorbLEScatter`` absorption stage.

    This condition should be placed in the condition pipeline **before**
    ``AbsorbLEScatter`` so that transitive LEScatter chains are collapsed
    into a single particle prior to shower absorption.

    Only ``kLEScatter`` × ``kLEScatter`` pairs are considered; all other
    semantic types are ignored.

    Pipeline
    --------
    1. ``get_candidates`` returns all unordered ``kLEScatter`` pairs for the
       initial (particle-level) proximity pass.
    2. ``get_rep_candidates`` returns directed ``(child, parent)`` pairs
       among current ``kLEScatter`` representatives, re-evaluated on every
       convergence pass.  The parent is chosen by the largest-PC /
       highest-energy rule described in :func:`_le_direction`.
    3. No ``post_filter`` override — every touching LEScatter pair is merged.
    """

    @property
    def name(self) -> str:
        return "CombineLEScatters"

    @property
    def description(self) -> str:
        return (
            "Consolidate transitively touching kLEScatter particles into one "
            "before shower absorption"
        )

    # ------------------------------------------------------------------
    # PartitionConditionBase interface
    # ------------------------------------------------------------------

    def get_candidates(
            self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Return all unordered pairs of ``kLEScatter`` particles.

        Used only in the legacy particle-level proximity path.  In the
        default partition-level path, :meth:`get_rep_candidates` is called
        instead.

        Parameters
        ----------
        partitioner : ParticlePartitioner

        Returns
        -------
        list of tuple[int, int]
            All ``(le_id_a, le_id_b)`` combinations, ``a < b`` by list order.
        """
        le = [p for p in partitioner.particles
              if p.sem_type == SemanticType.kLEScatter]
        candidates = [
            (le[i].id, le[j].id)
            for i in range(len(le))
            for j in range(i + 1, len(le))
        ]
        partitioner.diagnostics.record_candidates(
            f"{self.name}: Stage 1 Candidate Building", candidates
        )
        return candidates

    def get_rep_candidates(
            self,
            partitioner: 'ParticlePartitioner',
            rep_lookup: Dict[int, 'Particle']) -> List[Tuple[int, int]]:
        """
        Return directed ``(child_rep_id, parent_rep_id)`` pairs among the
        current unique ``kLEScatter`` representatives.

        Called on every convergence pass so that newly consolidated
        representatives (with grown point clouds) are re-evaluated for
        further merges.

        Direction rule: the representative with the larger point cloud is
        the parent; ties are broken by energy sum (see :func:`_le_direction`).

        Parameters
        ----------
        partitioner : ParticlePartitioner
        rep_lookup : dict[int, Particle]
            Maps original particle ID → current representative Particle.

        Returns
        -------
        list of tuple[int, int]
            ``(child_rep_id, parent_rep_id)`` for all unique pairs.
        """
        # Collect unique kLEScatter representatives from the live partition
        # state.  Using unique_reps() rather than iterating partitioner.particles
        # ensures that already-absorbed particles are never re-inspected:
        # an absorbed particle's rep_lookup entry points to its absorber, which
        # is deduplicated away by unique_reps(), leaving only active reps.
        unique_le = [
            rep
            for rep in self.unique_reps(rep_lookup)
            if rep.sem_type == SemanticType.kLEScatter
        ]

        if len(unique_le) < 2:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        D = partitioner.checker.D

        # Global point-cloud KDTree: concatenate all LE rep clouds into one
        # array labelled by rep index, then call query_pairs(D) once.  This
        # finds all touching (rep_i, rep_j) pairs in O(N log N) — strictly
        # better than O(n_reps²) centroid pre-filter followed by per-pair
        # cloud checks, regardless of whether point clouds are voxelized.
        def _xyz(rep):
            parts = rep._cloud_parts
            return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

        clouds   = [_xyz(r) for r in unique_le]
        nonempty = [(k, c) for k, c in enumerate(clouds) if len(c) > 0]

        if len(nonempty) < 2:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        rep_tags = np.repeat(
            [k for k, _ in nonempty],
            [len(c) for _, c in nonempty],
        ).astype(np.int32)
        pts = np.concatenate([c[:, :3] for _, c in nonempty], axis=0).astype(np.float32)

        from scipy.spatial import KDTree
        pt_pairs = KDTree(pts).query_pairs(r=D, output_type='ndarray')

        # Map point-pairs → rep-pairs, drop same-rep and deduplicate.
        if len(pt_pairs) == 0:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        ri = rep_tags[pt_pairs[:, 0]]
        rj = rep_tags[pt_pairs[:, 1]]
        mask = ri != rj
        if not mask.any():
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        # Canonical (min, max) ordering → unique set of rep-pairs.
        rep_pairs = np.unique(
            np.sort(np.stack([ri[mask], rj[mask]], axis=1), axis=1), axis=0
        )

        candidates: List[Tuple[int, int]] = []
        for i, j in rep_pairs:
            child, parent = _le_direction(unique_le[i], unique_le[j])
            candidates.append((child.id, parent.id))

        partitioner.diagnostics.record_candidates(
            f"{self.name}: Rep Candidate Building", candidates
        )
        return candidates
