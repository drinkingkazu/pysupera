from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from .base import PartitionConditionBase
from ..diagnostics import MergeOutcome
from ..utils import SemanticType

if TYPE_CHECKING:
    from ..partitioner import ParticlePartitioner
    from ..data import Particle


class AbsorbLEScatter(PartitionConditionBase):
    """
    Absorb each ``kLEScatter`` particle into at most one touching non-``kLEScatter`` neighbour.

    This condition targets low-energy scatter products
    (``sem_type == SemanticType.kLEScatter``) that are spatially adjacent to
    any other semantic class.  Each ``kLEScatter`` particle is allowed to
    merge with at most one partner to avoid over-clustering.

    Run :class:`CombineLEScatters` **before** this condition so that
    transitively connected LEScatter chains are consolidated into one particle
    prior to absorption.

    Pipeline
    --------
    1. ``get_candidates`` builds all pairs between ``SemanticType.kLEScatter``
       particles and particles with a different semantic type –
       O(n_kLEScatter × n_other).
    2. The base partitioner tests proximity on those candidates.
    3. ``post_filter`` enforces the one-merge-per-kLEScatter constraint on
       the touching pairs returned by the proximity check.
    """

    @property
    def name(self) -> str:
        return "AbsorbLEScatter"

    @property
    def description(self) -> str:
        return "Absorb kLEScatter particles into touching non-kLEScatter neighbours"

    def get_candidates(self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Build all pairs between ``kLEScatter`` particles and non-``kLEScatter`` particles.

        Partitions the full particle list into two sub-lists in O(n), then
        returns their Cartesian product in O(n_kLEScatter × n_other), avoiding
        the full O(n²) enumeration.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particles`` and ``diagnostics``.

        Returns
        -------
        list of tuple[int, int]
            Candidate pairs expressed as ``(kLEScatter_id, other_id)`` using
            particle IDs (not positional indices).
        """
        candidates = self._build_candidates(partitioner)
        partitioner.diagnostics.record_candidates(
            f"{self.name}: Stage 1 Candidate Building", candidates
        )
        return candidates

    def post_filter(self,
                    partitioner: 'ParticlePartitioner',
                    touching_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Allow each ``kLEScatter`` particle to merge with at most one partner.

        Groups *touching_pairs* by their ``kLEScatter`` member and keeps one
        touching non-``kLEScatter`` neighbour for each group: the partition
        of its own ancestor when that one touches (see
        :meth:`_home_rep_id`), otherwise the first.
        Surplus candidates are discarded; when diagnostics are enabled each
        rejection is recorded with reason ``ALREADY_MERGED_WITH_OTHER``.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particle_lookup`` and
            ``diagnostics``.
        touching_pairs : list of tuple[int, int]
            Pairs that passed the proximity check, as produced by
            ``ParticlePartitioner._batch_check_proximity_optimized``.

        Returns
        -------
        list of tuple[int, int]
            At most one pair per ``kLEScatter`` particle.
        """
        return self._select_one_merge_per_sem5(partitioner, touching_pairs)

    def get_rep_candidates(
            self,
            partitioner: 'ParticlePartitioner',
            rep_lookup: Dict[int, 'Particle']) -> List[Tuple[int, int]]:
        """
        Return directed pairs pairing each ``kLEScatter`` representative with
        every non-``kLEScatter`` representative.

        Pair ordering: ``(le_scatter_rep_id, other_rep_id)`` — the
        ``kLEScatter`` partition merges INTO the non-``kLEScatter`` partition,
        whose representative survives.

        Parameters
        ----------
        partitioner : ParticlePartitioner
        rep_lookup : dict[int, Particle]
            Maps original particle ID → current representative Particle.

        Returns
        -------
        list of tuple[int, int]
            ``(le_scatter_rep_id, other_rep_id)``.
        """
        # Deduplicate representatives
        unique_reps = self.unique_reps(rep_lookup)
        le_reps    = [r for r in unique_reps if r.sem_type == SemanticType.kLEScatter]
        other_reps = [r for r in unique_reps if r.sem_type != SemanticType.kLEScatter]

        if not le_reps or not other_reps:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        D = partitioner.checker.D

        # Global point-cloud KDTree: build separate LE and other point arrays
        # labelled by rep index, then call query_ball_tree(D) once to find all
        # (LE_point, other_point) pairs within D in O(N log N).
        # LE scatter clouds are always small (few hits/voxels), so the output
        # size is bounded by n_le_pts × local_density near touching boundaries.
        def _stack(reps):
            clouds   = []
            rep_tags = []
            for k, rep in enumerate(reps):
                parts = rep._cloud_parts
                c = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
                if len(c) == 0:
                    continue
                clouds.append(c[:, :3].astype(np.float32))
                rep_tags.append(np.full(len(c), k, dtype=np.int32))
            if not clouds:
                return None, None
            return np.concatenate(rep_tags), np.concatenate(clouds)

        le_tags, le_pts = _stack(le_reps)
        ot_tags, ot_pts = _stack(other_reps)

        if le_tags is None or ot_tags is None:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        from scipy.spatial import cKDTree
        # Every (LE point, other point) pair within D, as two index arrays.
        m = cKDTree(le_pts).sparse_distance_matrix(
            cKDTree(ot_pts), D, output_type="ndarray")
        pi = m["i"].astype(np.int64)
        pj = m["j"].astype(np.int64)

        # Map to rep pairs and deduplicate.  The order matters: when a
        # kLEScatter touches several reps and none is its home,
        # _select_one_merge_per_sem5 takes the first pair, i.e. the first in
        # this set's iteration order, which depends on the order the pairs
        # went in.  So they go in as query_ball_tree's sorted lists put them:
        # by LE point, then other point, each rep pair at its first
        # occurrence.
        n_ot = len(other_reps)
        rk = le_tags[pi].astype(np.int64) * n_ot + ot_tags[pj]
        rk = rk[np.argsort(pi * len(ot_pts) + pj)]
        _, first = np.unique(rk, return_index=True)
        touching: set = set()
        for k in rk[np.sort(first)].tolist():
            touching.add((k // n_ot, k % n_ot))

        candidates = [
            (le_reps[i].id, other_reps[j].id)
            for i, j in touching
        ]
        partitioner.diagnostics.record_candidates(
            f"{self.name}: Rep Candidate Building", candidates
        )
        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_candidates(self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Form the Cartesian product of ``kLEScatter`` and non-``kLEScatter`` sub-populations.

        Makes two O(n) passes over ``partitioner.particles`` to collect the
        two sub-lists, then iterates their product in
        O(n_kLEScatter × n_other).

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particles``.

        Returns
        -------
        list of tuple[int, int]
            All ``(kLEScatter_particle_id, other_particle_id)`` combinations.
        """
        sem5_particles  = [p for p in partitioner.particles if p.sem_type == SemanticType.kLEScatter]
        other_particles = [p for p in partitioner.particles if p.sem_type != SemanticType.kLEScatter]

        return [
            (sem5_p.id, other_p.id)
            for sem5_p in sem5_particles
            for other_p in other_particles
        ]

    def _select_one_merge_per_sem5(self,
                                   partitioner: 'ParticlePartitioner',
                                   touching_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Keep at most one touching partner per ``kLEScatter`` particle.

        Groups pairs by their ``kLEScatter`` member and selects, in each
        group, the pair whose partner is the particle's genealogical home
        (:meth:`_home_rep_id`) when it is among them, else the first pair.
        Records diagnostic rejections for the remainder.

        The preference matters for delta rays: one is born on its track and
        so touches it, but it often touches something else as well -- a
        larger delta, a shower crossing the track -- and the first pair is
        whichever the proximity search happened to list first.  Choosing
        among the partners it touches pulls nothing in from a distance.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particle_lookup`` and
            ``diagnostics``.
        touching_pairs : list of tuple[int, int]
            All pairs that passed the proximity check.

        Returns
        -------
        list of tuple[int, int]
            Filtered list with at most one entry per ``kLEScatter`` particle.
        """
        sem5_merges: dict = defaultdict(list)

        for id1, id2 in touching_pairs:
            p1 = partitioner.particle_lookup[id1]
            p2 = partitioner.particle_lookup[id2]

            if p1.sem_type == SemanticType.kLEScatter:
                sem5_merges[p1.id].append((id1, id2))
            elif p2.sem_type == SemanticType.kLEScatter:
                sem5_merges[id2].append((id1, id2))

        selected = []

        for sem5_id, pairs in sem5_merges.items():
            selected_pair = pairs[0]
            if len(pairs) > 1:
                home = self._home_rep_id(partitioner, sem5_id)
                for pair in pairs:
                    partner = pair[0] if pair[0] != sem5_id else pair[1]
                    if partner == home:
                        selected_pair = pair
                        break
            selected.append(selected_pair)

            if partitioner.diagnostics.enabled and len(pairs) > 1:
                selected_partner_id = (
                    selected_pair[0] if selected_pair[0] != sem5_id
                    else selected_pair[1]
                )
                for pair in pairs:
                    if pair == selected_pair:
                        continue
                    rejected_partner_id = pair[0] if pair[0] != sem5_id else pair[1]
                    partitioner.diagnostics.record(
                        pair[0], pair[1],
                        MergeOutcome.ALREADY_MERGED_WITH_OTHER,
                        f"SemID-5 particle {sem5_id} already merged with "
                        f"particle {selected_partner_id} "
                        f"(rejecting {rejected_partner_id})",
                        stage=f"{self.name}: Stage 3 Selection",
                    )

        return selected

    @staticmethod
    def _home_rep_id(partitioner: 'ParticlePartitioner', le_id: int):
        """
        The partition the ``kLEScatter`` partition *le_id* came from.

        Walks the parent chain of the particle *le_id* and returns the
        representative id of the first ancestor that sits in a different
        partition -- a delta ray's track, say.  Ancestors already merged
        into the same LE partition are skipped.  None when the chain ends
        first.
        """
        lookup = partitioner.particle_lookup
        reps = getattr(partitioner, "rep_lookup", None) or {}

        def rep_of(pid):
            r = reps.get(pid)
            return r.id if r is not None else pid

        own = rep_of(le_id)
        cur, seen = lookup.get(le_id), set()
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            par = cur.parent_id
            if par is None or par == cur.id or par not in lookup:
                return None
            home = rep_of(par)
            if home != own:
                return home
            cur = lookup[par]
        return None
