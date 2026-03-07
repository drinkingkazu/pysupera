from typing import Dict, List, Tuple, TYPE_CHECKING

from .base import PartitionConditionBase
from ..diagnostics import MergeOutcome

if TYPE_CHECKING:
    from ..partitioner import ParticlePartitioner
    from ..data import Particle


class TouchingEMShower(PartitionConditionBase):
    """
    Merge PDG-22/11 particles that are in a direct parent-child relationship
    and share a common ancestor, provided their point clouds touch.

    This condition targets electromagnetic shower fragments: a photon (PDG 22)
    or electron (PDG 11) that is the direct child of another photon or
    electron with the same PDG code.  The additional ancestor filter ensures
    that only fragments from the same original shower are merged.

    Pipeline
    --------
    1. ``get_candidates`` builds parent-child pairs where both members have
       PDG code 11 (electron) or 22 (photon) – O(n).
    2. ``get_candidates`` then filters those pairs to retain only pairs
       that share the same ``root_id`` – O(candidates).
    3. The base partitioner tests proximity on the remaining candidates.
    """

    @property
    def name(self) -> str:
        return "TouchingEMShower"

    @property
    def description(self) -> str:
        return "PDG 22/11 parent-child EM-shower merging"

    def get_candidates(self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Return parent-child PDG-11/22 pairs that share an ancestor.

        Iterates the full particle list once to collect direct parent-child
        edges (O(n)), then discards pairs with different ``root_id``
        values (O(candidates)).  Both filters run inside this method so
        that the list handed to the proximity checker is as small as
        possible.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particles``, ``particle_lookup``,
            and ``diagnostics``.

        Returns
        -------
        list of tuple[int, int]
            Pairs expressed as ``(parent_id, child_id)`` using particle IDs.
        """
        candidates = self._build_candidates(partitioner)
        partitioner.diagnostics.record_candidates(
            f"{self.name}: Stage 1 Candidate Building", candidates
        )
        candidates = self._filter_by_ancestor(partitioner, candidates)
        partitioner.diagnostics.record_candidates(
            f"{self.name}: Stage 2 Ancestor Filter", candidates
        )
        return candidates

    def get_rep_candidates(
            self,
            partitioner: 'ParticlePartitioner',
            rep_lookup: Dict[int, 'Particle']) -> List[Tuple[int, int]]:
        """
        Return directed parent-child pairs from partition representatives.

        Iterates the current unique partition representatives and emits
        ``(child_rep_id, parent_rep_id)`` whenever:

        * the child representative has PDG code 11 or 22,
        * the original parent of that child exists in *rep_lookup*
          (i.e. the parent particle belongs to some live partition),
        * the parent representative also has PDG code 11 or 22, and
        * child and parent representatives share the same ``root_id``.

        Because *rep_lookup* is updated after every merge, this method
        correctly handles chains: if A merged into B, the representative for
        A's former children now resolves to B.

        Parameters
        ----------
        partitioner : ParticlePartitioner
        rep_lookup : dict[int, Particle]
            Maps original particle ID → current representative Particle.

        Returns
        -------
        list of tuple[int, int]
            ``(child_rep_id, parent_rep_id)`` — child merges INTO parent.
        """
        target_pdgs = {11, 22}
        candidates = []

        for rep in self.unique_reps(rep_lookup):
            if rep.pdg not in target_pdgs:
                continue
            if rep.parent_id not in rep_lookup:
                # Original parent not in any live partition
                continue
            parent_rep = rep_lookup[rep.parent_id]
            if parent_rep.id == rep.id:
                continue  # already same partition
            if parent_rep.pdg not in target_pdgs:
                continue
            if rep.root_id != parent_rep.root_id:
                continue
            candidates.append((rep.id, parent_rep.id))

        partitioner.diagnostics.record_candidates(
            f"{self.name}: Rep Candidate Building", candidates
        )
        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_candidates(self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Collect direct parent-child pairs where both members are PDG 11 or 22.

        Iterates every particle once (O(n)) and emits ``(parent_id, child_id)``
        only when the child's PDG is in ``{11, 22}`` and its parent exists in
        the lookup with the same PDG.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particles`` and ``particle_lookup``.

        Returns
        -------
        list of tuple[int, int]
            Raw candidate pairs before ancestor filtering.
        """
        candidates = []
        target_pdgs = {22, 11}

        for p in partitioner.particles:
            if p.pdg not in target_pdgs:
                continue
            if p.parent_id is None or p.parent_id not in partitioner.particle_lookup:
                continue

            parent = partitioner.particle_lookup[p.parent_id]
            if parent.pdg == p.pdg:
                candidates.append((parent.id, p.id))

        return candidates

    def _filter_by_ancestor(self,
                             partitioner: 'ParticlePartitioner',
                             candidates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Keep only pairs that share the same ``root_id``.

        Iterates *candidates* once (O(candidates)) and discards any pair
        whose two particles have different ``root_id`` values.  When
        diagnostics are enabled, each rejected pair is recorded with reason
        ``DIFFERENT_ANCESTOR``.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particle_lookup`` and
            ``diagnostics``.
        candidates : list of tuple[int, int]
            Pairs produced by ``_build_candidates``.

        Returns
        -------
        list of tuple[int, int]
            Subset of *candidates* whose members share the same ancestor.
        """
        filtered = []

        for id1, id2 in candidates:
            p1 = partitioner.particle_lookup[id1]
            p2 = partitioner.particle_lookup[id2]

            if p1.root_id == p2.root_id:
                filtered.append((id1, id2))
            elif partitioner.diagnostics.enabled:
                partitioner.diagnostics.record(
                    id1, id2,
                    MergeOutcome.DIFFERENT_ANCESTOR,
                    f"root_id: {p1.root_id} vs {p2.root_id}",
                    stage=f"{self.name}: Stage 2 Ancestor Filter",
                )

        return filtered
