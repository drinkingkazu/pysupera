from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING

from .base import PartitionConditionBase
from ..diagnostics import MergeOutcome

if TYPE_CHECKING:
    from ..partitioner import ParticlePartitioner
    from ..data import Particle

_PHOTON_PDG = 22
_DECAY_PRODUCTS = {11, -11}   # electron and positron


class PhotonDecay(PartitionConditionBase):
    """
    Merge the charged decay products of a photon (PDG 22) into the photon partition.

    A photon may decay into an electron (PDG 11) and a positron (PDG -11).
    The photon itself typically carries an **empty** point cloud; its spatial
    extent is represented entirely by the clouds of its decay products.

    This condition recognises such photon-decay topologies and merges all
    PDG ±11 children of a photon particle into the photon's partition so
    that:

    * The photon is the **surviving representative** of the merged partition.
    * The photon's ``point_cloud`` grows to include every point from every
      absorbed child (and their own already-merged partitions).

    Because the photon has no spatial hits of its own, proximity checks are
    meaningless here.  The merge is driven purely by the parent-child
    relationship in the particle tree.  This condition therefore works
    **only** in the partition-level proximity path
    (``partition_level_proximity=True``) via
    :meth:`get_unconditional_merges`; the legacy particle-level path
    returns an empty candidate list.

    Pipeline
    --------
    1. :meth:`get_unconditional_merges` is called on every convergence pass.
    2. For each photon representative, iterate its direct children in
       ``partitioner.children_map``.
    3. For each qualifying child (PDG ±11), if it belongs to a different
       partition, emit ``(child_rep_id, photon_rep_id)`` for a no-proximity
       merge.
    4. :meth:`get_rep_candidates` and :meth:`get_candidates` both return
       empty lists (no proximity-based merges).

    Notes
    -----
    In a single event there may be zero, one, or many photons undergoing
    conversion.  The condition handles all of them in one pass.

    The condition does not restrict the photon to having *exactly* two
    children, so it also handles radiative corrections and other
    configurations that produce more than two PDG ±11 children from a
    single photon.
    """

    @property
    def name(self) -> str:
        return "PhotonDecay"

    @property
    def description(self) -> str:
        return "Photon (PDG 22) → e⁺e⁻ decay: merge charged children into photon partition"

    # ------------------------------------------------------------------
    # Unconditional (topology-only) merge — the main entry point
    # ------------------------------------------------------------------

    def get_unconditional_merges(
            self,
            partitioner: 'ParticlePartitioner',
            rep_lookup: Dict[int, 'Particle']) -> List[Tuple[int, int]]:
        """
        Return directed pairs ``(child_rep_id, photon_rep_id)`` for all
        PDG ±11 children of photon representatives.

        Called on every convergence pass with the live *rep_lookup*, so
        chains of merges resolve correctly across passes.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``children_map`` and
            ``particle_lookup``.
        rep_lookup : dict[int, Particle]
            Maps original particle ID → current representative Particle.

        Returns
        -------
        list of tuple[int, int]
            ``(child_rep_id, photon_rep_id)`` — child merges INTO photon.
            Empty when no qualifying photon partitions are found.
        """
        pairs: List[Tuple[int, int]] = []

        # Group PDG±11 particles present in the particle list by their
        # photon parent_id.  Used both to produce merge pairs and to record
        # sibling-level diagnostics so that get_decisions_for_pair(e-, e+)
        # returns a PhotonDecay entry explaining what happened.
        photon_children: Dict[int, List['Particle']] = defaultdict(list)
        for p in partitioner.particles:
            if p.pdg in _DECAY_PRODUCTS and p.parent_pdg == _PHOTON_PDG:
                photon_children[p.parent_id].append(p)

        _SIBLING_STAGE = f"{self.name}: Sibling Grouping"

        for photon_id, children in photon_children.items():
            photon_in_list = photon_id in partitioner.particle_lookup

            if photon_in_list:
                # Photon IS present — emit child→photon merge pairs.
                photon_rep = rep_lookup.get(photon_id)
                if photon_rep is None:
                    continue
                for child in children:
                    child_rep = rep_lookup.get(child.id)
                    if child_rep is None or child_rep.id == photon_rep.id:
                        continue
                    pairs.append((child_rep.id, photon_rep.id))

                # Record a sibling-pair entry so the user can see why
                # (e-, e+) shows up in the same photon partition.
                if partitioner.diagnostics.enabled:
                    sibling_reps: Dict[int, 'Particle'] = {}
                    for c in children:
                        rep = rep_lookup.get(c.id)
                        if rep is not None:
                            sibling_reps[rep.id] = rep
                    sib_list = list(sibling_reps.values())
                    for i, r1 in enumerate(sib_list):
                        for r2 in sib_list[i + 1:]:
                            a, b = min(r1.id, r2.id), max(r1.id, r2.id)
                            partitioner.diagnostics.record(
                                a, b,
                                MergeOutcome.MERGED,
                                f"PhotonDecay: both siblings will be merged into "
                                f"photon representative (id={photon_rep.id}) via "
                                f"individual child→photon unconditional merges",
                                stage=_SIBLING_STAGE,
                            )
            else:
                # Photon is ABSENT from the particle list — no merge possible.
                # Record a rejection for each sibling pair so the user knows
                # PhotonDecay ran and why it could not act.
                if partitioner.diagnostics.enabled:
                    sibling_reps: Dict[int, 'Particle'] = {}
                    for c in children:
                        rep = rep_lookup.get(c.id)
                        if rep is not None:
                            sibling_reps[rep.id] = rep
                    sib_list = list(sibling_reps.values())
                    for i, r1 in enumerate(sib_list):
                        for r2 in sib_list[i + 1:]:
                            a, b = min(r1.id, r2.id), max(r1.id, r2.id)
                            partitioner.diagnostics.record(
                                a, b,
                                MergeOutcome.NOT_IN_CANDIDATE_SET,
                                f"PhotonDecay: photon parent (id={photon_id}) is "
                                f"absent from the particle list — siblings cannot "
                                f"be merged without a photon representative",
                                stage=_SIBLING_STAGE,
                            )

        return pairs

    # ------------------------------------------------------------------
    # Proximity-based methods — not used for this condition
    # ------------------------------------------------------------------

    def get_candidates(
            self,
            partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Return an empty list.

        :class:`PhotonDecay` merges via :meth:`get_unconditional_merges`
        and does not participate in the legacy particle-level proximity path.

        Parameters
        ----------
        partitioner : ParticlePartitioner

        Returns
        -------
        list of tuple[int, int]
            Always empty.
        """
        return []

    def get_rep_candidates(
            self,
            partitioner: 'ParticlePartitioner',
            rep_lookup: Dict[int, 'Particle']) -> List[Tuple[int, int]]:
        """
        Return an empty list.

        All merges for this condition are unconditional (see
        :meth:`get_unconditional_merges`).

        Parameters
        ----------
        partitioner : ParticlePartitioner
        rep_lookup : dict[int, Particle]

        Returns
        -------
        list of tuple[int, int]
            Always empty.
        """
        return []
