from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING

from .base import PartitionConditionBase
from ..diagnostics import MergeOutcome
from ..utils import InteractionType, SemanticType

if TYPE_CHECKING:
    from ..partitioner import ParticlePartitioner
    from ..data import Particle

_PHOTON_PDG = 22
_DECAY_PRODUCTS = {11, -11}


def _is_own_fragment(child, parent) -> bool:
    """
    True when *child* is a defragmentation split-off of *parent*.

    Such a piece carries the parent's PDG (22) and, decisively, the parent's
    ``geant4_id``: it *is* the same Geant4 particle, cut in two because its
    deposits were too far apart to cluster.  Matching on ``geant4_id`` rather
    than PDG alone is what keeps genuine photon-from-photon secondaries --
    fluorescence X-rays, which are distinct particles -- out of this path.
    """
    if parent is None or int(child.pdg) != _PHOTON_PDG:
        return False
    cg = getattr(child, "geant4_id", None)
    pg = getattr(parent, "geant4_id", None)
    return cg is not None and pg is not None and int(cg) == int(pg)


def _is_pair_product(p) -> bool:
    """
    True when *p* is an electron or positron a photon converted into.

    Read from the process that created it (``kConversion``), not from
    ``pdg`` alone: a photon's Compton electrons and its photoelectron are
    also PDG 11 children of a photon, but they are not the shower.
    """
    return (int(p.pdg) in _DECAY_PRODUCTS
            and int(getattr(p, "_interaction_type", -1))
            == InteractionType.kConversion.value)


class PhotonDecay(PartitionConditionBase):
    """
    Merge the pair a photon (PDG 22) converted into into the photon partition.

    A photon may convert into an electron (PDG 11) and a positron (PDG -11).
    The photon itself carries **no** deposits of its own at the conversion;
    its spatial extent from there on is the clouds of its pair.

    This condition merges the ``kConversion`` children of a photon -- and
    only those -- into the photon's partition so that:

    * The photon is the **surviving representative** of the merged partition.
    * The photon's ``point_cloud`` grows to include every point from every
      absorbed child (and their own already-merged partitions).

    Because the photon has no spatial hits of its own, proximity checks are
    meaningless here.  The merge is driven purely by the parent-child
    relationship in the particle tree.

    A photon's Compton electrons and its photoelectron are left alone.  A
    shower begins at the pair production; a Compton electron is a separate
    particle, made where the photon scattered, before any conversion.  It is
    usually a small isolated blob, and merging it here made it part of a
    shower *fragment*, so it lost its LE label.  Whether it belongs to a
    shower *instance* is decided by genealogy in
    :func:`~pysupera.merge.merge_em_showers`.

    A photon that *does* carry hits -- Geant4 deposits argon atomic-relaxation
    energy at a photoabsorption vertex, so most photons carry one such point --
    may be cut into pieces by defragmentation.  Those pieces are PDG 22, not
    ±11, so they are matched on ``geant4_id`` instead and re-attached to the
    photon they came from.  This condition therefore works
    **only** in the partition-level proximity path
    (``partition_level_proximity=True``) via
    :meth:`get_unconditional_merges`; the legacy particle-level path
    returns an empty candidate list.

    Pipeline
    --------
    1. :meth:`get_unconditional_merges` is called on every convergence pass.
    2. For each photon representative, iterate its direct children in
       ``partitioner.children_map``.
    3. For each qualifying child (a ``kConversion`` e±, or a split-off
       piece of the photon itself), if it belongs to a different
       partition, emit ``(child_rep_id, photon_rep_id)`` for a no-proximity
       merge.
    4. :meth:`get_rep_candidates` and :meth:`get_candidates` both return
       empty lists (no proximity-based merges).

    Notes
    -----
    In a single event there may be zero, one, or many photons undergoing
    conversion.  The condition handles all of them in one pass.

    The condition does not restrict the photon to having *exactly* two
    conversion children.
    """

    @property
    def name(self) -> str:
        return "PhotonDecay"

    @property
    def description(self) -> str:
        return "Photon (PDG 22) → e⁺e⁻ conversion: merge the pair into the photon partition"

    # ------------------------------------------------------------------
    # Unconditional (topology-only) merge — the main entry point
    # ------------------------------------------------------------------

    def get_unconditional_merges(
            self,
            partitioner: 'ParticlePartitioner',
            rep_lookup: Dict[int, 'Particle']) -> List[Tuple[int, int]]:
        """
        Return directed pairs ``(child_rep_id, photon_rep_id)`` for the
        pair (``kConversion`` e±) of every photon representative, plus any
        PDG 22 child
        that is a defragmentation split-off of the photon itself (see
        :func:`_is_own_fragment`).

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

        # Group the conversion pairs present in the particle list by their
        # photon parent_id.  Used both to produce merge pairs and to record
        # sibling-level diagnostics so that get_decisions_for_pair(e-, e+)
        # returns a PhotonDecay entry explaining what happened.
        photon_children: Dict[int, List['Particle']] = defaultdict(list)
        for p in partitioner.particles:
            if p.parent_pdg != _PHOTON_PDG:
                continue
            if _is_pair_product(p):
                photon_children[p.parent_id].append(p)
            elif _is_own_fragment(p, partitioner.particle_lookup.get(p.parent_id)):
                # A piece of the photon itself.  Without this it could only
                # rejoin by drifting within distance_threshold of a sibling
                # electron -- which is a coin flip: in one measured case two
                # such pieces sat 0.01 mm and 5.30 mm from their nearest
                # sibling, against a 5.2 mm threshold.
                photon_children[p.parent_id].append(p)

        _SIBLING_STAGE = f"{self.name}: Sibling Grouping"

        for photon_id, children in photon_children.items():
            photon_in_list = photon_id in partitioner.particle_lookup

            if photon_in_list:
                # Photon IS present — emit child→photon merge pairs.
                photon_rep = rep_lookup.get(photon_id)
                if photon_rep is None:
                    continue
                # Promote the photon rep to kShower immediately, even if it
                # currently carries kLEScatter (which happens when the photon's
                # own point cloud is below the min_pc_size threshold so it was
                # initially assigned kLEScatter).  After merging its e+/e-
                # children in, the photon will have a non-zero point cloud and
                # correctly represents a shower.  Promoting BEFORE returning
                # the pairs ensures that CombineLEScatters and AbsorbLEScatter
                # (which run later in the same convergence pass) see the photon
                # as kShower and leave it alone, rather than absorbing it.
                # A photon with only pieces of itself to collect did not
                # convert, so there is no shower to promote it to.
                if (photon_rep.sem_type == SemanticType.kLEScatter
                        and any(_is_pair_product(c) for c in children)):
                    photon_rep.sem_type = SemanticType.kShower
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
