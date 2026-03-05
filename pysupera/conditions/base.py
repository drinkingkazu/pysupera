from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..partitioner import ParticlePartitioner
    from ..data import Particle


class PartitionConditionBase(ABC):
    """
    Abstract base class for particle-partitioning conditions.

    A *condition* encapsulates the domain-specific steps of the
    partitioning pipeline:

    1. **Candidate building** – ``get_candidates`` returns the subset of
       particle pairs that are worth checking for spatial proximity.  A
       well-written condition keeps this set much smaller than the full
       O(n²) pair space by exploiting particle metadata (PDG, ancestry,
       semantic type, etc.).

    2. **Post-filtering** – ``post_filter`` applies any additional
       selection logic *after* the proximity check.  The default
       implementation is a no-op (all touching pairs are accepted).

    3. **Unconditional merges** – ``get_unconditional_merges`` (optional)
       returns directed pairs that should be merged **without** a proximity
       check.  Use this for topological relationships where geometry is
       irrelevant (e.g. photon → e⁺e⁻ decay where the photon has an empty
       point cloud).  The default implementation returns an empty list.

    The base class deliberately holds no state so that condition objects
    can be reused across multiple ``ParticlePartitioner.partition`` calls.

    Subclasses must implement ``get_candidates`` and may override
    ``post_filter``, ``get_rep_candidates``, ``get_unconditional_merges``,
    ``name``, and ``description``.
    """

    @property
    def name(self) -> str:
        """
        Short identifier for this condition used in log output.

        Returns
        -------
        str
            Human-readable name.  Defaults to the class name.
        """
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """
        One-line description of the condition used in verbose headers.

        Returns
        -------
        str
            Descriptive string.  Defaults to an empty string.
        """
        return ""

    @abstractmethod
    def get_candidates(self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Build the set of candidate pairs to test for spatial proximity.

        This method should exploit particle metadata to keep the candidate
        list as small as possible.  The proximity check in
        ``ParticlePartitioner`` is called on the returned list, so every
        pair here incurs a geometric distance evaluation.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            The active partitioner instance.  Use ``partitioner.particles``,
            ``partitioner.particle_lookup``, ``partitioner.children_map``,
            ``partitioner.ancestor_map``, and ``partitioner.diagnostics``
            as needed.

        Returns
        -------
        list of tuple[int, int]
            Candidate pairs expressed as ``(id1, id2)`` using particle IDs
            (not positional indices in the particle list).
        """
        ...

    def post_filter(self,
                    partitioner: 'ParticlePartitioner',
                    touching_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Apply additional selection logic after the proximity check.

        Called by ``ParticlePartitioner.partition`` with the pairs that
        passed the proximity test.  The default implementation is a
        pass-through; override in subclasses that need to restrict which
        touching pairs are actually merged (e.g. one-merge-per-particle
        constraints).

        Parameters
        ----------
        partitioner : ParticlePartitioner
            The active partitioner instance, available for metadata lookups
            and diagnostic recording.
        touching_pairs : list of tuple[int, int]
            Pairs of particle IDs whose point clouds are within the distance
            threshold ``partitioner.D``.

        Returns
        -------
        list of tuple[int, int]
            The subset of *touching_pairs* that should actually be merged.
            Must be a subset (or equal to) *touching_pairs*.
        """
        return touching_pairs

    def get_rep_candidates(self,
                           partitioner: 'ParticlePartitioner',
                           rep_lookup: 'Dict[int, Particle]') -> List[Tuple[int, int]]:
        """
        Build directed proximity-candidate pairs at the partition level.

        Called during ``_build_partitions_incremental``.  Each pair
        ``(child_rep_id, parent_rep_id)`` will be tested for spatial
        proximity; if they are close enough the child partition is merged
        into the parent partition.

        The default implementation returns an empty list so that
        conditions that only provide unconditional merges do not need to
        override this method.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            The active partitioner instance.
        rep_lookup : dict[int, Particle]
            Maps each original particle ID to its current representative
            particle.

        Returns
        -------
        list of tuple[int, int]
            Directed pairs ``(child_rep_id, parent_rep_id)`` to test.
        """
        return []

    @staticmethod
    def unique_reps(rep_lookup: 'Dict[int, Particle]') -> List['Particle']:
        """
        Return the current set of unique partition representatives.

        The canonical helper for use inside :meth:`get_rep_candidates`.
        Deduplicates ``rep_lookup.values()`` by representative id, which
        naturally excludes particles that have already been absorbed into
        another partition: an absorbed particle's entry in *rep_lookup*
        points to its absorber, which then appears only once after
        deduplication.

        Subclasses should call this instead of iterating
        ``partitioner.particles`` directly, which would yield original
        (pre-merge) particles and require manual guards on the
        representative's attributes.

        Parameters
        ----------
        rep_lookup : dict[int, Particle]
            Maps each original particle ID to its current representative.

        Returns
        -------
        list of Particle
            One entry per live partition, the surviving representative.
        """
        return list({r.id: r for r in rep_lookup.values()}.values())

    def get_unconditional_merges(self,
                                 partitioner: 'ParticlePartitioner',
                                 rep_lookup: 'Dict[int, Particle]') -> List[Tuple[int, int]]:
        """
        Return directed merge pairs that bypass the proximity check.

        Use this for topological relationships where geometry is
        irrelevant (e.g. photon → e⁺e⁻ decay products when the photon
        has an empty point cloud).

        The default implementation returns an empty list.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            The active partitioner instance.
        rep_lookup : dict[int, Particle]
            Maps each original particle ID to its current representative
            particle.

        Returns
        -------
        list of tuple[int, int]
            Directed pairs ``(child_rep_id, parent_rep_id)`` to merge
            unconditionally.
        """
        return []
