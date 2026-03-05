# ============================================================================
# Diagnostic System for Tracking Partition Decisions
# ============================================================================

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict
import json

from .data import Particle


class MergeOutcome(Enum):
    """
    Enumeration of all possible outcomes for a particle-pair merge evaluation.

    Used by :class:`MergeDecision` and both diagnostic classes to tag every
    recorded outcome with a machine-readable, human-friendly label.

    Members
    -------
    NOT_IN_CANDIDATE_SET
        Pair was never proposed by the condition's ``get_candidates``
        method (failed a metadata pre-filter).
    DIFFERENT_PDG
        The two particles have different PDG codes.
    WRONG_PDG_VALUE
        One or both particle PDG codes are not in the condition's
        allowed set.
    DIFFERENT_ANCESTOR
        The two particles have different ``ancestor_id`` values.
    NO_PARENT_CHILD_RELATION
        Neither particle is the direct parent of the other.
    WRONG_SEMID
        One or both particles have a semantic type that does not match
        the condition's required value.
    CENTROID_TOO_FAR
        Particle centroids are farther apart than the proximity threshold
        (cheap pre-filter, may not be used by all conditions).
    BBOX_TOO_FAR
        Axis-aligned bounding boxes are separated by more than the
        proximity threshold.
    NOT_TOUCHING
        No point in cloud A lies within distance *D* of any point in
        cloud B (exact proximity check failed).
    ALREADY_MERGED_WITH_OTHER
        The pair was touching but one particle was already consumed by a
        higher-priority merge in the same condition.
    MERGED
        The pair was successfully merged.
    """
    
    # Filter-based rejections
    NOT_IN_CANDIDATE_SET = "Pair was not in candidate set (failed metadata filter)"
    DIFFERENT_PDG = "Particles have different PDG values"
    WRONG_PDG_VALUE = "Particle PDG not in target set"
    DIFFERENT_ANCESTOR = "Particles have different ancestor_id"
    NO_PARENT_CHILD_RELATION = "Particles are not in parent-child relationship"
    WRONG_SEMID = "Particle SemID does not match required value"
    
    # Geometric rejections
    CENTROID_TOO_FAR = "Centroids too far apart (pre-filter)"
    BBOX_TOO_FAR = "Bounding boxes too far apart"
    NOT_TOUCHING = "Point clouds not touching (no points within distance D)"
    
    # Selection rejections
    ALREADY_MERGED_WITH_OTHER = "SemID 4 particle already merged with another particle"
    
    # Success
    MERGED = "Particles were successfully merged"


@dataclass
class MergeDecision:
    """
    Immutable record of the merge outcome for one particle pair.

    Stored by both :class:`PartitionDiagnostics` and
    :class:`OptimizedPartitionDiagnostics`.

    Attributes
    ----------
    id1 : int
        ID of the first particle (always ≤ ``id2`` after normalisation).
    id2 : int
        ID of the second particle.
    reason : MergeOutcome
        The outcome tag for this pair.
    details : str or None, optional
        Free-text annotation, e.g. the measured distance or the
        specific stage value that triggered rejection.
    stage : str or None, optional
        Name of the pipeline stage at which the decision was made
        (e.g. ``"Stage 3: Proximity Check"``).
    """
    id1: int
    id2: int
    reason: MergeOutcome
    details: Optional[str] = None
    stage: Optional[str] = None

    def __str__(self):
        """
        Return a single-line human-readable summary of the decision.

        Returns
        -------
        str
            ``"Particles id1 <-> id2: <reason> (<details>) [Stage: <stage>]"``
            with optional fields omitted when ``None``.
        """
        result = f"Particles {self.id1} <-> {self.id2}: {self.reason.value}"
        if self.details:
            result += f" ({self.details})"
        if self.stage:
            result += f" [Stage: {self.stage}]"
        return result


class PartitionDiagnostics:
    """
    Full merge-decision tracker for the partitioning pipeline.

    Stores every :class:`MergeDecision` recorded during a partitioning run
    in ``decisions``, keyed by ``(stage, min(id1,id2), max(id1,id2))`` so
    that decisions from different conditions for the same pair are kept
    separately.  Stage-level aggregates are accumulated in ``stage_stats``.

    When *enabled* is ``False`` all calls are no-ops so the object can be
    constructed once and toggled without changing calling code.

    Attributes
    ----------
    enabled : bool
        Whether recording is active.
    decisions : dict[tuple[str, int, int], MergeDecision]
        All recorded decisions keyed by ``(stage, id1, id2)``.
        Use :meth:`get_decision` or :meth:`get_decisions_for_pair` to query.
    stage_stats : dict[str, dict[MergeOutcome, int]]
        Per-stage counts of each :class:`MergeOutcome`.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Parameters
        ----------
        enabled : bool, optional
            If ``False`` (default ``True``), all methods become no-ops.
        """
        self.enabled = enabled
        self.decisions: Dict[tuple, MergeDecision] = {}
        self.stage_stats: Dict[str, Dict[MergeOutcome, int]] = {}
        # Directed merge tracking: child (absorbed) → parent (survivor)
        self.absorbed_by_map: Dict[int, int] = {}
        # Reverse map: parent → list of directly absorbed children
        self.absorbed_particles: Dict[int, List[int]] = {}

    def record(self, id1: int, id2: int, reason: MergeOutcome,
               details: str = None, stage: str = None):
        """
        Record the merge outcome for a pair of particles.

        Parameters
        ----------
        id1 : int
            ID of the first particle.
        id2 : int
            ID of the second particle.
        reason : MergeOutcome
            Outcome tag.
        details : str or None, optional
            Supplementary annotation (e.g. measured distance).
        stage : str or None, optional
            Pipeline stage name.
        """
        if not self.enabled:
            return

        # Normalize order (always store smaller id first)
        if id1 > id2:
            id1, id2 = id2, id1

        key = (stage, id1, id2)
        decision = MergeDecision(id1, id2, reason, details, stage)

        # One decision per (stage, pair) — within a stage, latest write wins
        self.decisions[key] = decision

        # Track statistics by stage
        if stage:
            if stage not in self.stage_stats:
                self.stage_stats[stage] = {}
            self.stage_stats[stage][reason] = self.stage_stats[stage].get(reason, 0) + 1

    def record_actual_merge(self, child_id: int, parent_id: int):
        """
        Record that *child_id* was definitively absorbed into *parent_id*.

        Must be called **after** post-filtering and the Union-Find merge are
        both confirmed, so that exactly the pairs that were truly merged are
        captured.  This is the sole writer of :attr:`absorbed_by_map`.

        Parameters
        ----------
        child_id : int
            ID of the particle whose partition was absorbed (the child).
        parent_id : int
            ID of the surviving representative (the parent).
        """
        if not self.enabled:
            return
        self.absorbed_by_map[child_id] = parent_id
        self.absorbed_particles.setdefault(parent_id, []).append(child_id)

    def get_decision(self, id1: int, id2: int,
                     stage: str = None) -> Optional[MergeDecision]:
        """
        Return the recorded decision for a specific particle pair.

        Parameters
        ----------
        id1 : int
        id2 : int
        stage : str or None, optional
            If given, return only the decision recorded by that stage.
            If ``None``, return the decision from the *last* stage that
            recorded this pair (for backwards-compatible single-condition
            use), or ``None`` if the pair was never recorded.

        Returns
        -------
        MergeDecision or None
        """
        if id1 > id2:
            id1, id2 = id2, id1
        if stage is not None:
            return self.decisions.get((stage, id1, id2))
        # No stage specified — return the last recorded decision for the pair
        matched = [d for (s, a, b), d in self.decisions.items()
                   if a == id1 and b == id2]
        return matched[-1] if matched else None

    def get_decisions_for_pair(self, id1: int, id2: int) -> Dict[str, MergeDecision]:
        """
        Return all per-condition decisions recorded for a particle pair.

        Parameters
        ----------
        id1 : int
        id2 : int

        Returns
        -------
        dict[str, MergeDecision]
            Maps stage name → decision.  Empty dict if the pair was never
            evaluated.
        """
        if id1 > id2:
            id1, id2 = id2, id1
        return {s: d for (s, a, b), d in self.decisions.items()
                if a == id1 and b == id2}
    
    def why_not_merged(self, id1: int, id2: int) -> str:
        """
        Return a human-readable explanation of the merge decision.

        Parameters
        ----------
        id1 : int
        id2 : int

        Returns
        -------
        str
            Formatted decision string, or a "no decision recorded" message.
        """
        decision = self.get_decision(id1, id2)
        if decision is None:
            return f"No decision recorded for particles {id1} and {id2}"
        return str(decision)
    
    def print_decision(self, id1: int, id2: int):
        """
        Print the merge decision for a specific pair to stdout.

        Parameters
        ----------
        id1 : int
        id2 : int
        """
        print(self.why_not_merged(id1, id2))
    
    def get_all_decisions_for_particle(self, particle_id: int) -> List[MergeDecision]:
        """
        Return all recorded decisions that involve *particle_id*.

        Parameters
        ----------
        particle_id : int

        Returns
        -------
        list of MergeDecision
        """
        return [
            decision for (s, id1, id2), decision in self.decisions.items()
            if id1 == particle_id or id2 == particle_id
        ]
    
    def print_all_decisions_for_particle(self, particle_id: int):
        """
        Print all recorded decisions that involve *particle_id* to stdout.

        Parameters
        ----------
        particle_id : int
        """
        decisions = self.get_all_decisions_for_particle(particle_id)
        if not decisions:
            print(f"No decisions recorded for particle {particle_id}")
            return
        
        print(f"\nAll decisions for particle {particle_id}:")
        print("=" * 80)
        for decision in sorted(decisions, key=lambda d: (d.id1, d.id2)):
            print(f"  {decision}")

    def was_merged(self, particle_id: int) -> bool:
        """
        Return whether *particle_id* was absorbed into another partition.

        A particle is considered *merged* if it was the **child** (absorbed
        side) of at least one successful merge step.  The surviving
        representative of a partition — the particle that other partitions
        were merged into — returns ``False``, so within any non-singleton
        partition exactly the non-representative particles return ``True``.

        Parameters
        ----------
        particle_id : int

        Returns
        -------
        bool
        """
        return particle_id in self.absorbed_by_map

    def get_absorbed_by(self, particle_id: int) -> Optional[int]:
        """
        Return the ID of the particle whose partition absorbed *particle_id*.

        Parameters
        ----------
        particle_id : int

        Returns
        -------
        int or None
            The partner ID if *particle_id* was absorbed; ``None`` if it
            was not merged or was the surviving representative.
        """
        return self.absorbed_by_map.get(particle_id)

    def get_merge_partners(self, particle_id: int) -> List[int]:
        """
        Return the IDs of all particles directly paired with *particle_id*
        in any successful merge (regardless of direction).

        Parameters
        ----------
        particle_id : int

        Returns
        -------
        list of int
            Partner IDs.  Empty if *particle_id* was never part of any merge.
        """
        partners = []
        for (s, a, b), d in self.decisions.items():
            if d.reason != MergeOutcome.MERGED:
                continue
            if a == particle_id:
                partners.append(b)
            elif b == particle_id:
                partners.append(a)
        return partners

    def print_merge_summary(self, particle_id: int):
        """
        Print a concise merge summary for *particle_id* to stdout.

        Parameters
        ----------
        particle_id : int
        """
        absorbed_by = self.get_absorbed_by(particle_id)
        if absorbed_by is not None:
            print(f"Particle {particle_id}: absorbed into partition of particle {absorbed_by}")
        else:
            partners = self.get_merge_partners(particle_id)
            if partners:
                print(f"Particle {particle_id}: surviving representative; absorbed {partners}")
            else:
                print(f"Particle {particle_id}: not merged (singleton partition)")

    def get_summary(self) -> Dict:
        """
        Return aggregate statistics over all recorded decisions.

        Returns
        -------
        dict
            Keys:

            ``total_evaluations``
                Total number of ``(stage, pair)`` records stored (one pair
                evaluated by two conditions counts as 2).
            ``unique_pairs_evaluated``
                Number of distinct particle pairs that were evaluated by at
                least one condition.
            ``merged_pairs``
                Unique pairs for which *any* condition recorded ``MERGED``.
            ``rejected_pairs``
                Unique pairs for which *no* condition recorded ``MERGED``.
            ``rejection_reasons``
                Dict mapping reason string → count (across all conditions).
        """
        unique_pairs = {(a, b) for (s, a, b) in self.decisions}
        merged_pairs = {(a, b) for (s, a, b), d in self.decisions.items()
                        if d.reason == MergeOutcome.MERGED}
        summary = {
            'total_evaluations': len(self.decisions),
            'unique_pairs_evaluated': len(unique_pairs),
            'merged_pairs': len(merged_pairs),
            'rejected_pairs': len(unique_pairs - merged_pairs),
            'rejection_reasons': {}
        }
        
        for decision in self.decisions.values():
            reason = decision.reason.value
            summary['rejection_reasons'][reason] = summary['rejection_reasons'].get(reason, 0) + 1
        
        return summary
    
    def print_summary(self):
        """
        Print aggregate statistics for all recorded decisions to stdout.
        """
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("PARTITION DIAGNOSTICS SUMMARY")
        print("=" * 80)
        print(f"Total evaluations (stage × pair): {summary['total_evaluations']}")
        print(f"Unique pairs evaluated: {summary['unique_pairs_evaluated']}")
        print(f"Merged pairs: {summary['merged_pairs']}")
        print(f"Rejected pairs (no condition merged): {summary['rejected_pairs']}")
        print("\nRejection reasons:")
        
        for reason, count in sorted(summary['rejection_reasons'].items(), 
                                    key=lambda x: x[1], reverse=True):
            percentage = 100 * count / summary['total_pairs_evaluated']
            print(f"  {reason}: {count} ({percentage:.1f}%)")
        
        if self.stage_stats:
            print("\nBy stage:")
            for stage, stats in self.stage_stats.items():
                print(f"\n  {stage}:")
                for reason, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {reason.value}: {count}")
        
        print("=" * 80 + "\n")
    
    def export_to_json(self, filename: str):
        """
        Write all recorded decisions and summary stats to a JSON file.

        Parameters
        ----------
        filename : str
            Output path.  Existing files are overwritten.
        """
        data = {
            'decisions': [
                {
                    'id1': d.id1,
                    'id2': d.id2,
                    'reason': d.reason.value,
                    'details': d.details,
                    'stage': d.stage
                }
                for d in self.decisions.values()
            ],
            'summary': self.get_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Diagnostics exported to {filename}")
    
    def clear(self):
        """Discard all recorded decisions, stage statistics, and merge-direction data."""
        self.decisions.clear()
        self.stage_stats.clear()
        self.absorbed_by_map.clear()
        self.absorbed_particles.clear()


# ============================================================================
# Optimized Diagnostic System - Lazy Evaluation
# ============================================================================

class OptimizedPartitionDiagnostics:
    """
    Memory-efficient diagnostics using lazy evaluation.

    Unlike :class:`PartitionDiagnostics`, this class stores decisions only
    for pairs that were *actually evaluated* by the proximity checker
    (O(candidates) entries rather than O(n²)).  Reasons for pairs that were
    silently excluded by metadata pre-filters are reconstructed on demand
    when :meth:`why_not_merged` is called.

    Decisions are keyed by ``(stage, min_id, max_id)`` so that outcomes
    from different conditions for the same pair are kept separately and
    queryable via :meth:`get_decisions_for_pair`.

    Attributes
    ----------
    enabled : bool
        Whether recording is active.
    partitioner : ParticlePartitioner or None
        Back-reference to the owning partitioner, used for lazy reason
        reconstruction.
    evaluated_pairs : set of tuple[int, int]
        Normalised pairs ``(min_id, max_id)`` that were sent to the
        proximity checker (across all conditions).
    merge_decisions : dict[tuple[str, int, int], MergeDecision]
        Decisions keyed by ``(stage, id1, id2)``.
        Use :meth:`get_decisions_for_pair` to retrieve all decisions for a
        pair across every condition.
    stage_candidates : dict[str, set of tuple[int, int]]
        Candidate sets recorded per stage via :meth:`record_candidates`.
    stats : dict
        Running counters: ``total_evaluated``, ``merged``,
        ``rejected_by_stage``.
    """
    
    def __init__(self, enabled: bool = True, partitioner=None):
        """
        Parameters
        ----------
        enabled : bool, optional
            If ``False``, all methods become no-ops.  Default ``True``.
        partitioner : ParticlePartitioner or None, optional
            Reference to the owning partitioner, required for lazy
            rejection-reason reconstruction in :meth:`why_not_merged`.
        """
        self.enabled = enabled
        self.partitioner = partitioner  # Reference to partitioner for lazy queries

        # Only track pairs that were actually evaluated
        self.evaluated_pairs: Set[Tuple[int, int]] = set()
        self.merge_decisions: Dict[Tuple[int, int], MergeDecision] = {}

        # Track which stages ran and their candidate sets
        self.stage_candidates: Dict[str, Set[Tuple[int, int]]] = {}

        # Directed merge tracking: child (absorbed) → parent (survivor)
        self.absorbed_by_map: Dict[int, int] = {}
        # Reverse map: parent → list of directly absorbed children
        self.absorbed_particles: Dict[int, List[int]] = {}

        # Statistics
        self.stats = {
            'total_evaluated': 0,
            'merged': 0,
            'rejected_by_stage': defaultdict(int)
        }
    
    def record_candidates(self, stage: str, candidates: List[Tuple[int, int]]):
        """
        Store the candidate set proposed by a stage for later lazy lookup.

        Parameters
        ----------
        stage : str
            Stage name (used as key in :attr:`stage_candidates`).
        candidates : list of tuple[int, int]
            Pairs of particle IDs proposed as merge candidates.
        """
        if not self.enabled:
            return
        
        self.stage_candidates[stage] = set(
            (min(id1, id2), max(id1, id2)) for id1, id2 in candidates
        )
    
    def record(self, id1: int, id2: int, reason: MergeOutcome,
               details: str = None, stage: str = None):
        """
        Record the proximity-check outcome for an evaluated pair.

        Only pairs that were actually sent to the proximity checker should
        be recorded here.  Pairs excluded by metadata pre-filters are
        handled lazily via :meth:`why_not_merged`.

        Parameters
        ----------
        id1 : int
        id2 : int
        reason : MergeOutcome
        details : str or None, optional
        stage : str or None, optional
        """
        if not self.enabled:
            return

        # Normalize pair order
        pair = (min(id1, id2), max(id1, id2))

        # Key by (stage, id1, id2) so decisions from different conditions
        # for the same pair are stored independently
        key = (stage, pair[0], pair[1])
        self.merge_decisions[key] = MergeDecision(id1, id2, reason, details, stage)
        self.evaluated_pairs.add(pair)

        # Update statistics
        self.stats['total_evaluated'] += 1
        if reason == MergeOutcome.MERGED:
            self.stats['merged'] += 1
        else:
            self.stats['rejected_by_stage'][stage] += 1

    def record_actual_merge(self, child_id: int, parent_id: int):
        """
        Record that *child_id* was definitively absorbed into *parent_id*.

        Must be called **after** post-filtering and the Union-Find merge are
        both confirmed, so that exactly the pairs that were truly merged are
        captured.  This is the sole writer of :attr:`absorbed_by_map`.

        Parameters
        ----------
        child_id : int
            ID of the particle whose partition was absorbed (the child).
        parent_id : int
            ID of the surviving representative (the parent).
        """
        if not self.enabled:
            return
        self.absorbed_by_map[child_id] = parent_id
        self.absorbed_particles.setdefault(parent_id, []).append(child_id)

    def why_not_merged(self, id1: int, id2: int) -> str:
        """
        Return a human-readable explanation for why a pair was not merged.

        Uses lazy evaluation: if the pair was captured during a proximity
        check the stored :class:`MergeDecision` is returned directly;
        otherwise the reason is reconstructed on demand by examining which
        stage candidate sets the pair appeared in and what metadata filters
        would have rejected it.

        Parameters
        ----------
        id1, id2 : int

        Returns
        -------
        str
            A human-readable rejection explanation, or an informational
            message when diagnostics are disabled / the pair was never seen.
        """
        pair = (min(id1, id2), max(id1, id2))

        # Check if we have recorded decisions (any condition)
        matched = [d for (s, a, b), d in self.merge_decisions.items()
                   if a == pair[0] and b == pair[1]]
        if matched and not self.enabled:
            return "\n".join(str(d) for d in matched)

        # Use get_decisions_for_pair for the full (lazy) picture
        decisions = self.get_decisions_for_pair(id1, id2)
        if decisions:
            return "\n".join(str(d) for d in decisions.values())
        
        # Lazy evaluation: figure out why based on what stages ran
        if not self.enabled or self.partitioner is None:
            return "Diagnostics not enabled or partitioner reference not available"
        
        # Reconstruct the reason by checking each stage
        p1 = self.partitioner.particle_lookup.get(id1)
        p2 = self.partitioner.particle_lookup.get(id2)
        
        if p1 is None or p2 is None:
            return f"One or both particle IDs not found: {id1}, {id2}"
        
        # Check each stage to see why it was filtered out
        reason = self._lazy_determine_rejection_reason(p1, p2, pair)
        return reason
    
    def _lazy_determine_rejection_reason(self, p1: Particle, p2: Particle,
                                         pair: Tuple[int, int]) -> str:
        """
        Reconstruct the rejection reason by replaying stage membership.

        Iterates :attr:`stage_candidates` to find the earliest stage that
        did not include *pair*.  If the pair was never a candidate it
        delegates to :meth:`_determine_metadata_rejection` for a more
        specific explanation.

        Parameters
        ----------
        p1, p2 : Particle
            The two particles in question.
        pair : tuple[int, int]
            Normalised pair ``(min_id, max_id)``.

        Returns
        -------
        str
        """
        # Check if it was in any candidate set
        was_candidate = False
        earliest_rejection_stage = None
        
        for stage, candidates in self.stage_candidates.items():
            if pair in candidates:
                was_candidate = True
                break
            elif earliest_rejection_stage is None:
                earliest_rejection_stage = stage
        
        if not was_candidate and earliest_rejection_stage:
            # It was filtered out before being a candidate
            # Determine specific reason based on metadata
            return self._determine_metadata_rejection(p1, p2, earliest_rejection_stage)
        
        if was_candidate:
            # It was a candidate but got filtered later
            return f"Was a candidate but filtered in later stage (not touching or other reason)"
        
        return f"Pair ({p1.id}, {p2.id}) was never considered for merging"
    
    def _determine_metadata_rejection(self, p1: Particle, p2: Particle, stage: str) -> str:
        """
        Diagnose which metadata property caused pair rejection at *stage*.

        Checks parent-child relationship, PDG codes, ``ancestor_id``, and
        ``sem_type`` in turn and accumulates all violated conditions into a
        single human-readable string.

        Parameters
        ----------
        p1, p2 : Particle
        stage : str
            Name of the earliest stage at which the pair was absent.

        Returns
        -------
        str
        """
        reasons = []
        
        # Check parent-child relationship
        if not (p1.parent_id == p2.id or p2.parent_id == p1.id):
            reasons.append("not in parent-child relationship")
        
        # Check PDG
        if p1.pdg != p2.pdg:
            reasons.append(f"different PDG ({p1.pdg} vs {p2.pdg})")
        
        # Check ancestor
        if p1.ancestor_id != p2.ancestor_id:
            reasons.append(f"different ancestor_id ({p1.ancestor_id} vs {p2.ancestor_id})")
        
        # Check SemID
        if p1.sem_type != p2.sem_type:
            reasons.append(f"different SemID ({p1.sem_type} vs {p2.sem_type})")
        
        if reasons:
            return f"Particles {p1.id} <-> {p2.id}: Rejected at {stage} - " + ", ".join(reasons)
        else:
            return f"Particles {p1.id} <-> {p2.id}: Rejected at {stage} for unknown reason"
    
    def get_decisions_for_pair(self, id1: int, id2: int) -> Dict[str, MergeDecision]:
        """
        Return all per-condition decisions recorded for a particle pair.

        For pairs that reached the proximity checker the stored
        :class:`MergeDecision` is returned.  For pairs that were filtered out
        before reaching the proximity checker the decision is reconstructed
        lazily from :attr:`stage_candidates` and the particle metadata, so
        every stage that was active during the run appears in the result.

        Parameters
        ----------
        id1 : int
        id2 : int

        Returns
        -------
        dict[str, MergeDecision]
            Maps stage name → decision.  Contains one entry per stage
            recorded in :attr:`stage_candidates` plus any proximity-check
            decisions in :attr:`merge_decisions`.
        """
        a, b = min(id1, id2), max(id1, id2)
        pair = (a, b)

        # Directly recorded decisions (pairs that reached the proximity checker)
        result = {s: d for (s, x, y), d in self.merge_decisions.items()
                  if x == a and y == b}

        # Lazy reconstruction for every stage not already in result
        if self.enabled and self.partitioner is not None:
            p1 = self.partitioner.particle_lookup.get(id1)
            p2 = self.partitioner.particle_lookup.get(id2)
            if p1 is not None and p2 is not None:
                for stage, candidates in self.stage_candidates.items():
                    if stage in result:
                        continue
                    if pair in candidates:
                        # Pair passed this stage's filter but has no proximity
                        # decision — it either passed through to the next stage
                        # or was rejected there.
                        result[stage] = MergeDecision(
                            a, b,
                            MergeOutcome.NOT_TOUCHING,
                            "Passed candidate filter but not proximity-checked "
                            "(rejected at a subsequent stage or not touching)",
                            stage,
                        )
                    else:
                        # Pair was absent from this stage's candidate set —
                        # reconstruct the metadata reason.
                        details = self._determine_metadata_rejection(p1, p2, stage)
                        result[stage] = MergeDecision(
                            a, b,
                            MergeOutcome.NOT_IN_CANDIDATE_SET,
                            details,
                            stage,
                        )

        return result

    def was_merged(self, particle_id: int) -> bool:
        """
        Return whether *particle_id* was absorbed into another partition.

        A particle is considered *merged* if it was the **child** (absorbed
        side) of at least one merge step.  The surviving representative of
        a partition returns ``False``, so within any non-singleton partition
        exactly the non-representative particles return ``True``.

        Parameters
        ----------
        particle_id : int

        Returns
        -------
        bool
        """
        return particle_id in self.absorbed_by_map

    def get_absorbed_by(self, particle_id: int) -> Optional[int]:
        """
        Return the ID of the particle whose partition absorbed *particle_id*.

        Parameters
        ----------
        particle_id : int

        Returns
        -------
        int or None
            The partner ID if *particle_id* was absorbed; ``None`` if it
            was not merged or was the surviving representative.
        """
        return self.absorbed_by_map.get(particle_id)

    def get_merge_partners(self, particle_id: int) -> List[int]:
        """
        Return the IDs of all particles directly paired with *particle_id*
        in any confirmed merge (based on actual merge records, not proximity
        touching).

        Parameters
        ----------
        particle_id : int

        Returns
        -------
        list of int
            Partner IDs: the particle that absorbed *particle_id* (if any)
            plus all particles that *particle_id* directly absorbed.
        """
        partners = []
        parent = self.absorbed_by_map.get(particle_id)
        if parent is not None:
            partners.append(parent)
        partners.extend(self.absorbed_particles.get(particle_id, []))
        return partners

    def print_merge_summary(self, particle_id: int):
        """
        Print a concise merge summary for *particle_id* to stdout.

        Parameters
        ----------
        particle_id : int
        """
        absorbed_by = self.get_absorbed_by(particle_id)
        if absorbed_by is not None:
            print(f"Particle {particle_id}: absorbed into partition of particle {absorbed_by}")
        else:
            children = self.absorbed_particles.get(particle_id, [])
            if children:
                print(f"Particle {particle_id}: surviving representative; absorbed {children}")
            else:
                print(f"Particle {particle_id}: not merged (singleton partition)")

    def get_memory_usage_mb(self) -> float:
        """
        Estimate the RAM consumed by stored decisions and candidate sets.

        Returns
        -------
        float
            Approximate memory usage in mebibytes.
        """
        import sys

        decision_size = sys.getsizeof(MergeDecision(0, 0, MergeOutcome.MERGED))
        decisions_mem = len(self.merge_decisions) * decision_size
        candidates_mem = sum(len(s) * 16 for s in self.stage_candidates.values())

        total_bytes = decisions_mem + candidates_mem
        return total_bytes / (1024 * 1024)
    
    def print_summary(self):
        """
        Print a summary including evaluated/merged counts and memory usage.
        """
        print("\n" + "=" * 80)
        print("PARTITION DIAGNOSTICS SUMMARY (Optimized)")
        print("=" * 80)
        print(f"Pairs evaluated: {self.stats['total_evaluated']}")
        print(f"Merged pairs: {self.stats['merged']}")
        print(f"Rejected pairs: {self.stats['total_evaluated'] - self.stats['merged']}")
        print(f"Estimated memory usage: {self.get_memory_usage_mb():.2f} MB")
        
        if self.stats['rejected_by_stage']:
            print("\nRejections by stage:")
            for stage, count in self.stats['rejected_by_stage'].items():
                print(f"  {stage}: {count}")
        
        print("=" * 80 + "\n")
    
    def clear(self):
        """Discard all recorded decisions, candidate sets, statistics, and merge-direction data."""
        self.evaluated_pairs.clear()
        self.merge_decisions.clear()
        self.stage_candidates.clear()
        self.absorbed_by_map.clear()
        self.absorbed_particles.clear()
        self.stats = {
            'total_evaluated': 0,
            'merged': 0,
            'rejected_by_stage': defaultdict(int)
        }
