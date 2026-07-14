from .proxck import (
    ProximityChecker,
    UnionFind,
    CPUSingleThreadChecker,
    CPUMultiThreadChecker,
    GPUChecker,
    BulkGPUChecker,
    NumbaKernelChecker,
    CellHashCPUSingleThreadChecker,
    CellHashCPUMultiThreadChecker,
    CellHashGPUChecker,
)
from .data import Particle
from typing import List, Tuple, Callable, Set, Dict, Optional
import time
from collections import defaultdict
import numpy as np
import json

from .diagnostics import (
    MergeOutcome,
    MergeDecision,
    PartitionDiagnostics,
    OptimizedPartitionDiagnostics,
)
from .conditions import PartitionConditionBase
import copy


# ============================================================================
# Optimized Partitioner with Lazy Diagnostics
# ============================================================================

class ParticlePartitioner:
    """
    Groups a collection of particles into partitions by merging spatially
    touching pairs that satisfy configurable physical criteria.

    Partitioning proceeds in one or more *conditions*, each of which
    (1) builds a compact set of candidate pairs based on particle metadata
    (PDG code, semantic type, parent-child relationships, etc.),
    (2) filters candidates by spatial proximity using a pluggable compute
    backend (CPU single-thread, CPU multi-thread, or GPU), and
    (3) assembles merged groups via a Union-Find structure.

    An optional lightweight diagnostics system can record every merge
    decision so that the caller can later inspect *why* any two particles
    were or were not merged.

    Attributes
    ----------
    particles : list of Particle
        The full particle collection passed at construction time.
    D : float
        Distance threshold used by the proximity checker.
    backend_name : str
        Name of the active compute backend.
    particle_lookup : dict[int, Particle]
        Maps each particle ID to its ``Particle`` object.
    id_to_idx : dict[int, int]
        Maps each particle ID to its positional index in ``particles``.
    children_map : defaultdict(list)
        Maps a particle ID to the list of IDs of its direct children.
    ancestor_map : defaultdict(list)
        Maps an ancestor ID to the list of IDs of all its descendants.
    checker : ProximityChecker
        Active proximity-checking backend.
    diagnostics : OptimizedPartitionDiagnostics
        Diagnostic recorder (active only when *enable_diagnostics* is
        ``True``).
    """

    def __init__(self,
                 particles,
                 distance_threshold: float,
                 backend: str = 'gpu',
                 n_jobs: int = -1,
                 enable_diagnostics: bool = False,
                 check_completeness: bool = False,
                 verbose: bool = True,
                 **kwargs):
        
        if check_completeness:
            from .config import check_particle_list
            check_particle_list(particles)

        self.particles = particles
        self.D = distance_threshold
        self.backend_name = backend
        
        # Build lookup structures
        self.particle_lookup = {p.id: p for p in particles}
        self.id_to_idx = {p.id: idx for idx, p in enumerate(particles)}
        
        # Build relationships
        self._build_relationships()
        
        # Initialize OPTIMIZED diagnostics with reference to self
        self.diagnostics = OptimizedPartitionDiagnostics(
            enabled=enable_diagnostics,
            partitioner=self
        )
        
        # Initialize backend
        self.checker = self._create_backend(backend, n_jobs, **kwargs)
        self.checker._verbose = verbose
        self.checker.initialize(particles)

    def _build_relationships(self):
        """
        Populate ``children_map`` and ``ancestor_map`` from the particle list.

        Iterates over all particles once (O(n)) and fills two look-up
        dictionaries that avoid repeated linear scans during candidate
        building:

        * ``children_map[parent_id]`` – list of direct child IDs.
        * ``ancestor_map[root_id]`` – list of all descendant IDs that
          share the same ancestor.
        """
        self.children_map = defaultdict(list)
        self.ancestor_map = defaultdict(list)
        
        for p in self.particles:
            if p.parent_id is not None:
                self.children_map[p.parent_id].append(p.id)
            if p.root_id is not None:
                self.ancestor_map[p.root_id].append(p.id)
    
    def _create_backend(self, backend: str, n_jobs: int, **kwargs):
        """
        Instantiate and return the requested proximity-checking backend.

        The returned checker is already **initialised** with the particle
        list via a subsequent ``checker.initialize(particles)`` call in
        ``__init__``.

        Parameters
        ----------
        backend : str
            Name of the backend to construct.  Recognised values:

            * ``'cpu-single'`` – scipy KDTree, single thread.
            * ``'cpu-multi'`` – scipy KDTree + joblib thread pool.
            * ``'gpu'`` – RAPIDS cuML ``NearestNeighbors`` (brute-force).
            * ``'bulk-gpu'`` – CuPy chunked brute-force; stays fully
              GPU-resident; accepts ``chunk_size`` kwarg (default 512).
            * ``'numba'`` – batched ``@numba.cuda.jit`` kernel with
              shared-memory B-tiling; accepts ``block_size`` kwarg
              (default 128, must be a multiple of 32).
            * ``'cell-hash-cpu-single'`` – cell-hash, single thread,
              no scipy dependency.
            * ``'cell-hash-cpu-multi'`` – cell-hash + joblib threads.
            * ``'cell-hash-gpu'`` – cell-hash maps on CPU, distance
              kernels on GPU (CuPy).

        n_jobs : int
            Worker threads for ``'cpu-multi'`` and
            ``'cell-hash-cpu-multi'``.  ``-1`` means all available
            cores.  Ignored for other backends.
        **kwargs
            Backend-specific parameters forwarded from
            ``ParticlePartitioner.__init__``:

            * ``chunk_size`` (int) – used by ``'bulk-gpu'``.
            * ``block_size`` (int) – used by ``'numba'``.

        Returns
        -------
        ProximityChecker
            An uninitialised proximity checker of the requested type.

        Raises
        ------
        ValueError
            If *backend* is not one of the recognised values.
        """
        if backend == 'cpu-single':
            return CPUSingleThreadChecker(self.D)
        elif backend == 'cpu-multi':
            return CPUMultiThreadChecker(self.D, n_jobs=n_jobs)
        elif backend == 'gpu':
            return GPUChecker(self.D)
        elif backend == 'bulk-gpu':
            chunk_size = kwargs.get('chunk_size', 512)
            return BulkGPUChecker(self.D, chunk_size=chunk_size)
        elif backend == 'numba':
            block_size = kwargs.get('block_size', 128)
            return NumbaKernelChecker(self.D, block_size=block_size)
        elif backend == 'cell-hash-cpu-single':
            return CellHashCPUSingleThreadChecker(self.D)
        elif backend == 'cell-hash-cpu-multi':
            return CellHashCPUMultiThreadChecker(self.D, n_jobs=n_jobs)
        elif backend == 'cell-hash-gpu':
            return CellHashGPUChecker(self.D)
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}.  "
                f"Valid options: cpu-single, cpu-multi, gpu, bulk-gpu, numba, "
                f"cell-hash-cpu-single, cell-hash-cpu-multi, cell-hash-gpu."
            )
    
    # ========================================================================
    # Generic Partition Method
    # ========================================================================

    def partition(self, condition: PartitionConditionBase,
                  verbose: bool = True,
                  partition_level_proximity: bool = True) -> List[List]:
        """
        Partition particles using the supplied condition strategy.

        Orchestrates the three standard stages of the partitioning pipeline:

        1. **Candidate building** – delegates to ``condition.get_candidates``
           to obtain a compact set of pairs worth testing.
        2. **Proximity check** – tests each candidate pair for spatial
           proximity within distance ``D``.
        3. **Post-filtering** – delegates to ``condition.post_filter`` for
           any additional selection logic (e.g. one-merge-per-particle
           constraints).

        Parameters
        ----------
        condition : PartitionConditionBase
            A condition object implementing ``get_candidates`` and,
            optionally, ``post_filter``.
        verbose : bool, optional
            If ``True`` (default), prints stage-by-stage progress and a
            summary table to stdout.
        partition_level_proximity : bool, optional
            If ``True`` (default), proximity is evaluated between the merged
            point cloud of each **partition** (all particles in the same
            connected component) rather than between individual particle
            clouds.  Uses :meth:`_build_partitions_incremental` with
            iterative convergence.  If ``False``, uses the legacy
            particle-level proximity check.

        Returns
        -------
        list of list of Particle
            Each inner list is one partition. Unmerged particles appear as
            singleton lists.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Partitioning: {condition.description}")
            mode_label = "partition-level" if partition_level_proximity else "particle-level"
            print(f"Backend: {self.backend_name}  |  proximity: {mode_label}")
            print(f"{'='*70}")

        start_time = time.time()
        self.diagnostics.clear()

        if partition_level_proximity:
            partitions = self._build_partitions_incremental(
                [condition], verbose=verbose
            )
        else:
            if verbose:
                print("Stage 1: Building candidate pairs...")

            candidates = condition.get_candidates(self)

            if verbose:
                print(f"  Candidate pairs: {len(candidates)}")

            if len(candidates) == 0:
                return [[p] for p in self.particles]

            if verbose:
                print("Stage 2: Checking proximity...")

            touching_pairs = self._batch_check_proximity_optimized(candidates)

            if verbose:
                print(f"  Touching pairs found: {len(touching_pairs)}")

            merge_pairs = condition.post_filter(self, touching_pairs)

            if verbose and len(merge_pairs) != len(touching_pairs):
                print(f"  After post-filter: {len(merge_pairs)}")

            if verbose:
                print("Stage 3: Building partitions...")

            partitions = self._build_partitions(merge_pairs)

        elapsed = time.time() - start_time

        if verbose:
            self._print_partition_stats(partitions, elapsed)
            if self.diagnostics.enabled:
                self.diagnostics.print_summary()

        return partitions

    def _batch_check_proximity_optimized(self, candidates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Retain only candidate pairs whose point clouds are within distance ``D``.

        When diagnostics are **disabled**, delegates directly to
        ``checker.batch_check_proximity`` for maximum throughput. When
        diagnostics are **enabled**, falls back to per-pair queries so that
        each outcome (``MERGED`` or ``NOT_TOUCHING``) can be individually
        recorded.

        Parameters
        ----------
        candidates : list of tuple[int, int]
            Pairs of particle IDs ``(id1, id2)`` to test for proximity.

        Returns
        -------
        list of tuple[int, int]
            Subset of *candidates* for which the minimum distance between
            the two point clouds is at most ``self.D``.
        """
        merge_pairs = []
        
        if not self.diagnostics.enabled:
            # Fast path - no diagnostics overhead
            return self.checker.batch_check_proximity(candidates)
        
        # With diagnostics - check each pair and record decision
        for id1, id2 in candidates:
            is_touching = self.checker.check_proximity(id1, id2)
            
            if is_touching:
                merge_pairs.append((id1, id2))
                self.diagnostics.record(
                    id1, id2,
                    MergeOutcome.MERGED,
                    f"Point clouds touching (within {self.D}mm)",
                    stage="Stage 3: Proximity Check"
                )
            else:
                self.diagnostics.record(
                    id1, id2,
                    MergeOutcome.NOT_TOUCHING,
                    f"Minimum distance > {self.D}mm",
                    stage="Stage 3: Proximity Check"
                )
        
        return merge_pairs

    # ========================================================================
    # Combined Partitioning
    # ========================================================================
    
    def partition_combined(self,
                            conditions: List[PartitionConditionBase],
                            verbose: bool = True,
                            partition_level_proximity: bool = True) -> List[List]:
        """
        Apply multiple condition strategies and merge all their accepted pairs.

        Parameters
        ----------
        conditions : list of PartitionConditionBase
            Ordered list of condition objects to apply.
        verbose : bool, optional
            If ``True`` (default), prints per-condition progress and a
            final summary to stdout.
        partition_level_proximity : bool, optional
            If ``True`` (default), proximity is evaluated between the merged
            point cloud of each **partition** rather than between individual
            particle clouds.  All conditions share one incrementally-updated
            partition state, and the algorithm iterates until convergence.
            If ``False``, uses the legacy particle-level batch path.

        Returns
        -------
        list of list of Particle
            Each inner list is one partition. Unmerged particles appear as
            singleton lists.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Combined Partitioning: Applying {len(conditions)} conditions")
            mode_label = "partition-level" if partition_level_proximity else "particle-level"
            print(f"Backend: {self.backend_name}  |  proximity: {mode_label}")
            print(f"{'='*70}")

        start_time = time.time()
        self.diagnostics.clear()

        if partition_level_proximity:
            partitions = self._build_partitions_incremental(
                conditions, verbose=verbose
            )
        else:
            all_merge_pairs = []

            for i, condition in enumerate(conditions, 1):
                if verbose:
                    print(f"\n--- Condition {i}/{len(conditions)}: {condition.description} ---")

                candidates = condition.get_candidates(self)

                if verbose:
                    print(f"  Candidates: {len(candidates)}")

                touching_pairs = self._batch_check_proximity_optimized(candidates)
                merge_pairs = condition.post_filter(self, touching_pairs)

                if verbose:
                    print(f"  Merge pairs: {len(merge_pairs)}")

                all_merge_pairs.extend(merge_pairs)

            if verbose:
                print(f"\nBuilding final partitions from {len(all_merge_pairs)} total merge pairs...")

            partitions = self._build_partitions(all_merge_pairs)

        elapsed = time.time() - start_time

        if verbose:
            self._print_partition_stats(partitions, elapsed)
            if self.diagnostics.enabled:
                self.diagnostics.print_summary()

        return partitions
    
    # ========================================================================
    # Custom Partitioning
    # ========================================================================
    
    def partition_custom(self,
                         candidate_filter: Callable[[Particle, Particle], bool],
                         verbose: bool = True) -> List[List]:
        """
        Partition particles using an arbitrary pairwise filter function.

        Evaluates *candidate_filter* on every unordered pair of particles
        (O(n²) calls) to build the candidate set, then checks proximity and
        builds partitions in the usual way.

        .. warning::
            The O(n²) candidate-building step can be prohibitively slow for
            large particle collections. Prefer ``partition_condition_1``,
            ``partition_condition_2``, or ``partition_combined`` when the
            merging logic can be expressed through the built-in conditions.

        Parameters
        ----------
        candidate_filter : callable(Particle, Particle) -> bool
            A function that receives two distinct ``Particle`` objects and
            returns ``True`` if the pair should be considered as a merge
            candidate (spatial proximity is checked subsequently).
        verbose : bool, optional
            If ``True`` (default), prints stage-by-stage progress, a
            performance warning, and a final summary to stdout.

        Returns
        -------
        list of list of Particle
            Each inner list is one partition. Unmerged particles appear as
            singleton lists.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Custom Partitioning")
            print(f"Backend: {self.backend_name}")
            print(f"WARNING: This checks all O(n²) pairs - may be slow for large datasets")
            print(f"{'='*70}")
        
        start_time = time.time()
        self.diagnostics.clear()
        
        if verbose:
            print("Stage 1: Building candidate pairs with custom filter...")
        
        candidates = []
        n = len(self.particles)
        
        # This is O(n²) - unavoidable for custom filters
        for i, p1 in enumerate(self.particles):
            if verbose and i % 1000 == 0:
                print(f"  Processing particle {i}/{n}...")
            
            for p2 in self.particles[i+1:]:
                if candidate_filter(p1, p2):
                    candidates.append((p1.id, p2.id))
        
        self.diagnostics.record_candidates("Stage 1: Custom Filter", candidates)
        
        if verbose:
            print(f"  Candidate pairs: {len(candidates)}")
        
        if len(candidates) == 0:
            return [[p] for p in self.particles]
        
        if verbose:
            print("Stage 2: Checking proximity (touching)...")
        
        merge_pairs = self._batch_check_proximity_optimized(candidates)
        
        if verbose:
            print(f"  Touching pairs found: {len(merge_pairs)}")
            print("Stage 3: Building partitions...")
        
        partitions = self._build_partitions(merge_pairs)
        
        elapsed = time.time() - start_time
        
        if verbose:
            self._print_partition_stats(partitions, elapsed)
            if self.diagnostics.enabled:
                self.diagnostics.print_summary()
        
        return partitions
    
    # ========================================================================
    # Diagnostic Query Methods
    # ========================================================================
    
    def why_not_merged(self, id1: int, id2: int) -> str:
        """
        Return a human-readable explanation of why two particles were not merged.

        Delegates to ``OptimizedPartitionDiagnostics.why_not_merged``, which
        uses lazy evaluation: if the pair was never explicitly evaluated it
        reconstructs the rejection reason on-demand by re-examining particle
        metadata, keeping memory proportional to the number of *evaluated*
        pairs rather than O(n²).

        Parameters
        ----------
        id1 : int
            ID of the first particle.
        id2 : int
            ID of the second particle.

        Returns
        -------
        str
            Description of the merge decision, including the pipeline stage
            at which the pair was rejected and the specific reason.
        """
        return self.diagnostics.why_not_merged(id1, id2)
    
    def print_decision(self, id1: int, id2: int):
        """
        Print the merge decision for a specific particle pair to stdout.

        Convenience wrapper around ``why_not_merged``.

        Parameters
        ----------
        id1 : int
            ID of the first particle.
        id2 : int
            ID of the second particle.
        """
        print(self.why_not_merged(id1, id2))
    
    def print_all_decisions_for_particle(self, particle_id: int):
        """
        Print all recorded merge decisions that involve a given particle.

        Only pairs that were *actually evaluated* during the most recent
        partitioning run are stored. If diagnostics are disabled, or the
        particle was not part of any evaluated pair, a descriptive message
        is printed instead.

        Parameters
        ----------
        particle_id : int
            ID of the particle whose decisions should be displayed.
        """
        if not self.diagnostics.enabled:
            print("Diagnostics not enabled. Enable with enable_diagnostics=True")
            return
        
        decisions = [
            decision for decision in self.diagnostics.merge_decisions.values()
            if decision.id1 == particle_id or decision.id2 == particle_id
        ]
        
        if not decisions:
            print(f"\nNo recorded decisions for particle {particle_id}")
            print("Note: Only pairs that were actually evaluated are recorded.")
            print("Use why_not_merged(id1, id2) to query specific pairs.")
            return
        
        print(f"\nRecorded decisions for particle {particle_id}:")
        print("=" * 80)
        for decision in sorted(decisions, key=lambda d: (d.id1, d.id2)):
            print(f"  {decision}")
        print(f"\nTotal decisions recorded: {len(decisions)}")
    
    def print_diagnostics_summary(self):
        """
        Print an aggregate summary of all merge decisions to stdout.

        Displays total pairs evaluated, number of successful merges,
        rejection counts per stage, and estimated memory usage. Has no
        effect (prints a warning) when diagnostics are not enabled.
        """
        if not self.diagnostics.enabled:
            print("Diagnostics not enabled. Enable with enable_diagnostics=True")
            return
        
        self.diagnostics.print_summary()
    
    def export_diagnostics(self, filename: str):
        """
        Export the recorded merge decisions to a JSON file.

        Writes a JSON object containing every evaluated pair and its decision,
        aggregate statistics, and an estimate of the diagnostics memory
        footprint. Has no effect (prints a warning) when diagnostics are not
        enabled.

        Parameters
        ----------
        filename : str
            Path to the output JSON file. Existing files are overwritten.
        """
        if not self.diagnostics.enabled:
            print("Diagnostics not enabled. Enable with enable_diagnostics=True")
            return
        
        data = {
            'evaluated_pairs': len(self.diagnostics.merge_decisions),
            'decisions': [
                {
                    'id1': d.id1,
                    'id2': d.id2,
                    'reason': d.reason.value,
                    'details': d.details,
                    'stage': d.stage
                }
                for d in self.diagnostics.merge_decisions.values()
            ],
            'statistics': self.diagnostics.stats,
            'memory_usage_mb': self.diagnostics.get_memory_usage_mb()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Diagnostics exported to {filename}")
        print(f"Exported {len(self.diagnostics.merge_decisions)} evaluated pairs")
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _build_partitions(self, merge_pairs: List[Tuple[int, int]]) -> List[List]:
        """
        Assemble particle groups from a list of merge pairs using Union-Find.

        Initialises a ``UnionFind`` structure over all particles, calls
        ``union`` for every accepted pair, then collects particles by their
        root representative.

        Parameters
        ----------
        merge_pairs : list of tuple[int, int]
            Pairs of particle IDs ``(id1, id2)`` that should be placed in
            the same partition.

        Returns
        -------
        list of list of Particle
            Each inner list contains the ``Particle`` objects belonging to
            one partition. The order of partitions and of particles within
            each partition is not guaranteed.
        """
        n = len(self.particles)
        uf = UnionFind(n)
        
        for id1, id2 in merge_pairs:
            idx1 = self.id_to_idx[id1]
            idx2 = self.id_to_idx[id2]
            uf.union(idx1, idx2)
        
        partitions_dict = defaultdict(list)
        for idx, p in enumerate(self.particles):
            root = uf.find(idx)
            partitions_dict[root].append(p)

        # In the legacy (non-incremental) path no point clouds are merged, so
        # each original particle is its own representative.  Stamp member_ids
        # as a singleton list so downstream lookups (_frag_lookup / _inst_lookup)
        # can always use the member_ids branch rather than the id-fallback.
        for p in self.particles:
            p.member_ids = [p.id]
        self.rep_lookup = {p.id: p for p in self.particles}

        return list(partitions_dict.values())

    def _build_partitions_incremental(
        self,
        conditions: List[PartitionConditionBase],
        verbose: bool = True,
        max_passes: int = 10,
    ) -> List[List]:
        """
        Build partitions using **partition-level** proximity with iterative
        convergence and per-partition representative particles.

        Each partition is represented by exactly one :class:`~pysupera.data.Particle`
        object (the *representative*).  The representative retains all scalar
        attributes of the original particle (PDG, semantic type, ancestry, …)
        and carries a ``point_cloud`` that grows to include every point from
        every particle that has been merged into the partition.

        Merge direction
        ---------------
        For each candidate pair ``(child, parent)`` returned by
        :meth:`~pysupera.conditions.base.PartitionConditionBase.get_rep_candidates`,
        the *child's* partition is absorbed INTO the *parent's* partition:
        the parent representative survives and its ``point_cloud`` is updated;
        the child representative is discarded.  The ordering is entirely
        condition-specific — e.g. ``TouchingLEScatter`` always places the
        ``kLEScatter`` representative as the child.

        Algorithm
        ---------
        1. Initialise ``UnionFind`` and ``reps`` — one representative
           :class:`Particle` per index, keyed by UF root.
        2. Build ``rep_lookup`` — maps every original particle ID to the
           current representative of its partition.
        3. **Pass loop** (repeats until convergence or *max_passes*):

           For each condition:

           a. Call ``condition.get_rep_candidates(self, rep_lookup)`` to get
              directed ``(child_rep_id, parent_rep_id)`` pairs for this pass.
           b. For each pair whose UF roots differ, call
              ``checker.check_cloud_proximity(child_rep.point_cloud,
              parent_rep.point_cloud)``.
           c. Collect touching pairs, apply ``condition.post_filter``.
           d. For each accepted merge: grow parent cloud, call ``uf.union``,
              update ``rep_lookup`` for all child particles, clean up ``reps``.

        4. Terminate when a full pass produces zero new merges.

        Parameters
        ----------
        conditions : list of PartitionConditionBase
        verbose : bool, optional
        max_passes : int, optional
            Safety cap on the number of convergence passes.  In practice
            1–2 passes suffice for typical physics events.

        Returns
        -------
        list of list of Particle
        """
        n = len(self.particles)
        uf = UnionFind(n)

        # Build representative Particle objects: shallow copies whose point_cloud
        # will grow as partitions merge.  All scalar attributes (id, pdg, sem_type,
        # parent_id, root_id, …) remain those of the original particle.
        reps: Dict[int, Particle] = {}
        for idx, p in enumerate(self.particles):
            rep = copy.copy(p)
            rep.point_cloud = p.point_cloud[:, :3].astype(np.float32)
            rep._cloud_parts = [rep.point_cloud]  # deferred-concat chunks
            reps[idx] = rep   # keyed by UF root index

        # rep_lookup: original_particle_id → current representative Particle
        # Multiple original particles share the same representative once merged.
        rep_lookup: Dict[int, Particle] = {
            p.id: reps[self.id_to_idx[p.id]] for p in self.particles
        }

        # rep_members: id(rep) → list of original particle IDs in that partition.
        # Allows O(|partition|) rep_lookup update instead of O(n) full scan.
        rep_members: Dict[int, List[int]] = {
            id(reps[self.id_to_idx[p.id]]): [p.id] for p in self.particles
        }

        def _get_cloud(rep: Particle) -> np.ndarray:
            """Return the materialized point cloud, collapsing chunks lazily."""
            parts = rep._cloud_parts
            if len(parts) == 1:
                return parts[0]
            cloud = np.concatenate(parts, axis=0)
            rep._cloud_parts = [cloud]
            rep.point_cloud = cloud
            return cloud

        # Bbox cache: keyed by UF root index.  Updated in O(1) on every merge
        # as the merged bbox = (min of mins, max of maxes).  Used to skip
        # parent KDTree builds for pairs that are provably not touching.
        self._rep_bmin: Dict[int, np.ndarray] = {}
        self._rep_bmax: Dict[int, np.ndarray] = {}
        for idx, rep in reps.items():
            c = rep.point_cloud  # already xyz‑only float32
            if len(c) > 0:
                self._rep_bmin[idx] = c.min(axis=0)
                self._rep_bmax[idx] = c.max(axis=0)
            else:
                self._rep_bmin[idx] = np.array([ np.inf,  np.inf,  np.inf], dtype=np.float32)
                self._rep_bmax[idx] = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        self._D_sq = float(self.checker.D_squared)

        if verbose:
            print(f"  Partitioning {n} particles with {len(conditions)} condition(s)")

        total_merges = 0
        # Per-condition timing and merge-count accumulators (cumulative across passes).
        _cond_t: Dict[str, dict] = {
            cond.name: dict(time_s=0.0, n_unconditional=0, n_candidates=0,
                            n_resolved=0, n_touching=0, n_merged=0)
            for cond in conditions
        }

        for pass_num in range(1, max_passes + 1):
            pass_merges = 0

            for condition in conditions:
                _t_cond0 = time.perf_counter()  # per-condition wall-clock start
                # ---- Unconditional merges (e.g. photon decay: no proximity needed) ----
                unconditional = condition.get_unconditional_merges(self, rep_lookup)
                _n_unc = len(unconditional)
                for child_id, parent_id in unconditional:
                    child_idx  = self.id_to_idx[child_id]
                    parent_idx = self.id_to_idx[parent_id]
                    r_child  = uf.find(child_idx)
                    r_parent = uf.find(parent_idx)
                    if r_child == r_parent:
                        continue

                    child_rep  = reps[r_child]
                    parent_rep = reps[r_parent]

                    stage = f"{condition.name}: Unconditional Merge (pass {pass_num})"
                    # Deferred concat: accumulate chunks, no allocation yet
                    parent_rep._cloud_parts.extend(child_rep._cloud_parts)
                    if self.diagnostics.enabled:
                        self.diagnostics.record(
                            child_id, parent_id,
                            MergeOutcome.MERGED,
                            "Topology-based merge (no proximity check)",
                            stage=stage,
                        )
                    self.diagnostics.record_actual_merge(child_id, parent_id)
                    uf.union(r_child, r_parent)
                    new_root = uf.find(r_child)
                    # O(|partition|) update via reverse members map
                    child_members = rep_members.pop(id(child_rep))
                    for orig_id in child_members:
                        rep_lookup[orig_id] = parent_rep
                    rep_members[id(parent_rep)].extend(child_members)
                    if new_root == r_parent:
                        del reps[r_child]
                    else:
                        reps[new_root] = parent_rep
                        del reps[r_parent]
                    # Update bbox cache (O(1): element-wise min/max of the two bboxes).
                    self._rep_bmin[new_root] = np.minimum(self._rep_bmin.pop(r_child), self._rep_bmin.pop(r_parent))
                    self._rep_bmax[new_root] = np.maximum(self._rep_bmax.pop(r_child), self._rep_bmax.pop(r_parent))
                    pass_merges += 1

                # ---- Proximity-based candidates ----
                # Ask the condition which partition pairs are candidates this pass.
                # Returns (child_rep_id, parent_rep_id): child merges INTO parent.
                candidates = condition.get_rep_candidates(self, rep_lookup)
                _n_cands = len(candidates)
                touching_pairs: List[Tuple[int, int]] = []

                # Resolve to UF roots and deduplicate stale/duplicate rep-pairs.
                resolved: List[Tuple[int, int, int, int]] = []
                seen_pairs: Set[Tuple[int, int]] = set()
                for child_id, parent_id in candidates:
                    child_idx  = self.id_to_idx[child_id]
                    parent_idx = self.id_to_idx[parent_id]
                    r_child  = uf.find(child_idx)
                    r_parent = uf.find(parent_idx)
                    if r_child == r_parent:
                        continue
                    key = (r_child, r_parent)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    resolved.append((child_id, parent_id, r_child, r_parent))

                # Group by parent rep so each parent KDTree is built only once.
                # Bbox pre-filter: pairs whose bboxes are >D apart skip the KDTree
                # entirely (no cloud materialization for that parent group).
                by_parent: Dict[int, List[int]] = defaultdict(list)
                for i, (_, _, r_child, r_parent) in enumerate(resolved):
                    if r_child in self._rep_bmin and r_parent in self._rep_bmin:
                        delta = np.maximum(0.0, np.maximum(
                            self._rep_bmin[r_child] - self._rep_bmax[r_parent],
                            self._rep_bmin[r_parent] - self._rep_bmax[r_child]))
                        if float(np.dot(delta, delta)) > self._D_sq:
                            continue  # provably not touching; skip KDTree
                    by_parent[r_parent].append(i)

                hit = [False] * len(resolved)
                stage = f"{condition.name}: Partition Proximity (pass {pass_num})"
                # Collect groups, then dispatch all parents in parallel.
                # Multi-thread backends build each parent's KDTree in its own
                # thread; single-thread backend falls back to sequential.
                _groups: List = []
                _group_idxs: List = []
                for r_parent, idxs in by_parent.items():
                    parent_cloud = _get_cloud(reps[r_parent])
                    child_clouds  = [_get_cloud(reps[resolved[i][2]]) for i in idxs]
                    _groups.append((child_clouds, parent_cloud))
                    _group_idxs.append(idxs)
                for idxs, batch_results in zip(
                    _group_idxs,
                    self.checker.batch_check_multi_cloud_proximity(_groups),
                ):
                    for i, touching in zip(idxs, batch_results):
                        hit[i] = touching

                for (child_id, parent_id, _, _), touching in zip(resolved, hit):
                    if touching:
                        touching_pairs.append((child_id, parent_id))
                        if self.diagnostics.enabled:
                            self.diagnostics.record(
                                child_id, parent_id,
                                MergeOutcome.MERGED,
                                f"Partition clouds touching within {self.D}",
                                stage=stage,
                            )
                    elif self.diagnostics.enabled:
                        self.diagnostics.record(
                            child_id, parent_id,
                            MergeOutcome.NOT_TOUCHING,
                            f"Partition cloud distance > {self.D}",
                            stage=stage,
                        )

                # Post-filter (e.g. one-merge-per-kLEScatter)
                merge_pairs = condition.post_filter(self, touching_pairs)
                # Snapshot counts for print_timing (resolved / touching / merged after post-filter)
                _n_res   = len(resolved)
                _n_touch = len(touching_pairs)
                _n_merge = len(merge_pairs)

                # Apply accepted merges
                for child_id, parent_id in merge_pairs:
                    child_idx  = self.id_to_idx[child_id]
                    parent_idx = self.id_to_idx[parent_id]
                    r_child  = uf.find(child_idx)
                    r_parent = uf.find(parent_idx)
                    if r_child == r_parent:
                        continue   # already merged by an earlier pair in this batch

                    child_rep  = reps[r_child]
                    parent_rep = reps[r_parent]

                    # Deferred concat: accumulate chunks, no allocation yet
                    parent_rep._cloud_parts.extend(child_rep._cloud_parts)

                    # Record the confirmed merge direction for diagnostics
                    self.diagnostics.record_actual_merge(child_id, parent_id)

                    # Merge child into parent in the UF structure
                    uf.union(r_child, r_parent)
                    new_root = uf.find(r_child)  # equals r_parent OR r_child

                    # O(|partition|) update via reverse members map
                    child_members = rep_members.pop(id(child_rep))
                    for orig_id in child_members:
                        rep_lookup[orig_id] = parent_rep
                    rep_members[id(parent_rep)].extend(child_members)

                    # Tidy up reps dict: parent_rep must live under new_root.
                    if new_root == r_parent:
                        del reps[r_child]
                    else:
                        # UF elected r_child as new root due to rank; remap.
                        reps[new_root] = parent_rep   # reps[r_child] = parent_rep
                        del reps[r_parent]
                    # Update bbox cache (O(1)).
                    self._rep_bmin[new_root] = np.minimum(self._rep_bmin.pop(r_child), self._rep_bmin.pop(r_parent))
                    self._rep_bmax[new_root] = np.maximum(self._rep_bmax.pop(r_child), self._rep_bmax.pop(r_parent))

                    pass_merges += 1

                # Accumulate per-condition stats for Pipeline.print_timing
                _ct = _cond_t[condition.name]
                _ct['time_s']          += time.perf_counter() - _t_cond0
                _ct['n_unconditional'] += _n_unc
                _ct['n_candidates']    += _n_cands
                _ct['n_resolved']      += _n_res
                _ct['n_touching']      += _n_touch
                _ct['n_merged']        += _n_merge

            if verbose:
                print(f"  Incremental pass {pass_num}: {pass_merges} merge(s)")

            total_merges += pass_merges
            if pass_merges == 0:
                break

        if verbose:
            print(f"  Converged after {pass_num} pass(es), {total_merges} total merge(s)")

        # Expose per-condition stats to Pipeline.print_timing
        self._condition_timing = _cond_t

        partitions_dict: Dict = defaultdict(list)
        for idx, p in enumerate(self.particles):
            root = uf.find(idx)
            partitions_dict[root].append(p)

        # Materialize any lazily-deferred point clouds in the surviving reps.
        for rep in reps.values():
            if len(rep._cloud_parts) > 1:
                rep.point_cloud = np.concatenate(rep._cloud_parts, axis=0)
                rep._cloud_parts = [rep.point_cloud]

        # Stamp member_ids on every surviving representative so callers can
        # trace back which original preprocessed particle IDs were absorbed.
        for rep in reps.values():
            rep.member_ids = list(rep_members[id(rep)])

        # Persist rep_lookup so callers can retrieve the representative
        # Particle for any original particle ID via get_representative().
        self.rep_lookup = rep_lookup

        return list(partitions_dict.values())

    def get_representative(self, particle_id: int):
        """
        Return the representative :class:`~pysupera.data.Particle` for the
        partition that contains *particle_id*.

        The representative is the partition "root" produced by the last call to
        :meth:`partition` (or :meth:`partition_combined`).  It is a shallow copy
        of the original particle that survived as the absorbing side during
        merging, and whose ``point_cloud`` is the union of every constituent
        particle's point cloud.

        Because all members of a partition share the same representative, you
        can pass the ID of **any** particle in the partition::

            rep = alg.get_representative(ps[97][0].id)

        Parameters
        ----------
        particle_id : int
            ID of any particle in the event.

        Returns
        -------
        Particle
            The representative particle for that partition.

        Raises
        ------
        AttributeError
            If called before any :meth:`partition` or :meth:`partition_combined`
            invocation (``rep_lookup`` not yet built).
        KeyError
            If *particle_id* is not found in the current ``rep_lookup``.
        """
        if not hasattr(self, 'rep_lookup'):
            raise AttributeError(
                "rep_lookup is not available yet. "
                "Call partition() or partition_combined() first."
            )
        return self.rep_lookup[particle_id]

    def partition_by_sem_type(self, partitions: List[List]) -> dict:
        """
        Group a partition list by the semantic type of each partition's
        representative particle.

        The representative is looked up via :meth:`get_representative` using
        the first member of each partition as the key.

        Parameters
        ----------
        partitions : list of list of Particle
            As returned by :meth:`partition` or :meth:`partition_combined`.

        Returns
        -------
        dict[SemanticType, list[list[Particle]]]
            Keys are :class:`~pysupera.utils.SemanticType` members that
            are actually present; values are lists of partitions whose
            representative carries that semantic type.
        """
        from collections import defaultdict as _dd
        grouped = _dd(list)
        for part in partitions:
            rep = self.get_representative(part[0].id)
            grouped[rep.sem_type].append(part)
        return dict(grouped)

    def _print_partition_stats(self, partitions: List[List], elapsed_time: float):
        """
        Print a formatted statistics table for the resulting partitions.

        Displays total partition count, size distribution (min, max, mean,
        median), singleton count, large-partition count, a bucketed size
        histogram, and wall-clock elapsed time.

        Parameters
        ----------
        partitions : list of list of Particle
            Partitions as returned by ``_build_partitions``.
        elapsed_time : float
            Wall-clock time in seconds taken by the calling partitioning
            method, reported in the output footer.
        """
        sizes = [len(p) for p in partitions]
        
        print(f"\n{'='*70}")
        print(f"Partitioning Complete")
        print(f"{'='*70}")
        print(f"Total partitions: {len(partitions)}")
        print(f"Partition size - min: {min(sizes)}, max: {max(sizes)}, "
              f"mean: {np.mean(sizes):.2f}, median: {np.median(sizes):.0f}")
        print(f"Singleton partitions (size=1): {sum(1 for s in sizes if s == 1)}")
        print(f"Large partitions (size>10): {sum(1 for s in sizes if s > 10)}")
        
        # Show size distribution
        size_counts = defaultdict(int)
        for s in sizes:
            if s <= 5:
                size_counts[s] += 1
            elif s <= 10:
                size_counts['6-10'] += 1
            elif s <= 20:
                size_counts['11-20'] += 1
            else:
                size_counts['>20'] += 1
        
        print(f"\nPartition size distribution:")
        for size in sorted([k for k in size_counts.keys() if isinstance(k, int)]):
            print(f"  Size {size}: {size_counts[size]} partitions")
        for size in ['6-10', '11-20', '>20']:
            if size in size_counts:
                print(f"  Size {size}: {size_counts[size]} partitions")
        
        print(f"\nElapsed time: {elapsed_time:.3f} seconds")
        print(f"{'='*70}\n")
    
    def cleanup(self):
        """
        Release all resources held by the partitioner.

        Calls ``checker.cleanup()`` to free backend-specific resources
        (GPU memory, KD-trees, thread pools), clears the diagnostics
        buffer, and empties the internal lookup dictionaries. After calling
        this method the object should not be used for further partitioning.

        .. note::
            This method is called automatically when the partitioner is used
            as a context manager, so explicit calls are only needed outside
            a ``with`` block.
        """
        # Clean up backend (GPU memory, KD-trees, etc.)
        if hasattr(self, 'checker') and self.checker is not None:
            self.checker.cleanup()
            self.checker = None
        
        # Clear diagnostic data
        if hasattr(self, 'diagnostics') and self.diagnostics is not None:
            self.diagnostics.clear()
        
        # Clear lookup structures
        if hasattr(self, 'particle_lookup'):
            self.particle_lookup.clear()
        
        if hasattr(self, 'id_to_idx'):
            self.id_to_idx.clear()
        
        if hasattr(self, 'children_map'):
            self.children_map.clear()
        
        if hasattr(self, 'ancestor_map'):
            self.ancestor_map.clear()
        
        print("Partitioner resources cleaned up")

    # Context manager protocol for automatic cleanup
    def __enter__(self):
        """
        Enter the runtime context and return the partitioner itself.

        Returns
        -------
        ParticlePartitioner
            The partitioner instance, enabling the ``as`` clause in a
            ``with`` statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context and release all resources.

        Calls ``cleanup()`` unconditionally so that backend resources are
        freed even if an exception was raised inside the ``with`` block.

        Parameters
        ----------
        exc_type : type or None
            Exception class, or ``None`` if no exception occurred.
        exc_val : Exception or None
            Exception instance, or ``None`` if no exception occurred.
        exc_tb : traceback or None
            Traceback object, or ``None`` if no exception occurred.

        Returns
        -------
        bool
            Always ``False``; exceptions are propagated normally.
        """
        self.cleanup()
        return False  # Don't suppress exceptions