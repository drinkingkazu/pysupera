from .proxck import ProximityChecker, CPUSingleThreadChecker, CPUMultiThreadChecker, GPUChecker, UnionFind
from .data import Particle
from .utils import SemanticType
from typing import List, Tuple, Callable
import time
from collections import defaultdict
import numpy as np


# ============================================================================
# Enhanced Partitioner with Metadata-Based Conditions
# ============================================================================

class ParticlePartitioner:
    """
    Main partitioner class that works with any backend.
    Supports complex metadata-based partitioning conditions.
    """
    
    def __init__(self, 
                 particles, 
                 distance_threshold: float,
                 backend: str = 'gpu',
                 n_jobs: int = -1):
        """
        Initialize partitioner with specified backend.
        
        Args:
            particles: List of Particle objects
            distance_threshold: Distance threshold for "touching" (D)
            backend: 'cpu-single', 'cpu-multi', or 'gpu'
            n_jobs: Number of CPU threads (only for cpu-multi)
        """
        self.particles = particles
        self.D = distance_threshold
        self.backend_name = backend
        
        # Build lookup structures
        self.particle_lookup = {p.id: p for p in particles}
        self.id_to_idx = {p.id: idx for idx, p in enumerate(particles)}
        
        # Build parent-child and ancestor relationships
        self._build_relationships()
        
        # Initialize appropriate backend
        self.checker = self._create_backend(backend, n_jobs)
        self.checker.initialize(particles)
    
    def _build_relationships(self):
        """Build parent-child and ancestor lookup structures"""
        self.children_map = defaultdict(list)  # parent_id -> [child_ids]
        self.ancestor_map = defaultdict(list)  # ancestor_id -> [particle_ids]
        
        for p in self.particles:
            if p.parent_id is not None:
                self.children_map[p.parent_id].append(p.id)
            
            if p.ancestor_id is not None:
                self.ancestor_map[p.ancestor_id].append(p.id)
    
    def _create_backend(self, backend: str, n_jobs: int) -> ProximityChecker:
        """Factory method to create appropriate backend"""
        if backend == 'cpu-single':
            return CPUSingleThreadChecker(self.D)
        elif backend == 'cpu-multi':
            return CPUMultiThreadChecker(self.D, n_jobs=n_jobs)
        elif backend == 'gpu':
            return GPUChecker(self.D)
        else:
            raise ValueError(f"Unknown backend: {backend}. "
                           f"Choose from: 'cpu-single', 'cpu-multi', 'gpu'")
    
    # ========================================================================
    # Condition 1: PDG 22 or 11 with same ancestor
    # ========================================================================
    
    def partition_condition_1(self, verbose: bool = True) -> List[List]:
        """
        Condition 1: For particles with PDG value 22 or 11:
        - Identify immediate parent with the same PDG value
        - Merge if they are touching
        - Never merge particles with different ancestor_id
        
        Returns:
            List of partitions, where each partition is a list of Particle objects
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Partitioning Condition 1: PDG 22/11 parent-child merging")
            print(f"Backend: {self.backend_name}")
            print(f"{'='*70}")
        
        start_time = time.time()
        
        # Stage 1: Build candidate pairs
        if verbose:
            print("Stage 1: Building candidate pairs (parent-child, same PDG)...")
        
        candidates = self._build_candidates_condition_1()
        
        if verbose:
            print(f"  Candidate pairs: {len(candidates)}")
        
        if len(candidates) == 0:
            return [[p] for p in self.particles]
        
        # Stage 2: Filter by ancestor_id (before expensive proximity checks)
        if verbose:
            print("Stage 2: Filtering by ancestor_id...")
        
        ancestor_filtered = self._filter_by_ancestor(candidates)
        
        if verbose:
            print(f"  After ancestor filter: {len(ancestor_filtered)}")
        
        # Stage 3: Check proximity (touching)
        if verbose:
            print("Stage 3: Checking proximity (touching)...")
        
        merge_pairs = self.checker.batch_check_proximity(ancestor_filtered)
        
        if verbose:
            print(f"  Touching pairs found: {len(merge_pairs)}")
        
        # Stage 4: Build partitions
        if verbose:
            print("Stage 4: Building partitions with Union-Find...")
        
        partitions = self._build_partitions(merge_pairs)
        
        elapsed = time.time() - start_time
        
        if verbose:
            self._print_partition_stats(partitions, elapsed)
        
        return partitions
    
    def _build_candidates_condition_1(self) -> List[Tuple[int, int]]:
        """
        Build candidate pairs for condition 1:
        - Child has PDG 22 or 11
        - Parent has same PDG as child
        """
        candidates = []
        target_pdgs = {22, 11}
        
        for p in self.particles:
            # Check if particle has target PDG
            if p.pdg not in target_pdgs:
                continue
            
            # Check if has parent
            if p.parent_id is None or p.parent_id not in self.particle_lookup:
                continue

            # Check if the point cloud is empty
            if p.point_cloud.shape[0] == 0:
                continue
            
            parent = self.particle_lookup[p.parent_id]
            
            # Check if parent has same PDG
            if parent.pdg == p.pdg:
                candidates.append((parent.id, p.id))
        
        return candidates
    
    def _filter_by_ancestor(self, candidates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filter pairs to only include those with same ancestor_id"""
        filtered = []
        
        for id1, id2 in candidates:
            p1 = self.particle_lookup[id1]
            p2 = self.particle_lookup[id2]
            
            if p1.ancestor_id == p2.ancestor_id:
                filtered.append((id1, id2))
        
        return filtered
    
    # ========================================================================
    # Condition 2: SemID 4 merging with PDG 11/22
    # ========================================================================
    
    def partition_condition_2(self, verbose: bool = True) -> List[List]:
        """
        Condition 2: For particles with SemID type 4:
        - If touching a particle with PDG 11 or 22, merge them
        - Can merge with any one of multiple touching particles
        
        Returns:
            List of partitions, where each partition is a list of Particle objects
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Partitioning Condition 2: SemID 4 merging with PDG 11/22")
            print(f"Backend: {self.backend_name}")
            print(f"{'='*70}")
        
        start_time = time.time()
        
        # Stage 1: Build candidate pairs
        if verbose:
            print("Stage 1: Building candidate pairs (SemID 4 with PDG 11/22)...")
        
        candidates = self._build_candidates_condition_2()
        
        if verbose:
            print(f"  Candidate pairs: {len(candidates)}")
        
        if len(candidates) == 0:
            return [[p] for p in self.particles]
        
        # Stage 2: Check proximity (touching)
        if verbose:
            print("Stage 2: Checking proximity (touching)...")
        
        merge_pairs = self.checker.batch_check_proximity(candidates)
        
        if verbose:
            print(f"  Touching pairs found: {len(merge_pairs)}")
        
        # Stage 3: Select one merge per SemID 4 particle (if multiple options)
        if verbose:
            print("Stage 3: Selecting one merge per SemID 4 particle...")
        
        selected_pairs = self._select_one_merge_per_sem4(merge_pairs)
        
        if verbose:
            print(f"  Selected pairs: {len(selected_pairs)}")
        
        # Stage 4: Build partitions
        if verbose:
            print("Stage 4: Building partitions with Union-Find...")
        
        partitions = self._build_partitions(selected_pairs)
        
        elapsed = time.time() - start_time
        
        if verbose:
            self._print_partition_stats(partitions, elapsed)
        
        return partitions
    
    def _build_candidates_condition_2(self) -> List[Tuple[int, int]]:
        """
        Build candidate pairs for condition 2:
        - One particle has SemID 4
        - Other particle has PDG 11 or 22
        """
        candidates = []
        target_pdgs = {22, 11}
        
        # Find all SemID 4 particles with non-empty point cloud
        sem4_particles = [p for p in self.particles if p.sem_type == SemanticType.kMichel and p.point_cloud.shape[0] > 0 ]
        
        # Find all PDG 11/22 particles with non-empty point cloud
        pdg_particles = [p for p in self.particles if p.pdg in target_pdgs and p.point_cloud.shape[0] > 0]
        
        # Create all possible pairs
        for sem4_p in sem4_particles:
            for pdg_p in pdg_particles:
                candidates.append((sem4_p.id, pdg_p.id))
        
        return candidates
    
    def _select_one_merge_per_sem4(self, merge_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        For each SemID 4 particle that touches multiple PDG 11/22 particles,
        select only one to merge with (e.g., the first one found).
        """
        # Group by SemID 4 particle
        sem4_merges = defaultdict(list)
        
        for id1, id2 in merge_pairs:
            p1 = self.particle_lookup[id1]
            p2 = self.particle_lookup[id2]
            
            if p1.sem_type == SemanticType.kMichel:
                sem4_merges[p1.id].append((id1, id2))
            elif p2.sem_type == SemanticType.kMichel:
                sem4_merges[id2].append((id1, id2))
        
        # Select one merge per SemID 4 particle
        selected = []
        for sem4_id, pairs in sem4_merges.items():
            # Select first touching partner (could use other criteria)
            selected.append(pairs[0])
        
        return selected
    
    # ========================================================================
    # Combined Partitioning: Apply Multiple Conditions
    # ========================================================================
    
    def partition_combined(self, 
                          conditions: List[str] = ['condition_1', 'condition_2'],
                          verbose: bool = True) -> List[List]:
        """
        Apply multiple partitioning conditions sequentially.
        
        Args:
            conditions: List of condition names to apply in order
            verbose: Print progress information
        
        Returns:
            List of partitions after applying all conditions
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Combined Partitioning: Applying {len(conditions)} conditions")
            print(f"Backend: {self.backend_name}")
            print(f"{'='*70}")
        
        start_time = time.time()
        all_merge_pairs = []
        
        # Collect merge pairs from all conditions
        for i, condition_name in enumerate(conditions, 1):
            if verbose:
                print(f"\n--- Condition {i}/{len(conditions)}: {condition_name} ---")
            
            if condition_name == 'condition_1':
                candidates = self._build_candidates_condition_1()
                candidates = self._filter_by_ancestor(candidates)
            elif condition_name == 'condition_2':
                candidates = self._build_candidates_condition_2()
            else:
                raise ValueError(f"Unknown condition: {condition_name}")
            
            if verbose:
                print(f"  Candidates: {len(candidates)}")
            
            merge_pairs = self.checker.batch_check_proximity(candidates)
            
            if condition_name == 'condition_2':
                merge_pairs = self._select_one_merge_per_sem4(merge_pairs)
            
            if verbose:
                print(f"  Merge pairs: {len(merge_pairs)}")
            
            all_merge_pairs.extend(merge_pairs)
        
        # Build partitions from all merge pairs
        if verbose:
            print(f"\nBuilding final partitions from {len(all_merge_pairs)} total merge pairs...")
        
        partitions = self._build_partitions(all_merge_pairs)
        
        elapsed = time.time() - start_time
        
        if verbose:
            self._print_partition_stats(partitions, elapsed)
        
        return partitions
    
    # ========================================================================
    # General Purpose Partitioning with Custom Filters
    # ========================================================================
    
    def partition_custom(self,
                        candidate_filter: Callable[[Particle, Particle], bool],
                        verbose: bool = True) -> List[List]:
        """
        General-purpose partitioning with custom filter function.
        
        Args:
            candidate_filter: Function that takes two particles and returns True
                            if they should be considered for merging
            verbose: Print progress information
        
        Returns:
            List of partitions
        
        Example:
            # Custom condition: merge particles with same PDG and touching
            def my_filter(p1, p2):
                return p1.pdg == p2.pdg
            
            partitions = partitioner.partition_custom(my_filter)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Custom Partitioning")
            print(f"Backend: {self.backend_name}")
            print(f"{'='*70}")
        
        start_time = time.time()
        
        # Build all possible pairs and filter with custom function
        if verbose:
            print("Stage 1: Building candidate pairs with custom filter...")
        
        candidates = []
        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                if candidate_filter(p1, p2):
                    candidates.append((p1.id, p2.id))
        
        if verbose:
            print(f"  Candidate pairs: {len(candidates)}")
        
        if len(candidates) == 0:
            return [[p] for p in self.particles]
        
        # Check proximity
        if verbose:
            print("Stage 2: Checking proximity (touching)...")
        
        merge_pairs = self.checker.batch_check_proximity(candidates)
        
        if verbose:
            print(f"  Touching pairs found: {len(merge_pairs)}")
        
        # Build partitions
        if verbose:
            print("Stage 3: Building partitions...")
        
        partitions = self._build_partitions(merge_pairs)
        
        elapsed = time.time() - start_time
        
        if verbose:
            self._print_partition_stats(partitions, elapsed)
        
        return partitions

    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _build_partitions(self, merge_pairs: List[Tuple[int, int]]) -> List[List]:
        """
        Build final partitions using Union-Find algorithm.
        
        Args:
            merge_pairs: List of (id1, id2) tuples to merge
        
        Returns:
            List of partitions, where each partition is a list of Particle objects
        """
        n = len(self.particles)
        uf = UnionFind(n)
        
        # Perform unions for all merge pairs
        for id1, id2 in merge_pairs:
            idx1 = self.id_to_idx[id1]
            idx2 = self.id_to_idx[id2]
            uf.union(idx1, idx2)
        
        # Extract partitions by grouping particles with same root
        partitions_dict = defaultdict(list)
        for idx, p in enumerate(self.particles):
            root = uf.find(idx)
            partitions_dict[root].append(p)
        
        return list(partitions_dict.values())
    
    def _print_partition_stats(self, partitions: List[List], elapsed_time: float):
        """Print statistics about the partitions"""
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
        """Free backend resources"""
        self.checker.cleanup()