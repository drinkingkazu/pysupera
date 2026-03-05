import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Callable, Optional
import time

# ============================================================================
# Core Data Structures (Backend-Agnostic)
# ============================================================================

class UnionFind:
    """
    Union-Find (Disjoint Set Union) with path compression and union by rank.

    Supports amortised near-O(1) :meth:`find` and :meth:`union` operations
    via path compression (in :meth:`find`) and union by rank (in
    :meth:`union`).

    Parameters
    ----------
    n : int
        Number of elements; valid element indices are ``0 .. n-1``.
    """

    def __init__(self, n: int):
        """
        Parameters
        ----------
        n : int
            Number of elements to manage.
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """
        Return the root of the set containing *x* with path compression.

        Parameters
        ----------
        x : int

        Returns
        -------
        int
            Root representative of the set containing *x*.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing *x* and *y*.

        Uses union by rank to keep trees shallow.

        Parameters
        ----------
        x, y : int

        Returns
        -------
        bool
            ``True`` if the merge was performed (i.e. *x* and *y* were in
            different sets); ``False`` if they were already in the same set.
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

# ============================================================================
# Abstract Base Class for Proximity Checkers
# ============================================================================

class ProximityChecker(ABC):
    """
    Abstract base class for all proximity-checking backends.

    Subclasses implement the interface for a specific hardware / library
    combination (CPU single-thread, CPU multi-thread, GPU + RAPIDS, GPU +
    CuPy, GPU + Numba, cell-hash variants, …).  The contract is:

    1. :meth:`initialize` is called **once** with the full particle list.
    2. :meth:`check_proximity` or :meth:`batch_check_proximity` is called
       repeatedly to evaluate merge candidates.
    3. :meth:`cleanup` is called when the partitioner is done to release
       resources (GPU memory etc.).

    Parameters
    ----------
    distance_threshold : float
        Proximity distance *D*.  Two particles are considered *touching*
        if any point in one cloud lies within *D* of any point in the other.
    """

    def __init__(self, distance_threshold: float):
        """
        Parameters
        ----------
        distance_threshold : float
        """
        self.D = distance_threshold
        self.D_squared = distance_threshold ** 2
        # IDs of particles whose point cloud has zero points.  Every
        # check_proximity implementation returns False immediately for these.
        self.empty_ids: set = set()

    @abstractmethod
    def initialize(self, particles):
        """
        Build or upload any backend-specific data structures.

        Must be called exactly once before any proximity queries.

        Parameters
        ----------
        particles : list of Particle
        """
        pass

    @abstractmethod
    def check_proximity(self, id1, id2) -> bool:
        """
        Return ``True`` if clouds *id1* and *id2* are within *D* of each other.

        Parameters
        ----------
        id1, id2 : int
            Particle IDs as used in :meth:`initialize`.

        Returns
        -------
        bool
        """
        pass

    @abstractmethod
    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        """
        Return the subset of *candidate_pairs* whose clouds are within *D*.

        Parameters
        ----------
        candidate_pairs : list of tuple[int, int]

        Returns
        -------
        list of tuple[int, int]
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Release all backend-specific resources (memory, GPU buffers, etc.)."""
        pass

    def compute_bbox(self, point_cloud) -> dict:
        """
        Compute the axis-aligned bounding box of a point cloud.

        Parameters
        ----------
        point_cloud : ndarray, shape (n, ≥3)

        Returns
        -------
        dict
            ``{'min': ndarray (3,), 'max': ndarray (3,)}``.
        """
        if len(point_cloud) == 0:
            return {'min': None, 'max': None}
        return {
            'min': np.min(point_cloud[:, :3], axis=0),
            'max': np.max(point_cloud[:, :3], axis=0)
        }

    def compute_centroid(self, point_cloud) -> np.ndarray:
        """
        Compute the centroid (mean position) of a point cloud.

        Parameters
        ----------
        point_cloud : ndarray, shape (n, ≥3)

        Returns
        -------
        ndarray, shape (3,)
        """
        if len(point_cloud) == 0:
            return None
        return np.mean(point_cloud[:, :3], axis=0)

    def bbox_min_distance(self, bb1: dict, bb2: dict) -> float:
        """
        Return the minimum possible Euclidean distance between two bounding boxes.

        Returns 0 when the boxes overlap.

        Parameters
        ----------
        bb1, bb2 : dict
            Bounding boxes as returned by :meth:`compute_bbox`.

        Returns
        -------
        float
        """
        delta = np.maximum(0, np.maximum(bb1['min'] - bb2['max'],
                                         bb2['min'] - bb1['max']))
        return np.sqrt(np.sum(delta**2))

    def check_cloud_proximity(self, cloud_a: np.ndarray,
                               cloud_b: np.ndarray) -> bool:
        """
        Return ``True`` if any point of *cloud_a* lies within *D* of any
        point of *cloud_b*.

        Unlike :meth:`check_proximity`, this method operates on **raw numpy
        arrays** rather than pre-indexed particle IDs.  It is used by the
        partition-level proximity mode of the partitioner, where each cloud
        represents the merged point cloud of a whole partition rather than a
        single particle.

        The default implementation applies a cheap bounding-box rejection
        first, then queries a ``scipy.KDTree`` built on-the-fly for
        *cloud_b*.  If scipy is unavailable it falls back to chunked NumPy
        brute force.  GPU subclasses override this with a device-resident
        implementation.

        Parameters
        ----------
        cloud_a : ndarray, shape (n, ≥3)
            First point cloud.  Only the first three columns (x, y, z) are used.
        cloud_b : ndarray, shape (m, ≥3)
            Second point cloud.

        Returns
        -------
        bool
        """
        if len(cloud_a) == 0 or len(cloud_b) == 0:
            return False
        a = cloud_a[:, :3]
        b = cloud_b[:, :3]

        # Cheap bbox rejection
        min_a, max_a = a.min(axis=0), a.max(axis=0)
        min_b, max_b = b.min(axis=0), b.max(axis=0)
        delta = np.maximum(0.0, np.maximum(min_a - max_b, min_b - max_a))
        if float(np.dot(delta, delta)) > self.D_squared:
            return False

        try:
            from scipy.spatial import KDTree
            dists, _ = KDTree(b).query(a, k=1, distance_upper_bound=self.D + 1e-9)
            return bool(np.any(dists <= self.D))
        except ImportError:
            pass

        # Numpy chunked brute force fallback
        chunk = 256
        for start in range(0, len(a), chunk):
            a_chunk = a[start : start + chunk]
            diff = a_chunk[:, np.newaxis, :] - b[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            if float(dist_sq.min()) <= self.D_squared:
                return True
        return False

    def batch_check_cloud_proximity(
        self,
        child_clouds: List[np.ndarray],
        parent_cloud: np.ndarray,
    ) -> List[bool]:
        """
        Test whether each cloud in *child_clouds* is within *D* of *parent_cloud*.

        The default implementation calls :meth:`check_cloud_proximity` once per
        child.  CPU subclasses override this to build the parent KDTree once for
        all children, avoiding redundant tree construction.

        Parameters
        ----------
        child_clouds : list of ndarray, each shape (n_i, ≥3)
        parent_cloud : ndarray, shape (m, ≥3)

        Returns
        -------
        list of bool
        """
        return [self.check_cloud_proximity(c, parent_cloud) for c in child_clouds]


# ============================================================================
# CPU Single-Threaded Implementation
# ============================================================================

class CPUSingleThreadChecker(ProximityChecker):
    """
    Single-threaded CPU proximity checker using a scipy ``KDTree`` index.

    One ``KDTree`` is built per particle during :meth:`initialize`.  Each
    :meth:`check_proximity` call queries the tree of cloud B against every
    point of cloud A and returns ``True`` if the nearest-neighbour distance
    is ≤ *D*.

    Requires ``scipy``.
    """
    
    def __init__(self, distance_threshold: float):
        super().__init__(distance_threshold)
        self.kdtrees = {}
        self.clouds = {}
        self.bboxes = {}
        self.centroids = {}
    
    def initialize(self, particles):
        """Build KD-trees and precompute bounding boxes"""
        from scipy.spatial import KDTree
        
        print("Initializing CPU single-threaded backend...")
        for p in particles:
            cloud_xyz = p.point_cloud[:, :3].astype(np.float32)
            self.clouds[p.id] = cloud_xyz
            if len(cloud_xyz) == 0:
                self.empty_ids.add(p.id)
                self.bboxes[p.id]    = {'min': None, 'max': None}
                self.centroids[p.id] = None
                continue
            self.kdtrees[p.id]   = KDTree(cloud_xyz)
            self.bboxes[p.id]    = self.compute_bbox(p.point_cloud)
            self.centroids[p.id] = self.compute_centroid(p.point_cloud)

        print(f"  Initialized {len(particles)} particles")
    
    def check_proximity(self, id1, id2) -> bool:
        """Check if any point in cloud1 is within D of any point in cloud2"""
        if id1 in self.empty_ids or id2 in self.empty_ids:
            return False
        cloud1 = self.clouds[id1]
        tree2 = self.kdtrees[id2]
        
        # Query nearest neighbor for each point in cloud1
        distances, _ = tree2.query(cloud1, k=1)
        
        return np.min(distances) <= self.D
    
    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        """Process pairs sequentially"""
        merge_pairs = []
        
        for id1, id2 in candidate_pairs:
            if self.check_proximity(id1, id2):
                merge_pairs.append((id1, id2))
        
        return merge_pairs
    
    def batch_check_cloud_proximity(
        self,
        child_clouds: List[np.ndarray],
        parent_cloud: np.ndarray,
    ) -> List[bool]:
        """Build the parent KDTree once and query all children against it."""
        if len(parent_cloud) == 0:
            return [False] * len(child_clouds)
        from scipy.spatial import KDTree
        b = parent_cloud[:, :3]
        min_b, max_b = b.min(axis=0), b.max(axis=0)
        tree = KDTree(b)
        results: List[bool] = []
        for child_cloud in child_clouds:
            if len(child_cloud) == 0:
                results.append(False)
                continue
            a = child_cloud[:, :3]
            min_a, max_a = a.min(axis=0), a.max(axis=0)
            delta = np.maximum(0.0, np.maximum(min_a - max_b, min_b - max_a))
            if float(np.dot(delta, delta)) > self.D_squared:
                results.append(False)
                continue
            dists, _ = tree.query(a, k=1, distance_upper_bound=self.D + 1e-9)
            results.append(bool(np.any(dists <= self.D)))
        return results

    def cleanup(self):
        """Clear data structures"""
        self.kdtrees.clear()
        self.clouds.clear()
        self.bboxes.clear()
        self.centroids.clear()


# ============================================================================
# CPU Multi-Threaded Implementation
# ============================================================================

class CPUMultiThreadChecker(ProximityChecker):
    """
    Multi-threaded CPU proximity checker using scipy ``KDTree`` + ``joblib``.

    KDTree construction is parallelised over particles during
    :meth:`initialize`, and :meth:`batch_check_proximity` dispatches
    individual pair checks to a ``joblib`` thread pool.

    Requires ``scipy`` and ``joblib``.

    Parameters
    ----------
    distance_threshold : float
        Proximity distance *D*.
    n_jobs : int, optional
        Number of parallel threads.  ``-1`` (default) means *use all cores*.
    """
    
    def __init__(self, distance_threshold: float, n_jobs: int = -1):
        super().__init__(distance_threshold)
        self.n_jobs = n_jobs  # -1 means use all available cores
        self.kdtrees = {}
        self.clouds = {}
        self.bboxes = {}
        self.centroids = {}
    
    def initialize(self, particles):
        """Build KD-trees in parallel"""
        from scipy.spatial import KDTree
        from joblib import Parallel, delayed
        
        print(f"Initializing CPU multi-threaded backend (n_jobs={self.n_jobs})...")
        
        def build_structures(p):
            cloud_xyz = p.point_cloud[:, :3].astype(np.float32)
            if len(cloud_xyz) == 0:
                return (p.id, cloud_xyz, None,
                        {'min': None, 'max': None}, None)
            return (
                p.id,
                cloud_xyz,
                KDTree(cloud_xyz),
                self.compute_bbox(p.point_cloud),
                self.compute_centroid(p.point_cloud)
            )
        
        # Parallel initialization
        results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(build_structures)(p) for p in particles
        )
        
        # Store results
        for pid, cloud, tree, bbox, centroid in results:
            self.clouds[pid] = cloud
            if tree is None:
                self.empty_ids.add(pid)
            else:
                self.kdtrees[pid] = tree
            self.bboxes[pid]    = bbox
            self.centroids[pid] = centroid
        
        print(f"  Initialized {len(particles)} particles")
    
    def check_proximity(self, id1, id2) -> bool:
        """Check if any point in cloud1 is within D of any point in cloud2"""
        if id1 in self.empty_ids or id2 in self.empty_ids:
            return False
        cloud1 = self.clouds[id1]
        tree2 = self.kdtrees[id2]
        
        distances, _ = tree2.query(cloud1, k=1)
        return np.min(distances) <= self.D
    
    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        """Process pairs in parallel"""
        from joblib import Parallel, delayed
        
        def check_pair(id1, id2):
            if self.check_proximity(id1, id2):
                return (id1, id2)
            return None
        
        # Parallel processing
        results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(check_pair)(id1, id2) for id1, id2 in candidate_pairs
        )
        
        # Filter out None results
        merge_pairs = [r for r in results if r is not None]
        return merge_pairs
    
    def batch_check_cloud_proximity(
        self,
        child_clouds: List[np.ndarray],
        parent_cloud: np.ndarray,
    ) -> List[bool]:
        """Build the parent KDTree once and query all children against it."""
        if len(parent_cloud) == 0:
            return [False] * len(child_clouds)
        from scipy.spatial import KDTree
        b = parent_cloud[:, :3]
        min_b, max_b = b.min(axis=0), b.max(axis=0)
        tree = KDTree(b)
        results: List[bool] = []
        for child_cloud in child_clouds:
            if len(child_cloud) == 0:
                results.append(False)
                continue
            a = child_cloud[:, :3]
            min_a, max_a = a.min(axis=0), a.max(axis=0)
            delta = np.maximum(0.0, np.maximum(min_a - max_b, min_b - max_a))
            if float(np.dot(delta, delta)) > self.D_squared:
                results.append(False)
                continue
            dists, _ = tree.query(a, k=1, distance_upper_bound=self.D + 1e-9)
            results.append(bool(np.any(dists <= self.D)))
        return results

    def cleanup(self):
        """Clear data structures"""
        self.kdtrees.clear()
        self.clouds.clear()
        self.bboxes.clear()
        self.centroids.clear()


# ============================================================================
# GPU Implementation
# ============================================================================

class GPUChecker(ProximityChecker):
    """
    GPU proximity checker using RAPIDS cuML nearest-neighbour index.

    Requires ``cuml`` (RAPIDS) and ``cupy``.  Install with::

        pip install --extra-index-url https://pypi.nvidia.com cuml-cu12 cupy-cuda12x

    All point clouds are uploaded to the GPU during :meth:`initialize` and
    a ``cuml.neighbors.NearestNeighbors`` index (brute-force L2, k=1) is
    built for each particle.  Both clouds and indices remain GPU-resident
    for the lifetime of the object.

    :meth:`batch_check_proximity` runs a vectorised bounding-box prefilter
    across all candidate pairs in one GPU pass before falling through to
    per-pair cuML queries, minimising the number of index lookups performed.

    Parameters
    ----------
    distance_threshold : float
        Proximity distance *D*.
    """

    def __init__(self, distance_threshold: float):
        super().__init__(distance_threshold)
        self.gpu_clouds: dict = {}    # pid -> cp.ndarray (n, 3) float32
        self.nn_indices: dict = {}    # pid -> fitted NearestNeighbors model
        self.gpu_bb_min: dict = {}    # pid -> cp.ndarray (3,)
        self.gpu_bb_max: dict = {}    # pid -> cp.ndarray (3,)

    def initialize(self, particles) -> None:
        """
        Upload all point clouds to GPU and build cuML nearest-neighbour indices.

        One ``NearestNeighbors(n_neighbors=1, algorithm='brute',
        metric='euclidean')`` index is fitted per particle.  Using brute
        force here is intentional: cuML's brute-force backend launches a
        single fused CUDA kernel that is faster than tree traversal for the
        fixed-radius "any neighbour within D?" query when the index cloud is
        large and the query set is the full opposing cloud.

        Parameters
        ----------
        particles : list
            Objects exposing ``.id`` and ``.point_cloud`` (at least 3 cols).
        """
        import cupy as cp
        from cuml.neighbors import NearestNeighbors

        print("Initializing GPU backend (RAPIDS cuML)...")
        for p in particles:
            cloud = cp.asarray(p.point_cloud[:, :3].astype(np.float32))
            self.gpu_clouds[p.id] = cloud
            if len(cloud) == 0:
                self.empty_ids.add(p.id)
                self.gpu_bb_min[p.id] = cp.full(3,  cp.inf, dtype=cp.float32)
                self.gpu_bb_max[p.id] = cp.full(3, -cp.inf, dtype=cp.float32)
                continue
            self.gpu_bb_min[p.id] = cp.min(cloud, axis=0)
            self.gpu_bb_max[p.id] = cp.max(cloud, axis=0)

            nn = NearestNeighbors(n_neighbors=1, algorithm='brute',
                                  metric='euclidean', output_type='cupy')
            nn.fit(cloud)
            self.nn_indices[p.id] = nn

        total_pts = sum(len(c) for c in self.gpu_clouds.values())
        mempool = cp.get_default_memory_pool()
        print(f"  Initialized {len(particles)} particles "
              f"({total_pts:,} total points)")
        print(f"  GPU memory: {mempool.used_bytes() / 1024**2:.1f} MB")

    def check_proximity(self, id1, id2) -> bool:
        """
        Return ``True`` if any point of cloud *id1* lies within *D* of any
        point of cloud *id2*.

        Applies a fast bbox rejection on the GPU before issuing the cuML
        query.  The cuML ``kneighbors`` call returns distances as a CuPy
        array; ``cp.min`` reduces it to a scalar entirely on the GPU, and
        only the final ``float()`` cast synchronises with the CPU.

        Parameters
        ----------
        id1, id2 : int
            Particle IDs.

        Returns
        -------
        bool
        """
        if id1 in self.empty_ids or id2 in self.empty_ids:
            return False
        import cupy as cp

        # Bbox rejection — all GPU, one scalar sync
        delta = cp.maximum(
            0.0,
            cp.maximum(
                self.gpu_bb_min[id1] - self.gpu_bb_max[id2],
                self.gpu_bb_min[id2] - self.gpu_bb_max[id1],
            ),
        )
        if float(cp.sum(delta * delta)) > self.D_squared:
            return False

        # Query cloud1 against the index built on cloud2
        distances, _ = self.nn_indices[id2].kneighbors(self.gpu_clouds[id1])
        return float(cp.min(distances)) <= self.D

    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        """
        Return all pairs from *candidate_pairs* whose clouds are within *D*.

        Stage 1 vectorises all bounding-box comparisons in a single GPU pass
        so that cuML queries are only issued for pairs whose bounding boxes
        overlap within *D*.

        Parameters
        ----------
        candidate_pairs : list of tuple[int, int]

        Returns
        -------
        list of tuple[int, int]
        """
        import cupy as cp

        if not candidate_pairs:
            return []

        # Stage 1: vectorised bbox prefilter
        ids1 = [p[0] for p in candidate_pairs]
        ids2 = [p[1] for p in candidate_pairs]

        bb_min1 = cp.stack([self.gpu_bb_min[i] for i in ids1])  # (N, 3)
        bb_max1 = cp.stack([self.gpu_bb_max[i] for i in ids1])
        bb_min2 = cp.stack([self.gpu_bb_min[i] for i in ids2])
        bb_max2 = cp.stack([self.gpu_bb_max[i] for i in ids2])

        delta = cp.maximum(0.0, cp.maximum(bb_min1 - bb_max2,
                                           bb_min2 - bb_max1))  # (N, 3)
        bbox_dist_sq = cp.sum(delta * delta, axis=1)            # (N,)
        bbox_pass = (bbox_dist_sq <= self.D_squared).get()      # one transfer

        survivors = [pair for pair, ok in zip(candidate_pairs, bbox_pass) if ok]

        n_rejected = len(candidate_pairs) - len(survivors)
        if n_rejected:
            print(f"  GPUChecker Stage 1: bbox prefilter removed "
                  f"{n_rejected}/{len(candidate_pairs)} pairs")

        # Stage 2: cuML kneighbors query for survivors
        merge_pairs = []
        for id1, id2 in survivors:
            distances, _ = self.nn_indices[id2].kneighbors(self.gpu_clouds[id1])
            if float(cp.min(distances)) <= self.D:
                merge_pairs.append((id1, id2))

        return merge_pairs

    def cleanup(self) -> None:
        """Free all GPU memory."""
        import cupy as cp

        self.gpu_clouds.clear()
        self.nn_indices.clear()
        self.gpu_bb_min.clear()
        self.gpu_bb_max.clear()
        cp.get_default_memory_pool().free_all_blocks()
        print("GPU memory freed")

    def check_cloud_proximity(self, cloud_a: np.ndarray,
                               cloud_b: np.ndarray) -> bool:
        """
        Check partition-cloud proximity on GPU using cuML.

        Uploads both clouds to the GPU, applies a bounding-box rejection,
        then builds a temporary cuML ``NearestNeighbors`` index on *cloud_b*
        and queries it against *cloud_a*.
        """
        if len(cloud_a) == 0 or len(cloud_b) == 0:
            return False
        import cupy as cp
        from cuml.neighbors import NearestNeighbors

        a = cp.asarray(cloud_a[:, :3].astype(np.float32))
        b = cp.asarray(cloud_b[:, :3].astype(np.float32))

        delta = cp.maximum(0.0, cp.maximum(
            cp.min(a, axis=0) - cp.max(b, axis=0),
            cp.min(b, axis=0) - cp.max(a, axis=0),
        ))
        if float(cp.sum(delta * delta)) > self.D_squared:
            return False

        nn = NearestNeighbors(n_neighbors=1, algorithm='brute',
                              metric='euclidean', output_type='cupy')
        nn.fit(b)
        distances, _ = nn.kneighbors(a)
        return float(cp.min(distances)) <= self.D


# ============================================================================
# Cell-Hash Proximity Checkers  (scipy-free, GPU-compatible alternative)
# ============================================================================
#
# Algorithm
# ---------
# 1. Divide space into cubic voxels of side D.
# 2. Hash every point of cloud B by its voxel index (ix, iy, iz).
# 3. For each *unique occupied voxel* of cloud A, look up the 27 neighbouring
#    voxels in B's hash map and gather candidate B-points.
# 4. Compute exact squared distances between A-points in that voxel and the
#    B-candidates; return True on the first hit.
#
# Complexity
# ----------
# - Build:  O(n)       memory and time
# - Query:  O(n_cells_A * k_B)  where k_B ≈ points-per-cell (small constant)
#
# This gives bitwise-identical results to KDTree.query_ball_point for the
# fixed-radius "any neighbour within D?" query.
# ============================================================================


class _CellHashMixin:
    """
    Shared cell-hash logic (CPU path).

    The voxel index is a Python tuple ``(ix, iy, iz)`` stored in a
    ``defaultdict``; tuple-key lookup is O(1) average and avoids any
    integer-overflow risk from encoded single-integer keys.
    """

    @staticmethod
    def _build_hashmap(cloud: np.ndarray, inv_D: float) -> dict:
        """
        Map each voxel ``(ix, iy, iz)`` to the list of row indices of
        *cloud* points that fall inside it.

        Parameters
        ----------
        cloud : ndarray, shape (n, 3)
            Point cloud coordinates (float32).
        inv_D : float
            Reciprocal of the cell side length (= 1 / D).

        Returns
        -------
        dict
            ``{(ix, iy, iz): [row_index, ...]}``
        """
        cells = np.floor(cloud * inv_D).astype(np.int64)
        hashmap: dict = defaultdict(list)
        for i in range(len(cloud)):
            hashmap[(int(cells[i, 0]), int(cells[i, 1]), int(cells[i, 2]))].append(i)
        return hashmap

    @staticmethod
    def _query(cloud_a: np.ndarray, cloud_b: np.ndarray,
               hashmap_b: dict, D: float) -> bool:
        """
        Return ``True`` if any point of *cloud_a* lies within *D* of any
        point of *cloud_b*.

        Iterates over unique occupied voxels of *cloud_a*, gathers the
        B-candidates from the 27 neighbouring voxels, and evaluates exact
        squared distances using vectorised NumPy.

        Parameters
        ----------
        cloud_a : ndarray, shape (n_a, 3)
        cloud_b : ndarray, shape (n_b, 3)
        hashmap_b : dict
            Pre-built hash map for *cloud_b* (from ``_build_hashmap``).
        D : float
            Distance threshold.

        Returns
        -------
        bool
        """
        D_sq = D * D
        inv_D = 1.0 / D
        a_cells = np.floor(cloud_a * inv_D).astype(np.int64)
        unique_a_cells = np.unique(a_cells, axis=0)

        for cell in unique_a_cells:
            ax, ay, az = int(cell[0]), int(cell[1]), int(cell[2])

            b_indices: list = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        b_indices.extend(
                            hashmap_b.get((ax + dx, ay + dy, az + dz), ())
                        )

            if not b_indices:
                continue

            mask = np.all(a_cells == cell, axis=1)
            pts_a = cloud_a[mask]              # (k_a, 3)
            pts_b = cloud_b[b_indices]         # (k_b, 3)

            diff = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            if dist_sq.min() <= D_sq:
                return True

        return False


class CellHashCPUSingleThreadChecker(_CellHashMixin, ProximityChecker):
    """
    Single-threaded CPU cell-hash proximity checker.

    Drop-in replacement for ``CPUSingleThreadChecker`` that avoids the
    scipy dependency and scales to large point clouds without materialising
    an O(n*m) distance matrix.
    """

    def __init__(self, distance_threshold: float):
        ProximityChecker.__init__(self, distance_threshold)
        self._inv_D: float = 1.0 / distance_threshold
        self.clouds: dict = {}
        self.hashmaps: dict = {}
        self.bboxes: dict = {}
        self.centroids: dict = {}

    def initialize(self, particles) -> None:
        """Build voxel hash maps for every particle cloud."""
        print("Initializing CellHash CPU single-threaded backend...")
        for p in particles:
            cloud = p.point_cloud[:, :3].astype(np.float32)
            self.clouds[p.id] = cloud
            if len(cloud) == 0:
                self.empty_ids.add(p.id)
                self.bboxes[p.id]    = {'min': None, 'max': None}
                self.centroids[p.id] = None
                self.hashmaps[p.id]  = {}
                continue
            self.hashmaps[p.id]  = self._build_hashmap(cloud, self._inv_D)
            self.bboxes[p.id]    = self.compute_bbox(p.point_cloud)
            self.centroids[p.id] = self.compute_centroid(p.point_cloud)
        print(f"  Initialized {len(particles)} particles")

    def check_proximity(self, id1, id2) -> bool:
        """
        Return ``True`` if clouds *id1* and *id2* are within *D*.

        Delegates to :meth:`_CellHashMixin._query`.
        """
        if id1 in self.empty_ids or id2 in self.empty_ids:
            return False
        return self._query(
            self.clouds[id1], self.clouds[id2],
            self.hashmaps[id2], self.D,
        )

    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        """
        Check all candidate pairs sequentially and return the touching ones.

        Parameters
        ----------
        candidate_pairs : list of tuple[int, int]

        Returns
        -------
        list of tuple[int, int]
        """
        return [
            (id1, id2) for id1, id2 in candidate_pairs
            if self.check_proximity(id1, id2)
        ]

    def cleanup(self) -> None:
        """Discard all point clouds and hash maps."""
        self.clouds.clear()
        self.hashmaps.clear()
        self.bboxes.clear()
        self.centroids.clear()

    def check_cloud_proximity(self, cloud_a: np.ndarray,
                               cloud_b: np.ndarray) -> bool:
        """
        Check partition-cloud proximity using the cell-hash algorithm.

        Builds a fresh hash map for *cloud_b* and queries it against
        *cloud_a* using :meth:`_CellHashMixin._query`.  More efficient than
        the scipy-based default for large clouds because it avoids building
        a KDTree.
        """
        a = cloud_a[:, :3].astype(np.float32)
        b = cloud_b[:, :3].astype(np.float32)
        hmap = self._build_hashmap(b, self._inv_D)
        return self._query(a, b, hmap, self.D)


class CellHashCPUMultiThreadChecker(_CellHashMixin, ProximityChecker):
    """
    Multi-threaded CPU cell-hash proximity checker.

    Drop-in replacement for ``CPUMultiThreadChecker`` that avoids the
    scipy dependency.  Hash-map construction and pair evaluation are both
    parallelised with ``joblib`` threads.
    """

    def __init__(self, distance_threshold: float, n_jobs: int = -1):
        ProximityChecker.__init__(self, distance_threshold)
        self.n_jobs: int = n_jobs
        self._inv_D: float = 1.0 / distance_threshold
        self.clouds: dict = {}
        self.hashmaps: dict = {}
        self.bboxes: dict = {}
        self.centroids: dict = {}

    def initialize(self, particles) -> None:
        """Build voxel hash maps in parallel."""
        from joblib import Parallel, delayed

        print(f"Initializing CellHash CPU multi-threaded backend "
              f"(n_jobs={self.n_jobs})...")

        def _build(p):
            cloud = p.point_cloud[:, :3].astype(np.float32)
            if len(cloud) == 0:
                return (p.id, cloud, None,
                        {'min': None, 'max': None}, None)
            return (
                p.id, cloud,
                self._build_hashmap(cloud, self._inv_D),
                self.compute_bbox(p.point_cloud),
                self.compute_centroid(p.point_cloud),
            )

        results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(_build)(p) for p in particles
        )
        for pid, cloud, hmap, bbox, centroid in results:
            self.clouds[pid] = cloud
            if hmap is None:
                self.empty_ids.add(pid)
                self.hashmaps[pid] = {}
            else:
                self.hashmaps[pid] = hmap
            self.bboxes[pid]    = bbox
            self.centroids[pid] = centroid
        print(f"  Initialized {len(particles)} particles")

    def check_proximity(self, id1, id2) -> bool:
        """
        Return ``True`` if clouds *id1* and *id2* are within *D*.

        Delegates to :meth:`_CellHashMixin._query`.
        """
        if id1 in self.empty_ids or id2 in self.empty_ids:
            return False
        return self._query(
            self.clouds[id1], self.clouds[id2],
            self.hashmaps[id2], self.D,
        )

    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        """
        Check all candidate pairs in parallel and return the touching ones.

        Parameters
        ----------
        candidate_pairs : list of tuple[int, int]

        Returns
        -------
        list of tuple[int, int]
        """
        from joblib import Parallel, delayed

        def _check(id1, id2):
            return (id1, id2) if self.check_proximity(id1, id2) else None

        results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(_check)(id1, id2) for id1, id2 in candidate_pairs
        )
        return [r for r in results if r is not None]

    def cleanup(self) -> None:
        """Discard all point clouds and hash maps."""
        self.clouds.clear()
        self.hashmaps.clear()
        self.bboxes.clear()
        self.centroids.clear()

    def check_cloud_proximity(self, cloud_a: np.ndarray,
                               cloud_b: np.ndarray) -> bool:
        """
        Check partition-cloud proximity using the cell-hash algorithm.

        Same as :meth:`CellHashCPUSingleThreadChecker.check_cloud_proximity`
        — builds a fresh CPU hash map for *cloud_b* and queries it.
        Parallelism is not applied for a single two-cloud check.
        """
        a = cloud_a[:, :3].astype(np.float32)
        b = cloud_b[:, :3].astype(np.float32)
        hmap = self._build_hashmap(b, self._inv_D)
        return self._query(a, b, hmap, self.D)


class CellHashGPUChecker(_CellHashMixin, ProximityChecker):
    """
    Cell-hash GPU proximity checker using CuPy.

    The voxel hash map (control-flow-heavy dict lookup) is built and
    maintained on the CPU.  Point clouds are stored on the GPU; once
    candidate B-points are identified via the hash map, all distance
    computations are performed on the GPU.

    This requires only standard CuPy — no RAPIDS, no cupyx.scipy.spatial.

    CPU↔GPU transfer per ``check_proximity`` call
    ----------------------------------------------
    - **To CPU**: ``a_cells`` (int64 array of shape ``(n_a, 3)``), once per call.
    - **To GPU**: ``mask`` boolean array and ``b_indices`` index array —
      both small (proportional to points-per-cell, not total cloud size).
    """

    def __init__(self, distance_threshold: float):
        ProximityChecker.__init__(self, distance_threshold)
        self._inv_D: float = 1.0 / distance_threshold
        self.gpu_clouds: dict = {}
        self.hashmaps: dict = {}
        self.bboxes: dict = {}
        self.centroids: dict = {}

    def initialize(self, particles) -> None:
        """Transfer point clouds to the GPU and build CPU hash maps."""
        import cupy as cp

        print("Initializing CellHash GPU backend...")
        for p in particles:
            cloud = p.point_cloud[:, :3].astype(np.float32)
            self.gpu_clouds[p.id] = cp.asarray(cloud)
            if len(cloud) == 0:
                self.empty_ids.add(p.id)
                self.bboxes[p.id]    = {'min': None, 'max': None}
                self.centroids[p.id] = None
                self.hashmaps[p.id]  = {}
                continue
            self.hashmaps[p.id]   = self._build_hashmap(cloud, self._inv_D)
            self.bboxes[p.id]     = self.compute_bbox(p.point_cloud)
            self.centroids[p.id]  = self.compute_centroid(p.point_cloud)

        mempool = cp.get_default_memory_pool()
        print(f"  Initialized {len(particles)} particles")
        print(f"  GPU memory: {mempool.used_bytes() / 1024**2:.1f} MB")

    def check_proximity(self, id1, id2) -> bool:
        """
        Return ``True`` if the two GPU-resident clouds are within *D*.

        1. Compute voxel indices for cloud A on the GPU, then pull them to
           the CPU for hash-map lookup (one small transfer).
        2. For each unique occupied A-voxel, look up 27 neighbours in B's
           CPU hash map.
        3. Gather the matching A- and B-points entirely on the GPU and
           evaluate squared distances there.
        """
        if id1 in self.empty_ids or id2 in self.empty_ids:
            return False
        import cupy as cp

        cloud_a_gpu = self.gpu_clouds[id1]
        cloud_b_gpu = self.gpu_clouds[id2]
        hashmap_b   = self.hashmaps[id2]

        # Voxel indices for A — transfer to CPU for dict lookup
        a_cells_gpu = cp.floor(cloud_a_gpu * self._inv_D).astype(cp.int64)
        a_cells_cpu = a_cells_gpu.get()                    # (n_a, 3) numpy
        unique_a_cells = np.unique(a_cells_cpu, axis=0)   # (k, 3) numpy

        for cell in unique_a_cells:
            ax, ay, az = int(cell[0]), int(cell[1]), int(cell[2])

            b_indices: list = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        b_indices.extend(
                            hashmap_b.get((ax + dx, ay + dy, az + dz), ())
                        )

            if not b_indices:
                continue

            # Points of A in this voxel — boolean mask is tiny
            mask_cpu = np.all(a_cells_cpu == cell, axis=1)
            pts_a = cloud_a_gpu[cp.asarray(mask_cpu)]      # (k_a, 3) GPU
            pts_b = cloud_b_gpu[cp.asarray(b_indices)]     # (k_b, 3) GPU

            # Vectorised distances fully on GPU
            diff    = pts_a[:, cp.newaxis, :] - pts_b[cp.newaxis, :, :]
            dist_sq = cp.sum(diff * diff, axis=2)           # (k_a, k_b)

            if float(cp.min(dist_sq)) <= self.D_squared:
                return True

        return False

    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        return [
            (id1, id2) for id1, id2 in candidate_pairs
            if self.check_proximity(id1, id2)
        ]

    def cleanup(self) -> None:
        """Free GPU memory."""
        import cupy as cp

        self.gpu_clouds.clear()
        self.hashmaps.clear()
        self.bboxes.clear()
        self.centroids.clear()
        cp.get_default_memory_pool().free_all_blocks()
        print("GPU memory freed")

    def check_cloud_proximity(self, cloud_a: np.ndarray,
                               cloud_b: np.ndarray) -> bool:
        """
        Check partition-cloud proximity using CPU cell-hash for candidate
        selection and CuPy for distance computation.

        For arbitrary clouds that are not pre-uploaded, builds the hash map
        on the CPU (same as the CPU cell-hash classes) and performs the
        final distance computation on the GPU.
        """
        a = cloud_a[:, :3].astype(np.float32)
        b = cloud_b[:, :3].astype(np.float32)
        hmap = self._build_hashmap(b, self._inv_D)
        return self._query(a, b, hmap, self.D)


# ============================================================================
# Bulk GPU Proximity Checker
# ============================================================================
#
# Design goals
# ------------
# * All point clouds resident on the GPU from initialize() onward.
#   No PCIe data transfers after that point.
# * Correct for all cloud configurations — no assumptions about surface
#   locality, interior structure, or touching geometry.
# * GPU utilisation is maximised in batch_check_proximity() by vectorising
#   the bounding-box prefilter across all candidate pairs simultaneously,
#   before falling through to per-pair exact checks.
#
# Memory budget
# -------------
# Static (persistent across all calls):
#   n_total_points * 3 * 4 bytes
#   ~ 12–120 MB for 1M–10M points  (trivial on 40 GB)
#
# Working (transient, per-pair exact check):
#   chunk_size * n_b * 3 * 4 bytes
#   Default chunk_size=512, n_b=100k → 614 MB  (well within 40 GB)
#   Adjust chunk_size upward for larger throughput on smaller clouds.
#
# The only CPU synchronisation points are:
#   1. bbox_pass_mask.get()  — one (N_pairs,) bool array per batch call
#   2. float(cp.min(dist_sq)) — one scalar per chunk, needed for early exit
# ============================================================================


class BulkGPUChecker(ProximityChecker):
    """
    Fully GPU-resident proximity checker using CuPy chunked brute-force.

    All point clouds are uploaded during :meth:`initialize` and remain on
    the GPU for the lifetime of the object.  No PCIe array transfers occur
    after initialisation.  The algorithm is unconditionally correct — it
    makes no assumptions about cloud geometry, interior structure, or where
    touching occurs.

    Algorithm
    ---------
    :meth:`batch_check_proximity` operates in two stages:

    **Stage 1 — vectorised bbox prefilter (GPU).**
    Bounding-box ``min``/``max`` tensors for all candidate pairs are stacked
    into ``(N_pairs, 3)`` arrays and evaluated simultaneously.  Any pair
    whose bounding boxes are separated by more than *D* is rejected in
    *O*(*N_pairs*) GPU work with no per-pair Python loop.

    **Stage 2 — chunked exact check (GPU, early exit).**
    For each pair that survives stage 1, ``cloud_a`` is processed in row
    chunks of ``chunk_size``.  For each chunk the ``(C, n_b, 3)`` distance
    tensor is computed entirely on the GPU.  The moment any distance ≤ *D*
    is found the pair is accepted and remaining chunks are skipped.
    Because touching pairs contact at few points, the early exit fires in
    the first or second chunk in typical cases; the full scan is only
    performed for separated pairs, which are rare after the bbox prefilter.

    Parameters
    ----------
    distance_threshold : float
        Proximity distance *D*.
    chunk_size : int, optional
        Number of ``cloud_a`` rows processed per GPU kernel launch.
        Controls the peak working-memory footprint:
        ``chunk_size * max(n_b) * 12`` bytes.
        Default 512 gives ~614 MB for n_b = 100 000.
        Increase for smaller clouds; decrease if GPU memory is tighter.
    """

    def __init__(self, distance_threshold: float, chunk_size: int = 512):
        super().__init__(distance_threshold)
        self.chunk_size: int = chunk_size
        self.gpu_clouds: dict = {}   # pid -> cp.ndarray (n, 3) float32
        self.gpu_bb_min: dict = {}   # pid -> cp.ndarray (3,)   float32
        self.gpu_bb_max: dict = {}   # pid -> cp.ndarray (3,)   float32

    # ------------------------------------------------------------------
    # ProximityChecker interface
    # ------------------------------------------------------------------

    def initialize(self, particles) -> None:
        """
        Upload all point clouds to the GPU and precompute bounding boxes.

        Each particle's ``point_cloud`` is cast to float32, transferred to
        the GPU once, and its axis-aligned bounding box is computed there.
        The CPU-side arrays are not retained.

        Parameters
        ----------
        particles : list
            Particle objects exposing ``.id`` and ``.point_cloud`` (ndarray
            with at least 3 columns).
        """
        import cupy as cp

        print("Initializing BulkGPU backend...")
        for p in particles:
            cloud = cp.asarray(p.point_cloud[:, :3].astype(np.float32))
            self.gpu_clouds[p.id] = cloud
            if len(cloud) == 0:
                self.empty_ids.add(p.id)
                self.gpu_bb_min[p.id] = cp.full(3,  cp.inf, dtype=cp.float32)
                self.gpu_bb_max[p.id] = cp.full(3, -cp.inf, dtype=cp.float32)
                continue
            self.gpu_bb_min[p.id] = cp.min(cloud, axis=0)
            self.gpu_bb_max[p.id] = cp.max(cloud, axis=0)

        total_pts = sum(len(c) for c in self.gpu_clouds.values())
        mempool = cp.get_default_memory_pool()
        print(f"  Initialized {len(particles)} particles "
              f"({total_pts:,} total points)")
        print(f"  GPU memory: {mempool.used_bytes() / 1024**2:.1f} MB")

    def check_proximity(self, id1, id2) -> bool:
        """
        Return ``True`` if any point of cloud *id1* lies within *D* of any
        point of cloud *id2*.

        Applies a bbox prefilter first, then falls through to the chunked
        exact check.  No PCIe transfers are performed; all work is on the
        GPU with scalar synchronisation only.

        Parameters
        ----------
        id1, id2 : int
            Particle IDs as used in :meth:`initialize`.

        Returns
        -------
        bool
        """
        if id1 in self.empty_ids or id2 in self.empty_ids:
            return False
        import cupy as cp

        # Fast bbox rejection
        delta = cp.maximum(
            0.0,
            cp.maximum(
                self.gpu_bb_min[id1] - self.gpu_bb_max[id2],
                self.gpu_bb_min[id2] - self.gpu_bb_max[id1],
            ),
        )
        if float(cp.sum(delta * delta)) > self.D_squared:
            return False

        return self._exact_check(id1, id2)

    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        """
        Return all pairs from *candidate_pairs* whose clouds are within *D*.

        Uses a two-stage GPU pipeline for efficiency.

        **Stage 1** — all bounding-box comparisons are vectorised across the
        entire candidate list in a single GPU pass (no Python loop over pairs).

        **Stage 2** — only the survivors of stage 1 are examined with the
        chunked exact check.

        Parameters
        ----------
        candidate_pairs : list of tuple[int, int]
            Pairs of particle IDs to evaluate.

        Returns
        -------
        list of tuple[int, int]
            Subset of *candidate_pairs* that are within proximity *D*.
        """
        import cupy as cp

        if not candidate_pairs:
            return []

        # Stage 1: vectorised bbox prefilter — no Python loop over pairs
        ids1 = [p[0] for p in candidate_pairs]
        ids2 = [p[1] for p in candidate_pairs]

        bb_min1 = cp.stack([self.gpu_bb_min[i] for i in ids1])  # (N, 3)
        bb_max1 = cp.stack([self.gpu_bb_max[i] for i in ids1])  # (N, 3)
        bb_min2 = cp.stack([self.gpu_bb_min[i] for i in ids2])  # (N, 3)
        bb_max2 = cp.stack([self.gpu_bb_max[i] for i in ids2])  # (N, 3)

        delta = cp.maximum(0.0,
                           cp.maximum(bb_min1 - bb_max2,
                                      bb_min2 - bb_max1))       # (N, 3)
        bbox_dist_sq = cp.sum(delta * delta, axis=1)            # (N,)
        # Single (N,) bool transfer — only transfer in this method
        bbox_pass = (bbox_dist_sq <= self.D_squared).get()

        survivors = [pair for pair, ok in zip(candidate_pairs, bbox_pass) if ok]

        n_rejected = len(candidate_pairs) - len(survivors)
        if n_rejected:
            print(f"  BulkGPU Stage 1: bbox prefilter removed "
                  f"{n_rejected}/{len(candidate_pairs)} pairs")

        # Stage 2: chunked exact check for survivors
        return [pair for pair in survivors if self._exact_check(pair[0], pair[1])]

    def cleanup(self) -> None:
        """Free all GPU memory allocated during :meth:`initialize`."""
        import cupy as cp

        self.gpu_clouds.clear()
        self.gpu_bb_min.clear()
        self.gpu_bb_max.clear()
        cp.get_default_memory_pool().free_all_blocks()
        print("GPU memory freed")

    def check_cloud_proximity(self, cloud_a: np.ndarray,
                               cloud_b: np.ndarray) -> bool:
        """
        Check partition-cloud proximity on GPU using CuPy chunked brute force.

        Uploads both clouds, applies bbox rejection, then evaluates exact
        distances in row-chunks of :attr:`chunk_size` with early exit.
        """
        import cupy as cp

        a = cp.asarray(cloud_a[:, :3].astype(np.float32))
        b = cp.asarray(cloud_b[:, :3].astype(np.float32))

        delta = cp.maximum(0.0, cp.maximum(
            cp.min(a, axis=0) - cp.max(b, axis=0),
            cp.min(b, axis=0) - cp.max(a, axis=0),
        ))
        if float(cp.sum(delta * delta)) > self.D_squared:
            return False

        for start in range(0, len(a), self.chunk_size):
            chunk = a[start : start + self.chunk_size]
            diff    = chunk[:, cp.newaxis, :] - b[cp.newaxis, :, :]
            dist_sq = cp.sum(diff * diff, axis=2)
            if float(cp.min(dist_sq)) <= self.D_squared:
                return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _exact_check(self, id1: int, id2: int) -> bool:
        """
        Chunked exact proximity check, fully on GPU with early exit.

        Processes ``cloud_a`` in row-chunks of :attr:`chunk_size`.  For each
        chunk the ``(C, n_b, 3)`` pairwise-difference tensor is computed and
        reduced to ``(C, n_b)`` squared distances on the GPU.  Returns
        ``True`` as soon as any distance ≤ *D* is found, skipping the
        remaining chunks.

        The only CPU synchronisation is a single scalar ``float()`` cast per
        chunk (needed to make the early-exit decision in Python).

        Parameters
        ----------
        id1, id2 : int
            Particle IDs of the two clouds to compare.

        Returns
        -------
        bool
        """
        import cupy as cp

        cloud_a = self.gpu_clouds[id1]   # (n_a, 3) GPU
        cloud_b = self.gpu_clouds[id2]   # (n_b, 3) GPU

        for start in range(0, len(cloud_a), self.chunk_size):
            chunk = cloud_a[start : start + self.chunk_size]    # (C, 3)
            diff    = chunk[:, cp.newaxis, :] - cloud_b[cp.newaxis, :, :]
            dist_sq = cp.sum(diff * diff, axis=2)               # (C, n_b)
            # Scalar sync — only CPU↔GPU communication in this method
            if float(cp.min(dist_sq)) <= self.D_squared:
                return True

        return False


# ============================================================================
# Numba JIT Proximity Checker
# ============================================================================
#
# Design goals
# ------------
# * Eliminate the per-chunk/per-pair Python loop and scalar CPU sync that
#   bottleneck BulkGPUChecker and GPUChecker.
# * Process all candidate pairs in a single batched kernel launch.
# * Use shared-memory B-tiling and a per-pair early-exit flag so touching
#   pairs return as soon as the first hit is found without scanning the
#   remainder of cloud A.
#
# Flat storage layout
# -------------------
# All point clouds are concatenated into one (N_total, 3) float32 GPU array.
# Two companion int dicts store the start index and size of each cloud:
#
#   flat_points[ starts[pid] : starts[pid] + sizes[pid] ]  ->  cloud for pid
#
# Kernel grid / block layout
# --------------------------
#   Grid  : (n_pairs, ceil(max_n_a / block_size))
#   Block : (block_size,)
#
#   blockIdx.x  = pair index  p
#   blockIdx.y  = A-row-tile index
#   threadIdx.x = index within the A-tile
#
#   Global A-row index = blockIdx.y * blockDim.x + threadIdx.x
#
# Shared memory per block
# -----------------------
#   tile_b : (256, 3) float32  = 3 kB
#   found  : (1,)     int32
#
# For each A-point the thread iterates over B in tiles of 256:
#   1. Cooperatively load 256 points of B into shared memory.
#   2. Compute distances for all tile-B points against this A-point.
#   3. On first hit: cuda.atomic.max(&found[0], 1) then break.
# After all B-tiles: thread 0 atomically marks result[p] = 1 if found.
#
# The only CPU sync after initialization is:
#   1. One (N_pairs,) bool .get() for the bbox prefilter.
#   2. One (N_survivors,) int32 .get() for the kernel result.
# ============================================================================

# Lazily compiled kernel (avoids importing numba at module load time)
_NUMBA_KERNEL = None


def _get_numba_kernel():
    """
    Return the lazily-compiled ``@numba.cuda.jit`` batched proximity kernel.

    The kernel is compiled from Python source **once** (on first call) and
    cached in the module-level ``_NUMBA_KERNEL`` variable.  Subsequent calls
    return the already-compiled function object so that PTX compilation cost
    is paid only once per interpreter session.

    Returns
    -------
    numba.cuda.compiler.CUDAKernel
        The ``_batched_proximity_kernel`` CUDA function, ready to launch.

    Notes
    -----
    Importing ``numba`` is deferred inside this function to avoid a hard
    import-time dependency.  The function raises ``ImportError`` if Numba is
    not installed.
    """
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is not None:
        return _NUMBA_KERNEL

    from numba import cuda, float32 as nb_f32, int32 as nb_i32

    @cuda.jit
    def _batched_proximity_kernel(flat, a_start, a_size,
                                   b_start, b_size, D_sq, result):
        """
        Batched proximity kernel.

        Each block is responsible for one (pair, A-tile) combination.
        Threads cooperatively load a tile of B-cloud into shared memory,
        then each thread checks its own A-point against every tile-B point.
        ``cuda.atomic.max`` sets the per-block ``found`` flag on the first hit,
        allowing the B-tile loop to short-circuit.
        """
        p = cuda.blockIdx.x
        if p >= result.shape[0]:
            return
        # Skip if another A-tile block already found a hit for this pair
        if result[p]:
            return

        # Shared memory: B-tile (256 points × 3 coords) + int32 hit flag
        tile_b = cuda.shared.array(shape=(256, 3), dtype=nb_f32)
        found  = cuda.shared.array(shape=(1,),     dtype=nb_i32)

        tx = cuda.threadIdx.x
        if tx == 0:
            found[0] = 0
        cuda.syncthreads()

        a_off = a_start[p]
        n_a   = a_size[p]
        b_off = b_start[p]
        n_b   = b_size[p]

        a_row   = cuda.blockIdx.y * cuda.blockDim.x + tx
        a_valid = a_row < n_a

        bt = 0
        while bt < n_b:
            if found[0]:
                break

            # Cooperatively load a tile of B into shared memory
            b_local  = tx
            b_global = bt + b_local
            if b_global < n_b:
                tile_b[b_local, 0] = flat[b_off + b_global, 0]
                tile_b[b_local, 1] = flat[b_off + b_global, 1]
                tile_b[b_local, 2] = flat[b_off + b_global, 2]
            cuda.syncthreads()

            if a_valid and not found[0]:
                ax = flat[a_off + a_row, 0]
                ay = flat[a_off + a_row, 1]
                az = flat[a_off + a_row, 2]
                tile_sz = min(256, n_b - bt)
                for k in range(tile_sz):
                    dx = ax - tile_b[k, 0]
                    dy = ay - tile_b[k, 1]
                    dz = az - tile_b[k, 2]
                    if dx*dx + dy*dy + dz*dz <= D_sq:
                        cuda.atomic.max(found, 0, 1)
                        break

            cuda.syncthreads()
            bt += 256

        if tx == 0 and found[0]:
            cuda.atomic.max(result, p, 1)

    _NUMBA_KERNEL = _batched_proximity_kernel
    return _NUMBA_KERNEL


class NumbaKernelChecker(ProximityChecker):
    """
    Proximity checker using a batched ``@numba.cuda.jit`` kernel.

    Requires **Numba** and **CuPy**.  Install with::

        pip install numba cupy-cuda12x

    Compared to :class:`BulkGPUChecker` this class eliminates:

    * The per-chunk Python loop in ``_exact_check``.
    * The per-chunk ``float(cp.min(...))`` CPU synchronisation.
    * Per-pair Python overhead in ``batch_check_proximity``.

    All point clouds are stored in a single flat ``(N_total, 3)`` float32
    CuPy array (one PCIe upload at :meth:`initialize`, zero thereafter).
    The Numba kernel receives CuPy arrays directly via the
    ``__cuda_array_interface__`` protocol.

    The kernel uses shared-memory B-tiling (256 points per tile) and a
    per-block hit flag updated with ``cuda.atomic.max``.  All candidate
    pairs that survive the bounding-box prefilter are processed in a
    **single kernel launch**.

    The only CPU synchronisation after initialisation is:

    1. One ``(N_pairs,)`` bool ``.get()`` for the bbox prefilter result.
    2. One ``(N_survivors,)`` int32 ``.get()`` for the kernel result.

    Grid / block layout
    -------------------
    * ``grid  = (N_survivors, ceil(max_n_a / block_size))``
    * ``block = (block_size,)``  (default 128)

    The Y-dimension tiles over A-cloud rows so every point in A receives a
    thread regardless of cloud-size variation across pairs.

    Parameters
    ----------
    distance_threshold : float
        Proximity distance *D*.
    block_size : int, optional
        CUDA block dimension.  Must be a multiple of 32.  Default 128.
    """

    def __init__(self, distance_threshold: float, block_size: int = 128):
        super().__init__(distance_threshold)
        if block_size % 32 != 0:
            raise ValueError(f"block_size must be a multiple of 32, got {block_size}")
        self.block_size: int = block_size

        self._flat: object = None     # cp.ndarray (N_total, 3) float32
        self._starts: dict = {}       # pid -> int  start row in _flat
        self._sizes:  dict = {}       # pid -> int  number of rows
        self._gpu_bb_min: dict = {}   # pid -> cp.ndarray (3,)
        self._gpu_bb_max: dict = {}   # pid -> cp.ndarray (3,)

    # ------------------------------------------------------------------
    # ProximityChecker interface
    # ------------------------------------------------------------------

    def initialize(self, particles) -> None:
        """
        Concatenate all point clouds into a single flat GPU array and
        record per-cloud start/size offsets.

        The Numba kernel is compiled (JIT-ted) on the first call to
        :meth:`_launch` so the PTX compilation cost is paid once.

        Parameters
        ----------
        particles : list
            Objects with ``.id`` and ``.point_cloud`` (≥ 3 columns).
        """
        import cupy as cp

        print("Initializing NumbaKernel backend...")
        total_pts = sum(len(p.point_cloud) for p in particles)
        flat_cpu  = np.empty((total_pts, 3), dtype=np.float32)

        cursor = 0
        for p in particles:
            n = len(p.point_cloud)
            flat_cpu[cursor : cursor + n] = p.point_cloud[:, :3].astype(np.float32)
            self._starts[p.id] = cursor
            self._sizes[p.id]  = n
            cursor += n

        self._flat = cp.asarray(flat_cpu)   # single PCIe upload

        for p in particles:
            s, n = self._starts[p.id], self._sizes[p.id]
            if n == 0:
                self.empty_ids.add(p.id)
                self._gpu_bb_min[p.id] = cp.full(3,  cp.inf, dtype=cp.float32)
                self._gpu_bb_max[p.id] = cp.full(3, -cp.inf, dtype=cp.float32)
                continue
            cloud = self._flat[s : s + n]
            self._gpu_bb_min[p.id] = cp.min(cloud, axis=0)
            self._gpu_bb_max[p.id] = cp.max(cloud, axis=0)

        mempool = cp.get_default_memory_pool()
        print(f"  Initialized {len(particles)} particles "
              f"({total_pts:,} total points)")
        print(f"  GPU memory: {mempool.used_bytes() / 1024**2:.1f} MB")

    def check_proximity(self, id1, id2) -> bool:
        """Return ``True`` if any point of cloud *id1* lies within *D* of *id2*."""
        if id1 in self.empty_ids or id2 in self.empty_ids:
            return False
        import cupy as cp

        delta = cp.maximum(
            0.0,
            cp.maximum(
                self._gpu_bb_min[id1] - self._gpu_bb_max[id2],
                self._gpu_bb_min[id2] - self._gpu_bb_max[id1],
            ),
        )
        if float(cp.sum(delta * delta)) > self.D_squared:
            return False

        result = cp.zeros(1, dtype=cp.int32)
        self._launch(
            cp.array([self._starts[id1]], dtype=cp.int32),
            cp.array([self._sizes[id1]],  dtype=cp.int32),
            cp.array([self._starts[id2]], dtype=cp.int32),
            cp.array([self._sizes[id2]],  dtype=cp.int32),
            result,
        )
        return bool(result.get()[0])

    def batch_check_proximity(self, candidate_pairs: List[Tuple]) -> List[Tuple]:
        """
        Two-stage batched check.

        Stage 1: vectorised bbox prefilter — one GPU pass, one transfer.
        Stage 2: single ``@cuda.jit`` kernel launch for all survivors —
        one transfer for the ``(N_survivors,)`` result array.
        """
        import cupy as cp

        if not candidate_pairs:
            return []

        ids1 = [p[0] for p in candidate_pairs]
        ids2 = [p[1] for p in candidate_pairs]

        bb_min1 = cp.stack([self._gpu_bb_min[i] for i in ids1])
        bb_max1 = cp.stack([self._gpu_bb_max[i] for i in ids1])
        bb_min2 = cp.stack([self._gpu_bb_min[i] for i in ids2])
        bb_max2 = cp.stack([self._gpu_bb_max[i] for i in ids2])

        delta        = cp.maximum(0.0, cp.maximum(bb_min1 - bb_max2,
                                                   bb_min2 - bb_max1))
        bbox_dist_sq = cp.sum(delta * delta, axis=1)
        bbox_pass    = (bbox_dist_sq <= self.D_squared).get()   # one transfer

        survivors = [p for p, ok in zip(candidate_pairs, bbox_pass) if ok]
        n_rej = len(candidate_pairs) - len(survivors)
        if n_rej:
            print(f"  NumbaKernel Stage 1: bbox prefilter removed "
                  f"{n_rej}/{len(candidate_pairs)} pairs")

        if not survivors:
            return []

        a_starts = cp.array([self._starts[p[0]] for p in survivors], dtype=cp.int32)
        a_sizes  = cp.array([self._sizes[p[0]]  for p in survivors], dtype=cp.int32)
        b_starts = cp.array([self._starts[p[1]] for p in survivors], dtype=cp.int32)
        b_sizes  = cp.array([self._sizes[p[1]]  for p in survivors], dtype=cp.int32)
        result   = cp.zeros(len(survivors), dtype=cp.int32)

        self._launch(a_starts, a_sizes, b_starts, b_sizes, result)

        hit = result.get()                                       # one transfer
        return [p for p, h in zip(survivors, hit) if h]

    def cleanup(self) -> None:
        """Free all GPU memory."""
        import cupy as cp

        self._flat = None
        self._starts.clear()
        self._sizes.clear()
        self._gpu_bb_min.clear()
        self._gpu_bb_max.clear()
        cp.get_default_memory_pool().free_all_blocks()
        print("GPU memory freed")

    def check_cloud_proximity(self, cloud_a: np.ndarray,
                               cloud_b: np.ndarray) -> bool:
        """
        Check partition-cloud proximity using CuPy chunked brute force.

        Uploads both clouds on demand (they are not pre-stored in
        :attr:`_flat`), applies bbox rejection, then evaluates exact
        distances in 512-row chunks with early exit.
        """
        import cupy as cp

        a = cp.asarray(cloud_a[:, :3].astype(np.float32))
        b = cp.asarray(cloud_b[:, :3].astype(np.float32))

        delta = cp.maximum(0.0, cp.maximum(
            cp.min(a, axis=0) - cp.max(b, axis=0),
            cp.min(b, axis=0) - cp.max(a, axis=0),
        ))
        if float(cp.sum(delta * delta)) > self.D_squared:
            return False

        chunk = 512
        for start in range(0, len(a), chunk):
            piece   = a[start : start + chunk]
            diff    = piece[:, cp.newaxis, :] - b[cp.newaxis, :, :]
            dist_sq = cp.sum(diff * diff, axis=2)
            if float(cp.min(dist_sq)) <= self.D_squared:
                return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _launch(self, a_starts, a_sizes, b_starts, b_sizes, result) -> None:
        """Configure grid/block dims and invoke the ``@cuda.jit`` kernel."""
        import cupy as cp

        n_pairs = len(a_starts)
        max_n_a = int(cp.max(a_sizes))
        grid_y  = (max_n_a + self.block_size - 1) // self.block_size

        kernel = _get_numba_kernel()
        kernel[(n_pairs, grid_y), (self.block_size,)](
            self._flat,
            a_starts, a_sizes,
            b_starts, b_sizes,
            np.float32(self.D_squared),
            result,
        )
