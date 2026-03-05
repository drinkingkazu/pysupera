"""
Modular pre-processing backends for particle point-cloud defragmentation.

A particle's point cloud is *fragmented* if an eps-ball connected-components
algorithm (eps = ``distance_threshold``) finds more than one cluster.  For
fragmented particles, small clusters (size ≤ ``min_pc_size``) are detached as
new ``kLEScatter`` particles; large clusters are retained in the original.

Selection via config
--------------------
Set ``particle.preprocessor.name`` to one of:

``scipy`` (default, CPU)
    Per-particle scipy cKDTree + sparse connected_components.  Fast early-exit
    checks (bounding-box diagonal, N≤2) skip the graph algorithm for the
    majority of non-fragmented particles.

``gpu`` (GPU, no RAPIDS required)
    Same per-particle logic as ``scipy`` for the early-exit path; for
    particles that survive screening, the pairwise distance matrix is computed
    on GPU with CuPy (a single matrix-multiply-like broadcast) and the
    resulting adjacency is transferred back to CPU for connected_components.
    Best when point clouds contain many hundreds of points; add a
    ``min_pts_for_gpu`` override (default 64) to keep small clouds on CPU.

``rapids`` (fully GPU, requires cuDF + cuGraph)
    For particles that survive the early-exit screen, builds an edge list
    entirely on GPU with CuPy and runs cuGraph connected_components without
    any CPU round-trip.  Requires the RAPIDS stack (same as the existing
    GPUChecker / BulkGPUChecker partitioner backends).
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from .data import Particle
from .utils import SemanticType, PointFeature


# ============================================================================
# Diagnostic record — defragmentation
# ============================================================================

@dataclass
class FragmentRecord:
    """
    Diagnostic record for one particle processed by
    :meth:`DefragmentBase.process`.

    Attributes
    ----------
    particle_id : int
        ID of the particle that was inspected.
    n_points : int
        Total number of points in the original point cloud.
    early_exit : str or None
        Name of the fast-path that short-circuited the CC algorithm, or
        ``None`` if the full algorithm ran.  Possible values:

        ``'single_pt'``         len(pc) \u2264 1 \u2014 trivially one cluster.
        ``'two_pt_connected'``  exactly 2 points within *eps* of each other.
        ``'bbox'``              bounding-box diagonal \u2264 *eps*, so provably
                                one cluster without any graph work.
    n_clusters : int
        Number of clusters found (1 for early-exit particles).
    cluster_sizes : list of int
        Size of each cluster, sorted descending.
    kept_pts : int
        Points retained in the original particle after splitting
        (equals *n_points* for non-fragmented particles).
    spawned_ids : list of int
        IDs of new kLEScatter particles created from small fragments.
        Empty for non-fragmented particles.
    """
    particle_id  : int
    n_points     : int
    early_exit   : Optional[str]
    n_clusters   : int
    cluster_sizes: list
    kept_pts     : int
    spawned_ids  : list = field(default_factory=list)

    @property
    def fragmented(self) -> bool:
        """``True`` if the cloud was split into more than one cluster."""
        return self.n_clusters > 1

    def __str__(self) -> str:
        tag = f"FRAGMENTED ({self.n_clusters} clusters)" if self.fragmented \
              else "not fragmented"
        exit_str = f" [early-exit: {self.early_exit}]" if self.early_exit else ""
        sizes_str = ", ".join(str(s) for s in self.cluster_sizes)
        spawn_str = f", spawned ids: {self.spawned_ids}" if self.spawned_ids else ""
        return (
            f"  particle {self.particle_id:>8d} | "
            f"{self.n_points:>6d} pts | "
            f"{tag}{exit_str} | "
            f"cluster sizes: [{sizes_str}] | "
            f"kept pts: {self.kept_pts}{spawn_str}"
        )


# ============================================================================
# Diagnostic record — duplicate-point merging
# ============================================================================

@dataclass
class MergeRecord:
    """
    Diagnostic record for one particle processed by
    :meth:`MergeDuplicatesProcessor.process`.

    Attributes
    ----------
    particle_id : int
        ID of the particle that was processed.
    n_before : int
        Number of points before merging.
    n_after : int
        Number of unique-coordinate points after merging.
    """
    particle_id : int
    n_before    : int
    n_after     : int

    @property
    def n_merged(self) -> int:
        """Number of points that were collapsed into existing ones."""
        return self.n_before - self.n_after

    @property
    def had_duplicates(self) -> bool:
        """``True`` if at least one duplicate coordinate was found."""
        return self.n_before > self.n_after

    def __str__(self) -> str:
        tag = f"MERGED {self.n_merged} duplicate(s)" if self.had_duplicates \
              else "no duplicates"
        return (
            f"  particle {self.particle_id:>8d} | "
            f"{self.n_before:>6d} pts before | "
            f"{self.n_after:>6d} pts after | {tag}"
        )


# ============================================================================
# Shared helper
# ============================================================================

def _split_fragments(
    p: Particle,
    labels: np.ndarray,
    min_pc_size: int,
    next_id: int,
) -> tuple[Particle | None, list[Particle], int]:
    """
    Given a particle and per-point cluster labels, split out small fragments.

    Parameters
    ----------
    p : Particle
        Original particle whose point cloud has already been clustered.
    labels : np.ndarray, shape (N,)
        Integer cluster label for each point (0-based, no -1 noise).
    min_pc_size : int
        Fragments with ``size <= min_pc_size`` become new kLEScatter particles.
    next_id : int
        The next available particle id for spawned particles.

    Returns
    -------
    kept : Particle or None
        Original particle, possibly with point_cloud trimmed to large-fragment
        points only.  ``None`` if all fragments were small.
    spawned : list of Particle
        Newly created kLEScatter particles, one per small fragment.
    next_id : int
        Updated next available id.
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        return p, [], next_id     # single cluster — nothing to split

    pc = p.point_cloud
    large_mask = np.zeros(len(pc), dtype=bool)
    spawned: list[Particle] = []

    for lbl in unique_labels:
        frag_mask = labels == lbl
        frag_size = int(frag_mask.sum())

        if frag_size > min_pc_size:
            large_mask |= frag_mask
        else:
            frag_pc = pc[frag_mask]
            new_p = Particle(
                id           = next_id,
                parent_id    = p.id,
                ancestor_id  = p.ancestor_id,
                pdg          = p.pdg,
                parent_pdg   = p.parent_pdg,
                process_type = p._process_type,
                point_cloud  = frag_pc,
            )
            new_p.sem_type = SemanticType.kLEScatter
            spawned.append(new_p)
            next_id += 1

    if large_mask.any():
        p.point_cloud = pc[large_mask]
        return p, spawned, next_id
    else:
        return None, spawned, next_id


# ============================================================================
# Abstract base class
# ============================================================================

class DefragmentBase(ABC):
    """
    Abstract base for all defragmentation backends.

    Subclasses implement :meth:`_get_labels` for one particle's point cloud;
    the early-exit screening and particle-splitting bookkeeping is shared in
    :meth:`process` on this base class.

    Parameters
    ----------
    distance_threshold : float
        Neighbourhood radius (same units as point-cloud coordinates).
    min_pc_size : int
        Fragments with size ``<= min_pc_size`` are split off as kLEScatter.
    verbose : bool, optional
        When ``True``, :meth:`process` prints one line per particle that
        reaches the full connected-components algorithm (i.e. survived all
        fast early-exit checks) and a summary line at the end.  Particles
        that are trivially non-fragmented via early exits are not printed.
        Default ``False``.
    """

    def __init__(self, distance_threshold: float, min_pc_size: int,
                 verbose: bool = False,
                 sem_types=None):
        self.eps         = float(distance_threshold)
        self.eps2        = self.eps ** 2
        self.min_pc_size = int(min_pc_size)
        self.verbose     = bool(verbose)
        # sem_types: frozenset of SemanticType values to process, or None for all.
        # Particles whose sem_type is not in the set are passed through unchanged.
        if sem_types is None or len(sem_types) == 0:
            self.sem_types: Optional[frozenset] = None
        else:
            self.sem_types = frozenset(sem_types)
        #: List of :class:`FragmentRecord` objects from the most recent
        #: :meth:`process` call.  Always populated regardless of *verbose*.
        self.last_diagnostics: List[FragmentRecord] = []

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_labels(self, xyz: np.ndarray) -> np.ndarray:
        """
        Compute per-point cluster labels for *xyz*.

        Only called for point clouds that survive the fast early-exit
        screening in :meth:`process` (N ≥ 3, bounding-box diagonal > eps).

        Parameters
        ----------
        xyz : np.ndarray, shape (N, 3)
            Spatial coordinates. N ≥ 3 guaranteed.

        Returns
        -------
        labels : np.ndarray of int, shape (N,)
            0-based contiguous cluster index per point.
        """

    # ------------------------------------------------------------------
    # Shared processing loop (early exits + fragment splitting)
    # ------------------------------------------------------------------

    def process(self, particles: List[Particle],
                verbose: Optional[bool] = None) -> List[Particle]:
        """
        Defragment an event's particle list.

        Non-fragmented particles are returned unchanged.  Fragmented ones
        are trimmed to their large-fragment points; small fragments become
        new kLEScatter particles appended at the end.

        A :class:`FragmentRecord` is created for **every** particle and
        stored in :attr:`last_diagnostics`, regardless of whether *verbose*
        is enabled.  Only particles that reach the full CC algorithm (i.e.
        that survived all fast early-exit checks) are included; trivially
        non-fragmented particles would dominate the list and carry no useful
        information.

        Parameters
        ----------
        particles : list of Particle
        verbose : bool or None, optional
            Override the instance-level *verbose* flag for this call.
            ``None`` (default) uses the value set at construction time.

        Returns
        -------
        list of Particle
        """
        if not particles:
            self.last_diagnostics = []
            return particles

        be_verbose = self.verbose if verbose is None else bool(verbose)
        self.last_diagnostics = []

        max_id  = max(p.id for p in particles)
        next_id = max_id + 1
        result: list[Particle]  = []
        spawned: list[Particle] = []

        n_skipped      = 0
        n_early_exit   = 0
        n_cc_checked   = 0
        n_fragmented   = 0
        n_spawned_total = 0

        for p in particles:
            # Skip particles whose semantic type is not in the allowed set.
            if self.sem_types is not None and p.sem_type not in self.sem_types:
                result.append(p)
                n_skipped += 1
                continue

            pc = p.point_cloud

            # ---- fast early exits (cheapest first) -----------------------

            # 1. Single point: trivially one cluster
            if len(pc) <= 1:
                result.append(p)
                n_early_exit += 1
                continue

            xyz = pc[:, :3]

            # 2. Two points: O(1) distance comparison
            if len(xyz) == 2:
                if ((xyz[0] - xyz[1]) ** 2).sum() <= self.eps2:
                    result.append(p)
                    n_early_exit += 1
                    continue
                # Two isolated points — labels are [0, 1] by definition
                labels = np.array([0, 1], dtype=np.intp)
                exit_reason = None   # did not early-exit; fell through to CC logic
            else:
                # 3. Bounding-box diagonal: if the whole cloud fits inside a
                #    ball of radius eps, all pairs are within eps → 1 cluster
                span = xyz.max(axis=0) - xyz.min(axis=0)
                if (span ** 2).sum() <= self.eps2:
                    result.append(p)
                    n_early_exit += 1
                    continue

                # 4. Subclass-specific connected-components algorithm
                exit_reason = None
                labels = self._get_labels(xyz)

            # ---- reached CC — build diagnostic record -------------------
            n_cc_checked += 1
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_sizes = sorted(counts.tolist(), reverse=True)
            n_clusters = len(unique_labels)

            # ---- split large vs. small fragments -------------------------
            kept, new_particles, next_id = _split_fragments(
                p, labels, self.min_pc_size, next_id
            )
            kept_pts    = int(kept.point_cloud.shape[0]) if kept is not None else 0
            spawned_ids = [sp.id for sp in new_particles]

            rec = FragmentRecord(
                particle_id   = p.id,
                n_points      = len(pc),
                early_exit    = exit_reason,
                n_clusters    = n_clusters,
                cluster_sizes = cluster_sizes,
                kept_pts      = kept_pts,
                spawned_ids   = spawned_ids,
            )
            self.last_diagnostics.append(rec)

            if be_verbose:
                print(rec)

            if kept is not None:
                result.append(kept)
            spawned.extend(new_particles)

            if rec.fragmented:
                n_fragmented    += 1
                n_spawned_total += len(new_particles)

        if be_verbose:
            sem_label = (
                "all sem_types"
                if self.sem_types is None
                else "|".join(st.name for st in sorted(self.sem_types, key=int))
            )
            print(
                f"[defragment] {len(particles)} particles | "
                f"sem_types: {sem_label} | "
                f"{n_skipped} skipped (sem_type filter) | "
                f"{n_early_exit} early-exit (single cluster) | "
                f"{n_cc_checked} CC checked | "
                f"{n_fragmented} fragmented | "
                f"{n_spawned_total} new kLEScatter spawned"
            )

        return result + spawned


# ============================================================================
# Duplicate-coordinate merging
# ============================================================================

# Merge rules applied to feature columns beyond xyz (index >= 3).
# Each entry is (column_index, initial_value, numpy_ufunc).
# Columns beyond the array width are silently skipped.
#
#   time   (col 3) — minimum: earliest hit time survives
#   energy (col 4) — sum:     total deposited energy
#   dedx   (col 5) — maximum: peak ionisation density
_MERGE_RULES: list = [
    (PointFeature.time,   np.inf,  np.minimum),
    (PointFeature.energy, 0.0,     np.add),
    (PointFeature.dedx,  -np.inf,  np.maximum),
]


def _merge_point_cloud(pc: np.ndarray) -> np.ndarray:
    """
    Collapse rows that share identical 3-D coordinates into a single row.

    Coordinates (columns 0–2) are used as the grouping key.  Feature
    columns beyond index 2 are aggregated according to ``_MERGE_RULES``:
    time → min, energy → sum, dedx → max.  Any extra columns beyond
    ``PointFeature.dedx`` are left at zero.

    Parameters
    ----------
    pc : np.ndarray, shape (N, F), F \u2265 3

    Returns
    -------
    np.ndarray, shape (M, F), M \u2264 N
        The original array if no duplicates were found (same object),
        otherwise a new array with merged rows.
    """
    unique_xyz, inv = np.unique(pc[:, :3], axis=0, return_inverse=True)
    n_out = len(unique_xyz)

    if n_out == len(pc):
        return pc   # no duplicates — return the original array unchanged

    n_cols = pc.shape[1]
    out    = np.zeros((n_out, n_cols), dtype=pc.dtype)
    out[:, :3] = unique_xyz

    for col, init_val, ufunc in _MERGE_RULES:
        col = int(col)
        if col >= n_cols:
            continue
        out[:, col] = init_val
        ufunc.at(out[:, col], inv, pc[:, col])

    return out


class MergeDuplicatesProcessor:
    """
    Pre-processing stage that collapses point-cloud rows sharing the same
    3-D coordinate ``(x, y, z)`` into a single row.

    This is intended for pixelated (regularly sampled) detector readouts
    where multiple simulation hits can land on the same voxel.  Running
    it before :class:`DefragmentBase` ensures that ``min_pc_size`` counts
    unique spatial positions rather than raw hit multiplicity.

    Feature aggregation (applied to all columns after the first three):

    ============  ======================================================
    Column        Rule
    ============  ======================================================
    time (3)      minimum — the earliest hit time is retained
    energy (4)    sum — total deposited energy
    dedx (5)      maximum — peak ionisation density
    any extras    zero — placeholder; extend ``_MERGE_RULES`` to change
    ============  ======================================================

    Parameters
    ----------
    verbose : bool, optional
        When ``True``, :meth:`process` prints one line per particle
        that had at least one duplicate coordinate, plus a summary.
        Default ``False``.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = bool(verbose)
        #: List of :class:`MergeRecord` from the most recent
        #: :meth:`process` call (only particles *with* duplicates are
        #: recorded, since duplicate-free particles carry no information).
        self.last_diagnostics: List[MergeRecord] = []

    def process(self, particles: List[Particle],
                verbose: Optional[bool] = None) -> List[Particle]:
        """
        Merge duplicate-coordinate points in every particle's point cloud.

        Particles with no duplicate coordinates are returned unchanged
        (same object, same array).  A :class:`MergeRecord` is appended to
        :attr:`last_diagnostics` only for particles that *had* duplicates.

        Parameters
        ----------
        particles : list of Particle
        verbose : bool or None, optional
            Per-call override of the instance *verbose* flag.

        Returns
        -------
        list of Particle
        """
        be_verbose = self.verbose if verbose is None else bool(verbose)
        self.last_diagnostics = []

        n_total_before = 0
        n_total_after  = 0
        n_affected     = 0

        for p in particles:
            pc        = p.point_cloud
            n_before  = len(pc)
            merged    = _merge_point_cloud(pc)
            n_after   = len(merged)

            n_total_before += n_before
            n_total_after  += n_after

            if n_after < n_before:
                p.point_cloud = merged
                n_affected   += 1
                rec = MergeRecord(
                    particle_id = p.id,
                    n_before    = n_before,
                    n_after     = n_after,
                )
                self.last_diagnostics.append(rec)
                if be_verbose:
                    print(rec)

        if be_verbose:
            print(
                f"[merge_duplicates] {len(particles)} particles | "
                f"{n_affected} had duplicates | "
                f"{n_total_before - n_total_after} points removed | "
                f"{n_total_before} \u2192 {n_total_after} total pts"
            )

        return particles


# ============================================================================
# Backend: scipy  (CPU, default)
# ============================================================================

class ScipyDefragmenter(DefragmentBase):
    """
    CPU defragmentation using ``scipy.spatial.cKDTree`` + sparse
    ``connected_components``.

    This is mathematically identical to ``DBSCAN(eps, min_samples=1)``
    but avoids sklearn's per-call Python overhead.  Recommended default.
    """

    def _get_labels(self, xyz: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        n = len(xyz)
        pairs = cKDTree(xyz).query_pairs(self.eps, output_type='ndarray')

        if len(pairs) == 0:
            # No edges → every point is its own cluster
            return np.arange(n, dtype=np.intp)

        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(n, n),
        )
        _, labels = connected_components(graph, directed=False)
        return labels


# ============================================================================
# Backend: gpu  (CuPy distance matrix, CPU connected_components)
# ============================================================================

class GPUDefragmenter(DefragmentBase):
    """
    GPU-accelerated defragmentation using CuPy for pairwise distance
    computation; connected_components runs on CPU after a small integer
    adjacency matrix is transferred back.

    For particles with ``N < min_pts_for_gpu`` points the scipy path is used
    instead (GPU launch overhead outweighs the benefit for tiny clouds).

    The GPU step is a single O(N²) broadcast::

        diff  = xyz_gpu[:, None, :] - xyz_gpu[None, :, :]   # (N, N, 3)
        dist2 = (diff ** 2).sum(axis=-1)                     # (N, N)
        adj   = dist2 <= eps²                                # (N, N) bool

    This is exactly the adjacency matrix of the proximity graph.

    Parameters
    ----------
    distance_threshold : float
    min_pc_size : int
    min_pts_for_gpu : int, optional
        Point-cloud size threshold below which the scipy CPU path is used.
        Default is 64.
    verbose : bool, optional
        Passed through to :class:`DefragmentBase`. Default ``False``.
    """

    def __init__(self, distance_threshold: float, min_pc_size: int,
                 min_pts_for_gpu: int = 64, verbose: bool = False,
                 sem_types=None):
        super().__init__(distance_threshold, min_pc_size, verbose=verbose,
                         sem_types=sem_types)
        self.min_pts_for_gpu = int(min_pts_for_gpu)
        self._scipy = ScipyDefragmenter(distance_threshold, min_pc_size)

    def _get_labels(self, xyz: np.ndarray) -> np.ndarray:
        if len(xyz) < self.min_pts_for_gpu:
            return self._scipy._get_labels(xyz)
        return self._gpu_labels(xyz)

    def _gpu_labels(self, xyz: np.ndarray) -> np.ndarray:
        import cupy as cp
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        n   = len(xyz)
        eps2_gpu = cp.float32(self.eps2)

        xyz_gpu = cp.asarray(xyz.astype(np.float32))     # (N, 3) on GPU

        # Broadcast pairwise squared distance (N, N) — O(N²) GPU memory
        diff  = xyz_gpu[:, None, :] - xyz_gpu[None, :, :]   # (N, N, 3)
        dist2 = (diff * diff).sum(axis=-1)                   # (N, N)

        # Adjacency: True where within eps, excluding self-loops
        adj = dist2 <= eps2_gpu
        cp.fill_diagonal(adj, False)

        # Transfer sparse edge list to CPU for connected_components
        rows_gpu, cols_gpu = cp.where(adj)
        rows = cp.asnumpy(rows_gpu)
        cols = cp.asnumpy(cols_gpu)

        del xyz_gpu, diff, dist2, adj, rows_gpu, cols_gpu   # free GPU memory

        if len(rows) == 0:
            return np.arange(n, dtype=np.intp)

        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(n, n),
        )
        _, labels = connected_components(graph, directed=False)
        return labels


# ============================================================================
# Backend: rapids  (fully GPU — CuPy edges + cuGraph CC)
# ============================================================================

class RAPIDSDefragmenter(DefragmentBase):
    """
    Fully GPU defragmentation: edge list computed with CuPy, connected
    components via cuGraph (no CPU round-trip for the graph algorithm).

    Requires the RAPIDS stack (``cupy``, ``cudf``, ``cugraph``).

    The same ``min_pts_for_gpu`` threshold as ``GPUDefragmenter`` routes
    small clouds to the scipy CPU path to avoid kernel-launch overhead.

    Parameters
    ----------
    distance_threshold : float
    min_pc_size : int
    min_pts_for_gpu : int, optional
        Default 64.
    verbose : bool, optional
        Passed through to :class:`DefragmentBase`. Default ``False``.
    """

    def __init__(self, distance_threshold: float, min_pc_size: int,
                 min_pts_for_gpu: int = 64, verbose: bool = False,
                 sem_types=None):
        super().__init__(distance_threshold, min_pc_size, verbose=verbose,
                         sem_types=sem_types)
        self.min_pts_for_gpu = int(min_pts_for_gpu)
        self._scipy = ScipyDefragmenter(distance_threshold, min_pc_size)

    def _get_labels(self, xyz: np.ndarray) -> np.ndarray:
        if len(xyz) < self.min_pts_for_gpu:
            return self._scipy._get_labels(xyz)
        return self._rapids_labels(xyz)

    def _rapids_labels(self, xyz: np.ndarray) -> np.ndarray:
        import cupy as cp
        import cudf
        import cugraph

        n        = len(xyz)
        eps2_gpu = cp.float32(self.eps2)
        xyz_gpu  = cp.asarray(xyz.astype(np.float32))

        # Pairwise adjacency on GPU (upper triangle only to halve work)
        diff  = xyz_gpu[:, None, :] - xyz_gpu[None, :, :]
        dist2 = (diff * diff).sum(axis=-1)
        cp.fill_diagonal(dist2, self.eps2 + 1.0)          # exclude self-loops
        adj   = dist2 <= eps2_gpu

        src_gpu, dst_gpu = cp.where(adj)
        del xyz_gpu, diff, dist2, adj

        if len(src_gpu) == 0:
            return np.arange(n, dtype=np.intp)

        # Build cuGraph graph and run connected_components entirely on GPU
        edge_df = cudf.DataFrame({
            'src': cudf.Series(src_gpu),
            'dst': cudf.Series(dst_gpu),
        })
        G = cugraph.Graph()
        G.from_cudf_edgelist(edge_df, source='src', destination='dst')

        cc_df    = cugraph.connected_components(G)         # cuDF DataFrame
        cc_df    = cc_df.sort_values('vertex').reset_index(drop=True)
        labels   = cc_df['labels'].to_numpy().astype(np.intp)

        # cuGraph may not include isolated vertices (no edges) — pad if needed
        if len(labels) < n:
            full_labels = np.arange(n, dtype=np.intp)
            vertices = cc_df['vertex'].to_numpy()
            full_labels[vertices] = labels
            # Remap to contiguous ids
            _, labels = np.unique(full_labels, return_inverse=True)

        return labels
