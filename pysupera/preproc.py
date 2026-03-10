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
                root_id  = p.root_id,
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
                 sem_types=None,
                 n_jobs: int = 1):
        self.eps         = float(distance_threshold)
        self.eps2        = self.eps ** 2
        self.min_pc_size = int(min_pc_size)
        self.verbose     = bool(verbose)
        self.n_jobs      = int(n_jobs)
        # sem_types: frozenset of SemanticType values to process, or None for all.
        # Particles whose sem_type is not in the set are passed through unchanged.
        if sem_types is None or len(sem_types) == 0:
            self.sem_types: Optional[frozenset] = None
        else:
            self.sem_types = frozenset(sem_types)
        #: List of :class:`FragmentRecord` objects from the most recent
        #: :meth:`process` call.  Always populated regardless of *verbose*.
        self.last_diagnostics: List[FragmentRecord] = []
        #: Summary counts from the most recent :meth:`process` call.
        self.last_stats: dict = {}

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
            self.last_stats = {'n_particles': 0, 'n_skipped': 0,
                               'n_early_exit': 0, 'n_cc_checked': 0,
                               'n_fragmented': 0, 'n_spawned': 0,
                               'n_jobs': self.n_jobs,
                               'cc_pts_median': 0, 'cc_pts_max': 0, 'cc_pts_total': 0}
            return particles

        be_verbose = self.verbose if verbose is None else bool(verbose)
        self.last_diagnostics = []

        max_id  = max(p.id for p in particles)
        next_id = max_id + 1
        spawned: list[Particle] = []

        n_skipped      = 0
        n_early_exit   = 0
        n_cc_checked   = 0
        n_fragmented   = 0
        n_spawned_total = 0

        # ---------------------------------------------------------------
        # First pass: fast early-exit checks; collect particles needing CC
        # ---------------------------------------------------------------
        # result_slots preserves input order; None = to be filled after CC.
        result_slots: list = [None] * len(particles)
        # needs_cc: (original_index, particle, xyz)
        # precomp:  original_index → pre-computed labels (2-pt case)
        needs_cc: list = []
        precomp:  dict = {}

        for i, p in enumerate(particles):
            # Skip particles whose semantic type is not in the allowed set.
            if self.sem_types is not None and p.sem_type not in self.sem_types:
                result_slots[i] = p
                n_skipped += 1
                continue

            pc = p.point_cloud

            # ---- fast early exits (cheapest first) ---------------------

            # 1. Single point: trivially one cluster
            if len(pc) <= 1:
                result_slots[i] = p
                n_early_exit += 1
                continue

            xyz = pc[:, :3]

            # 2. Two points: O(1) distance comparison
            if len(xyz) == 2:
                if ((xyz[0] - xyz[1]) ** 2).sum() <= self.eps2:
                    result_slots[i] = p
                    n_early_exit += 1
                    continue
                # Two isolated points — labels known without CC
                precomp[i] = np.array([0, 1], dtype=np.intp)
                needs_cc.append((i, p, xyz))
                n_cc_checked += 1
                continue

            # 3. Bounding-box diagonal: if the whole cloud fits inside a
            #    ball of radius eps, all pairs are within eps → 1 cluster
            span = xyz.max(axis=0) - xyz.min(axis=0)
            if (span ** 2).sum() <= self.eps2:
                result_slots[i] = p
                n_early_exit += 1
                continue

            # 4. Needs full CC
            needs_cc.append((i, p, xyz))
            n_cc_checked += 1

        # ---------------------------------------------------------------
        # Parallel (or sequential) CC computation
        # ---------------------------------------------------------------
        to_compute = [(idx, p, xyz) for idx, p, xyz in needs_cc
                      if idx not in precomp]

        if to_compute:
            if self.n_jobs != 1 and len(to_compute) > 1:
                import math
                from joblib import Parallel, delayed
                # Determine effective worker count (joblib uses cpu_count for -1)
                import os
                n_workers = (
                    os.cpu_count() or 1
                    if self.n_jobs < 0
                    else self.n_jobs
                )
                # Chunk into n_workers groups to eliminate per-task overhead.
                # Each worker processes its chunk serially — joblib dispatches
                # only n_workers tasks instead of len(to_compute).
                chunk_size = max(1, math.ceil(len(to_compute) / n_workers))
                chunks = [
                    to_compute[i : i + chunk_size]
                    for i in range(0, len(to_compute), chunk_size)
                ]

                def _run_chunk(chunk):
                    return [self._get_labels(xyz) for _, _, xyz in chunk]

                nested = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(_run_chunk)(chunk) for chunk in chunks
                )
                labels_list = [lbl for group in nested for lbl in group]
            else:
                labels_list = [self._get_labels(xyz) for _, _, xyz in to_compute]

            for (idx, _, _), labels in zip(to_compute, labels_list):
                precomp[idx] = labels

        # ---------------------------------------------------------------
        # Second pass: split fragments (sequential — needs next_id order)
        # ---------------------------------------------------------------
        for idx, p, xyz in needs_cc:
            pc     = p.point_cloud
            labels = precomp[idx]

            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_sizes = sorted(counts.tolist(), reverse=True)
            n_clusters = len(unique_labels)

            kept, new_particles, next_id = _split_fragments(
                p, labels, self.min_pc_size, next_id
            )
            kept_pts    = int(kept.point_cloud.shape[0]) if kept is not None else 0
            spawned_ids = [sp.id for sp in new_particles]

            rec = FragmentRecord(
                particle_id   = p.id,
                n_points      = len(pc),
                early_exit    = None,
                n_clusters    = n_clusters,
                cluster_sizes = cluster_sizes,
                kept_pts      = kept_pts,
                spawned_ids   = spawned_ids,
            )
            self.last_diagnostics.append(rec)

            if be_verbose:
                print(rec)

            if kept is not None:
                result_slots[idx] = kept
            # else: result_slots[idx] remains None → excluded from output

            spawned.extend(new_particles)

            if rec.fragmented:
                n_fragmented    += 1
                n_spawned_total += len(new_particles)

        result = [p for p in result_slots if p is not None]

        cc_sizes = [len(xyz) for _, _, xyz in needs_cc]
        self.last_stats = {
            'n_particles'   : len(particles),
            'n_skipped'     : n_skipped,
            'n_early_exit'  : n_early_exit,
            'n_cc_checked'  : n_cc_checked,
            'n_fragmented'  : n_fragmented,
            'n_spawned'     : n_spawned_total,
            'n_jobs'        : self.n_jobs,
            'cc_pts_median' : int(np.median(cc_sizes)) if cc_sizes else 0,
            'cc_pts_max'    : int(np.max(cc_sizes)) if cc_sizes else 0,
            'cc_pts_total'  : int(np.sum(cc_sizes)) if cc_sizes else 0,
        }

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
        #: Summary counts from the most recent :meth:`process` call.
        self.last_stats: dict = {}

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

        n_affected = 0

        # --- batch: one np.unique call over all particles ----------------
        all_pcs  = [p.point_cloud for p in particles]
        lengths  = [len(pc) for pc in all_pcs]
        total    = sum(lengths)
        n_cols   = next((pc.shape[1] for pc in all_pcs if len(pc) > 0), None)

        if n_cols is None or total == 0:
            self.last_stats = {'n_particles': len(particles), 'n_affected': 0,
                               'pts_before': 0, 'pts_after': 0}
            if be_verbose:
                print(
                    f"[merge_duplicates] {len(particles)} particles | "
                    f"0 had duplicates | 0 points removed | 0 \u2192 0 total pts"
                )
            return particles

        n_total_before = total
        all_pts = np.concatenate(all_pcs)           # (total, n_cols)

        # key: (float64-pid, x, y, z) — respects exact float equality for xyz
        pids = np.repeat(np.arange(len(particles), dtype=np.float64), lengths)
        keys = np.empty((total, 4), dtype=np.float64)
        keys[:, 0]  = pids
        keys[:, 1:] = all_pts[:, :3]

        unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
        n_out = len(unique_keys)

        # Fast path: no duplicates at all — skip aggregation entirely
        if n_out == total:
            self.last_stats = {
                'n_particles' : len(particles),
                'n_affected'  : 0,
                'pts_before'  : total,
                'pts_after'   : total,
            }
            if be_verbose:
                print(
                    f"[merge_duplicates] {len(particles)} particles | "
                    f"0 had duplicates | 0 points removed | "
                    f"{total} \u2192 {total} total pts"
                )
            return particles

        out = np.zeros((n_out, n_cols), dtype=all_pts.dtype)
        out[:, :3] = unique_keys[:, 1:4].astype(all_pts.dtype)
        for col, init_val, ufunc in _MERGE_RULES:
            col = int(col)
            if col >= n_cols:
                continue
            out[:, col] = init_val
            ufunc.at(out[:, col], inv, all_pts[:, col])

        # split back per particle — unique_keys is sorted so pid is monotone
        pid_out    = unique_keys[:, 0].astype(np.int64)
        boundaries = np.searchsorted(pid_out, np.arange(len(particles) + 1))
        n_total_after = int(n_out)

        for i, p in enumerate(particles):
            start, end = int(boundaries[i]), int(boundaries[i + 1])
            n_before = lengths[i]
            n_after  = end - start
            if n_after < n_before:
                p.point_cloud = out[start:end]
                n_affected   += 1
                rec = MergeRecord(
                    particle_id = p.id,
                    n_before    = n_before,
                    n_after     = n_after,
                )
                self.last_diagnostics.append(rec)
                if be_verbose:
                    print(rec)

        self.last_stats = {
            'n_particles'  : len(particles),
            'n_affected'   : n_affected,
            'pts_before'   : n_total_before,
            'pts_after'    : n_total_after,
        }

        if be_verbose:
            print(
                f"[merge_duplicates] {len(particles)} particles | "
                f"{n_affected} had duplicates | "
                f"{n_total_before - n_total_after} points removed | "
                f"{n_total_before} \u2192 {n_total_after} total pts"
            )

        return particles


# ============================================================================
# Voxelization
# ============================================================================

@dataclass
class VoxelizeRecord:
    """
    Diagnostic record for one particle processed by
    :meth:`VoxelizeProcessor.process`.

    Attributes
    ----------
    particle_id : int
        ID of the particle that was processed.
    n_before : int
        Number of points before voxelization.
    n_after : int
        Number of voxels (points after merging).
    """
    particle_id : int
    n_before    : int
    n_after     : int

    @property
    def n_merged(self) -> int:
        """Number of points collapsed."""
        return self.n_before - self.n_after

    def __str__(self) -> str:
        return (
            f"  particle {self.particle_id:>8d} | "
            f"{self.n_before:>6d} pts → {self.n_after:>6d} voxels "
            f"({self.n_merged} merged)"
        )


def _voxelize_point_cloud(
    pc: np.ndarray,
    voxel_size: np.ndarray,
    origin: np.ndarray | None,
) -> np.ndarray:
    """
    Bin each point into a regular 3-D grid and merge points that share the
    same voxel.

    Parameters
    ----------
    pc : np.ndarray, shape (N, F), F >= 3
        Point-cloud array.  Columns 0–2 are x, y, z.
    voxel_size : np.ndarray, shape (3,)
        Side lengths of each voxel cell along x, y, z.
    origin : np.ndarray of shape (3,) or None
        Lower corner from which voxel indices are computed.  When ``None``
        the per-cloud minimum of the input coordinates is used so that the
        grid is always tightly aligned to the data.

    Returns
    -------
    np.ndarray, shape (M, F), M <= N
        New array where each row represents one non-empty voxel.  The
        x, y, z columns hold the voxel-centre coordinates; remaining
        feature columns are aggregated by the same rules as
        :func:`_merge_point_cloud` (time=min, energy=sum, dedx=max).
        Returns the original array unchanged when every voxel already
        contains exactly one point.
    """
    coords = pc[:, :3]
    if len(coords) == 0:
        return pc   # nothing to voxelize
    org    = coords.min(axis=0) if origin is None else origin

    # Integer voxel indices for every point
    idx = np.floor((coords - org) / voxel_size).astype(np.int64)

    # Unique voxels and group membership
    unique_idx, inv = np.unique(idx, axis=0, return_inverse=True)
    n_out = len(unique_idx)

    if n_out == len(pc):
        return pc   # already one point per voxel — no copy needed

    n_cols = pc.shape[1]
    out    = np.zeros((n_out, n_cols), dtype=pc.dtype)

    # Voxel-centre coordinates
    out[:, :3] = (unique_idx + 0.5) * voxel_size + org

    for col, init_val, ufunc in _MERGE_RULES:
        col = int(col)
        if col >= n_cols:
            continue
        out[:, col] = init_val
        ufunc.at(out[:, col], inv, pc[:, col])

    return out


class VoxelizeProcessor:
    """
    Pre-processing stage that bins each particle's point cloud onto a
    regular 3-D grid and merges all points that fall into the same voxel.

    This is a strict generalisation of :class:`MergeDuplicatesProcessor`:
    with an infinitesimally small ``voxel_size`` the two are equivalent.
    When voxelization is enabled, running :class:`MergeDuplicatesProcessor`
    afterwards is redundant.

    By setting ``merge_duplicates=True`` the voxelizer signals to the
    :class:`~pysupera.config.Pipeline` that it already covers duplicate-
    coordinate merging and the separate :class:`MergeDuplicatesProcessor`
    step should be skipped.  This is always correct: two points that share
    exactly the same ``(x, y, z)`` always fall into the same voxel cell,
    so they are guaranteed to be collapsed by the voxelization pass.

    Voxel-centre positions
    ~~~~~~~~~~~~~~~~~~~~~~
    For a point at coordinate ``x`` the voxel index is::

        i = floor( (x - origin_x) / voxel_size_x )

    and its centre is placed at::

        x_centre = (i + 0.5) * voxel_size_x + origin_x

    Feature aggregation (columns beyond index 2)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Identical to :class:`MergeDuplicatesProcessor`:
    time → min, energy → sum, dedx → max.

    Parameters
    ----------
    voxel_size : float or sequence of float
        Voxel side length.  A single float gives an isotropic grid;
        a 3-element sequence ``[dx, dy, dz]`` gives an anisotropic grid.
    origin : sequence of float or None, optional
        Lower corner ``[x0, y0, z0]`` from which voxel indices are
        computed.  ``None`` (default) uses the per-cloud coordinate
        minimum so that the grid is tightly aligned to each particle's
        data independently.  Use a fixed value when a globally consistent
        grid is needed (e.g. detector boundaries).
    verbose : bool, optional
        When ``True``, :meth:`process` prints one line per affected
        particle and an end-of-batch summary.  Default ``False``.
    merge_duplicates : bool, optional
        When ``True``, signals that this step also subsumes a preceding
        :class:`MergeDuplicatesProcessor`.  The voxelization algorithm is
        unchanged (exact-coordinate duplicates already land in the same
        voxel). Setting this to ``True`` lets the
        :class:`~pysupera.config.Pipeline` skip a redundant separate
        merge-duplicates pass.  Default ``False``.
    """

    def __init__(
        self,
        voxel_size: float | list,
        origin: list | None = None,
        verbose: bool = False,
        merge_duplicates: bool = False,
    ) -> None:
        vs = np.asarray(voxel_size, dtype=float)
        if vs.ndim == 0:
            vs = np.broadcast_to(vs, (3,)).copy()
        if vs.shape != (3,):
            raise ValueError(
                f"voxel_size must be a scalar or a 3-element sequence, "
                f"got shape {vs.shape}."
            )
        if np.any(vs <= 0):
            raise ValueError(f"All voxel_size values must be > 0, got {vs}.")
        self.voxel_size: np.ndarray = vs

        if origin is not None:
            org = np.asarray(origin, dtype=float)
            if org.shape != (3,):
                raise ValueError(
                    f"origin must be a 3-element sequence or None, "
                    f"got shape {org.shape}."
                )
            self.origin: np.ndarray | None = org
        else:
            self.origin = None

        self.verbose = bool(verbose)
        #: When ``True`` the Pipeline skips the separate MergeDuplicatesProcessor.
        self.merge_duplicates: bool = bool(merge_duplicates)
        #: :class:`VoxelizeRecord` list from the most recent :meth:`process`
        #: call.  Only particles that had at least one merge are recorded.
        self.last_diagnostics: List[VoxelizeRecord] = []
        #: Summary counts from the most recent :meth:`process` call.
        self.last_stats: dict = {}

    def process(
        self,
        particles: List[Particle],
        verbose: Optional[bool] = None,
    ) -> List[Particle]:
        """
        Voxelize every particle's point cloud in-place.

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

        n_affected = 0

        # --- batch: one np.unique call over all particles ----------------
        all_pcs  = [p.point_cloud for p in particles]
        lengths  = [len(pc) for pc in all_pcs]
        total    = sum(lengths)
        n_cols   = next((pc.shape[1] for pc in all_pcs if len(pc) > 0), None)

        if n_cols is None or total == 0:
            self.last_stats = {'n_particles': len(particles), 'n_affected': 0,
                               'pts_before': 0, 'pts_after': 0}
            if be_verbose:
                print(
                    f"[voxelize] {len(particles)} particles | "
                    f"0 affected | 0 points merged | 0 \u2192 0 total pts"
                )
            return particles

        n_total_before = total
        all_pts = np.concatenate(all_pcs)           # (total, n_cols)
        coords  = all_pts[:, :3]

        # Per-particle origins (preserves per-particle grid alignment)
        if self.origin is None:
            # compute per-particle min; empty particles get zeros
            starts = np.cumsum([0] + lengths[:-1])
            orgs   = np.array([
                coords[s:s + l].min(axis=0) if l > 0 else np.zeros(3)
                for s, l in zip(starts, lengths)
            ])                                      # (n_parts, 3)
        else:
            orgs = np.broadcast_to(self.origin, (len(particles), 3))

        # Per-point origin (replicated from particle)
        pt_orgs  = np.repeat(orgs, lengths, axis=0)  # (total, 3)

        # Integer voxel indices per point
        vox_idx = np.floor(
            (coords - pt_orgs) / self.voxel_size
        ).astype(np.int64)                            # (total, 3)

        # Key: (pid, vx, vy, vz) as int64
        pids = np.repeat(np.arange(len(particles), dtype=np.int64), lengths)
        keys = np.concatenate([pids[:, None], vox_idx], axis=1)  # (total, 4)

        unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
        n_out = len(unique_keys)

        # Voxel-centre coordinates per output row
        out_pid  = unique_keys[:, 0]                  # (n_out,)
        out_orgs = orgs[out_pid]                      # (n_out, 3)
        out = np.zeros((n_out, n_cols), dtype=all_pts.dtype)
        out[:, :3] = (
            (unique_keys[:, 1:4] + 0.5) * self.voxel_size + out_orgs
        ).astype(all_pts.dtype)

        for col, init_val, ufunc in _MERGE_RULES:
            col = int(col)
            if col >= n_cols:
                continue
            out[:, col] = init_val
            ufunc.at(out[:, col], inv, all_pts[:, col])

        # split back per particle — unique_keys sorted so pid is monotone
        boundaries  = np.searchsorted(out_pid, np.arange(len(particles) + 1))
        n_total_after = int(n_out)

        for i, p in enumerate(particles):
            start, end = int(boundaries[i]), int(boundaries[i + 1])
            n_before = lengths[i]
            n_after  = end - start
            if n_after < n_before:
                p.point_cloud = out[start:end]
                n_affected   += 1
                rec = VoxelizeRecord(
                    particle_id = p.id,
                    n_before    = n_before,
                    n_after     = n_after,
                )
                self.last_diagnostics.append(rec)
                if be_verbose:
                    print(rec)

        self.last_stats = {
            'n_particles'  : len(particles),
            'n_affected'   : n_affected,
            'pts_before'   : n_total_before,
            'pts_after'    : n_total_after,
        }

        if be_verbose:
            print(
                f"[voxelize] {len(particles)} particles | "
                f"{n_affected} affected | "
                f"{n_total_before - n_total_after} points merged | "
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
                 sem_types=None, n_jobs: int = 1):
        super().__init__(distance_threshold, min_pc_size, verbose=verbose,
                         sem_types=sem_types, n_jobs=n_jobs)
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
                 sem_types=None, n_jobs: int = 1):
        super().__init__(distance_threshold, min_pc_size, verbose=verbose,
                         sem_types=sem_types, n_jobs=n_jobs)
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
