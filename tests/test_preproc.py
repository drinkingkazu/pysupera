"""Tests for MergeDuplicatesProcessor and ScipyDefragmenter."""
import numpy as np
import pytest

from pysupera.preproc import MergeDuplicatesProcessor, ScipyDefragmenter, VoxelizeProcessor
from pysupera.utils import SemanticType, PointFeature
from tests.conftest import make_particle, cloud, PT_PRIMARY, PT_TRACK, PT_NEUTRON


# ============================================================================
# Helpers
# ============================================================================

def pc6(*rows):
    """Build a 6-column point cloud from (x, y, z, time, energy, dedx) rows."""
    return np.array(rows, dtype=np.float32)


# ============================================================================
# MergeDuplicatesProcessor
# ============================================================================

class TestMergeDuplicatesProcessor:
    def setup_method(self):
        self.proc = MergeDuplicatesProcessor()

    def test_no_duplicates_unchanged(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0, 0, 0, 0, 1, 0.5],
                                 [1, 0, 0, 1, 2, 0.3]))
        original_len = len(p.point_cloud)
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == original_len

    def test_duplicate_xyz_collapses_to_one_row(self):
        # Two rows with same xyz → collapsed to one
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([1, 2, 3, 0.1, 5.0, 1.0],
                                 [1, 2, 3, 0.5, 3.0, 2.0]))  # same xyz
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == 1

    def test_time_takes_minimum(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0, 0, 0, 10.0, 1.0, 0.5],
                                 [0, 0, 0,  2.0, 1.0, 0.5]))
        self.proc.process([p])
        assert p.point_cloud[0, 3] == pytest.approx(2.0)

    def test_energy_takes_sum(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0, 0, 0, 1.0, 3.0, 0.5],
                                 [0, 0, 0, 1.0, 4.0, 0.5]))
        self.proc.process([p])
        assert p.point_cloud[0, 4] == pytest.approx(7.0)

    def test_dedx_takes_maximum(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0, 0, 0, 1.0, 1.0, 1.5],
                                 [0, 0, 0, 1.0, 1.0, 3.2]))
        self.proc.process([p])
        assert p.point_cloud[0, 5] == pytest.approx(3.2)

    def test_empty_point_cloud_passes_through(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=np.zeros((0, 6), dtype=np.float32))
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == 0

    def test_single_point_unchanged(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([5, 5, 5, 1.0, 2.0, 0.5]))
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == 1

    def test_multiple_particles_processed_independently(self):
        p1 = make_particle(1, PT_PRIMARY, pdg=11,
                           pc=pc6([0, 0, 0, 1, 1, 1], [0, 0, 0, 2, 2, 2]))
        p2 = make_particle(2, PT_TRACK,   pdg=13,
                           pc=pc6([9, 9, 9, 1, 1, 1], [8, 8, 8, 2, 2, 2]))
        result = self.proc.process([p1, p2])
        assert len(result[0].point_cloud) == 1   # p1 had duplicate
        assert len(result[1].point_cloud) == 2   # p2 had no duplicate

    def test_last_stats_populated_no_duplicates(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0, 0, 0, 1, 2, 0.5],
                                 [1, 0, 0, 2, 3, 0.3]))
        self.proc.process([p])
        s = self.proc.last_stats
        assert {'n_particles', 'n_affected', 'pts_before', 'pts_after'} <= s.keys()
        assert s['n_affected'] == 0
        assert s['pts_before'] == s['pts_after'] == 2

    def test_last_stats_populated_with_duplicates(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0, 0, 0, 0, 1, 0.5],
                                 [0, 0, 0, 1, 2, 0.3]))  # duplicate xyz
        self.proc.process([p])
        s = self.proc.last_stats
        assert s['n_affected'] == 1
        assert s['pts_before'] == 2
        assert s['pts_after'] == 1

    def test_diagnostic_records_only_duplicates(self):
        p_dup  = make_particle(1, PT_PRIMARY, pdg=11,
                               pc=pc6([0, 0, 0, 1, 1, 1], [0, 0, 0, 2, 2, 2]))
        p_clean = make_particle(2, PT_TRACK,   pdg=13,
                                pc=pc6([9, 9, 9, 1, 1, 1], [8, 8, 8, 2, 2, 2]))
        self.proc.process([p_dup, p_clean])
        assert len(self.proc.last_diagnostics) == 1
        assert self.proc.last_diagnostics[0].particle_id == 1


# ============================================================================
# ScipyDefragmenter
# ============================================================================

def _cluster_pts(n, origin, spread=0.05):
    """Return n points tightly clustered around `origin`."""
    rng = np.random.default_rng(42)
    xyz = rng.uniform(0, spread, size=(n, 3)).astype(np.float32)
    xyz += np.array(origin, dtype=np.float32)
    return xyz


class TestScipyDefragmenter:
    EPS       = 5.0
    MIN_SIZE  = 3

    def defrag(self, sem_types=None):
        return ScipyDefragmenter(self.EPS, self.MIN_SIZE, sem_types=sem_types)

    # ── non-fragmented inputs ─────────────────────────────────────────────

    def test_contiguous_cloud_unchanged(self):
        pts = _cluster_pts(20, (0, 0, 0))
        p = make_particle(1, PT_PRIMARY, pdg=11, pc=pts)
        result = self.defrag().process([p])
        assert len(result) == 1
        assert len(result[0].point_cloud) == 20

    def test_empty_cloud_passes_through(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=np.zeros((0, 3), dtype=np.float32))
        result = self.defrag().process([p])
        assert len(result) == 1

    def test_single_point_cloud_passes_through(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=np.array([[1, 2, 3]], dtype=np.float32))
        result = self.defrag().process([p])
        assert len(result) == 1

    # ── fragmented: both clusters large ──────────────────────────────────

    def test_two_large_fragments_stay_together(self):
        # Both clusters are larger than min_pc_size → no splitting
        pts = np.vstack([
            _cluster_pts(10, (  0, 0, 0)),
            _cluster_pts(10, (100, 0, 0)),   # far away
        ])
        p = make_particle(1, PT_PRIMARY, pdg=11, pc=pts)
        result = self.defrag().process([p])
        # Both large → kept particle gets both, no spawning
        assert len(result) == 1
        assert len(result[0].point_cloud) == 20

    # ── fragmented: small fragment splits off ─────────────────────────────

    def test_small_fragment_splits_into_klescatter(self):
        # Large cluster (10 pts) + small cluster (2 pts, ≤ min_size=3) far apart
        large = _cluster_pts(10, (0, 0, 0))
        small = _cluster_pts(2, (100, 0, 0))
        pts = np.vstack([large, small])
        p = make_particle(1, PT_PRIMARY, pdg=11, pc=pts)
        result = self.defrag().process([p])
        # 1 kept particle + 1 spawned kLEScatter
        assert len(result) == 2
        kept    = next(r for r in result if r.id == 1)
        spawned = next(r for r in result if r.id != 1)
        assert len(kept.point_cloud) == 10
        assert len(spawned.point_cloud) == 2
        assert spawned.sem_type == SemanticType.kLEScatter

    def test_spawned_particle_has_correct_parentage(self):
        large = _cluster_pts(10, (0, 0, 0))
        small = _cluster_pts(2, (100, 0, 0))
        p = make_particle(42, PT_PRIMARY, pdg=11, pc=np.vstack([large, small]),
                          root_id=99)
        result = self.defrag().process([p])
        spawned = next(r for r in result if r.id != 42)
        assert spawned.parent_id == 42
        assert spawned.root_id == 99

    # ── sem_types filter ─────────────────────────────────────────────────

    def test_sem_types_filter_skips_non_matching(self):
        # kTrack particle (interaction_type=kTrack) should be skipped when sem_types=[kShower]
        large = _cluster_pts(10, (0, 0, 0))
        small = _cluster_pts(2, (100, 0, 0))
        p_track = make_particle(1, PT_TRACK, pdg=13, pc=np.vstack([large, small]))
        assert p_track.sem_type == SemanticType.kTrack

        defrag = self.defrag(sem_types=[SemanticType.kShower])
        result = defrag.process([p_track])
        # kTrack not in filter → passed through unchanged; no splitting
        assert len(result) == 1
        assert len(result[0].point_cloud) == 12   # original total

    def test_sem_types_filter_processes_matching(self):
        # kShower particle is processed even with sem_types=[kShower]
        large = _cluster_pts(10, (0, 0, 0))
        small = _cluster_pts(2, (100, 0, 0))
        p_shower = make_particle(1, PT_PRIMARY, pdg=11,
                                 pc=np.vstack([large, small]))
        assert p_shower.sem_type == SemanticType.kShower

        defrag = self.defrag(sem_types=[SemanticType.kShower])
        result = defrag.process([p_shower])
        assert len(result) == 2   # split happened

    def test_sem_types_none_processes_all(self):
        large = _cluster_pts(10, (0, 0, 0))
        small = _cluster_pts(2, (100, 0, 0))
        p = make_particle(1, PT_TRACK, pdg=13, pc=np.vstack([large, small]))
        result = self.defrag(sem_types=None).process([p])
        assert len(result) == 2   # kTrack still processed when sem_types=None

    # ── last_stats ────────────────────────────────────────────────────────

    def test_last_stats_populated_no_fragmentation(self):
        # Single contiguous cloud — bounding box early-exit applies
        pts = _cluster_pts(20, (0, 0, 0))
        p = make_particle(1, PT_PRIMARY, pdg=11, pc=pts)
        defrag = self.defrag()
        defrag.process([p])
        s = defrag.last_stats
        required = {'n_particles', 'n_skipped', 'n_early_exit', 'n_cc_checked',
                    'n_fragmented', 'n_spawned', 'n_jobs',
                    'cc_pts_median', 'cc_pts_max', 'cc_pts_total'}
        assert required <= s.keys()
        assert s['n_particles'] == 1
        assert s['n_fragmented'] == 0
        assert s['n_spawned'] == 0

    def test_last_stats_fragmented_counters(self):
        large = _cluster_pts(10, (0, 0, 0))
        small = _cluster_pts(2, (100, 0, 0))
        p = make_particle(1, PT_PRIMARY, pdg=11, pc=np.vstack([large, small]))
        defrag = self.defrag()
        defrag.process([p])
        s = defrag.last_stats
        assert s['n_fragmented'] == 1
        assert s['n_spawned'] == 1
        assert s['cc_pts_total'] > 0

    # ── diagnostic records ───────────────────────────────────────────────

    def test_fragmented_particle_has_diagnostic_record(self):
        large = _cluster_pts(10, (0, 0, 0))
        small = _cluster_pts(2, (100, 0, 0))
        p = make_particle(1, PT_PRIMARY, pdg=11, pc=np.vstack([large, small]))
        defrag = self.defrag()
        defrag.process([p])
        assert len(defrag.last_diagnostics) == 1
        assert defrag.last_diagnostics[0].fragmented is True

    def test_non_fragmented_particle_no_diagnostic(self):
        p = make_particle(1, PT_PRIMARY, pdg=11, pc=_cluster_pts(20, (0, 0, 0)))
        defrag = self.defrag()
        defrag.process([p])
        assert len(defrag.last_diagnostics) == 0


# ============================================================================
# VoxelizeProcessor
# ============================================================================

class TestVoxelizeProcessor:
    """Tests for VoxelizeProcessor (isotropic and anisotropic voxelization)."""

    def setup_method(self):
        # Isotropic 1-unit voxels anchored at the origin for predictability.
        self.proc = VoxelizeProcessor(voxel_size=1.0, origin=[0.0, 0.0, 0.0])

    # ── basic collapsing ─────────────────────────────────────────────────

    def test_two_points_same_voxel_collapse_to_one(self):
        # (0.2,0,0) and (0.7,0,0) both lie in voxel (0,0,0)
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.2, 0.0, 0.0, 1.0, 2.0, 0.5],
                                 [0.7, 0.0, 0.0, 3.0, 4.0, 1.5]))
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == 1

    def test_two_points_different_voxels_unchanged(self):
        # (0.2,0,0) → voxel (0,0,0); (1.5,0,0) → voxel (1,0,0)
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.2, 0.0, 0.0, 1.0, 2.0, 0.5],
                                 [1.5, 0.0, 0.0, 3.0, 4.0, 1.5]))
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == 2

    def test_voxel_centre_position(self):
        # voxel_size=1, origin=(0,0,0) → voxel (0,0,0) centre is (0.5,0.5,0.5)
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.3, 0.4, 0.2, 1.0, 1.0, 1.0],
                                 [0.8, 0.1, 0.9, 2.0, 2.0, 2.0]))
        result = self.proc.process([p])
        np.testing.assert_allclose(result[0].point_cloud[0, :3],
                                   [0.5, 0.5, 0.5], atol=1e-5)

    # ── feature aggregation ───────────────────────────────────────────────

    def test_time_takes_minimum(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.2, 0.0, 0.0, 10.0, 1.0, 0.5],
                                 [0.7, 0.0, 0.0,  2.0, 1.0, 0.5]))
        result = self.proc.process([p])
        assert result[0].point_cloud[0, 3] == pytest.approx(2.0)

    def test_energy_takes_sum(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.2, 0.0, 0.0, 1.0, 3.0, 0.5],
                                 [0.7, 0.0, 0.0, 1.0, 4.0, 0.5]))
        result = self.proc.process([p])
        assert result[0].point_cloud[0, 4] == pytest.approx(7.0)

    def test_dedx_takes_maximum(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.2, 0.0, 0.0, 1.0, 1.0, 1.5],
                                 [0.7, 0.0, 0.0, 1.0, 1.0, 3.2]))
        result = self.proc.process([p])
        assert result[0].point_cloud[0, 5] == pytest.approx(3.2)

    # ── edge cases ────────────────────────────────────────────────────────

    def test_empty_cloud_passes_through(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=np.zeros((0, 6), dtype=np.float32))
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == 0

    def test_single_point_passes_through_unchanged(self):
        # A single point has nothing to merge; original coordinates are preserved.
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.3, 0.3, 0.3, 1.0, 1.0, 1.0]))
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == 1
        np.testing.assert_allclose(result[0].point_cloud[0, :3],
                                   [0.3, 0.3, 0.3], atol=1e-6)

    def test_multiple_particles_processed_independently(self):
        # p1: two points in the same voxel → collapse
        p1 = make_particle(1, PT_PRIMARY, pdg=11,
                           pc=pc6([0.2, 0.0, 0.0, 1.0, 1.0, 1.0],
                                  [0.7, 0.0, 0.0, 2.0, 2.0, 2.0]))
        # p2: two points in different voxels → no collapse
        p2 = make_particle(2, PT_TRACK, pdg=13,
                           pc=pc6([0.2, 0.0, 0.0, 1.0, 1.0, 1.0],
                                  [1.5, 0.0, 0.0, 2.0, 2.0, 2.0]))
        result = self.proc.process([p1, p2])
        assert len(result[0].point_cloud) == 1  # p1 collapsed
        assert len(result[1].point_cloud) == 2  # p2 unchanged

    # ── anisotropic voxel ────────────────────────────────────────────────

    def test_anisotropic_voxel_size(self):
        # dx=1, dy=2, dz=3; origin=(0,0,0)
        proc = VoxelizeProcessor(voxel_size=[1.0, 2.0, 3.0],
                                 origin=[0.0, 0.0, 0.0])
        # Both points fall in voxel (0,0,0): floor(0.2/1)=0, floor(0.5/2)=0, floor(0.5/3)=0
        # and floor(0.7/1)=0, floor(1.9/2)=0, floor(2.9/3)=0
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.2, 0.5, 0.5, 1.0, 2.0, 0.5],
                                 [0.7, 1.9, 2.9, 2.0, 3.0, 1.5]))
        result = proc.process([p])
        assert len(result[0].point_cloud) == 1
        # centre of voxel (0,0,0) with anisotropic grid: (0.5*1, 0.5*2, 0.5*3)
        np.testing.assert_allclose(result[0].point_cloud[0, :3],
                                   [0.5, 1.0, 1.5], atol=1e-5)

    # ── fixed vs per-particle origin ─────────────────────────────────────

    def test_fixed_origin_grid_alignment(self):
        # voxel_size=1, origin=(0,0,0): both points in voxel (1,0,0)
        # → merged centre at (1.5, 0.5, 0.5)
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([1.2, 0.0, 0.0, 1.0, 1.0, 1.0],
                                 [1.8, 0.0, 0.0, 2.0, 2.0, 2.0]))
        result = self.proc.process([p])
        assert len(result[0].point_cloud) == 1
        np.testing.assert_allclose(result[0].point_cloud[0, :3],
                                   [1.5, 0.5, 0.5], atol=1e-5)

    # ── merge_duplicates flag ────────────────────────────────────────────

    def test_merge_duplicates_flag_default_false(self):
        proc = VoxelizeProcessor(voxel_size=1.0)
        assert proc.merge_duplicates is False

    def test_merge_duplicates_flag_can_be_set(self):
        proc = VoxelizeProcessor(voxel_size=1.0, merge_duplicates=True)
        assert proc.merge_duplicates is True

    # ── last_stats ────────────────────────────────────────────────────────

    def test_last_stats_populated_after_merge(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.2, 0.0, 0.0, 1.0, 2.0, 0.5],
                                 [0.7, 0.0, 0.0, 3.0, 4.0, 1.5]))
        self.proc.process([p])
        s = self.proc.last_stats
        assert {'n_particles', 'n_affected', 'pts_before', 'pts_after'} <= s.keys()
        assert s['n_particles'] == 1
        assert s['n_affected'] == 1
        assert s['pts_before'] == 2
        assert s['pts_after'] == 1

    def test_last_stats_no_merge_when_all_separate_voxels(self):
        p = make_particle(1, PT_PRIMARY, pdg=11,
                          pc=pc6([0.2, 0.0, 0.0, 1.0, 2.0, 0.5],
                                 [1.5, 0.0, 0.0, 3.0, 4.0, 1.5]))
        self.proc.process([p])
        s = self.proc.last_stats
        assert s['n_affected'] == 0
        assert s['pts_before'] == s['pts_after']

    # ── diagnostics ──────────────────────────────────────────────────────

    def test_diagnostics_record_only_affected_particles(self):
        p_merged = make_particle(1, PT_PRIMARY, pdg=11,
                                 pc=pc6([0.2, 0.0, 0.0, 1.0, 1.0, 1.0],
                                        [0.7, 0.0, 0.0, 2.0, 2.0, 2.0]))
        p_clean  = make_particle(2, PT_TRACK,   pdg=13,
                                 pc=pc6([0.2, 0.0, 0.0, 1.0, 1.0, 1.0],
                                        [1.5, 0.0, 0.0, 2.0, 2.0, 2.0]))
        self.proc.process([p_merged, p_clean])
        assert len(self.proc.last_diagnostics) == 1
        assert self.proc.last_diagnostics[0].particle_id == 1

    # ── invalid constructor args ──────────────────────────────────────────

    def test_zero_voxel_size_raises(self):
        with pytest.raises(ValueError):
            VoxelizeProcessor(voxel_size=0.0)

    def test_negative_voxel_size_raises(self):
        with pytest.raises(ValueError):
            VoxelizeProcessor(voxel_size=-1.0)


# ============================================================================
# Voxelization → MergeDuplicatesProcessor roundtrip with mapping verification
# ============================================================================

def _pc7(*rows):
    """7-column point cloud: (x, y, z, time, energy, dedx, input_id)."""
    return np.array(rows, dtype=np.float32)


class TestVoxelizationMappingRoundtrip:
    """
    End-to-end mapping test.

    The pipeline under test is:

        input points (with explicit PointFeature.id in col 6)
            │
            ▼
        VoxelizeProcessor(store_mapping=True)   ← CSR records produced
            │
            ▼
        MergeDuplicatesProcessor                ← should be a no-op afterwards
            │
            ▼
            verify forward and inverse point → voxel mappings

    Two particles are used, with deterministic voxel assignments under a
    unit (1×1×1) grid anchored at the origin:

    Particle 1 (pid=1, 4 input points):
        input_id=0  (0.2, 0, 0, t=0, e=1.0, dedx=0.1)  → voxel (0,0,0) · centre (0.5,0.5,0.5)
        input_id=1  (0.7, 0, 0, t=0, e=2.0, dedx=0.2)  → voxel (0,0,0) ┘  energy = 3.0
        input_id=2  (1.2, 0, 0, t=0, e=3.0, dedx=0.3)  → voxel (1,0,0) · centre (1.5,0.5,0.5)
        input_id=3  (1.8, 0, 0, t=0, e=4.0, dedx=0.4)  → voxel (1,0,0) ┘  energy = 7.0

    Particle 2 (pid=2, 3 input points — identity mapping):
        input_id=10 (0.5, 0, 0, t=0, e=5.0, dedx=0.5)  → voxel (0,0,0)
        input_id=11 (1.5, 0, 0, t=0, e=6.0, dedx=0.6)  → voxel (1,0,0)
        input_id=12 (2.5, 0, 0, t=0, e=7.0, dedx=0.7)  → voxel (2,0,0)
    """

    # ------------------------------------------------------------------ fixtures

    def _make_particles(self):
        p1 = make_particle(
            1, PT_PRIMARY, pdg=11,
            pc=_pc7(
                [0.2, 0.0, 0.0, 0.0, 1.0, 0.1,  0],
                [0.7, 0.0, 0.0, 0.0, 2.0, 0.2,  1],
                [1.2, 0.0, 0.0, 0.0, 3.0, 0.3,  2],
                [1.8, 0.0, 0.0, 0.0, 4.0, 0.4,  3],
            ),
        )
        p2 = make_particle(
            2, PT_TRACK, pdg=13,
            pc=_pc7(
                [0.5, 0.0, 0.0, 0.0, 5.0, 0.5, 10],
                [1.5, 0.0, 0.0, 0.0, 6.0, 0.6, 11],
                [2.5, 0.0, 0.0, 0.0, 7.0, 0.7, 12],
            ),
        )
        return p1, p2

    def _vox(self):
        return VoxelizeProcessor(
            voxel_size=1.0,
            origin=[0.0, 0.0, 0.0],
            store_mapping=True,
        )

    def _rec(self, diagnostics, pid):
        return next(r for r in diagnostics if r.particle_id == pid)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _per_voxel_ids(rec):
        """Return a list of sets: per_voxel_ids[v] = {input IDs that merged into voxel v}."""
        return [
            set(int(x) for x in rec.input_ids[rec.voxel_offsets[v] : rec.voxel_offsets[v + 1]])
            for v in range(rec.n_after)
        ]

    @staticmethod
    def _inverse_map(rec):
        """Return dict {input_id: voxel_index} derived from the record."""
        inv = {}
        for v in range(rec.n_after):
            for iid in rec.input_ids[rec.voxel_offsets[v] : rec.voxel_offsets[v + 1]]:
                inv[int(iid)] = v
        return inv

    # ------------------------------------------------------------------ record counts

    def test_record_counts_after_voxelization(self):
        """store_mapping=True must produce one record per particle."""
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        pids = {r.particle_id for r in vox.last_diagnostics}
        assert pids == {1, 2}

    def test_n_before_n_after_particle1(self):
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        rec = self._rec(vox.last_diagnostics, 1)
        assert rec.n_before == 4
        assert rec.n_after  == 2

    def test_n_before_n_after_particle2_identity(self):
        """Particle 2 has all points in separate voxels → n_before == n_after."""
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        rec = self._rec(vox.last_diagnostics, 2)
        assert rec.n_before == 3
        assert rec.n_after  == 3

    # ------------------------------------------------------------------ forward mapping

    def test_forward_mapping_groups_particle1(self):
        """
        Particle 1: the two output voxels must partition input IDs into
        exactly {0, 1} and {2, 3}.
        """
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        rec = self._rec(vox.last_diagnostics, 1)
        groups = self._per_voxel_ids(rec)

        assert len(groups) == 2
        assert all(len(g) == 2 for g in groups), "Each voxel must have exactly 2 input points"
        assert {frozenset(g) for g in groups} == {frozenset({0, 1}), frozenset({2, 3})}

    def test_forward_mapping_covers_all_input_ids(self):
        """Union of all per-voxel input ID sets must equal the full input ID set."""
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        for pid, expected in [(1, {0, 1, 2, 3}), (2, {10, 11, 12})]:
            rec    = self._rec(vox.last_diagnostics, pid)
            groups = self._per_voxel_ids(rec)
            assert set().union(*groups) == expected, (
                f"Particle {pid}: union of voxel ID groups does not match expected"
            )

    def test_forward_mapping_identity_particle2(self):
        """Particle 2: each output voxel maps to exactly one distinct input ID."""
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        rec    = self._rec(vox.last_diagnostics, 2)
        groups = self._per_voxel_ids(rec)
        assert all(len(g) == 1 for g in groups)
        assert set().union(*groups) == {10, 11, 12}

    # ------------------------------------------------------------------ inverse mapping

    def test_inverse_mapping_covers_every_input(self):
        """
        Inverse map (input_id → voxel_index) must have exactly n_before entries,
        meaning every input point has a unique owning voxel.
        """
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        for rec in vox.last_diagnostics:
            inv = self._inverse_map(rec)
            assert len(inv) == rec.n_before, (
                f"Particle {rec.particle_id}: inverse map size {len(inv)} "
                f"!= n_before {rec.n_before}"
            )

    def test_inverse_mapping_no_input_assigned_to_two_voxels(self):
        """No input ID may appear in more than one voxel."""
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        for rec in vox.last_diagnostics:
            seen = {}
            for v in range(rec.n_after):
                for iid in rec.input_ids[rec.voxel_offsets[v] : rec.voxel_offsets[v + 1]]:
                    key = int(iid)
                    assert key not in seen, (
                        f"Input ID {key} assigned to voxels {seen[key]} and {v}"
                    )
                    seen[key] = v

    def test_inverse_mapping_correct_voxel_for_particle1(self):
        """
        For particle 1:
          input_id ∈ {0, 1} → same voxel (the (0,0,0)-voxel)
          input_id ∈ {2, 3} → same voxel (the (1,0,0)-voxel)
          the two voxels must be different.
        """
        p1, p2 = self._make_particles()
        vox = self._vox()
        vox.process([p1, p2])
        rec = self._rec(vox.last_diagnostics, 1)
        inv = self._inverse_map(rec)

        assert inv[0] == inv[1], "IDs 0 and 1 must map to the same voxel"
        assert inv[2] == inv[3], "IDs 2 and 3 must map to the same voxel"
        assert inv[0] != inv[2], "voxel (0,0,0) and voxel (1,0,0) must be different"

    # ------------------------------------------------------------------ energy consistency

    def test_energy_in_record_matches_input_energies(self):
        """
        For each voxel, the sum of rec.input_energies must equal the merged
        output voxel's energy column.
        """
        _EN = int(PointFeature.energy)
        p1, p2 = self._make_particles()
        vox = self._vox()
        results = vox.process([p1, p2])

        # Build lookup: particle_id → output point cloud
        out_clouds = {p.id: p.point_cloud for p in results}

        for rec in vox.last_diagnostics:
            cloud_out = out_clouds[rec.particle_id]
            for v in range(rec.n_after):
                sl       = slice(int(rec.voxel_offsets[v]), int(rec.voxel_offsets[v + 1]))
                eng_sum  = float(rec.input_energies[sl].sum())
                vox_eng  = float(cloud_out[v, _EN])
                np.testing.assert_allclose(
                    eng_sum, vox_eng, rtol=1e-5,
                    err_msg=f"Particle {rec.particle_id} voxel {v}: "
                            f"input energy sum {eng_sum} != voxel energy {vox_eng}",
                )

    def test_energy_known_values_particle1(self):
        """Explicit check: particle 1 voxel energies must be 3.0 and 7.0."""
        _EN = int(PointFeature.energy)
        p1, p2 = self._make_particles()
        vox = self._vox()
        results = vox.process([p1, p2])
        out_energies = sorted(float(e) for e in results[0].point_cloud[:, _EN])
        assert out_energies == pytest.approx([3.0, 7.0])

    # ------------------------------------------------------------------ merge-dups is a no-op

    def test_merge_duplicates_no_op_after_voxelization(self):
        """
        Running MergeDuplicatesProcessor after VoxelizeProcessor must leave
        point counts unchanged because voxelization already placed outputs at
        unique voxel-centre coordinates.
        """
        p1, p2 = self._make_particles()
        vox  = self._vox()
        mdup = MergeDuplicatesProcessor()

        results = vox.process([p1, p2])
        counts_before_merge = [len(p.point_cloud) for p in results]

        mdup.process(results)
        counts_after_merge = [len(p.point_cloud) for p in results]

        assert counts_before_merge == counts_after_merge

    def test_merge_duplicates_stats_show_no_affected_particles(self):
        """MergeDuplicatesProcessor.last_stats must report 0 affected particles."""
        p1, p2 = self._make_particles()
        vox  = self._vox()
        mdup = MergeDuplicatesProcessor()

        results = vox.process([p1, p2])
        mdup.process(results)

        assert mdup.last_stats['n_affected']  == 0
        assert mdup.last_stats['pts_before']  == mdup.last_stats['pts_after']

    # ------------------------------------------------------------------ output voxel IDs

    def test_output_voxel_col6_are_unique_per_particle(self):
        """
        After voxelization, col 6 (PointFeature.id) of each output cloud
        must contain unique integers — one per output voxel.
        """
        _ID = int(PointFeature.id)
        p1, p2 = self._make_particles()
        vox = self._vox()
        results = vox.process([p1, p2])
        for p in results:
            ids = p.point_cloud[:, _ID]
            assert len(ids) == len(set(ids.astype(int).tolist())), (
                f"Particle {p.id}: output voxel IDs are not unique: {ids}"
            )
