"""Tests for MergeDuplicatesProcessor and ScipyDefragmenter."""
import numpy as np
import pytest

from pysupera.preproc import MergeDuplicatesProcessor, ScipyDefragmenter
from pysupera.utils import SemanticType
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
        # kTrack particle (process_type=0) should be skipped when sem_types=[kShower]
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
