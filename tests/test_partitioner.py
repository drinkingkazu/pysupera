"""Integration tests for ParticlePartitioner."""
import numpy as np
import pytest

from pysupera.conditions.photon_decay import PhotonDecay
from pysupera.conditions.touching_em_shower import TouchingEMShower
from pysupera.conditions.combine_le_scatters import CombineLEScatters
from pysupera.conditions.absorb_le_scatter import AbsorbLEScatter
from pysupera.utils import SemanticType
from tests.conftest import (
    make_particle, cloud, make_partitioner,
    PT_PRIMARY, PT_TRACK, PT_IONIZATION, PT_COMPTON, PT_PHOTON,
)


# ============================================================================
# Basic partition structure
# ============================================================================

class TestPartitionBasics:
    def test_single_particle_stays_singleton(self):
        p = make_particle(1, PT_PRIMARY, pdg=11)
        prt = make_partitioner([p])
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert len(parts) == 1
        assert len(parts[0]) == 1

    def test_unrelated_far_particles_stay_separate(self):
        p1 = make_particle(1, PT_IONIZATION, pdg=11, offset=(0,   0, 0))
        p2 = make_particle(2, PT_IONIZATION, pdg=11, offset=(100, 0, 0))
        prt = make_partitioner([p1, p2], D=5.0)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert len(parts) == 2

    def test_all_particles_appear_in_result(self):
        particles = [
            make_particle(i, PT_IONIZATION, pdg=11, offset=(i * 50, 0, 0))
            for i in range(5)
        ]
        prt = make_partitioner(particles, D=5.0)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        all_ids = {p.id for part in parts for p in part}
        assert all_ids == {p.id for p in particles}

    def test_partition_count_never_exceeds_particle_count(self):
        particles = [make_particle(i, PT_IONIZATION, pdg=11) for i in range(6)]
        prt = make_partitioner(particles, D=5.0)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        assert len(parts) <= len(particles)

    def test_each_particle_in_exactly_one_partition(self):
        particles = [
            make_particle(i, PT_IONIZATION, pdg=11, offset=(i * 0.05, 0, 0))
            for i in range(4)
        ]
        prt = make_partitioner(particles, D=5.0)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        all_ids = [p.id for part in parts for p in part]
        assert len(all_ids) == len(set(all_ids))   # no duplicates


# ============================================================================
# get_representative
# ============================================================================

class TestGetRepresentative:
    def test_get_representative_raises_before_partition(self):
        p = make_particle(1, PT_PRIMARY, pdg=11)
        prt = make_partitioner([p])
        with pytest.raises(AttributeError):
            prt.get_representative(1)

    def test_get_representative_returns_particle(self):
        p = make_particle(1, PT_PRIMARY, pdg=11)
        prt = make_partitioner([p])
        prt.partition(AbsorbLEScatter(), verbose=False)
        rep = prt.get_representative(1)
        assert rep is not None

    def test_all_members_share_same_representative(self):
        le1 = make_particle(1, PT_IONIZATION, pdg=11, offset=(0,    0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        prt = make_partitioner([le1, le2])
        parts = prt.partition(CombineLEScatters(), verbose=False)
        rep1 = prt.get_representative(1)
        rep2 = prt.get_representative(2)
        assert rep1.id == rep2.id

    def test_representative_cloud_contains_all_points(self):
        le1 = make_particle(1, PT_IONIZATION, pdg=11, n_pts=5,  offset=(0,    0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11, n_pts=8,  offset=(0.05, 0, 0))
        prt = make_partitioner([le1, le2])
        prt.partition(CombineLEScatters(), verbose=False)
        rep = prt.get_representative(1)
        assert len(rep.point_cloud) == 5 + 8

    def test_unknown_particle_id_raises_keyerror(self):
        p = make_particle(1, PT_PRIMARY, pdg=11)
        prt = make_partitioner([p])
        prt.partition(AbsorbLEScatter(), verbose=False)
        with pytest.raises(KeyError):
            prt.get_representative(9999)

    def test_singleton_representative_is_itself(self):
        p = make_particle(42, PT_PRIMARY, pdg=11)
        prt = make_partitioner([p])
        prt.partition(AbsorbLEScatter(), verbose=False)
        rep = prt.get_representative(42)
        assert rep.id == 42


# ============================================================================
# partition_by_sem_type
# ============================================================================

class TestPartitionBySemType:
    def test_returns_dict(self):
        p = make_particle(1, PT_PRIMARY, pdg=11)
        prt = make_partitioner([p])
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        result = prt.partition_by_sem_type(parts)
        assert isinstance(result, dict)

    def test_keys_are_semantic_types(self):
        p = make_particle(1, PT_PRIMARY, pdg=11)
        prt = make_partitioner([p])
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        result = prt.partition_by_sem_type(parts)
        for key in result:
            assert isinstance(key, SemanticType)

    def test_total_partition_count_preserved(self):
        particles = [
            make_particle(1, PT_PRIMARY,    pdg=11, offset=(0,   0, 0)),
            make_particle(2, PT_TRACK,      pdg=13, offset=(100, 0, 0)),
            make_particle(3, PT_IONIZATION, pdg=11, offset=(200, 0, 0)),
        ]
        prt = make_partitioner(particles, D=5.0)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        by_type = prt.partition_by_sem_type(parts)
        total = sum(len(v) for v in by_type.values())
        assert total == len(parts)

    def test_shower_and_track_appear_as_separate_keys(self):
        # Particles far apart so they can't merge
        shower = make_particle(1, PT_PRIMARY, pdg=11, offset=(0,   0, 0))
        track  = make_particle(2, PT_TRACK,   pdg=13, offset=(100, 0, 0))
        prt = make_partitioner([shower, track], D=5.0)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        by_type = prt.partition_by_sem_type(parts)
        assert SemanticType.kShower in by_type
        assert SemanticType.kTrack  in by_type

    def test_only_present_types_are_keys(self):
        # Only kLEScatter particles → only kLEScatter key in result
        le1 = make_particle(1, PT_IONIZATION, pdg=11, offset=(0,   0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11, offset=(100, 0, 0))
        prt = make_partitioner([le1, le2], D=5.0)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        by_type = prt.partition_by_sem_type(parts)
        assert set(by_type.keys()) == {SemanticType.kLEScatter}


# ============================================================================
# Full pipeline integration
# ============================================================================

class TestFullPipeline:
    def _run_pipeline(self, particles, D=5.0):
        """Run the standard 4-condition pipeline via partition_combined."""
        from pysupera.partitioner import ParticlePartitioner
        prt = ParticlePartitioner(particles, distance_threshold=D, backend='cpu-single')
        parts = prt.partition_combined(
            [PhotonDecay(), TouchingEMShower(), CombineLEScatters(), AbsorbLEScatter()],
            verbose=False,
        )
        return prt, parts

    def test_all_particles_preserved(self):
        particles = [
            make_particle(1, PT_PHOTON, pdg=22,
                          pc=np.zeros((0, 3), dtype=np.float32),
                          parent_id=1, root_id=1),
            make_particle(2, PT_PRIMARY, pdg=11, parent_pdg=22,
                          parent_id=1, root_id=1),
            make_particle(3, PT_IONIZATION, pdg=11, offset=(50, 0, 0)),
        ]
        _, parts = self._run_pipeline(particles)
        all_ids = {p.id for part in parts for p in part}
        assert all_ids == {1, 2, 3}

    def test_photon_and_electron_merged_in_pipeline(self):
        photon   = make_particle(1, PT_PHOTON, pdg=22,
                                  pc=np.zeros((0, 3), dtype=np.float32),
                                  parent_id=1, root_id=1)
        electron = make_particle(2, PT_PRIMARY, pdg=11, parent_pdg=22,
                                  parent_id=1, root_id=1)
        _, parts = self._run_pipeline([photon, electron])
        # After PhotonDecay, they should be in 1 partition
        all_ids_per_part = [frozenset(p.id for p in part) for part in parts]
        assert frozenset({1, 2}) in all_ids_per_part

    def test_le_scatter_absorbed_at_end(self):
        shower = make_particle(1, PT_PRIMARY,    pdg=11, offset=(0,    0, 0))
        le     = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        _, parts = self._run_pipeline([shower, le])
        # After AbsorbLEScatter, le is merged into shower
        assert len(parts) == 1

    def test_partition_count_monotonically_non_increasing(self):
        """Re-running AbsorbLEScatter on already-absorbed result doesn't create new partitions."""
        from pysupera.partitioner import ParticlePartitioner
        shower = make_particle(1, PT_PRIMARY,    pdg=11, offset=(0,    0, 0))
        le     = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        prt = ParticlePartitioner([shower, le], distance_threshold=5.0, backend='cpu-single')
        parts1 = prt.partition(AbsorbLEScatter(), verbose=False)
        parts2 = prt.partition(AbsorbLEScatter(), verbose=False)
        assert len(parts2) <= len(parts1)
