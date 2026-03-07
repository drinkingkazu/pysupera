"""Tests for partitioning conditions."""
import numpy as np
import pytest

from pysupera.conditions.photon_decay import PhotonDecay
from pysupera.conditions.touching_em_shower import TouchingEMShower
from pysupera.conditions.combine_le_scatters import CombineLEScatters
from pysupera.conditions.absorb_le_scatter import AbsorbLEScatter
from pysupera.utils import SemanticType
from tests.conftest import (
    make_particle, cloud,
    PT_PRIMARY, PT_TRACK, PT_IONIZATION, PT_COMPTON, PT_PHOTON, PT_DELTA,
)


def _part(*particles, D=5.0):
    """Build a ParticlePartitioner for the given particles."""
    from tests.conftest import make_partitioner
    return make_partitioner(list(particles), D=D)


def _find_partition(partitions, pid):
    """Return the partition that contains the particle with the given id."""
    for part in partitions:
        if any(p.id == pid for p in part):
            return part
    raise KeyError(pid)


def _same_partition(partitions, pid_a, pid_b):
    """True iff pid_a and pid_b are in the same partition."""
    return _find_partition(partitions, pid_a) is _find_partition(partitions, pid_b)


# ============================================================================
# PhotonDecay
# ============================================================================

class TestPhotonDecay:
    def test_electron_child_of_photon_merges(self):
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, root_id=1)
        electron = make_particle(2, PT_PRIMARY, pdg=11, parent_pdg=22,
                                  parent_id=1, root_id=1,
                                  offset=(0, 0, 0))
        prt = _part(photon, electron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_positron_child_of_photon_merges(self):
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, root_id=1)
        positron = make_particle(2, PT_PRIMARY, pdg=-11, parent_pdg=22,
                                  parent_id=1, root_id=1,
                                  offset=(0.1, 0, 0))
        prt = _part(photon, positron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_both_decay_products_in_same_partition(self):
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, root_id=1)
        electron = make_particle(2, PT_PRIMARY, pdg=11, parent_pdg=22,
                                  parent_id=1, root_id=1)
        positron = make_particle(3, PT_PRIMARY, pdg=-11, parent_pdg=22,
                                  parent_id=1, root_id=1,
                                  offset=(0.2, 0, 0))
        prt = _part(photon, electron, positron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert _same_partition(parts, 1, 2)
        assert _same_partition(parts, 1, 3)

    def test_non_photon_parent_no_merge(self):
        # parent is a track (parent_pdg=13, not 22) → no PhotonDecay merge
        track = make_particle(1, PT_TRACK, pdg=13,
                               parent_id=1, root_id=1,
                               offset=(0, 0, 0))
        electron = make_particle(2, PT_PRIMARY, pdg=11, parent_pdg=13,
                                  parent_id=1, root_id=1,
                                  offset=(50, 0, 0))  # far away
        prt = _part(track, electron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_result_has_fewer_or_equal_partitions_than_particles(self):
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, root_id=1)
        electron = make_particle(2, PT_PRIMARY, pdg=11,
                                  parent_id=1, root_id=1)
        prt = _part(photon, electron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert len(parts) <= 2


# ============================================================================
# TouchingEMShower
# ============================================================================

class TestTouchingEMShower:
    def test_touching_parent_child_electrons_merge(self):
        parent = make_particle(1, PT_PRIMARY, pdg=11,
                                parent_id=1, root_id=1,
                                offset=(0, 0, 0))
        child = make_particle(2, PT_COMPTON, pdg=11,
                               parent_id=1, root_id=1,
                               offset=(0.05, 0, 0))  # very close → touching
        prt = _part(parent, child)
        parts = prt.partition(TouchingEMShower(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_different_ancestor_no_merge(self):
        parent = make_particle(1, PT_PRIMARY, pdg=11,
                                parent_id=1, root_id=1,
                                offset=(0, 0, 0))
        # child has different root_id → TouchingEMShower filter rejects
        child = make_particle(2, PT_COMPTON, pdg=11,
                               parent_id=1, root_id=99,   # different ancestor
                               offset=(0.05, 0, 0))
        prt = _part(parent, child)
        parts = prt.partition(TouchingEMShower(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_non_touching_parent_child_not_merged(self):
        parent = make_particle(1, PT_PRIMARY, pdg=11,
                                parent_id=1, root_id=1,
                                offset=(0, 0, 0))
        child = make_particle(2, PT_COMPTON, pdg=11,
                               parent_id=1, root_id=1,
                               offset=(100, 0, 0))  # far away
        prt = _part(parent, child, D=5.0)
        parts = prt.partition(TouchingEMShower(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_track_particle_not_included_as_candidate(self):
        # A track (pdg=13) parent should not produce EM shower candidates
        parent = make_particle(1, PT_TRACK, pdg=13,
                                parent_id=1, root_id=1,
                                offset=(0, 0, 0))
        child = make_particle(2, PT_COMPTON, pdg=11,
                               parent_id=1, root_id=1,
                               offset=(0.05, 0, 0))
        prt = _part(parent, child)
        parts = prt.partition(TouchingEMShower(), verbose=False)
        assert not _same_partition(parts, 1, 2)


# ============================================================================
# CombineLEScatters
# ============================================================================

class TestCombineLEScatters:
    def test_two_touching_le_scatters_merge(self):
        le1 = make_particle(1, PT_IONIZATION, pdg=11,
                             parent_id=1, root_id=1,
                             offset=(0, 0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11,
                             parent_id=2, root_id=2,
                             offset=(0.05, 0, 0))  # touching
        assert le1.sem_type == SemanticType.kLEScatter
        assert le2.sem_type == SemanticType.kLEScatter
        prt = _part(le1, le2)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_non_touching_le_scatters_not_merged(self):
        le1 = make_particle(1, PT_IONIZATION, pdg=11, offset=(0, 0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11, offset=(100, 0, 0))
        prt = _part(le1, le2, D=5.0)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_shower_not_absorbed_by_combine_le(self):
        # CombineLEScatters only looks at LE-LE pairs → shower unaffected
        le = make_particle(1, PT_IONIZATION, pdg=11, offset=(0, 0, 0))
        shower = make_particle(2, PT_PRIMARY, pdg=11, offset=(0.05, 0, 0))
        assert shower.sem_type == SemanticType.kShower
        prt = _part(le, shower)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_transitive_chain_all_merged(self):
        # A touches B, B touches C, A does not touch C → expect 1 partition
        le_a = make_particle(1, PT_IONIZATION, pdg=11, offset=(0,    0, 0))
        le_b = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        le_c = make_particle(3, PT_IONIZATION, pdg=11, offset=(0.10, 0, 0))
        prt = _part(le_a, le_b, le_c, D=0.1)   # D just enough to link neighbors
        parts = prt.partition(CombineLEScatters(), verbose=False)
        # All three should be in one partition
        assert _same_partition(parts, 1, 2)
        assert _same_partition(parts, 2, 3)

    def test_representative_is_largest_cloud(self):
        # le2 has more points → should be representative after merge
        le1 = make_particle(1, PT_IONIZATION, pdg=11, n_pts=3,  offset=(0,    0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11, n_pts=15, offset=(0.05, 0, 0))
        prt = _part(le1, le2)
        prt.partition(CombineLEScatters(), verbose=False)
        rep1 = prt.get_representative(1)
        rep2 = prt.get_representative(2)
        assert rep1.id == rep2.id   # same representative
        assert rep1.id == 2         # le2 is representative (largest cloud)


# ============================================================================
# AbsorbLEScatter
# ============================================================================

class TestAbsorbLEScatter:
    def test_le_scatter_absorbed_by_touching_shower(self):
        shower = make_particle(1, PT_PRIMARY, pdg=11, offset=(0,    0, 0))
        le     = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        assert shower.sem_type == SemanticType.kShower
        assert le.sem_type     == SemanticType.kLEScatter
        prt = _part(shower, le)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_isolated_le_scatter_stays_singleton(self):
        le = make_particle(1, PT_IONIZATION, pdg=11, offset=(0, 0, 0))
        # No other particle nearby
        prt = _part(le)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert len(parts) == 1
        assert len(parts[0]) == 1

    def test_le_scatter_not_close_to_shower_not_absorbed(self):
        shower = make_particle(1, PT_PRIMARY, pdg=11, offset=(0,   0, 0))
        le     = make_particle(2, PT_IONIZATION, pdg=11, offset=(100, 0, 0))
        prt = _part(shower, le, D=5.0)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_le_scatter_absorbed_by_at_most_one_neighbor(self):
        # Two showers close to one LE scatter → LE absorbed into exactly one
        shower_a = make_particle(1, PT_PRIMARY, pdg=11, offset=(0,    0, 0))
        shower_b = make_particle(2, PT_PRIMARY, pdg=11, offset=(0.2,  0, 0))
        le       = make_particle(3, PT_IONIZATION, pdg=11, offset=(0.1, 0, 0))
        prt = _part(shower_a, shower_b, le)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        # All three may end up together (via le), but le is in exactly one partition
        le_part = _find_partition(parts, 3)
        assert le_part is not None
        # shower_a and shower_b may or may not be together; le belongs to one
        assert len(le_part) >= 2

    def test_le_scatter_absorbed_into_track(self):
        track = make_particle(1, PT_TRACK, pdg=13, offset=(0, 0, 0))
        le    = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        assert track.sem_type == SemanticType.kTrack
        prt = _part(track, le)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert _same_partition(parts, 1, 2)
