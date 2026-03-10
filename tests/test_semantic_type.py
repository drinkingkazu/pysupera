"""Tests for SetSemanticType classification logic."""
import numpy as np
import pytest

from pysupera.utils import SetSemanticType, SemanticType, InteractionType

# Aliases so test bodies stay readable
PT_TRACK        = InteractionType.kTrack
PT_NEUTRON      = InteractionType.kNeutron
PT_NUCLEUS      = InteractionType.kNucleus
PT_PHOTON       = InteractionType.kPhoton
PT_PRIMARY      = InteractionType.kPrimary
PT_COMPTON      = InteractionType.kCompton
PT_DELTA        = InteractionType.kDelta
PT_CONVERSION   = InteractionType.kConversion
PT_IONIZATION   = InteractionType.kIonization
PT_PHOTO_ELEC   = InteractionType.kPhotoElectron
PT_DECAY        = InteractionType.kDecay
PT_OTHER_SHOWER = InteractionType.kOtherShower
PT_INVALID      = InteractionType.kInvalidProcess

# Point cloud helpers
LARGE_PC  = np.zeros((20, 3), dtype=np.float32)   # 20 pts — "large"
SMALL_PC  = np.zeros((2, 3),  dtype=np.float32)    # 2 pts  — "small"
EMPTY_PC  = np.zeros((0, 3),  dtype=np.float32)    # 0 pts
MIN_SIZE  = 5                                        # threshold used in delta/shower cases


class TestTrackAndPrimary:
    def test_ktrack_process_is_ktrack(self):
        assert SetSemanticType(PT_TRACK, 13, 0, LARGE_PC) == SemanticType.kTrack

    def test_primary_pdg_electron_is_kshower(self):
        assert SetSemanticType(PT_PRIMARY, 11, 0, LARGE_PC) == SemanticType.kShower

    def test_primary_pdg_photon_is_kshower(self):
        assert SetSemanticType(PT_PRIMARY, 22, 0, LARGE_PC) == SemanticType.kShower

    def test_primary_pdg_proton_is_ktrack(self):
        assert SetSemanticType(PT_PRIMARY, 2212, 0, LARGE_PC) == SemanticType.kTrack

    def test_primary_pdg_muon_is_ktrack(self):
        assert SetSemanticType(PT_PRIMARY, 13, 0, LARGE_PC) == SemanticType.kTrack


class TestDelta:
    def test_delta_large_cloud_is_kdelta(self):
        result = SetSemanticType(PT_DELTA, 11, 0, LARGE_PC, point_cloud_size=MIN_SIZE)
        assert result == SemanticType.kDelta

    def test_delta_small_cloud_is_klescatter(self):
        small = np.zeros((3, 3), dtype=np.float32)  # 3 < MIN_SIZE=5
        result = SetSemanticType(PT_DELTA, 11, 0, small, point_cloud_size=MIN_SIZE)
        assert result == SemanticType.kLEScatter

    def test_delta_no_threshold_is_kdelta(self):
        # threshold=-1 → all sizes treated as "large"
        result = SetSemanticType(PT_DELTA, 11, 0, SMALL_PC, point_cloud_size=-1)
        assert result == SemanticType.kDelta


class TestDecay:
    def test_michel_electron_from_muon(self):
        # pdg=11 (e-), parent_pdg=13 (mu-)
        result = SetSemanticType(PT_DECAY, 11, 13, LARGE_PC)
        assert result == SemanticType.kMichel

    def test_michel_positron_from_antimuon(self):
        result = SetSemanticType(PT_DECAY, 11, -13, LARGE_PC)
        assert result == SemanticType.kMichel

    def test_decay_electron_non_muon_parent_is_kshower(self):
        # pdg=11, parent_pdg=211 (pion), not Michel
        result = SetSemanticType(PT_DECAY, 11, 211, LARGE_PC)
        assert result == SemanticType.kShower

    def test_decay_photon_is_kshower(self):
        result = SetSemanticType(PT_DECAY, 22, 13, LARGE_PC)
        assert result == SemanticType.kShower

    def test_decay_pion_is_ktrack(self):
        result = SetSemanticType(PT_DECAY, 211, 13, LARGE_PC)
        assert result == SemanticType.kTrack


class TestLEScatterProcesses:
    def test_neutron_is_klescatter(self):
        assert SetSemanticType(PT_NEUTRON, 2112, 0, LARGE_PC) == SemanticType.kLEScatter

    def test_ionization_is_klescatter(self):
        assert SetSemanticType(PT_IONIZATION, 11, 0, LARGE_PC) == SemanticType.kLEScatter

    def test_photoelectron_is_klescatter(self):
        assert SetSemanticType(PT_PHOTO_ELEC, 11, 0, LARGE_PC) == SemanticType.kLEScatter


class TestShowerProcesses:
    def test_photon_process_electron_large_is_kshower(self):
        result = SetSemanticType(PT_PHOTON, 11, 0, LARGE_PC, point_cloud_size=MIN_SIZE)
        assert result == SemanticType.kShower

    def test_photon_process_electron_small_is_klescatter(self):
        small = np.zeros((3, 3), dtype=np.float32)
        result = SetSemanticType(PT_PHOTON, 11, 0, small, point_cloud_size=MIN_SIZE)
        assert result == SemanticType.kLEScatter

    def test_compton_photon_large_is_kshower(self):
        result = SetSemanticType(PT_COMPTON, 22, 0, LARGE_PC, point_cloud_size=MIN_SIZE)
        assert result == SemanticType.kShower

    def test_conversion_electron_large_is_kshower(self):
        result = SetSemanticType(PT_CONVERSION, 11, 0, LARGE_PC, point_cloud_size=MIN_SIZE)
        assert result == SemanticType.kShower


class TestNucleusAndInvalid:
    def test_nucleus_large_is_ktrack(self):
        result = SetSemanticType(PT_NUCLEUS, 1000060120, 0, LARGE_PC, point_cloud_size=MIN_SIZE)
        assert result == SemanticType.kTrack

    def test_nucleus_small_is_klescatter(self):
        small = np.zeros((3, 3), dtype=np.float32)
        result = SetSemanticType(PT_NUCLEUS, 1000060120, 0, small, point_cloud_size=MIN_SIZE)
        assert result == SemanticType.kLEScatter

    def test_invalid_process_is_kunknown(self):
        assert SetSemanticType(PT_INVALID, 11, 0, LARGE_PC) == SemanticType.kUnknown
