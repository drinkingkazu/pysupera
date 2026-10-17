"""
A primary is a particle of the generator's list (the primary table).

Such a particle is kPrimary, so it is never LE, whatever its size and
whatever its proc_start says -- Geant4 records no creator process for a
primary.  The table is cross-checked against parent_track_id == -1.
"""
import numpy as np
import pytest

from pysupera.readers.format_edepsim_h5 import (
    _get_interaction_type, describe_primary_check, primary_check_failed,
    primary_mask)
from pysupera.utils import InteractionType as IT
from pysupera.utils import SemanticType as ST
from pysupera.utils import SetSemanticType

h5py = pytest.importorskip("h5py")

PART = np.dtype([(c, "f4") for c in (
    "x", "y", "z", "t", "px", "py", "pz", "ke", "end_x", "end_y", "end_z",
    "end_t", "end_px", "end_py", "end_pz", "end_ke")]
    + [(c, "i4") for c in ("track_id", "parent_track_id",
                           "ancestor_track_id", "pdg")]
    + [("mass", "f4"), ("proc_start", "i4"), ("subproc_start", "i4"),
       ("proc_end", "i4"), ("subproc_end", "i4")])
STEP = np.dtype([(c, "f4") for c in ("x", "y", "z", "t", "theta", "phi", "p")]
                + [(c, "i4") for c in ("pdg", "track_id", "ancestor_track_id")]
                + [("de", "f4"), ("dx", "f4"), ("proc_start", "i4"),
                   ("subproc_start", "i2"), ("proc_stop", "i4"),
                   ("subproc_stop", "i2")])
VERT = np.dtype([(c, "f4") for c in ("x", "y", "z", "t")]
                + [("interaction_id", "i2")])
PRIM = np.dtype([("track_id", "i4"), ("pdg", "i4"), ("interaction_id", "i4")])
ASS = np.dtype([("start", "u8"), ("end", "u8")])


def parts(rows):
    """rows: (track_id, parent_track_id, pdg, proc_start, subproc_start)."""
    a = np.zeros(len(rows), dtype=PART)
    for k, (t, par, pdg, pr, sub) in enumerate(rows):
        a[k]["track_id"], a[k]["parent_track_id"] = t, par
        a[k]["ancestor_track_id"] = t if par == -1 else 1
        a[k]["pdg"], a[k]["proc_start"], a[k]["subproc_start"] = pdg, pr, sub
        a[k]["mass"] = 105.66 if abs(pdg) == 13 else 0.511
        a[k]["ke"] = 1.7
    return a


# The event of the production failure: a primary mu+ of 1.7 MeV whose
# proc_start/subproc_start read 7/401 (General / StepLimit), and a primary
# e+ whose fields read 2/13 (EM / Compton).
EVENT = parts([(1, -1, -13, 7, 401), (2, -1, -11, 2, 13),
               (3, 1, -11, 6, 201), (4, 2, 22, 2, 3)])


class TestPrimaryMask:

    def test_the_table_decides(self):
        mask, check = primary_mask(EVENT, [1, 2])
        assert mask.tolist() == [True, True, False, False]
        assert not primary_check_failed(check)

    def test_without_the_table_the_parentless_rows_are_primaries(self):
        mask, check = primary_mask(EVENT)
        assert mask.tolist() == [True, True, False, False]
        assert check is None and not primary_check_failed(check)

    def test_each_kind_of_disagreement_is_reported(self):
        mask, check = primary_mask(EVENT, [1, 3, 99])
        assert mask.tolist() == [True, False, True, False]
        assert check["parentless_not_primary"].tolist() == [2]
        assert check["primary_with_parent"].tolist() == [3]
        assert check["primary_not_in_particles"].tolist() == [99]
        assert primary_check_failed(check)
        text = describe_primary_check(check)
        assert "track 2 with parent_track_id -1 but not in the primary table" in text
        assert "track 3 in the primary table but with a parent" in text
        assert "track 99 in the primary table but not among the particles" in text


class TestPrimaryIsNeverLE:

    def test_primaries_are_kprimary_whatever_their_process_fields(self):
        it = _get_interaction_type(EVENT, is_primary=primary_mask(EVENT, [1, 2])[0])
        assert [IT(int(v)) for v in it[:2]] == [IT.kPrimary, IT.kPrimary]
        assert IT(int(it[2])) is IT.kDecay

    def test_the_default_is_the_parentless_rule(self):
        it = _get_interaction_type(EVENT)
        assert [IT(int(v)) for v in it[:2]] == [IT.kPrimary, IT.kPrimary]

    def test_a_small_primary_muon_is_a_track(self):
        tiny = np.zeros((3, 6), dtype=np.float32)
        assert SetSemanticType(IT.kPrimary, -13, 0, tiny, 63) is ST.kTrack
        # As a non-primary of the same size it would be LE.
        assert SetSemanticType(IT.kTrack, -13, 0, tiny, 63) is ST.kLEScatter

    def test_a_primary_photon_or_electron_is_a_shower(self):
        empty = np.zeros((0, 6), dtype=np.float32)
        assert SetSemanticType(IT.kPrimary, 22, 0, empty, 63) is ST.kShower
        assert SetSemanticType(IT.kPrimary, -11, 0, empty, 63) is ST.kShower

    @pytest.mark.parametrize("pdg", [2112, 1000180400, 111])
    def test_a_primary_neutron_nucleus_or_pi0_is_a_track(self, pdg):
        ev = parts([(1, -1, pdg, 0, 0), (2, 1, pdg, 4, 111)])
        it = _get_interaction_type(ev, is_primary=np.array([True, False]))
        assert IT(int(it[0])) is IT.kPrimary
        empty = np.zeros((0, 6), dtype=np.float32)
        assert SetSemanticType(IT.kPrimary, pdg, 0, empty, 63) is ST.kTrack

    def test_a_secondary_neutron_or_nucleus_keeps_its_rule(self):
        ev = parts([(1, -1, 13, 0, 0), (2, 1, 2112, 4, 111),
                    (3, 1, 1000180400, 4, 111)])
        it = _get_interaction_type(ev, is_primary=np.array([True, False, False]))
        assert [IT(int(v)) for v in it[1:]] == [IT.kNeutron, IT.kNucleus]


def write_edepsim(path, prim_ids=None, steps=True):
    """
    A one-event EDepSim file: EVENT, one step each for tracks 1 and 3 (none
    at all with ``steps=False``).
    """
    st = np.zeros(2 if steps else 0, dtype=STEP)
    ass = np.zeros(len(EVENT), dtype=ASS)
    if steps:
        st["track_id"] = [1, 3]
        st["de"], st["dx"] = 1.0, 0.1
        ass["start"] = [0, 1, 1, 2]
        ass["end"] = [1, 1, 2, 2]
    vert = np.zeros(1, dtype=VERT)
    with h5py.File(path, "w") as f:
        for key, a, dt in (("particle/geant4", EVENT, PART),
                           ("vertex/geant4", vert, VERT),
                           ("pstep/lar_vol", st, STEP),
                           ("ass/particle_pstep_lar_vol", ass, ASS)):
            d = f.create_dataset(key, (1,), dtype=h5py.vlen_dtype(dt))
            d[0] = a
        if prim_ids is not None:
            pr = np.zeros(len(prim_ids), dtype=PRIM)
            pr["track_id"] = prim_ids
            d = f.create_dataset("primary/geant4", (1,), dtype=h5py.vlen_dtype(PRIM))
            d[0] = pr


class TestReader:

    def read(self, path, **kw):
        from pysupera.readers import EDepSimHDF5Reader
        with EDepSimHDF5Reader(str(path), min_pc_size=5, **kw) as r:
            return r[0], r.last_primary_check

    def test_the_reader_takes_primaries_from_the_table(self, tmp_path):
        write_edepsim(tmp_path / "e.h5", prim_ids=[1, 2])
        ps, check = self.read(tmp_path / "e.h5")
        assert [IT(p._interaction_type) for p in ps[:2]] == [IT.kPrimary] * 2
        assert ps[0].sem_type is ST.kTrack          # one step, still a track
        assert not primary_check_failed(check)

    def test_the_reader_reports_a_disagreement(self, tmp_path):
        write_edepsim(tmp_path / "e.h5", prim_ids=[1])
        ps, check = self.read(tmp_path / "e.h5")
        assert IT(ps[1]._interaction_type) is not IT.kPrimary
        assert check["parentless_not_primary"].tolist() == [2]

    def test_without_the_table_the_reader_falls_back(self, tmp_path):
        write_edepsim(tmp_path / "e.h5")
        ps, check = self.read(tmp_path / "e.h5")
        assert check is None
        assert IT(ps[0]._interaction_type) is IT.kPrimary
