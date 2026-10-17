"""
The rules that decide instances and their kinematics.

* A shower starts at the pair production: a photon's Compton electrons and
  photoelectron are not its fragment, and join its *instance* only when the
  photon is itself inside a shower.  LE fragments may join, never lead.
* An LE particle touching several things is absorbed into its own ancestor
  when that is one of them (delta rays).
* A stand-alone split-off piece of a track goes back to its track.
* first_* / last_* are made per instance, from the steps it collected.
"""
import numpy as np
import pytest

from pysupera.conditions.absorb_le_scatter import AbsorbLEScatter
from pysupera.kinematics import instance_kinematics, step_particles
from pysupera.layout import Group, fragment_le, snapshot_groups
from pysupera.merge import (attach_split_pieces, find_split_pieces,
                            merge_em_showers, relabel_split_pieces)
from pysupera.utils import SemanticType as ST
from tests.conftest import (
    make_particle, make_partitioner,
    PT_COMPTON, PT_CONVERSION, PT_DECAY, PT_DELTA, PT_NEUTRON, PT_PHOTON,
    PT_PRIMARY, PT_TRACK,
)

EMPTY = np.zeros((0, 3), dtype=np.float32)


def fragments(*parts):
    """Each particle its own fragment, as step 1 leaves the unmerged ones."""
    for p in parts:
        p.member_ids = [p.id]
    return list(parts)


def instance_of(instances, pid):
    return next(i for i in instances if pid in i.member_ids)


def line(x0, x1, n=40, y=0.0):
    x = np.linspace(x0, x1, n, dtype=np.float32)
    return np.c_[x, np.full(n, y, np.float32), np.zeros(n, np.float32)]


# ---------------------------------------------------------------------------
# Showers
# ---------------------------------------------------------------------------

class TestShowerStartsAtPairProduction:

    def event(self):
        """
        Photon 1 Compton-scatters out electron 4, then converts into 2 and 3.
        Electron 2 radiates photon 5, which Compton-scatters out the small
        blob 6.  Step 1 put the pair into the photon's fragment.
        """
        g1 = make_particle(1, PT_PRIMARY, pdg=22, pc=EMPTY)
        e2 = make_particle(2, PT_CONVERSION, pdg=11, parent_id=1, parent_pdg=22,
                           ancestor_id=1, n_pts=30)
        e3 = make_particle(3, PT_CONVERSION, pdg=-11, parent_id=1,
                           parent_pdg=22, ancestor_id=1, n_pts=30)
        c4 = make_particle(4, PT_COMPTON, pdg=11, parent_id=1, parent_pdg=22,
                           ancestor_id=1, n_pts=30, offset=(500, 0, 0))
        g5 = make_particle(5, PT_PHOTON, pdg=22, parent_id=2, parent_pdg=11,
                           ancestor_id=1, pc=EMPTY)
        c6 = make_particle(6, PT_COMPTON, pdg=11, parent_id=5, parent_pdg=22,
                           ancestor_id=1, n_pts=2, offset=(300, 0, 0))
        c4.sem_type, c6.sem_type = ST.kShower, ST.kLEScatter
        g1.sem_type = ST.kShower                    # PhotonDecay promoted it
        g1.member_ids = [1, 2, 3]
        for p in (c4, g5, c6):
            p.member_ids = [p.id]
        return [g1, c4, g5, c6]

    def test_the_initiating_photons_compton_electron_is_its_own_instance(self):
        insts = merge_em_showers(self.event())
        assert instance_of(insts, 4) is not instance_of(insts, 1)
        assert instance_of(insts, 4).id == 4

    def test_compton_blob_of_a_photon_inside_the_shower_joins_it(self):
        frags = self.event()
        groups = snapshot_groups(frags)
        insts = merge_em_showers(frags)
        shower = instance_of(insts, 1)
        assert {5, 6} <= set(shower.member_ids)
        # ... and stays LE: it joined by genealogy, not by touching
        assert fragment_le(self._particles(frags), groups)[6] is True

    @staticmethod
    def _particles(frags):
        out = {}
        for f in frags:
            out[f.id] = f
        return list(out.values())

    def test_an_le_fragment_never_heads_a_shower(self):
        le = make_particle(1, PT_COMPTON, pdg=11, n_pts=2)
        big = make_particle(2, PT_PRIMARY, pdg=11, parent_id=1, parent_pdg=11,
                            ancestor_id=1, n_pts=30)
        le.sem_type, big.sem_type = ST.kLEScatter, ST.kShower
        insts = merge_em_showers(fragments(le, big))
        assert instance_of(insts, 2).id == 2
        assert instance_of(insts, 1).id == 1

    def test_a_parent_absorbed_into_its_descendants_shower_loses_nothing(self):
        """
        A stopped muon 1, too small to be a track, absorbed by proximity into
        the shower of photon 3, which its Michel positron 2 radiated: 2's
        parent fragment is 3's, and 3's is 2's.  Seen in production.
        """
        mu = make_particle(1, PT_PRIMARY, pdg=-13, n_pts=4)
        e2 = make_particle(2, PT_DECAY, pdg=-11, parent_id=1, parent_pdg=-13,
                           ancestor_id=1, n_pts=30)
        g3 = make_particle(3, PT_PHOTON, pdg=22, parent_id=2, parent_pdg=-11,
                           ancestor_id=1, pc=EMPTY)
        mu.sem_type, e2.sem_type, g3.sem_type = (ST.kLEScatter, ST.kMichel,
                                                 ST.kShower)
        e2.member_ids, g3.member_ids = [2], [3, 1]
        insts = merge_em_showers([e2, g3])
        members = sorted(m for i in insts for m in i.member_ids)
        assert members == [1, 2, 3]
        assert len(insts) == 1 and insts[0].id == 2


# ---------------------------------------------------------------------------
# Absorption prefers the particle's own ancestor
# ---------------------------------------------------------------------------

class TestDeltaRayGoesToItsTrack:

    def particles(self):
        mu = make_particle(1, PT_TRACK, pdg=13, pc=line(0, 100))
        sh = make_particle(2, PT_PRIMARY, pdg=11, pc=line(40, 60, y=3.0))
        mu.sem_type, sh.sem_type = ST.kTrack, ST.kShower
        d = make_particle(3, PT_DELTA, pdg=11, parent_id=1, parent_pdg=13,
                          ancestor_id=1,
                          pc=np.array([[50, 1.5, 0], [50.5, 1.5, 0]],
                                      dtype=np.float32))
        d.sem_type = ST.kLEScatter
        return mu, sh, d

    @pytest.mark.parametrize("pairs", [[(3, 2), (3, 1)], [(3, 1), (3, 2)]])
    def test_the_parent_wins_whatever_the_order(self, pairs):
        prt = make_partitioner(list(self.particles()), D=5.0)
        assert AbsorbLEScatter().post_filter(prt, pairs) == [(3, 1)]

    def test_without_the_parent_among_them_the_first_is_kept(self):
        prt = make_partitioner(list(self.particles()), D=5.0)
        assert AbsorbLEScatter().post_filter(prt, [(3, 2)]) == [(3, 2)]

    def test_in_the_partitioner(self):
        prt = make_partitioner(list(self.particles()), D=5.0)
        prt.partition_combined([AbsorbLEScatter()], verbose=False)
        assert prt.rep_lookup[3] is prt.rep_lookup[1]


# ---------------------------------------------------------------------------
# Split-off pieces of tracks
# ---------------------------------------------------------------------------

class TestSplitPiecesGoBackToTheirTrack:

    @staticmethod
    def piece_of(src, pid, offset):
        p = make_particle(pid, src._interaction_type, pdg=int(src.pdg),
                          parent_id=src.id, parent_pdg=int(src.pdg),
                          ancestor_id=src.ancestor_id, n_pts=3, offset=offset)
        p.geant4_id = src.geant4_id
        p.sem_type = ST.kLEScatter               # what defragmentation sets
        return p

    def test_a_stub_past_a_gap_joins_the_track_and_is_not_le(self):
        mu = make_particle(1, PT_TRACK, pdg=13, pc=line(0, 100))
        stub = self.piece_of(mu, 2, (300, 0, 0))
        frags = fragments(mu, stub)
        pairs = find_split_pieces(frags, [mu, stub])
        assert [(f.id, h.id) for f, h in pairs] == [(2, 1)]
        relabel_split_pieces(pairs)
        groups = snapshot_groups(frags)
        assert fragment_le([mu, stub], groups)[2] is False
        assert stub.sem_type == ST.kTrack
        insts = attach_split_pieces(merge_em_showers(frags), pairs)
        assert len(insts) == 1 and sorted(insts[0].member_ids) == [1, 2]

    def test_a_neutrons_blobs_are_left_alone(self):
        n = make_particle(1, PT_NEUTRON, pdg=2112, n_pts=8)
        blob = self.piece_of(n, 2, (300, 0, 0))
        assert find_split_pieces(fragments(n, blob), [n, blob]) == []

    def test_a_piece_absorbed_elsewhere_stays_there(self):
        mu = make_particle(1, PT_TRACK, pdg=13, pc=line(0, 100))
        pi = make_particle(3, PT_TRACK, pdg=211, pc=line(300, 400))
        stub = self.piece_of(mu, 2, (300, 0, 0))
        mu.member_ids = [1]
        pi.member_ids = [3, 2]                   # proximity took the stub
        assert find_split_pieces([mu, pi], [mu, pi, stub]) == []


# ---------------------------------------------------------------------------
# Instance kinematics
# ---------------------------------------------------------------------------

STEP_DTYPE = [(c, "f4") for c in ("x", "y", "z", "t", "theta", "phi", "p",
                                  "dx")]


def steps_along_z(zs, ts, p, dx=1.0):
    s = np.zeros(len(zs), dtype=STEP_DTYPE)
    s["z"], s["t"], s["p"], s["dx"] = zs, ts, p, dx
    return s


class TestInstanceKinematics:

    def test_an_absorbed_blob_of_another_particle_is_not_the_last_point(self):
        mu = make_particle(0, PT_TRACK, pdg=13, n_pts=5)
        blob = make_particle(1, PT_COMPTON, pdg=11, n_pts=2)
        mu.geant4_id, blob.geant4_id = 10, 20
        mu.sem_type = ST.kTrack
        blob.sem_type = ST.kLEScatter
        steps = steps_along_z([1, 2, 3, 2.5], [1, 2, 3, 50], [300, 290, 280, 1])
        kin = dict(geant4_trackid=np.array([10, 20]),
                   mass=np.array([105.658, 0.511]),
                   start_mom=np.zeros((2, 4)))
        g = Group(0, [0, 1], ST.kTrack.value)
        out = instance_kinematics([g], [mu, blob], steps,
                                  step_owner=np.array([0, 0, 0, 1]),
                                  step_pid=np.array([0, 0, 0, 0]),
                                  track_kin=kin)[0]
        assert out["first_pos"][2] == pytest.approx(0.5)     # 1 - dx/2
        assert out["last_pos"][2] == pytest.approx(3.5)      # not the blob
        assert out["last_mom"][2] == pytest.approx(280)

    def test_an_le_only_instance_uses_all_its_steps(self):
        blob = make_particle(0, PT_COMPTON, pdg=11, n_pts=2)
        blob.geant4_id, blob.sem_type = 20, ST.kLEScatter
        steps = steps_along_z([4, 5], [1, 2], [1, 1])
        kin = dict(geant4_trackid=np.array([20]), mass=np.array([0.511]),
                   start_mom=np.zeros((1, 4)))
        out = instance_kinematics([Group(0, [0], ST.kLEScatter.value)], [blob],
                                  steps, np.array([0, 0]), np.array([0, 0]),
                                  kin)[0]
        assert out["first_pos"][2] == pytest.approx(3.5)
        assert out["last_pos"][2] == pytest.approx(5.5)

    def shower(self):
        g = make_particle(0, PT_PRIMARY, pdg=22, pc=EMPTY)
        em = make_particle(1, PT_CONVERSION, pdg=11, parent_id=0,
                           parent_pdg=22, n_pts=30)
        ep = make_particle(2, PT_CONVERSION, pdg=-11, parent_id=0,
                           parent_pdg=22, n_pts=30)
        late = make_particle(3, PT_PRIMARY, pdg=11, parent_id=1,
                             parent_pdg=11, n_pts=30)
        for p, t in zip((g, em, ep, late), (1, 2, 3, 4)):
            p.geant4_id, p.sem_type = t, ST.kShower
        start_mom = np.array([[0, 0, 10, 10], [1, 0, 4, 4.2],
                              [-1, 0, 5, 5.3], [0, 0, 1, 1.1]], dtype=float)
        kin = dict(geant4_trackid=np.array([1, 2, 3, 4]),
                   mass=np.array([0, 0.511, 0.511, 0.511]),
                   start_mom=start_mom)
        steps = steps_along_z([10, 10, 40], [1, 1, 5], [4, 5, 1])
        group = Group(0, [0, 1, 2, 3], ST.kShower.value)
        return [g, em, ep, late], steps, kin, group

    def test_a_shower_has_the_pair_momentum_and_no_last_point(self):
        parts, steps, kin, g = self.shower()
        out = instance_kinematics([g], parts, steps, np.array([1, 2, 3]),
                                  np.array([1, 2, 3]), kin)[0]
        np.testing.assert_allclose(out["first_mom"], [0, 0, 9, 9.5])
        assert out["first_pos"][2] == pytest.approx(9.5)
        assert np.isnan(out["last_pos"]).all() and np.isnan(out["last_mom"]).all()

    def test_a_low_energy_pair_still_starts_the_shower(self):
        parts, steps, kin, g = self.shower()
        parts[1].sem_type = parts[2].sem_type = ST.kLEScatter
        out = instance_kinematics([g], parts, steps, np.array([1, 2, 3]),
                                  np.array([1, 2, 3]), kin)[0]
        assert out["first_pos"][2] == pytest.approx(9.5)
        np.testing.assert_allclose(out["first_mom"], [0, 0, 9, 9.5])

    def test_a_pair_the_readout_missed_leaves_the_steps_own_momentum(self):
        parts, steps, kin, g = self.shower()
        out = instance_kinematics([g], parts, steps, np.array([1, 2, 3]),
                                  np.array([1, 2, 3]), kin,
                                  counted=np.array([False, False, True]))[0]
        assert out["first_pos"][2] == pytest.approx(39.5)
        assert out["first_mom"][2] == pytest.approx(1.0)

    def test_steps_follow_the_voxels_that_hold_them(self):
        a = make_particle(0, PT_TRACK, pdg=13, n_pts=1)
        b = make_particle(1, PT_TRACK, pdg=13, n_pts=1)
        a.voxmap = (np.array([0, 2]), np.array([0, 3]))
        b.voxmap = (np.array([0, 1]), np.array([1]))
        np.testing.assert_array_equal(step_particles(5, [a, b]),
                                      [0, 1, -1, 0, -1])
        np.testing.assert_array_equal(
            step_particles(3, [a, b], group=np.array([1, -1, 0]),
                           group_owner=np.array([1, 0])), [0, -1, 1])


def test_a_converting_photon_ends_with_its_momentum_before_converting():
    from pysupera.readers.format_edepsim_h5 import particle_kinematics
    f4 = [(c, "f4") for c in ("x", "y", "z", "t", "px", "py", "pz", "ke",
                              "end_x", "end_y", "end_z", "end_t", "end_px",
                              "end_py", "end_pz", "end_ke", "mass")]
    i4 = [(c, "i4") for c in ("track_id", "parent_track_id", "pdg",
                              "proc_start", "subproc_start", "proc_end",
                              "subproc_end")]
    P = np.zeros(2, dtype=f4 + i4)
    P["track_id"], P["parent_track_id"] = [1, 2], [-1, 1]
    P["pdg"] = [22, 11]
    P["mass"] = [0.0, 0.511]
    P["pz"][0], P["ke"][0] = 10.0, 10.0                 # photon along z
    P["proc_end"][0], P["subproc_end"][0] = 2, 14       # converts; G4 says 0
    P["px"][1], P["pz"][1], P["ke"][1] = 1.0, 2.0, 1.5  # its Compton electron
    P["proc_start"][1], P["subproc_start"][1] = 2, 13
    k = particle_kinematics(P)
    np.testing.assert_allclose(k["end_mom"][0], [-1, 0, 8, 8.5], rtol=1e-6)
    np.testing.assert_allclose(k["end_mom"][1], [0, 0, 0, 0.511], rtol=1e-6)


def test_when_the_non_le_particles_left_nothing_all_steps_count():
    from pysupera.kinematics import instance_kinematics
    mu = make_particle(0, PT_TRACK, pdg=13, n_pts=5)
    blob = make_particle(1, PT_COMPTON, pdg=11, n_pts=2)
    mu.geant4_id, blob.geant4_id = 10, 20
    mu.sem_type, blob.sem_type = ST.kTrack, ST.kLEScatter
    steps = steps_along_z([1, 7], [1, 2], [300, 1])
    kin = dict(geant4_trackid=np.array([10, 20]),
               mass=np.array([105.658, 0.511]), start_mom=np.zeros((2, 4)))
    out = instance_kinematics([Group(0, [0, 1], ST.kTrack.value)], [mu, blob],
                              steps, np.array([0, 1]), np.array([0, 0]), kin,
                              counted=np.array([False, True]))[0]
    assert out["first_pos"][2] == pytest.approx(6.5)
