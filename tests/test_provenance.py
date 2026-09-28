"""
Tests for hit provenance.

The table compresses a per-deposit truth into one owner per group, which is
only valid while no group straddles a defragmentation split.  Most of what
matters here is that the compression refuses to lie when it cannot hold.
"""
import numpy as np
import pytest

from pysupera.provenance import (build_group_owners, remap_owners,
                                 hits_of_groups, groups_of_particles,
                                 GroupOwnershipError, NO_OWNER)


class FakeParticle:
    """Carries only what build_group_owners reads: id and voxmap."""

    def __init__(self, pid, deposits):
        self.id = pid
        n = len(deposits)
        self.voxmap = (np.arange(n + 1, dtype=np.int64),
                       np.asarray(deposits, dtype=np.int64))


# deposits 0,1 -> group 0; 2,3 -> group 1; 4 -> group 2
D2G = np.array([0, 0, 1, 1, 2], dtype=np.int64)


class TestBuildGroupOwners:

    def test_each_group_gets_its_particle(self):
        ps = [FakeParticle(7, [0, 1]), FakeParticle(9, [2, 3])]
        owner = build_group_owners(ps, D2G, 3)
        assert owner[0] == 7
        assert owner[1] == 9

    def test_untouched_groups_are_unowned(self):
        owner = build_group_owners([FakeParticle(7, [0])], D2G, 3)
        assert owner[1] == NO_OWNER and owner[2] == NO_OWNER

    def test_a_particle_may_own_several_groups(self):
        owner = build_group_owners([FakeParticle(7, [0, 2, 4])], D2G, 3)
        assert owner.tolist() == [7, 7, 7]

    def test_particles_without_a_voxmap_are_skipped(self):
        p = FakeParticle(7, [0]); p.voxmap = None
        assert (build_group_owners([p], D2G, 3) == NO_OWNER).all()

    def test_empty_event(self):
        assert len(build_group_owners([], D2G, 3)) == 3

    def test_out_of_range_groups_are_ignored(self):
        # a deposit mapped past the end must not write out of bounds
        owner = build_group_owners([FakeParticle(7, [4])], D2G, 2)
        assert owner.tolist() == [NO_OWNER, NO_OWNER]


class TestInvariant:
    """A straddling group means the table cannot express the truth."""

    def _straddle(self):
        # both particles hold a deposit of group 0
        return [FakeParticle(7, [0]), FakeParticle(9, [1])]

    def test_a_straddling_group_raises(self):
        with pytest.raises(GroupOwnershipError, match="two pysupera particles"):
            build_group_owners(self._straddle(), D2G, 3)

    def test_the_error_names_both_particles(self):
        with pytest.raises(GroupOwnershipError) as e:
            build_group_owners(self._straddle(), D2G, 3)
        assert "7" in str(e.value) and "9" in str(e.value)

    def test_the_error_points_at_the_exact_alternative(self):
        with pytest.raises(GroupOwnershipError) as e:
            build_group_owners(self._straddle(), D2G, 3)
        assert "store_mapping" in str(e.value)

    def test_check_false_keeps_the_first_owner(self):
        owner = build_group_owners(self._straddle(), D2G, 3, check=False)
        assert owner[0] == 7


class TestRemap:

    def test_maps_into_the_other_id_space(self):
        owner = np.array([7, 9, NO_OWNER], dtype=np.int32)
        assert remap_owners(owner, {7: 100, 9: 200}).tolist() == [100, 200, -1]

    def test_unmapped_particles_become_unowned(self):
        owner = np.array([7, 9], dtype=np.int32)
        assert remap_owners(owner, {7: 100}).tolist() == [100, -1]

    def test_unowned_stays_unowned(self):
        owner = np.full(3, NO_OWNER, dtype=np.int32)
        assert (remap_owners(owner, {7: 1}) == NO_OWNER).all()

    def test_several_particles_may_share_an_owner(self):
        # the usual case: many particles in one instance
        owner = np.array([7, 9], dtype=np.int32)
        assert remap_owners(owner, {7: 5, 9: 5}).tolist() == [5, 5]


class TestQueries:

    def test_hits_of_groups_selects_by_group(self):
        gid = np.array([0, 1, 0, 2, 1])
        assert hits_of_groups(gid, [0]).tolist() == [0, 2]
        assert hits_of_groups(gid, [1, 2]).tolist() == [1, 3, 4]

    def test_hits_of_no_groups_is_empty(self):
        assert len(hits_of_groups(np.array([0, 1]), [])) == 0

    def test_groups_of_particles(self):
        owner = np.array([7, 9, 7, NO_OWNER], dtype=np.int32)
        assert groups_of_particles(owner, [7]).tolist() == [0, 2]
        assert groups_of_particles(owner, [7, 9]).tolist() == [0, 1, 2]

    def test_groups_of_no_particles_is_empty(self):
        assert len(groups_of_particles(np.array([7]), [])) == 0

    def test_unowned_is_never_selected(self):
        owner = np.array([NO_OWNER, NO_OWNER], dtype=np.int32)
        assert len(groups_of_particles(owner, [NO_OWNER])) == 0


class TestAttachTruthClouds:
    """The true segments behind the hits follow their group's owner."""

    @staticmethod
    def segs():
        # x, y, z [mm], t [us], dE [MeV], dX [cm]; groups 0, 0, 1, 2, -1
        flat = np.array([[0.1, 0, 0, 5, 1, .03], [0.2, 0, 0, 3, 2, .03],
                         [9, 9, 9, 1, 4, .03], [50, 50, 50, 0, 8, .03],
                         [70, 70, 70, 0, 16, .03]], dtype=np.float32)
        return flat, np.array([0, 0, 1, 2, -1], dtype=np.int64)

    def test_segments_go_to_the_owner_of_their_group(self):
        from pysupera.provenance import attach_truth_clouds
        ps = [FakeParticle(7, []), FakeParticle(9, [])]
        owner = np.array([7, 9, NO_OWNER], dtype=np.int32)
        stats = attach_truth_clouds(ps, owner, *self.segs())
        np.testing.assert_array_equal(ps[0].truth_cloud[:, 4], [1, 2])
        np.testing.assert_array_equal(ps[1].truth_cloud[:, 4], [4])
        # group 2 is unowned and -1 is no group: neither is attached
        assert stats == {"n_segments": 5, "n_owned": 3, "n_rows": 3}

    def test_voxelizing_merges_with_the_deposit_rules(self):
        from pysupera.provenance import attach_truth_clouds
        ps = [FakeParticle(7, [])]
        owner = np.array([7, NO_OWNER, NO_OWNER], dtype=np.int32)
        attach_truth_clouds(ps, owner, *self.segs(), voxel_size=3.0,
                            origin=[0, 0, 0])
        tc = ps[0].truth_cloud
        assert len(tc) == 1                          # both segments in one cell
        np.testing.assert_allclose(tc[0, :3], [1.5, 1.5, 1.5])   # cell centre
        assert tc[0, 3] == 3                         # t: earliest
        assert tc[0, 4] == 3                         # dE summed
        assert tc[0, 5] == pytest.approx(0.06)       # dX summed

    def test_voxels_take_direction_and_momentum_from_the_earliest_step(self):
        """
        With the nine truth columns, a voxel's theta, phi and |p| are those of
        its earliest step -- not zero, and not an average.
        """
        from pysupera.provenance import attach_truth_clouds
        # x, y, z, t, dE, dX, theta, phi, p; two steps in one 3 mm cell
        flat = np.array([[0.1, 0, 0, 5, 1, .03, 1.0, 2.0, 50.0],
                         [0.2, 0, 0, 3, 2, .03, 0.5, -1.0, 80.0]], dtype=np.float32)
        ps = [FakeParticle(7, [])]
        attach_truth_clouds(ps, np.array([7], dtype=np.int32), flat,
                            np.array([0, 0], dtype=np.int64),
                            voxel_size=3.0, origin=[0, 0, 0])
        tc = ps[0].truth_cloud
        assert len(tc) == 1 and tc[0, 3] == 3            # earliest t
        np.testing.assert_allclose(tc[0, 6:], [0.5, -1.0, 80.0])
        assert tc[0, 4] == 3                             # dE summed

    def test_particles_owning_nothing_get_an_empty_cloud(self):
        from pysupera.provenance import attach_truth_clouds
        ps = [FakeParticle(7, []), FakeParticle(8, [])]
        owner = np.array([7, 7, 7], dtype=np.int32)
        attach_truth_clouds(ps, owner, *self.segs())
        assert ps[1].truth_cloud.shape == (0, 6)
