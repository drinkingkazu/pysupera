"""
Round-trip tests for the 3.0.0 format.

These check the two properties the schema exists to provide: that every level's
points are a single contiguous slice, and that a stored particle's genealogy
can always be walked without leaving the file.
"""
import numpy as np
import pytest
import h5py

from pysupera.layout import build_layout, snapshot_groups
from pysupera.io_v3 import (open_writer_v3, read_events_v3, EventStoreV3,
                            FORMAT_VERSION_V3, COLUMNS)
from tests.conftest import make_particle, PT_PRIMARY, PT_TRACK


def chain_event():
    """Two fragments merged into one instance, plus a point-less ancestor.

    pi0(0) -- no points -- is the ancestor of photon(1); photon(1) and its
    partner(2) form one fragment, electron(3) another, and step 2 merges both
    into a single instance headed by photon(1).
    """
    pi0 = make_particle(0, PT_PRIMARY, pdg=111, parent_id=0, ancestor_id=0,
                        pc=np.zeros((0, 6), dtype=np.float32), interaction_id=0)
    g1 = make_particle(1, PT_TRACK, pdg=22, parent_id=0, ancestor_id=0,
                       n_pts=3, interaction_id=0)
    g2 = make_particle(2, PT_TRACK, pdg=11, parent_id=1, ancestor_id=0,
                       n_pts=2, interaction_id=0)
    e3 = make_particle(3, PT_TRACK, pdg=11, parent_id=1, ancestor_id=0,
                       n_pts=4, interaction_id=0)
    particles = [pi0, g1, g2, e3]

    # Fragments are snapshotted first, exactly as the real pipeline must:
    # merge_em_showers reuses the surviving rep objects for the instances, so
    # reading fragment membership afterwards would give the instance's.
    g1.member_ids = [1, 2]
    e3.member_ids = [3]
    pi0.member_ids = [0]
    frag_groups = snapshot_groups([pi0, g1, e3])

    g1.member_ids = [0, 1, 2, 3]          # what the merge would leave behind
    inst_groups = snapshot_groups([g1])
    return particles, frag_groups, inst_groups


def write(tmp_path, name="v3.h5", events=None):
    events = events or [chain_event()]
    path = str(tmp_path / name)
    with open_writer_v3(path) as w:
        for parts, frag_groups, inst_groups in events:
            w.append_event(build_layout(parts, frag_groups, inst_groups))
    return path


class TestFileShape:

    def test_version_and_counts(self, tmp_path):
        path = write(tmp_path)
        with h5py.File(path) as f:
            assert f["format_version"][()].decode() == FORMAT_VERSION_V3
            assert int(f["n_events"][()]) == 1
            assert f["points/flat"].shape[1] == 6
            for c in COLUMNS:
                assert f"particles/{c}" in f

    def test_all_points_are_stored_even_for_unstored_particles(self, tmp_path):
        parts, frags, insts = chain_event()
        total = sum(len(p.point_cloud) for p in parts)
        path = write(tmp_path, events=[(parts, frags, insts)])
        with h5py.File(path) as f:
            assert f["points/flat"].shape[0] == total

    def test_fenceposts(self, tmp_path):
        ev = [chain_event(), chain_event()]
        path = write(tmp_path, events=ev)
        with h5py.File(path) as f:
            assert len(f["events/offsets"]) == 3
            assert len(f["points/offsets"]) == 3
            assert f["events/offsets"][-1] == f["particles/id"].shape[0]
            assert f["points/offsets"][-1] == f["points/flat"].shape[0]


class TestGroupSemantics:

    def test_rep_marked_by_self_reference(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
        assert v.is_fragment.sum() >= 1
        assert v.is_instance.sum() == 1
        inst_row = int(np.flatnonzero(v.is_instance)[0])
        assert v["inst_id"][inst_row] == v["id"][inst_row]

    def test_merge_counts(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
        r = int(np.flatnonzero(v.is_instance)[0])
        assert v["inst_merge_count"][r] == 4        # all four particles
        assert v["frag_merge_count"][r] == 2        # its own fragment only

    def test_instance_split_separates_le(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
        for r in np.flatnonzero(v.is_instance):
            a = int(v["inst_pc_start"][r]); sp = int(v["inst_pc_le_start"][r])
            e = int(v["inst_pc_end"][r])
            assert a <= sp <= e
            assert len(v.points_of(r, "inst", le=False)) == sp - a
            assert len(v.points_of(r, "inst", le=True)) == e - sp
            assert len(v.points_of(r, "inst")) == e - a

    def test_fragment_sides_sit_inside_their_instance(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
        inst = {int(v["id"][r]): r for r in np.flatnonzero(v.is_instance)}
        for r in np.flatnonzero(v.is_fragment):
            ir = inst.get(int(v["inst_id"][r]))
            if ir is None:
                continue
            lo, hi = int(v["inst_pc_start"][ir]), int(v["inst_pc_end"][ir])
            for pre in ("frag_pc", "frag_pc_le"):
                a, b = int(v[f"{pre}_start"][r]), int(v[f"{pre}_end"][r])
                if a >= 0:
                    assert lo <= a <= b <= hi

    def test_instance_points_cover_every_member(self, tmp_path):
        parts, frags, insts = chain_event()
        total = sum(len(p.point_cloud) for p in parts)
        path = write(tmp_path, events=[(parts, frags, insts)])
        with read_events_v3(path) as s:
            v = s[0]
            r = int(np.flatnonzero(v.is_instance)[0])
            pts = v.points_of(r, "inst")
        assert len(pts) == total

    def test_non_rep_has_sentinel(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
        non_inst = ~v.is_instance
        assert np.all(v["inst_id"][non_inst] == -1)


class TestGenealogy:

    def test_parent_always_resolves(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
        ids = set(v["id"].tolist())
        assert all(int(p) in ids for p in v["parent_id"])

    def test_walk_terminates_on_ancestor_id(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
        idx = {int(i): k for k, i in enumerate(v["id"])}
        for k, i in enumerate(v["id"]):
            cur, seen = int(i), set()
            while cur not in seen:
                seen.add(cur)
                nxt = int(v["parent_id"][idx[cur]])
                if nxt == cur:
                    break
                cur = nxt
            assert cur == int(v["ancestor_id"][k])


class TestMultiEvent:

    def test_view_is_self_consistent_per_event(self, tmp_path):
        # Ranges are stored absolute but rebased by the view, so every index
        # a caller sees -- ids and point ranges alike -- is event-local.
        path = write(tmp_path, events=[chain_event(), chain_event()])
        with read_events_v3(path) as s:
            a, b = s[0], s[1]
        np.testing.assert_array_equal(a["id"], b["id"])
        np.testing.assert_array_equal(a["pc_start"], b["pc_start"])
        assert int(a["pc_start"].min()) == 0
        import h5py
        with h5py.File(path) as f:                    # absolute on disk
            n0 = int(f["points/offsets"][1])
            starts = f["particles/pc_start"][:]
            assert int(starts.max()) >= n0

    def test_rebased_ranges_index_the_event_points(self, tmp_path):
        path = write(tmp_path, events=[chain_event(), chain_event()])
        with read_events_v3(path) as s:
            v = s[1]
        for k in range(len(v)):
            a, b = int(v["pc_start"][k]), int(v["pc_end"][k])
            assert 0 <= a <= b <= len(v.points)

    def test_points_slice_directly(self, tmp_path):
        path = write(tmp_path, events=[chain_event(), chain_event()])
        with read_events_v3(path) as s:
            v = s[1]
            r = int(np.flatnonzero(v.is_instance)[0])
            got = v.points_of(r, "inst")
        assert len(got) == sum(len(p.point_cloud) for p in chain_event()[0])


class TestErrors:

    def test_rejects_2x_file(self, tmp_path):
        from pysupera import write_events
        old = str(tmp_path / "old.h5")
        write_events(old, [[make_particle(0, PT_TRACK)]])
        with pytest.raises(ValueError, match="not 3.x"):
            EventStoreV3(old)

    def test_event_index_out_of_range(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            with pytest.raises(IndexError):
                s[5]


class TestPointCoverage:

    def test_every_point_is_reachable_from_some_stored_row(self, tmp_path):
        # Only ~9% of particles get rows, but no point may be orphaned: each
        # must fall inside some stored fragment's or instance's range.  A
        # change to the selection rule that dropped a fragment owning points
        # would leave dead weight in the file, and this is what would catch it.
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
        covered = np.zeros(len(v.points), bool)
        for k in np.flatnonzero(v.is_fragment):
            for pre in ("frag_pc", "frag_pc_le"):
                a, b = int(v[f"{pre}_start"][k]), int(v[f"{pre}_end"][k])
                if a >= 0:
                    covered[a:b] = True
        assert covered.all(), f"{(~covered).sum()} points unreachable"


class TestMergeMutationTrap:

    def test_snapshot_is_taken_not_aliased(self):
        # merge_em_showers reuses rep objects, so a snapshot must copy the
        # member list rather than hold a reference to it.
        from pysupera.layout import snapshot_groups
        p = make_particle(0, PT_TRACK)
        p.member_ids = [0, 1]
        before = snapshot_groups([p])
        p.member_ids = [0, 1, 2, 3]            # what the merge does in place
        after = snapshot_groups([p])
        assert before[0].member_ids == [0, 1]
        assert after[0].member_ids == [0, 1, 2, 3]

    def test_fragment_membership_is_not_read_from_mutated_reps(self, tmp_path):
        # Build the layout from a correct fragment snapshot while the rep
        # object already carries the instance's membership.  frag_merge_count
        # must reflect the fragment, not the instance.
        parts, frag_groups, inst_groups = chain_event()
        layout = build_layout(parts, frag_groups, inst_groups)
        c = layout.columns
        r = int(np.flatnonzero(c["inst_id"] == c["id"])[0])
        assert c["inst_merge_count"][r] == 4
        assert c["frag_merge_count"][r] == 2


def two_interaction_event():
    """Two independent interactions, so renumbering has something to do."""
    import copy
    parts, frags, insts = [], [], []
    for k, (base, vtx) in enumerate(((0, 3), (10, 7))):
        a = make_particle(base, PT_PRIMARY, pdg=22, parent_id=base,
                          ancestor_id=base, n_pts=2, interaction_id=vtx)
        b = make_particle(base + 1, PT_TRACK, pdg=11, parent_id=base,
                          ancestor_id=base, n_pts=3, interaction_id=vtx)
        a.member_ids = [base, base + 1]
        b.member_ids = [base + 1]
        parts += [a, b]
        frags += [a, b]
        i = copy.copy(a); i.member_ids = [base, base + 1]
        insts.append(i)
    from pysupera.layout import snapshot_groups
    return parts, snapshot_groups(frags), snapshot_groups(insts)


class TestInteractions:

    def _view(self, tmp_path, vertices=None):
        parts, fg, ig = two_interaction_event()
        path = str(tmp_path / "int.h5")
        with open_writer_v3(path) as w:
            w.append_event(build_layout(parts, fg, ig, vertices=vertices))
        return read_events_v3(path)

    def test_interaction_id_is_a_dense_row_index(self, tmp_path):
        # Input vertex ids are 3 and 7; stored ids must be 0 and 1.
        with self._view(tmp_path) as s:
            v = s[0]
        assert sorted(set(v["interaction_id"].tolist())) == [0, 1]
        assert len(v.interactions["vertex_id"]) == 2

    def test_original_vertex_id_is_preserved(self, tmp_path):
        with self._view(tmp_path) as s:
            v = s[0]
        assert v.interactions["vertex_id"].tolist() == [3, 7]

    def test_particle_range_holds_only_that_interaction(self, tmp_path):
        with self._view(tmp_path) as s:
            v = s[0]
        I = v.interactions
        for k in range(len(I["vertex_id"])):
            rows = v["interaction_id"][I["part_start"][k]:I["part_end"][k]]
            assert set(rows.tolist()) == {k}

    def test_point_ranges_partition_the_event(self, tmp_path):
        with self._view(tmp_path) as s:
            v = s[0]
        I = v.interactions
        covered = np.zeros(len(v.points), bool)
        for a, b in zip(I["pc_start"], I["pc_end"]):
            assert not covered[int(a):int(b)].any()      # no overlap
            covered[int(a):int(b)] = True
        assert covered.all()                             # and no gap

    def test_vertex_position_round_trips(self, tmp_path):
        verts = np.zeros(8, dtype=[("x", "f4"), ("y", "f4"),
                                   ("z", "f4"), ("t", "f4"),
                                   ("energy_sum", "f4"), ("ke_sum", "f4")])
        verts["x"][3], verts["y"][3], verts["z"][3] = 1.5, 2.5, 3.5
        verts["x"][7] = -4.0
        with self._view(tmp_path, vertices=verts) as s:
            v = s[0]
        I = v.interactions
        assert I["x"][0] == pytest.approx(1.5)
        assert I["y"][0] == pytest.approx(2.5)
        assert I["x"][1] == pytest.approx(-4.0)

    def test_interaction_of_resolves(self, tmp_path):
        with self._view(tmp_path) as s:
            v = s[0]
        got = v.interaction_of(0)
        assert got["vertex_id"] == v.interactions["vertex_id"][
            int(v["interaction_id"][0])]
