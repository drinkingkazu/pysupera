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


class TestRangesHoldOnlyOwnRows:
    """
    A group's range is computed from where its members sit, so it is only
    right while they sit together.  These pin the two ways that failed.
    """

    @staticmethod
    def le(p):
        from pysupera.utils import SemanticType
        p.sem_type = SemanticType.kLEScatter
        return p

    def test_empty_fragment_of_an_unwritten_instance_claims_nothing(self):
        """
        Fragment 10659 in a real event: a photon and its Compton electrons,
        all without points (their interaction happened after the readout
        window), merged by PhotonDecay.  No instance of theirs is written,
        so each member sorted under its own id -- around a particle that did
        have points -- and the range from first to last member claimed them.
        """
        empty = np.zeros((0, 6), dtype=np.float32)
        a1 = self.le(make_particle(1, PT_TRACK, pc=empty, interaction_id=0))
        b2 = self.le(make_particle(2, PT_TRACK, n_pts=5, interaction_id=0))
        a3 = self.le(make_particle(3, PT_TRACK, pc=empty, interaction_id=0))
        a1.member_ids, b2.member_ids = [1, 3], [2]
        frags = snapshot_groups([a1, b2])
        lay = build_layout([a1, b2, a3], frags, [])
        cols = {int(i): k for k, i in enumerate(lay.columns["id"])}
        # the empty fragment gets no row, and no range anywhere claims more
        # than the 5 rows that exist
        assert 1 not in cols
        k = cols[2]
        c = lay.columns
        assert (c["frag_pc_le_start"][k], c["frag_pc_le_end"][k]) == (0, 5)

    def test_a_cross_interaction_absorption_is_stored_with_the_absorber(self):
        """
        AbsorbLEScatter can take an LE particle from interaction 0 into a
        track from interaction 1.  The absorber is the representative: the
        points become its fragment's, instance's and interaction's, stored
        in one block.  The absorbed particle keeps its own identity -- its
        row stays with interaction 0 and carries interaction 0.
        """
        from pysupera.io_v3 import check_ranges
        b = make_particle(1, PT_TRACK, n_pts=3, interaction_id=1)
        a = self.le(make_particle(2, PT_TRACK, n_pts=2, interaction_id=0))
        c = make_particle(3, PT_TRACK, n_pts=4, interaction_id=0,
                          parent_id=2, ancestor_id=2)   # keeps a's row stored
        b.member_ids, c.member_ids = [1, 2], [3]
        frags = snapshot_groups([b, c])
        insts = snapshot_groups([b, c])
        lay = build_layout([b, a, c], frags, insts)
        col, row = lay.columns, {int(i): k for k, i in
                                 enumerate(lay.columns["id"])}
        rb = row[1]
        # a's points are b's: inside b's instance (LE side) and fragment
        assert col["inst_pc_le_end"][rb] - col["inst_pc_le_start"][rb] == 2
        assert col["frag_pc_le_end"][rb] - col["frag_pc_le_start"][rb] == 2
        # the absorbed particle's row keeps interaction 0, beside c's
        assert 2 in row
        inter = lay.interactions
        k0 = int(np.flatnonzero(inter["vertex_id"] == 0)[0])
        k1 = int(np.flatnonzero(inter["vertex_id"] == 1)[0])
        assert col["interaction_id"][row[2]] == k0
        assert inter["part_start"][k0] <= row[2] < inter["part_end"][k0]
        assert inter["part_start"][k0] <= row[3] < inter["part_end"][k0]
        # and the points count toward interaction 1: 3 of b's and 2 of a's
        assert inter["pc_end"][k1] - inter["pc_start"][k1] == 5
        assert inter["pc_end"][k0] - inter["pc_start"][k0] == 4

    def test_a_group_whose_members_sort_apart_is_refused(self):
        """
        The invariant behind every range: a group's members sit together.
        A fragment whose members were put in two different instances breaks
        it, and the layout must refuse rather than write a range that
        swallows another particle's rows.
        """
        from pysupera.layout import LayoutError
        p1 = make_particle(1, PT_TRACK, n_pts=3, interaction_id=0)
        p2 = make_particle(2, PT_TRACK, n_pts=3, interaction_id=0)
        p3 = make_particle(3, PT_TRACK, n_pts=3, interaction_id=0)
        # fragment {1, 3}, but 1 and 3 in different instances, with 2 in
        # 1's: the instance order puts 2 between the fragment's members
        p1.member_ids, p2.member_ids = [1, 3], [2]
        frags = snapshot_groups([p1, p2])
        p1.member_ids, p3.member_ids = [1, 2], [3]
        insts = snapshot_groups([p1, p3])
        with pytest.raises(LayoutError, match="not contiguous"):
            build_layout([p1, p2, p3], frags, insts)


class TestCheckRanges:

    def test_a_clean_file_passes(self, tmp_path):
        from pysupera.io_v3 import check_ranges
        assert check_ranges(write(tmp_path, events=[chain_event()] * 2)) == []

    def test_an_overlapping_fragment_slice_is_reported(self, tmp_path):
        from pysupera.io_v3 import check_ranges
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
            frags = np.flatnonzero(v.is_fragment & (v["frag_pc_end"] > v["frag_pc_start"]))
        assert len(frags) >= 2
        # stretch one fragment's range over its neighbour, as the bug did
        with h5py.File(path, "a") as f:
            a, b = frags[:2]
            lo = min(f["particles/frag_pc_start"][a], f["particles/frag_pc_start"][b])
            hi = max(f["particles/frag_pc_end"][a], f["particles/frag_pc_end"][b])
            f["particles/frag_pc_start"][a] = lo
            f["particles/frag_pc_end"][a] = hi
        found = check_ranges(path)
        assert found and found[0][:3] == (0, "points", "frag")


class TestGroupOfPoints:

    def test_every_point_gets_its_fragment_and_instance(self, tmp_path):
        path = write(tmp_path)
        with read_events_v3(path) as s:
            v = s[0]
            frag = v.group_of_points("frag")
            inst = v.group_of_points("inst")
            assert len(frag) == len(v.points) == len(inst)
            # chain_event: fragment {1, 2} (5 points), fragment {3} (4),
            # one instance headed by 1 holding all 9
            assert sorted(np.unique(frag, return_counts=True)[1].tolist()) == [4, 5]
            assert set(frag.tolist()) == {1, 3}
            assert (inst == 1).all()

class TestPointsStorage:
    """How points/flat is stored: invisible to a reader, apart from precision."""

    def test_round_mantissa_bounds_the_relative_error(self):
        from pysupera.io_v3 import round_mantissa
        rng = np.random.default_rng(1)
        a = (rng.standard_normal(100_000) * 10.0 ** rng.uniform(-3, 5, 100_000)
             ).astype(np.float32)
        r = round_mantissa(a, 7)
        assert r.dtype == np.float32
        assert np.max(np.abs(r - a) / np.abs(a)) <= 2.0 ** -8 + 1e-7
        special = np.array([0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
        out = round_mantissa(special, 7)
        assert out[0] == 0 and np.isinf(out[1]) and np.isinf(out[2]) and np.isnan(out[3])

    def test_column_chunks_and_rounding_are_plain_hdf5(self, tmp_path):
        """A bare h5py read sees an ordinary (N, 6) float32 array."""
        import hdf5plugin  # noqa: F401 -- the filter plugin, as for LZ4
        parts, frags, insts = chain_event()
        for p in parts:
            if len(p.point_cloud):
                pc = np.zeros((len(p.point_cloud), 6), dtype=np.float32)
                pc[:, :3] = p.point_cloud[:, :3]
                pc[:, 4] = 1234.5678
                p.point_cloud = pc
        path = str(tmp_path / "cols.h5")
        with open_writer_v3(path, points=dict(chunks="columns", bitshuffle=True,
                                              energy_mantissa_bits=7)) as w:
            w.append_event(build_layout(parts, frags, insts))
        with h5py.File(path) as f:
            d = f["points/flat"]
            assert d.shape[1] == 6 and d.dtype == np.float32
            assert d.chunks[1] == 1 and "32008" in d._filters
            q = d[:, 4]
        assert np.all(np.abs(q - 1234.5678) / 1234.5678 <= 2.0 ** -8)

    def test_repack_keeps_the_bitshuffle_filter(self, tmp_path):
        from pysupera.io import repack
        path = str(tmp_path / "cols.h5")
        with open_writer_v3(path, points=dict(chunks="columns", bitshuffle=True)) as w:
            parts, frags, insts = chain_event()
            w.append_event(build_layout(parts, frags, insts))
        repack(path)
        with h5py.File(path) as f:
            assert "32008" in f["points/flat"]._filters
            assert f["points/flat"].chunks[1] == 1



class TestColumnsAndHitLabels:
    """points/columns names points/flat; hit_labels aligns with the hits CSR."""

    def test_point_columns_are_named_in_the_file(self, tmp_path):
        from pysupera.io_v3 import DEFAULT_POINT_COLUMNS
        cols = ("x", "y", "z", "t", "dE", "dX", "theta", "phi", "p")
        for columns, want in ((None, DEFAULT_POINT_COLUMNS), (cols, cols)):
            path = str(tmp_path / f"c{len(want)}.h5")
            with open_writer_v3(path, columns=columns) as w:
                parts, frags, insts = chain_event()
                w.append_event(build_layout(parts, frags, insts))
            with h5py.File(path) as f:
                assert f["points/flat"].shape[1] == len(want)
                names = tuple(c.decode() if isinstance(c, bytes) else c
                              for c in f["points/columns"][()])
                assert names == want
            with read_events_v3(path) as st:
                assert st.point_columns == want

    @staticmethod
    def labels(n0, n1, sensor=False):
        out = []
        for v, n in ((0, n0), (1, n1)):
            d = {"volume": v, "plane": 0, "source": "Pixel",
                 "fragment_id": np.arange(n, dtype=np.int32) - 1,
                 "instance_id": np.full(n, 7, dtype=np.int32),
                 "is_le": np.arange(n) % 2 == 0}
            if sensor:
                d["on_sensor"] = d["fragment_id"] >= 0
            out.append(d)
        return out

    def test_hit_labels_round_trip_per_event(self, tmp_path):
        path = str(tmp_path / "lab.h5")
        with open_writer_v3(path) as w:
            for n0, n1 in ((3, 2), (1, 4)):
                parts, frags, insts = chain_event()
                w.append_event(build_layout(parts, frags, insts))
                w.append_hit_labels(self.labels(n0, n1, sensor=True))
        with h5py.File(path) as f:
            g = f["hit_labels/volume1/plane0"]
            assert g.attrs["source"] == "Pixel"
            np.testing.assert_array_equal(g["offsets"][:], [0, 2, 6])
            assert g["is_le"].dtype == bool
        with read_events_v3(path) as st:
            ev1 = st.hit_labels(1)
            assert set(ev1) == {(0, 0), (1, 0)}
            np.testing.assert_array_equal(ev1[(1, 0)]["fragment_id"], [-1, 0, 1, 2])
            np.testing.assert_array_equal(ev1[(1, 0)]["on_sensor"],
                                          [False, True, True, True])

    def test_a_plane_appearing_late_is_refused(self, tmp_path):
        path = str(tmp_path / "lab.h5")
        with open_writer_v3(path) as w:
            parts, frags, insts = chain_event()
            w.append_event(build_layout(parts, frags, insts))
            w.append_hit_labels(self.labels(1, 1)[:1])
            with pytest.raises(ValueError, match="first appear"):
                w.append_hit_labels(self.labels(1, 1))

    def test_a_fragment_row_names_its_instance(self, tmp_path):
        parts, frags, insts = chain_event()
        lay = build_layout(parts, frags, insts)
        c = lay.columns
        row = {int(i): k for k, i in enumerate(c["id"])}
        # chain_event: fragments {1, 2} and {3} both inside instance 1
        assert c["frag_inst_id"][row[1]] == 1 and c["frag_inst_id"][row[3]] == 1

    def test_keep_gives_a_row_to_a_pointless_fragment(self):
        """
        A fragment a hit is labelled with must get a row even with no points.
        Particle 0 of chain_event has none; drop it from every ancestry by
        making it its own primary and it only gets a row through keep.
        """
        parts, frags, insts = chain_event()
        for p in parts:
            if int(p.id) == 1:
                p.parent_id = 1
        ids = lambda lay: {int(i) for i in lay.columns["id"]}
        assert 0 not in ids(build_layout(parts, frags, insts))
        assert 0 in ids(build_layout(parts, frags, insts, keep={0}))
