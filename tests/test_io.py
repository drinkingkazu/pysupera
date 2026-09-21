"""
Round-trip tests for the HDF5 writer and reader.

These pin the on-disk format (2.3.0) so that a schema change has to break a
test rather than a downstream analysis.  The IO layer previously had no
coverage at all, which made it the riskiest part of the codebase to touch.

Reading is checked two ways on purpose: through the public API
(:class:`~pysupera.io.EventStore`), which is what consumers use, and through
raw h5py, which pins the actual dataset layout that other tools -- the WebGL
viewer among them -- depend on.
"""
import numpy as np
import pytest
import h5py

from pysupera import (Particle, open_writer, write_events, read_events,
                      EventStore, voxmap_path)
from pysupera.io import FORMAT_VERSION, _PC_NDIM, repack
from pysupera.utils import SemanticType
from tests.conftest import make_particle, PT_PRIMARY, PT_TRACK, PT_DECAY


SCALARS = ("id", "geant4_id", "parent_id", "ancestor_id", "pdg", "parent_pdg",
           "interaction_id", "interaction_type")


def rep(pid, members, n_pts=4, **kw):
    """A representative particle with a member list, as the rep levels expect."""
    p = make_particle(pid, kw.pop("itype", PT_PRIMARY), n_pts=n_pts, **kw)
    p.member_ids = list(members)
    p.parent_frag_id = kw.get("parent_frag_id", -1)
    p.parent_inst_id = kw.get("parent_inst_id", -1)
    return p


def simple_event(ids, n_pts=5):
    return [make_particle(i, PT_TRACK, n_pts=n_pts) for i in ids]


# ============================================================================
# Particle round trip
# ============================================================================

class TestParticleRoundTrip:

    def test_scalar_fields_survive(self, tmp_path):
        p = make_particle(7, PT_DECAY, pdg=13, parent_pdg=211,
                          parent_id=3, ancestor_id=1, interaction_id=4)
        p.geant4_id = 4242
        path = str(tmp_path / "a.h5")
        write_events(path, [[p]])
        with read_events(path) as store:
            q = store[0][0]
        assert (q.id, q.geant4_id, q.parent_id, q.ancestor_id) == (7, 4242, 3, 1)
        assert (q.pdg, q.parent_pdg) == (13, 211)
        assert q._interaction_id == 4
        assert q._interaction_type == int(p._interaction_type)

    def test_sem_type_is_stored_not_rederived(self, tmp_path):
        # The writer stores sem_type and the reader must trust it rather than
        # re-running the classifier, which could disagree after preprocessing.
        p = make_particle(1, PT_TRACK, n_pts=20)
        p.sem_type = SemanticType.kMichel
        path = str(tmp_path / "b.h5")
        write_events(path, [[p]])
        with read_events(path) as store:
            assert store[0][0].sem_type is SemanticType.kMichel

    def test_point_cloud_values_survive(self, tmp_path):
        pc = np.arange(18, dtype=np.float32).reshape(3, 6)
        p = make_particle(1, PT_TRACK, pc=pc)
        path = str(tmp_path / "c.h5")
        write_events(path, [[p]])
        with read_events(path) as store:
            out = store[0][0].point_cloud
        assert out.shape == (3, _PC_NDIM)
        np.testing.assert_allclose(out, pc)

    def test_zero_point_particle(self, tmp_path):
        p = make_particle(1, PT_TRACK, pc=np.zeros((0, 6), dtype=np.float32))
        path = str(tmp_path / "d.h5")
        write_events(path, [[p]])
        with read_events(path) as store:
            q = store[0][0]
        assert len(q.point_cloud) == 0
        assert q.id == 1

    def test_empty_event(self, tmp_path):
        path = str(tmp_path / "e.h5")
        write_events(path, [[], simple_event([1, 2]), []])
        with read_events(path) as store:
            assert len(store) == 3
            assert store[0] == [] and store[2] == []
            assert [p.id for p in store[1]] == [1, 2]

    def test_multi_event_boundaries(self, tmp_path):
        events = [simple_event([0, 1, 2]), simple_event([0, 1]),
                  simple_event([0, 1, 2, 3])]
        path = str(tmp_path / "f.h5")
        write_events(path, events)
        with read_events(path) as store:
            assert len(store) == 3
            assert [len(store[i]) for i in range(3)] == [3, 2, 4]
            # ids restart per event
            assert [p.id for p in store[2]] == [0, 1, 2, 3]

    def test_incremental_writer_matches_bulk(self, tmp_path):
        events = [simple_event([0, 1]), simple_event([0, 1, 2])]
        bulk, incr = str(tmp_path / "g1.h5"), str(tmp_path / "g2.h5")
        write_events(bulk, events)
        with open_writer(incr) as w:
            for ev in events:
                w.append_event(ev)
        with h5py.File(bulk) as a, h5py.File(incr) as b:
            for name in SCALARS:
                np.testing.assert_array_equal(a[f"particles/{name}"][:],
                                              b[f"particles/{name}"][:])
            np.testing.assert_allclose(a["points/flat"][:], b["points/flat"][:])


# ============================================================================
# Point-cloud width
# ============================================================================

class TestPointCloudWidth:

    def test_written_width_is_pc_ndim(self, tmp_path):
        path = str(tmp_path / "w.h5")
        write_events(path, [[make_particle(1, PT_TRACK, n_pts=3)]])
        with h5py.File(path) as f:
            assert f["points/flat"].shape[1] == _PC_NDIM

    def test_narrow_cloud_is_zero_padded(self, tmp_path):
        pc = np.array([[1., 2., 3.]], dtype=np.float32)      # xyz only
        path = str(tmp_path / "n.h5")
        write_events(path, [[make_particle(1, PT_TRACK, pc=pc)]])
        with read_events(path) as store:
            out = store[0][0].point_cloud
        assert out.shape == (1, _PC_NDIM)
        np.testing.assert_allclose(out[0, :3], [1., 2., 3.])
        np.testing.assert_allclose(out[0, 3:], 0.0)

    def test_wide_cloud_is_truncated(self, tmp_path):
        pc = np.arange(9, dtype=np.float32).reshape(1, 9)
        path = str(tmp_path / "t.h5")
        write_events(path, [[make_particle(1, PT_TRACK, pc=pc)]])
        with read_events(path) as store:
            out = store[0][0].point_cloud
        assert out.shape == (1, _PC_NDIM)
        np.testing.assert_allclose(out[0], pc[0, :_PC_NDIM])


# ============================================================================
# Representative levels
# ============================================================================

class TestRepresentativeLevels:

    def _write(self, path):
        parts = simple_event([0, 1, 2, 3])
        frags = [rep(0, [0, 1]), rep(2, [2, 3])]
        insts = [rep(0, [0, 1, 2, 3])]
        frags[1].parent_frag_id = 0
        insts[0].parent_inst_id = 0
        with open_writer(path) as w:
            w.append_fragments(frags)
            w.append_instances(insts)
            w.append_event(parts)
        return parts, frags, insts

    def test_rows_and_ids(self, tmp_path):
        path = str(tmp_path / "r.h5")
        self._write(path)
        with h5py.File(path) as f:
            assert f["particle_fragments/id"].shape[0] == 2
            assert f["particle_instances/id"].shape[0] == 1
            np.testing.assert_array_equal(f["particle_fragments/id"][:], [0, 2])

    def test_member_lists(self, tmp_path):
        path = str(tmp_path / "m.h5")
        self._write(path)
        with h5py.File(path) as f:
            mo = f["particle_fragments/member_offsets"][:]
            mem = f["particle_fragments_members/flat"][:]
            assert mo.tolist() == [0, 2, 4]
            np.testing.assert_array_equal(mem, [0, 1, 2, 3])

    def test_group_parent_pointers(self, tmp_path):
        path = str(tmp_path / "p.h5")
        self._write(path)
        with h5py.File(path) as f:
            assert f["particle_fragments/parent_frag_id"][:].tolist() == [-1, 0]
            assert f["particle_instances/parent_inst_id"][:].tolist() == [0]

    def test_empty_rep_level_keeps_offsets_consistent(self, tmp_path):
        # write_fragments=false passes an empty list; the event fencepost must
        # still advance so the file stays internally consistent.
        path = str(tmp_path / "z.h5")
        with open_writer(path) as w:
            w.append_fragments([])
            w.append_instances([])
            w.append_event(simple_event([0, 1]))
        with h5py.File(path) as f:
            assert f["particle_fragments/id"].shape[0] == 0
            assert f["frag_events/offsets"][:].tolist() == [0, 0]
            assert int(f["n_events"][()]) == 1


# ============================================================================
# Event clouds
# ============================================================================

class TestEventClouds:

    def test_round_trip_and_split(self, tmp_path):
        nle = np.arange(18, dtype=np.float32).reshape(2, 9)
        le = np.arange(9, dtype=np.float32).reshape(1, 9)
        path = str(tmp_path / "ec.h5")
        with open_writer(path) as w:
            w.append_event_clouds(nle, le)
            w.append_event(simple_event([0]))
        with h5py.File(path) as f:
            np.testing.assert_allclose(f["non_le_cloud/flat"][:], nle)
            np.testing.assert_allclose(f["le_scatter_cloud/flat"][:], le)
            assert f["non_le_cloud/offsets"][:].tolist() == [0, 2]
            assert f["le_scatter_cloud/offsets"][:].tolist() == [0, 1]


# ============================================================================
# Offsets
# ============================================================================

class TestOffsets:

    def test_fenceposts_are_monotonic_and_sized(self, tmp_path):
        events = [simple_event([0, 1]), simple_event([0, 1, 2])]
        path = str(tmp_path / "o.h5")
        write_events(path, events)
        with h5py.File(path) as f:
            n = int(f["n_events"][()])
            ev = f["events/offsets"][:]
            pc = f["particles/pc_offsets"][:]
            assert len(ev) == n + 1
            assert ev.tolist() == sorted(ev.tolist())
            assert ev[-1] == f["particles/id"].shape[0]
            assert len(pc) == f["particles/id"].shape[0] + 1
            assert pc[-1] == f["points/flat"].shape[0]

    def test_format_version_recorded(self, tmp_path):
        path = str(tmp_path / "v.h5")
        write_events(path, [simple_event([0])])
        with h5py.File(path) as f:
            assert f["format_version"][()].decode() == FORMAT_VERSION


# ============================================================================
# Compression
# ============================================================================

class TestCompression:

    @pytest.mark.parametrize("comp", ["lzf", "gzip", "lz4", None])
    def test_filter_does_not_change_data(self, tmp_path, comp):
        events = [simple_event([0, 1, 2], n_pts=8)]
        ref = str(tmp_path / "ref.h5")
        write_events(ref, events, compression="lzf")
        other = str(tmp_path / f"{comp}.h5")
        write_events(other, events, compression=comp)
        with read_events(ref) as a, read_events(other) as b:
            pa, pb = a[0], b[0]
        assert [p.id for p in pa] == [p.id for p in pb]
        for x, y in zip(pa, pb):
            np.testing.assert_allclose(x.point_cloud, y.point_cloud)


# ============================================================================
# repack
# ============================================================================

class TestRepack:

    def _file(self, tmp_path, name="src.h5"):
        path = str(tmp_path / name)
        write_events(path, [simple_event([0, 1, 2], n_pts=6),
                            simple_event([0, 1], n_pts=6)])
        return path

    def test_in_place_preserves_data(self, tmp_path):
        path = self._file(tmp_path)
        with read_events(path) as s:
            before = [p.point_cloud.copy() for p in s[0]]
        repack(path, verify=True, verbose=False)
        with read_events(path) as s:
            after = [p.point_cloud for p in s[0]]
        assert len(before) == len(after)
        for x, y in zip(before, after):
            np.testing.assert_allclose(x, y)

    def test_to_dst_leaves_source_alone(self, tmp_path):
        src = self._file(tmp_path)
        dst = str(tmp_path / "dst.h5")
        repack(src, dst=dst, verify=True, verbose=False)
        with read_events(src) as a, read_events(dst) as b:
            assert len(a) == len(b) == 2
            np.testing.assert_allclose(a[0][0].point_cloud, b[0][0].point_cloud)

    def test_switches_compression(self, tmp_path):
        src = self._file(tmp_path)
        dst = str(tmp_path / "gz.h5")
        repack(src, dst=dst, compression="gzip", verify=True, verbose=False)
        with h5py.File(dst) as f:
            assert f["points/flat"].compression == "gzip"

    def test_preserves_filter_by_default(self, tmp_path):
        src = str(tmp_path / "gz_src.h5")
        write_events(src, [simple_event([0, 1])], compression="gzip")
        dst = str(tmp_path / "out.h5")
        repack(src, dst=dst, verify=True, verbose=False)
        with h5py.File(dst) as f:
            assert f["points/flat"].compression == "gzip"

    def test_rechunk_changes_layout_not_data(self, tmp_path):
        src = self._file(tmp_path)
        dst = str(tmp_path / "rc.h5")
        repack(src, dst=dst, rechunk=True, verify=True, verbose=False)
        with read_events(src) as a, read_events(dst) as b:
            np.testing.assert_allclose(a[0][0].point_cloud, b[0][0].point_cloud)

    def test_dst_equal_to_src_is_rejected(self, tmp_path):
        src = self._file(tmp_path)
        with pytest.raises(ValueError, match="must differ"):
            repack(src, dst=src, verbose=False)

    def test_verify_catches_corruption(self, tmp_path):
        # Corrupt the copy after writing to prove verify actually compares.
        src = self._file(tmp_path)
        dst = str(tmp_path / "bad.h5")
        repack(src, dst=dst, verify=True, verbose=False)
        with h5py.File(dst, "r+") as f:
            f["points/flat"][0, 0] += 1.0
        from pysupera.io import _verify_same_datasets
        with pytest.raises(ValueError, match="verification failed"):
            _verify_same_datasets(src, dst)


# ============================================================================
# Backward compatibility
# ============================================================================

class TestBackwardCompatibility:

    def test_missing_geant4_id_falls_back_to_id(self, tmp_path):
        # Files written before 2.2.0 have no geant4_id column.  Simulate one
        # by deleting the dataset, and check the reader substitutes `id`.
        path = str(tmp_path / "old.h5")
        write_events(path, [simple_event([0, 1, 2])])
        with h5py.File(path, "r+") as f:
            del f["particles/geant4_id"]
        with read_events(path) as store:
            ps = store[0]
        assert [p.geant4_id for p in ps] == [p.id for p in ps] == [0, 1, 2]


# ============================================================================
# repack: auto-flip and scope
# ============================================================================

import pytest as _pytest                                        # noqa: E402
from pysupera.io import repack, dominant_filter                 # noqa: E402


def _mk(path, compression, n=64):
    import h5py
    with h5py.File(path, "w") as f:
        g = f.create_group("g")
        kw = {} if compression is None else {"compression": compression}
        g.create_dataset("a", data=np.arange(n, dtype=np.int64),
                         chunks=(16,), **kw)
        g.create_dataset("b", data=np.arange(n, dtype=np.float32),
                         chunks=(16,), **kw)
        f.create_dataset("scalar", data=np.int64(n))
    return str(path)


class TestDominantFilter:

    def test_reports_gzip(self, tmp_path):
        assert dominant_filter(_mk(tmp_path / "g.h5", "gzip")) == "gzip"

    def test_reports_none(self, tmp_path):
        assert dominant_filter(_mk(tmp_path / "n.h5", None)) == "none"

    def test_ignores_unchunked_scalars(self, tmp_path):
        # the scalar dataset carries no filter and must not sway the vote
        assert dominant_filter(_mk(tmp_path / "g.h5", "gzip")) == "gzip"


class TestRepackAuto:
    """``auto`` exists so a round trip needs no memory of the current state."""

    def test_gzip_flips_to_lz4(self, tmp_path):
        src = _mk(tmp_path / "g.h5", "gzip")
        dst = str(tmp_path / "out.h5")
        repack(src, compression="auto", dst=dst, verbose=False, verify=True)
        assert dominant_filter(dst) == "lz4"

    def test_the_flip_round_trips(self, tmp_path):
        src = _mk(tmp_path / "g.h5", "gzip")
        a = str(tmp_path / "a.h5"); b = str(tmp_path / "b.h5")
        repack(src, compression="auto", dst=a, verbose=False)
        repack(a, compression="auto", dst=b, verbose=False, verify=True)
        assert dominant_filter(b) == "gzip"

    def test_uncompressed_cannot_be_flipped(self, tmp_path):
        src = _mk(tmp_path / "n.h5", None)
        with _pytest.raises(ValueError, match="auto-flip"):
            repack(src, compression="auto", dst=str(tmp_path / "o.h5"),
                   verbose=False)

    def test_data_survives_the_flip(self, tmp_path):
        import h5py
        src = _mk(tmp_path / "g.h5", "gzip")
        dst = str(tmp_path / "out.h5")
        repack(src, compression="auto", dst=dst, verbose=False, verify=True)
        with h5py.File(dst, "r") as f:
            assert np.array_equal(f["g/a"][:], np.arange(64, dtype=np.int64))


class TestRepackScope:

    def test_source_leaves_uncompressed_alone(self, tmp_path):
        src = _mk(tmp_path / "n.h5", None)
        dst = str(tmp_path / "out.h5")
        repack(src, compression="gzip", dst=dst, verbose=False, verify=True)
        assert dominant_filter(dst) == "none"

    def test_all_compresses_everything(self, tmp_path):
        src = _mk(tmp_path / "n.h5", None)
        dst = str(tmp_path / "out.h5")
        repack(src, compression="gzip", dst=dst, scope="all",
               verbose=False, verify=True)
        assert dominant_filter(dst) == "gzip"

    def test_scope_does_not_affect_already_compressed(self, tmp_path):
        src = _mk(tmp_path / "g.h5", "gzip")
        for scope in ("source", "all"):
            dst = str(tmp_path / f"{scope}.h5")
            repack(src, compression="lzf", dst=dst, scope=scope,
                   verbose=False, verify=True)
            assert dominant_filter(dst) == "lzf"

    def test_a_bad_scope_is_rejected(self, tmp_path):
        src = _mk(tmp_path / "g.h5", "gzip")
        with _pytest.raises(ValueError, match="scope"):
            repack(src, compression="gzip", dst=str(tmp_path / "o.h5"),
                   scope="everything", verbose=False)
