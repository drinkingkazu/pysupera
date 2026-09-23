"""
JAXTPC wire and pixel readout.

JAXTPC simulates two LArTPC geometries.  A wire readout records three 2-D
projections (``U``/``V``/``Y`` planes, each hit a wire and a drift tick); a
pixel readout records one natively 3-D image (a single ``Pixel`` plane, each
hit two pixel indices and a drift tick).

pysupera reads both through one code path, and these tests pin down why that
is allowed to be true: *visibility is a property of a group*, and a group is
above threshold or not regardless of how the readout is arranged.  The
central test is :func:`test_visibility_identical_across_readouts`, which
feeds the same groups through both geometries and demands the same answer.

What does differ is anything that draws hits, so the plane discovery and the
hit-centre selection in ``pysupera-hits-subset`` are tested per geometry.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pysupera.hits_subset import KEEP, extract, _plane_names
from pysupera.readers.format_jaxtpc import (
    READOUT_PLANE_NAMES,
    get_visible_segment_indices_by_track,
    read_readout_type,
)


# ---------------------------------------------------------------------------
# Synthetic hits files
# ---------------------------------------------------------------------------
# Small enough to read at a glance, and shaped exactly like JAXTPC's output:
# per-volume lookup tables beside one subgroup per readout plane.

#: deposit index -> group, for the single volume in every fixture below.
DEPOSIT_TO_GROUP = np.array([0, 0, 1, 2, 2, 3, 4, 4, 5], dtype=np.int32)

#: group -> Geant4 track id.
GROUP_TO_TRACK = np.array([10, 11, 11, 12, 13, 14], dtype=np.int32)

#: Groups that made it above threshold.  Split across the wire planes below
#: so that no single plane sees all of them -- visibility is the union, and a
#: fixture where one plane already holds the answer would not prove that.
VISIBLE_GROUPS = (0, 2, 4)


def _write_plane(vol, name, groups, kind):
    """One readout-plane subgroup, with the centres its geometry stores."""
    g = vol.create_group(name)
    groups = np.asarray(groups, dtype=np.int32)
    n = len(groups)
    g.create_dataset("group_ids", data=groups)
    g.create_dataset("peak_charges", data=np.full(n, 7.0, dtype=np.float32))
    g.create_dataset("center_times", data=np.arange(n, dtype=np.int16) + 100)
    if kind == "pixel":
        g.create_dataset("center_py", data=np.arange(n, dtype=np.int16) + 20)
        g.create_dataset("center_pz", data=np.arange(n, dtype=np.int16) + 30)
    else:
        g.create_dataset("center_wires", data=np.arange(n, dtype=np.int16) + 40)
    # Bulk CSR arrays the viewer never opens; present so the subset tool has
    # something it is supposed to leave behind.
    g.create_dataset("charges_i16", data=np.ones(n * 4, dtype=np.int16))
    g.create_dataset("delta_times", data=np.zeros(n * 4, dtype=np.int8))
    return g


def write_hits(path, kind, readout_attr=True, plane_names=None, n_events=1):
    """A JAXTPC-shaped hits file for one readout geometry."""
    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        if readout_attr:
            cfg.attrs["readout_type"] = kind
        if plane_names is None:
            plane_names = READOUT_PLANE_NAMES[kind]
        for ev in range(n_events):
            vol = f.create_group(f"event_{ev:03d}/volume_0")
            vol.create_dataset("deposit_to_group", data=DEPOSIT_TO_GROUP)
            vol.create_dataset("group_to_track", data=GROUP_TO_TRACK)
            if kind == "pixel":
                _write_plane(vol, plane_names[0], VISIBLE_GROUPS, kind)
            else:
                # The same groups, scattered over the three projections: each
                # plane sees some of them, and only the union is the answer.
                per = ([VISIBLE_GROUPS[0]], [VISIBLE_GROUPS[1]],
                       [VISIBLE_GROUPS[0], VISIBLE_GROUPS[2]])
                for nm, gs in zip(plane_names, per):
                    _write_plane(vol, nm, gs, kind)
    return path


@pytest.fixture
def wire_hits(tmp_path):
    return write_hits(tmp_path / "wire_hits.h5", "wire")


@pytest.fixture
def pixel_hits(tmp_path):
    return write_hits(tmp_path / "pixel_hits.h5", "pixel")


# ---------------------------------------------------------------------------
# Readout-type detection
# ---------------------------------------------------------------------------

def test_read_readout_type(wire_hits, pixel_hits):
    with h5py.File(wire_hits, "r") as f:
        assert read_readout_type(f) == "wire"
    with h5py.File(pixel_hits, "r") as f:
        assert read_readout_type(f) == "pixel"


def test_readout_type_defaults_to_wire(tmp_path):
    """
    JAXTPC wrote hits files before ``readout_type`` existed, and all of them
    are wire.  Its own loader makes the same assumption, so a missing
    attribute must not raise.
    """
    p = write_hits(tmp_path / "old.h5", "wire", readout_attr=False)
    with h5py.File(p, "r") as f:
        assert read_readout_type(f) == "wire"


def test_readout_type_without_config_group(tmp_path):
    with h5py.File(tmp_path / "bare.h5", "w") as f:
        f.create_group("event_000")
    with h5py.File(tmp_path / "bare.h5", "r") as f:
        assert read_readout_type(f) == "wire"


# ---------------------------------------------------------------------------
# The claim that one code path is enough
# ---------------------------------------------------------------------------

def test_visibility_identical_across_readouts(wire_hits, pixel_hits):
    """
    The same groups above threshold must give the same visible segments,
    whichever geometry recorded them.  This is the whole reason pixel mode
    needed no new reader: the filter reads ``group_ids`` and nothing else.
    """
    seg_volumes = [{"n_actual": len(DEPOSIT_TO_GROUP)}]
    out = {}
    for tag, path in (("wire", wire_hits), ("pixel", pixel_hits)):
        with h5py.File(path, "r") as f:
            out[tag] = get_visible_segment_indices_by_track(
                seg_volumes, f, "event_000")[0]

    assert out["wire"].keys() == out["pixel"].keys()
    for tid in out["wire"]:
        np.testing.assert_array_equal(out["wire"][tid], out["pixel"][tid])

    # And the answer is the one the fixture describes, not merely a matching
    # pair of wrong ones: groups 0, 2, 4 -> tracks 10, 11, 13.
    expected_deposits = np.flatnonzero(np.isin(DEPOSIT_TO_GROUP, VISIBLE_GROUPS))
    got = np.sort(np.concatenate(list(out["pixel"].values())))
    np.testing.assert_array_equal(got, expected_deposits)
    assert set(out["pixel"]) == {10, 11, 13}


def test_planes_discovered_structurally(tmp_path):
    """
    A plane is any subgroup holding ``group_ids``.  A readout whose plane
    names this code has never seen must still resolve, because the names are
    JAXTPC's to change and visibility does not depend on them.
    """
    p = write_hits(tmp_path / "odd.h5", "pixel", plane_names=("Anode0",))
    with h5py.File(p, "r") as f:
        vis = get_visible_segment_indices_by_track(
            [{"n_actual": len(DEPOSIT_TO_GROUP)}], f, "event_000")[0]
    assert set(vis) == {10, 11, 13}


def test_no_plane_subgroup_raises(tmp_path):
    """A volume with no plane at all is a schema error, not an empty event."""
    with h5py.File(tmp_path / "noplane.h5", "w") as f:
        vol = f.create_group("event_000/volume_0")
        vol.create_dataset("deposit_to_group", data=DEPOSIT_TO_GROUP)
        vol.create_dataset("group_to_track", data=GROUP_TO_TRACK)
    with h5py.File(tmp_path / "noplane.h5", "r") as f:
        with pytest.raises(KeyError, match="Pixel"):
            get_visible_segment_indices_by_track(
                [{"n_actual": len(DEPOSIT_TO_GROUP)}], f, "event_000")


# ---------------------------------------------------------------------------
# pysupera-hits-subset
# ---------------------------------------------------------------------------

def test_subset_keeps_pixel_centres(pixel_hits, tmp_path):
    dst = tmp_path / "small.h5"
    info = extract(str(pixel_hits), str(dst), n_events=-1, verbose=False)
    assert info["readout"] == "pixel"
    assert info["planes"] == ["Pixel"]
    with h5py.File(dst, "r") as f:
        got = set(f["event_000/volume_0/Pixel"].keys())
    assert got == {"group_ids", "center_times", "peak_charges",
                   "center_py", "center_pz"}


def test_subset_keeps_wire_centres(wire_hits, tmp_path):
    dst = tmp_path / "small.h5"
    info = extract(str(wire_hits), str(dst), n_events=-1, verbose=False)
    assert info["readout"] == "wire"
    assert info["planes"] == ["U", "V", "Y"]
    with h5py.File(dst, "r") as f:
        got = set(f["event_000/volume_0/U"].keys())
    assert got == {"group_ids", "center_times", "peak_charges", "center_wires"}


def test_subset_drops_the_bulk_arrays(pixel_hits, tmp_path):
    """The CSR arrays are the file's bulk and the viewer never opens them."""
    dst = tmp_path / "small.h5"
    extract(str(pixel_hits), str(dst), n_events=-1, verbose=False)
    with h5py.File(dst, "r") as f:
        keys = set(f["event_000/volume_0/Pixel"].keys())
    assert not keys & {"charges_i16", "delta_times", "delta_py", "delta_pz"}
    assert "charges_i16" not in KEEP


def test_subset_carries_readout_type(pixel_hits, tmp_path):
    """
    The subset is what the browser viewer actually opens, so it has to keep
    saying which readout it is -- otherwise the viewer has to guess.
    """
    dst = tmp_path / "small.h5"
    extract(str(pixel_hits), str(dst), n_events=-1, verbose=False)
    with h5py.File(dst, "r") as f:
        assert read_readout_type(f) == "pixel"


def test_subset_limits_events(tmp_path):
    src = write_hits(tmp_path / "many.h5", "pixel", n_events=5)
    dst = tmp_path / "two.h5"
    info = extract(str(src), str(dst), n_events=2, verbose=False)
    assert info["events"] == 2
    with h5py.File(dst, "r") as f:
        assert sorted(k for k in f if k.startswith("event_")) == \
            ["event_000", "event_001"]


def test_plane_names_helper(pixel_hits):
    with h5py.File(pixel_hits, "r") as f:
        assert _plane_names(f["event_000/volume_0"]) == ["Pixel"]
