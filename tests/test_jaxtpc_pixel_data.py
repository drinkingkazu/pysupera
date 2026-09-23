"""
Smoke check of pixel readout against a real JAXTPC production directory.

The synthetic fixtures in ``test_jaxtpc_readout.py`` pin down the logic; they
cannot catch JAXTPC changing the schema underneath us, because they *are* our
idea of the schema.  This module reads an actual pixel batch instead, and is
skipped wherever that batch is not on disk -- so it costs nothing in CI and
fails loudly on the machine that has the data.

Point it somewhere else with::

    PYSUPERA_JAXTPC_PIXEL_DIR=/path/to/batch \\
    PYSUPERA_EDEPSIM_H5=/path/to/edepsim.h5 pytest tests/

The directory is expected in JAXTPC's current production layout::

    <dir>/step/<run>/<name>_step_<NNNN>_<SS>.h5
    <dir>/hits/<run>/<name>_hits_<NNNN>_<SS>.h5
"""

import os
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pysupera.hits_subset import extract
from pysupera.readers.format_jaxtpc import read_readout_type

# Default location: the scratch tree this was developed against.
_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_DIR = _REPO.parent / "data" / "jaxtpc_pixel"
_DEFAULT_EDEPSIM = _REPO.parent / "data" / "kazu.h5"


def _one(dirname: str, root: Path) -> Path | None:
    """The single step/hits file under *root*/<dirname>, or None."""
    found = sorted((root / dirname).rglob("*.h5"))
    return found[0] if len(found) == 1 else (found[0] if found else None)


def _paths():
    root = Path(os.environ.get("PYSUPERA_JAXTPC_PIXEL_DIR", _DEFAULT_DIR))
    edep = Path(os.environ.get("PYSUPERA_EDEPSIM_H5", _DEFAULT_EDEPSIM))
    if not root.is_dir() or not edep.is_file():
        return None
    step, hits = _one("step", root), _one("hits", root)
    if step is None or hits is None:
        return None
    return edep, step, hits


_P = _paths()
pytestmark = pytest.mark.skipif(
    _P is None,
    reason=f"no JAXTPC pixel batch (looked in {_DEFAULT_DIR}); "
           f"set PYSUPERA_JAXTPC_PIXEL_DIR to run")


#: The geometry of the detector this batch was simulated with, from
#: config/cubic_pixel_config.yaml.  Stated, never fitted -- that is the whole
#: contract these tests exist to hold.
PITCH_MM = 4.32
DRIFT_DIRECTION = [-1, 1]


def _run_pysupera():
    """
    The console script, wherever pip put it.

    ``shutil.which`` is not enough: the entry point lands in ``~/.local/bin``
    for a user install, which is often off PATH inside a container.
    """
    import shutil
    exe = shutil.which("run_pysupera")
    if exe:
        return exe
    local = Path.home() / ".local" / "bin" / "run_pysupera"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    pytest.skip("run_pysupera not found on PATH or in ~/.local/bin")


def _sensor():
    """The sensor file, which states the drift velocity and sampling period."""
    root = _P[1].parent.parent.parent
    found = sorted((root / "sensor").rglob("*.h5"))
    return str(found[0]) if found else None


def hit_kwargs(**over):
    """Reader kwargs for hit mode with the geometry configured."""
    edep, step, hits = (str(x) for x in _P)
    kw = dict(edepsim_path=edep, seg_path=step, inst_path=hits,
              point_source="hits", sensor_path=_sensor(),
              pixel_pitch_mm=PITCH_MM,
              pixel_drift_direction=list(DRIFT_DIRECTION))
    kw.update(over)
    return kw


@pytest.fixture(scope="module")
def reader():
    from pysupera.readers import JaxtpcHDF5Reader
    edep, step, hits = _P
    with JaxtpcHDF5Reader(edepsim_path=str(edep), seg_path=str(step),
                          inst_path=str(hits)) as r:
        yield r


def test_it_is_a_pixel_batch(reader):
    assert reader.readout_type == "pixel"
    with h5py.File(_P[2], "r") as f:
        assert read_readout_type(f) == "pixel"
        # One plane per volume, named Pixel -- not three projections.
        vol = f["event_000/volume_0"]
        planes = [k for k in vol if isinstance(vol[k], h5py.Group)]
        assert planes == ["Pixel"]
        assert "center_py" in vol["Pixel"] and "center_wires" not in vol["Pixel"]


def test_reads_visible_point_clouds(reader):
    particles = reader[0]
    assert len(particles) > 0
    with_points = [p for p in particles if len(p.point_cloud)]
    assert with_points, "every particle came back empty"
    # Visibility is a filter, not a wipe: some deposits survive and some do not.
    stats = reader.last_mask_stats
    assert 0 < stats["n_visible"] < stats["n_total"]


def test_deposit_ids_index_the_group_table(reader):
    """
    Each particle carries the *input deposit* row behind every point, and
    ``last_deposit_to_group`` is indexed by exactly that.  If the two ever
    drift apart the group table silently describes the wrong deposits, so
    check the bound rather than trust it.
    """
    particles = reader[0]
    d2g = reader.last_deposit_to_group
    assert d2g is not None and len(d2g)
    for p in particles:
        dep = getattr(p, "deposit_id", None)
        if dep is None or not len(dep):
            continue
        assert len(dep) == len(p.point_cloud)
        assert dep.min() >= 0 and dep.max() < len(d2g)


def test_every_visible_deposit_belongs_to_a_readout_group(reader):
    """
    The reader's contract: a deposit is in a point cloud only because its
    group produced a hit.  Verify it the long way round, straight from the
    file, rather than by re-running the same function.
    """
    particles = reader[0]
    d2g = reader.last_deposit_to_group

    with h5py.File(_P[2], "r") as f:
        ev = f["event_000"]
        active, off = set(), 0
        for vname in sorted(k for k in ev if k.startswith("volume_")):
            vol = ev[vname]
            for g in (vol[k] for k in vol if isinstance(vol[k], h5py.Group)):
                active.update((g["group_ids"][:].astype(np.int64) + off).tolist())
            off += len(vol["group_to_track"])

    dep = np.concatenate([p.deposit_id for p in particles
                          if getattr(p, "deposit_id", None) is not None
                          and len(p.deposit_id)])
    assert len(dep)
    assert set(np.unique(d2g[dep]).tolist()) <= active


def test_subset_of_the_real_file(tmp_path):
    dst = tmp_path / "pixel_small.h5"
    info = extract(str(_P[2]), str(dst), n_events=1, verbose=False)
    assert info["readout"] == "pixel"
    assert info["planes"] == ["Pixel"]
    # The point of the tool: the CSR bulk is what makes the file big.
    assert info["size_after"] < info["size_before"] / 20
    with h5py.File(dst, "r") as f:
        pg = f["event_000/volume_0/Pixel"]
        assert {"center_py", "center_pz", "center_times",
                "peak_charges", "group_ids"} <= set(pg.keys())
        assert read_readout_type(f) == "pixel"


# ---------------------------------------------------------------------------
# point_source=hits — the detected image
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hit_reader():
    from pysupera.readers import JaxtpcHDF5Reader
    with JaxtpcHDF5Reader(**hit_kwargs()) as r:
        yield r


def test_the_configured_geometry_is_applied_and_verified(hit_reader):
    """
    The configured constants must be used *as given* -- not re-measured --
    and then checked against the truth deposits.  Both halves matter: a
    reader that quietly substituted its own fit would pass a residual test
    no matter what was configured.
    """
    hit_reader[0]
    reports = hit_reader.pixel_geometry_report
    assert len(reports) == 2
    for rep, expect_dir, expect_anode in zip(reports, DRIFT_DIRECTION,
                                             (-2160.0, 2160.0)):
        assert rep["source"] == "stated"
        # exactly what was configured, to the bit
        assert rep["pitch_mm"] == PITCH_MM
        assert rep["drift_direction"] == expect_dir
        assert rep["x_anode_mm"] == pytest.approx(expect_anode, abs=1e-6)
        assert rep["reference_tick"] == 0.0
        # and it reproduces the truth: x continuous, y and z quantised
        assert rep["residual_mm"]["x"] < 1.0
        assert rep["residual_mm"]["y"] < PITCH_MM
        assert rep["residual_mm"]["z"] < PITCH_MM


def test_the_data_agrees_with_the_configured_geometry(hit_reader):
    """
    The audit, reported alongside but never applied.  The detector YAML says
    0.432 cm and 0.16 cm/us at 2 MHz; the hits should say the same.
    """
    hit_reader[0]
    for rep, expect_dir in zip(hit_reader.pixel_geometry_report,
                               DRIFT_DIRECTION):
        m = rep["measured"]
        assert m["pitch_mm"] == pytest.approx(4.32, abs=0.01)
        assert m["mm_per_tick"] == pytest.approx(0.80, abs=0.01)
        assert m["drift_velocity_mm_us"] == pytest.approx(1.60, abs=0.01)
        assert m["time_step_us"] == pytest.approx(0.50, abs=0.01)
        assert m["drift_direction"] == expect_dir


def test_hit_mode_refuses_to_guess_the_geometry():
    """
    Without the configured constants hit mode must stop, not fall back to
    measuring them -- a constant fitted to the truth it is then compared
    against cannot be caught when it is wrong.  The error has to carry the
    values the data is consistent with, or it is not actionable.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    from pysupera.readers.pixel_hits import PixelGeometryError
    kw = hit_kwargs(pixel_pitch_mm=None)
    with JaxtpcHDF5Reader(**kw) as r:
        with pytest.raises(PixelGeometryError) as exc:
            r[0]
    msg = str(exc.value)
    assert "pixel_pitch_mm" in msg
    assert "4.32" in msg                       # what the data says
    assert "pixel_geometry_from_fit" in msg    # and the way out


@pytest.mark.parametrize("bad,off_axis", [
    (dict(pixel_pitch_mm=4.0), "y"),
    (dict(pixel_drift_direction=[1, 1]), "x"),
    (dict(drift_velocity_mm_us=1.4), "x"),
])
def test_a_wrong_configured_constant_is_caught(bad, off_axis):
    """
    The point of applying stated constants rather than fitted ones: a wrong
    one now produces a residual instead of being absorbed.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    from pysupera.readers.pixel_hits import PixelGeometryError
    with JaxtpcHDF5Reader(**hit_kwargs(**bad)) as r:
        with pytest.raises(PixelGeometryError, match="does not reproduce"):
            r[0]


def test_fitting_the_geometry_is_opt_in_and_warns():
    """Available for an unfamiliar batch, but it must say what it costs."""
    from pysupera.readers import JaxtpcHDF5Reader
    kw = hit_kwargs(pixel_pitch_mm=None, pixel_geometry_from_fit=True)
    with JaxtpcHDF5Reader(**kw) as r:
        with pytest.warns(RuntimeWarning, match="circular"):
            r[0]
        assert r.pixel_geometry_report[0]["source"] == "fitted"


def test_hit_mode_attaches_every_hit(hit_reader):
    particles = hit_reader[0]
    st = hit_reader.last_hit_stats
    assert st["n_hits"] > 1_000_000        # the detected image is not small
    assert st["n_unmatched"] == 0
    assert sum(len(p.point_cloud) for p in particles) == st["n_attached"]
    # A hit belongs to one group and a group to one track, so no hit is shared.
    assert hit_reader.last_deposit_to_group is not None
    assert len(hit_reader.last_deposit_to_group) == st["n_attached"]


def test_hit_mode_point_is_its_own_provenance(hit_reader):
    """
    In deposit mode a point's group is reached through the deposit index; in
    hit mode the point *is* the hit, so ``deposit_id`` indexes the group map
    directly.  Downstream group provenance relies on that alignment.
    """
    particles = hit_reader[0]
    g = hit_reader.last_deposit_to_group
    for p in particles:
        dep = getattr(p, "deposit_id", None)
        if dep is None or not len(dep):
            continue
        assert len(dep) == len(p.point_cloud)
        assert dep.min() >= 0 and dep.max() < len(g)


def test_true_t0_lands_on_the_truth_cloud():
    """
    Removing t0 should put the detected hits back on the deposits that made
    them, to about a pixel.  The nominal convention should not -- t0 differs
    across the interactions of one event by up to 1094 us, which is 175 cm
    of drift.  This is the difference the two conventions exist to express,
    so pin it.
    """
    pytest.importorskip("scipy")
    from scipy.spatial import cKDTree
    from pysupera.readers import JaxtpcHDF5Reader
    edep, step, hits = (str(x) for x in _P)
    with JaxtpcHDF5Reader(edepsim_path=edep, seg_path=step,
                          inst_path=hits) as r:
        truth = {p.id: p.point_cloud for p in r[0] if len(p.point_cloud)}

    got = {}
    for mode in ("true_t0", "nominal"):
        with JaxtpcHDF5Reader(**hit_kwargs(hit_x_from=mode)) as r:
            big = sorted((p for p in r[0] if len(p.point_cloud) > 500),
                         key=lambda p: -len(p.point_cloud))[:10]
            d = []
            for p in big:
                t = truth.get(p.id)
                if t is None or len(t) < 20:
                    continue
                dist, _ = cKDTree(t[:, :3]).query(p.point_cloud[:, :3], k=1)
                d.append(float(np.median(dist)))
        got[mode] = float(np.median(d))

    # ~1 pixel pitch once t0 is gone; hundreds of mm while it is there.
    assert got["true_t0"] < 15.0
    assert got["nominal"] > 50.0


def test_hit_energy_conventions_differ_as_documented():
    from pysupera.readers import JaxtpcHDF5Reader
    from pysupera.utils import PointFeature
    with JaxtpcHDF5Reader(**hit_kwargs(hit_energy="charge")) as r:
        q = np.concatenate([p.point_cloud[:, PointFeature.energy]
                            for p in r[0] if len(p.point_cloud)])
    with JaxtpcHDF5Reader(**hit_kwargs(hit_energy="true_de")) as r:
        ps = r[0]
        e = np.concatenate([p.point_cloud[:, PointFeature.energy]
                            for p in ps if len(p.point_cloud)])
        dx = np.concatenate([p.point_cloud[:, PointFeature.dx]
                             for p in ps if len(p.point_cloud)])

    # The readout response is bipolar, so measured charge goes negative.
    assert (q < 0).any(), "expected some negative charge from a bipolar response"
    # Apportioned truth energy is a sum of non-negative deposits.
    assert (e >= 0).all()
    # Neither convention invents a path length.
    assert not dx.any()


def test_hit_mode_refuses_a_wire_file(tmp_path):
    """
    There is no 3-D image to convert in a wire readout, so asking for one has
    to say why rather than produce something.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    from pysupera.readers.pixel_hits import PixelGeometryError
    from tests.test_jaxtpc_readout import write_hits

    wire = write_hits(tmp_path / "wire_hits.h5", "wire")
    with JaxtpcHDF5Reader(**hit_kwargs(inst_path=str(wire))) as r:
        with pytest.raises((PixelGeometryError, KeyError)):
            r[0]


def test_the_default_reference_tick_agrees_with_the_data(hit_reader):
    """
    The reference tick is configured, not measured -- but the data still has
    an opinion, and it is reported.  This batch has pre_window_us = 0, so the
    configured 0 should sit on top of what the hits imply; a disagreement
    would mean the window does not open at Geant4 t=0.
    """
    hit_reader[0]
    for rep in hit_reader.pixel_geometry_report:
        assert rep["reference_tick"] == 0.0
        assert rep["reference_tick_derived"] == pytest.approx(0.0, abs=1.0)
        assert rep["reference_tick_offset_mm"] == pytest.approx(0.0, abs=1.0)


def test_an_explicit_reference_tick_moves_the_whole_cloud():
    """
    A stated reference tick must displace every x by exactly its own worth
    -- and toward each volume's own anode, which is why this compares point
    by point.  Averaged over the detector the two volumes drift opposite
    ways and the shift very nearly cancels, so a global mean would pass
    while the geometry was mirrored.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    from pysupera.utils import PointFeature
    def xs(ref):
        with JaxtpcHDF5Reader(**hit_kwargs(hit_reference_tick=ref)) as r:
            pc = [p.point_cloud for p in r[0] if len(p.point_cloud)]
            return np.concatenate(pc)[:, PointFeature.x]

    d = xs(100.0) - xs(0.0)
    # 100 ticks * 0.8 mm, signed by each volume's drift direction
    np.testing.assert_allclose(np.abs(d), 80.0, atol=0.05)
    assert (d < 0).any() and (d > 0).any(), "both drift directions expected"


def test_a_stated_reference_tick_is_not_treated_as_a_measurement():
    """
    The calibration residual exists to say whether the pitch, velocity and
    drift direction describe this detector.  A reference tick is a choice
    about where t=0 sits, not a measurement, so asking "what if the trigger
    were 100 ticks later" must not be reported as a broken detector -- the
    distance from the data is reported in mm instead.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    with JaxtpcHDF5Reader(**hit_kwargs(hit_reference_tick=100.0)) as r:
        r[0]
        for rep in r.pixel_geometry_report:
            assert rep["residual_mm"]["x"] < 1.0          # geometry still fine
            assert rep["reference_tick"] == 100.0
            assert rep["reference_tick_offset_mm"] == pytest.approx(80.0, abs=1.0)


def test_the_window_is_what_makes_deposits_invisible():
    """
    In this batch 'visible' turns out to mean 'arrived while the readout was
    open' rather than 'above threshold': the window is exactly one full
    drift, so any positive t0 walks the cathode-side end of an interaction
    past the last tick.  Worth pinning, because it decides how the two point
    sources should be compared -- they lose the same charge for the same
    reason.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    from pysupera.readers.format_jaxtpc import get_visible_segment_indices_by_track
    r = JaxtpcHDF5Reader(**hit_kwargs())
    try:
        r[0]                                    # triggers the calibration
        geoms, n_steps = r.pixel_geometry, r._num_time_steps()
        assert n_steps
        segs = r._load_seg_volumes("event_000")
        vis = get_visible_segment_indices_by_track(segs, r._inst_file,
                                                   "event_000")
        for v, (sv, vv) in enumerate(zip(segs, vis)):
            n = int(sv["n_actual"])
            seen = np.zeros(n, bool)
            for idx in vv.values():
                seen[idx] = True
            g = geoms[v]
            drift = np.abs(sv["positions_mm"][:n, 0]
                           - g.x_anode_mm) / g.drift_velocity_mm_us
            want = g.reference_tick + (drift + sv["t0_us"][:n]) / g.time_step_us
            past = want > n_steps - 1
            masked = ~seen
            assert masked.any()
            # almost every masked deposit is one the window never reached
            assert past[masked].mean() > 0.9
    finally:
        r.close()


def test_the_reader_reports_a_shift_that_recovers_the_truth(hit_reader):
    """
    Over the real batch: adding the shift to the nominal x must land on what
    the true_t0 convention would have produced, to float32 precision.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    from pysupera.utils import PointFeature

    ps = hit_reader[0]
    nominal = np.concatenate([p.point_cloud[:, PointFeature.x]
                              for p in ps if len(p.point_cloud)])
    shift = hit_reader.last_true_x_shift
    with JaxtpcHDF5Reader(**hit_kwargs(hit_x_from="true_t0")) as r:
        true_x = np.concatenate([p.point_cloud[:, PointFeature.x]
                                 for p in r[0] if len(p.point_cloud)])
        assert not r.last_true_x_shift.any()      # already true, nothing to add

    assert len(shift) == len(nominal)
    np.testing.assert_allclose(nominal + shift, true_x, atol=0.01)
    # it is a real displacement, not a rounding artefact
    assert np.abs(shift).max() > 1000.0


def test_the_shift_takes_few_distinct_values(hit_reader):
    """
    Why the column is affordable: it is drift_direction * t0 * v, so it has
    one value per (interaction, volume) pair rather than one per hit.  If
    this ever grew to thousands the storage argument would need revisiting.
    """
    hit_reader[0]
    n = len(np.unique(np.round(hit_reader.last_true_x_shift, 3)))
    assert n < 100, f"{n} distinct shifts -- expected a few tens"


def test_written_output_carries_the_shift(tmp_path):
    """
    End to end: the column survives voxelisation and the write, and reading
    x back and adding it puts the cloud inside the detector again -- the
    nominal x does not have to be.
    """
    from pysupera.io_v3 import read_events_v3
    import subprocess

    exe = _run_pysupera()
    edep, step, hits = (str(x) for x in _P)
    out = tmp_path / "hits.h5"
    cmd = [exe, "reader=jaxtpc_pixel",
           f"io.input_path={edep}", f"io.output_path={out}",
           f"reader.jaxtpc_seg_path={step}", f"reader.jaxtpc_inst_path={hits}",
           f"reader.jaxtpc_sensor_path={_sensor()}",
           "reader.point_source=hits",
           f"reader.pixel_pitch_mm={PITCH_MM}",
           "reader.pixel_drift_direction=[-1,1]",
           "check_group_ownership=false", "distance_threshold=6.2",
           "max_events=1", "progress=false", "report=false",
           f"hydra.run.dir={tmp_path}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]

    with read_events_v3(str(out)) as store:
        view = store[0]
        shift = store.true_x_shift(0)
        assert shift is not None and len(shift) == len(view.points)
        x = view.points[:, 0]
        true_x = x + shift
        # the true positions sit inside the detector; the nominal ones are
        # displaced by each interaction's own t0 and need not
        assert true_x.min() > -2160.0 and true_x.max() < 2160.0
        assert np.abs(shift).max() > 100.0


def test_deposit_mode_writes_the_column_as_zero(tmp_path):
    """
    The relation true_x = x + true_x_shift should hold for every file, so a
    deposit-mode run writes zeros rather than omitting the column.
    """
    from pysupera.io_v3 import read_events_v3
    import subprocess

    exe = _run_pysupera()
    edep, step, hits = (str(x) for x in _P)
    out = tmp_path / "dep.h5"
    r = subprocess.run(
        [exe, "reader=jaxtpc_pixel", f"io.input_path={edep}",
         f"io.output_path={out}", f"reader.jaxtpc_seg_path={step}",
         f"reader.jaxtpc_inst_path={hits}", "max_events=1",
         "progress=false", "report=false", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with read_events_v3(str(out)) as store:
        shift = store.true_x_shift(0)
        assert shift is not None and not shift.any()


# ---------------------------------------------------------------------------
# pysupera-truth-subset, and the run metadata the viewers read
# ---------------------------------------------------------------------------

def test_truth_subset_carries_what_the_viewer_needs(tmp_path):
    """
    vis_truth.html labels a truth deposit by walking deposit -> group ->
    fragment, so it needs the deposit positions from the step file and
    deposit_to_group from the hits file -- and it cannot read either, both
    being Blosc.  The extract has to bring all of it across as gzip.
    """
    from pysupera.truth_subset import extract
    _edep, step, hits = (str(x) for x in _P)
    dst = tmp_path / "truth.h5"
    info = extract(step, hits, str(dst), n_events=1, verbose=False)
    assert info["events"] == 1 and info["deposits"] > 0
    with h5py.File(dst, "r") as f:
        vol = f["event_000/volume_0"]
        assert {"positions", "de", "charge", "deposit_to_group"} <= set(vol.keys())
        n = int(vol.attrs["n_actual"])
        # every array is truncated to the real row count, not the padded shape
        for k in ("positions", "de", "charge", "deposit_to_group"):
            assert vol[k].shape[0] == n
        # the attributes needed to put positions back into mm
        for a in ("pos_step_mm", "pos_origin_x", "pos_origin_y", "pos_origin_z"):
            assert a in vol.attrs
        # gzip, or the browser cannot open it
        assert vol["positions"].compression == "gzip"
    # and it is worth doing: two Blosc files of tens of MB become one small one
    assert info["size_after"] < (info["size_step"] + info["size_hits"]) / 5


def test_truth_subset_charge_and_de_are_different_quantities(tmp_path):
    """
    Both are kept because they answer different questions -- dE is energy in
    MeV, charge is the ionisation electrons that survived recombination --
    and the viewer thresholds on one while showing the other.
    """
    from pysupera.truth_subset import extract
    _edep, step, hits = (str(x) for x in _P)
    dst = tmp_path / "truth.h5"
    extract(step, hits, str(dst), n_events=1, verbose=False)
    with h5py.File(dst, "r") as f:
        de = f["event_000/volume_0/de"][:]
        ch = f["event_000/volume_0/charge"][:]
    assert de.dtype == np.float32 and ch.dtype == np.float32   # not float16
    assert np.median(de) < 1.0 and np.median(ch) > 100.0       # MeV vs electrons


def test_the_deposit_to_group_chain_closes(tmp_path):
    """
    The join vis_truth.html performs: a deposit's group, shifted by the
    volume offset, must index the output's groups/fragment_id.  If those two
    numbering spaces ever drift apart the viewer paints truth deposits with
    another object's label and nothing raises.
    """
    from pysupera.truth_subset import extract
    from pysupera.io_v3 import read_events_v3
    import subprocess

    exe = _run_pysupera()
    edep, step, hits = (str(x) for x in _P)
    out = tmp_path / "out.h5"
    r = subprocess.run(
        [exe, "reader=jaxtpc_pixel", f"io.input_path={edep}",
         f"io.output_path={out}", f"reader.jaxtpc_seg_path={step}",
         f"reader.jaxtpc_inst_path={hits}",
         f"reader.jaxtpc_sensor_path={_sensor()}", "reader.point_source=hits",
         f"reader.pixel_pitch_mm={PITCH_MM}",
         "reader.pixel_drift_direction=[-1,1]",
         "check_group_ownership=false", "distance_threshold=6.2",
         "max_events=1", "progress=false", "report=false",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]

    truth = tmp_path / "truth.h5"
    extract(step, hits, str(truth), n_events=1, verbose=False)

    with h5py.File(out, "r") as fo, h5py.File(truth, "r") as ft:
        go = fo["groups/offsets"][:]
        vi = fo["groups/volume_offsets_index"][:]
        ga, gb = int(go[0]), int(go[1])
        frag = fo["groups/fragment_id"][ga:gb]
        voff = fo["groups/volume_offsets"][int(vi[0]):int(vi[1])]
        labelled = total = 0
        for v, vol in enumerate(sorted(k for k in ft["event_000"]
                                       if k.startswith("volume_"))):
            g = ft[f"event_000/{vol}/deposit_to_group"][:].astype(np.int64)
            g = g + int(voff[v])
            total += len(g)
            # every group number must land inside the table, not past its end
            assert g.min() >= 0 and g.max() < len(frag), \
                f"{vol}: group {g.max()} outside a table of {len(frag)}"
            labelled += int((frag[g] >= 0).sum())
    # most deposits are claimed; the rest are the ones the readout never saw
    assert 0.3 < labelled / total < 1.0


def test_output_records_how_it_was_made(tmp_path):
    """
    Nothing in the point columns says whether the energy column holds true dE
    or measured charge, and a viewer inferring it from negative values gets a
    quiet event wrong.  So the run writes it down.
    """
    from pysupera.io_v3 import read_events_v3
    import subprocess
    exe = _run_pysupera()
    edep, step, hits = (str(x) for x in _P)
    out = tmp_path / "meta.h5"
    r = subprocess.run(
        [exe, "reader=jaxtpc_pixel", f"io.input_path={edep}",
         f"io.output_path={out}", f"reader.jaxtpc_seg_path={step}",
         f"reader.jaxtpc_inst_path={hits}",
         f"reader.jaxtpc_sensor_path={_sensor()}", "reader.point_source=hits",
         f"reader.pixel_pitch_mm={PITCH_MM}",
         "reader.pixel_drift_direction=[-1,1]",
         "check_group_ownership=false", "max_events=1",
         "progress=false", "report=false", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with h5py.File(out, "r") as f:
        a = {k: str(v) for k, v in f.attrs.items()}
    assert a.get("point_source") == "hits"
    assert a.get("hit_energy") == "charge"
    assert a.get("hit_x_from") == "nominal"
    assert a.get("readout_type") == "pixel"
