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
    <dir>/sensor/<run>/<name>_sensor_<NNNN>_<SS>.h5

It has to be written by a JAXTPC whose ``group_sizes`` is uint16: an older
batch is refused on read (see ``test_a_batch_with_wrapped_group_sizes_is_refused``).
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
_DEFAULT_DIR = _REPO.parent / "data" / "output_pixel"
#: The same batch as written before the uint8 group_sizes fix, if kept.
_WRAPPED_DIR = _REPO.parent / "data" / "jaxtpc_pixel"
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


def _wire():
    """A JAXTPC wire batch for the deposit path, or None when absent."""
    root = Path(os.environ.get("PYSUPERA_JAXTPC_WIRE_DIR",
                               str(_REPO.parent / "data" / "output_wireplane")))
    edep = Path(os.environ.get("PYSUPERA_WIRE_EDEPSIM_H5", str(_DEFAULT_EDEPSIM)))
    step, hits = _one("step", root), _one("hits", root)
    if not root.is_dir() or not edep.is_file() or step is None or hits is None:
        return None
    return str(edep), str(step), str(hits)


def test_it_is_a_pixel_batch():
    from pysupera.readers import JaxtpcHDF5Reader
    with JaxtpcHDF5Reader(**hit_kwargs()) as r:
        assert r.readout_type == "pixel"
        assert r.point_source == "hits"
    with h5py.File(_P[2], "r") as f:
        assert read_readout_type(f) == "pixel"
        # One plane per volume, named Pixel -- not three projections.
        vol = f["event_000/volume_0"]
        planes = [k for k in vol if isinstance(vol[k], h5py.Group)]
        assert planes == ["Pixel"]
        assert "center_py" in vol["Pixel"] and "center_wires" not in vol["Pixel"]


def test_a_pixel_batch_is_not_read_as_deposits():
    """A pixel batch is read as its hits; there is no deposit mode for it."""
    from pysupera.readers import JaxtpcHDF5Reader
    edep, step, hits = (str(x) for x in _P)
    with pytest.raises(ValueError, match="pixel readout"):
        JaxtpcHDF5Reader(edepsim_path=edep, seg_path=step, inst_path=hits,
                         point_source="deposits")


def test_the_readout_decides_the_point_source():
    """Left unset, the reader follows the file: hits for pixel."""
    from pysupera.readers import JaxtpcHDF5Reader
    kw = hit_kwargs(); kw.pop("point_source")
    with JaxtpcHDF5Reader(**kw) as r:
        assert r.point_source == "hits"


def test_subset_of_the_real_file(tmp_path):
    dst = tmp_path / "pixel_small.h5"
    info = extract(str(_P[2]), str(dst), n_events=1, verbose=False, hits=False)
    assert info["readout"] == "pixel"
    assert info["planes"] == ["Pixel"]
    # The point of the tool: the CSR bulk is what makes the file big.
    assert info["size_after"] < info["size_before"] / 20
    with h5py.File(dst, "r") as f:
        pg = f["event_000/volume_0/Pixel"]
        assert {"center_py", "center_pz", "center_times",
                "peak_charges", "group_ids"} <= set(pg.keys())
        assert read_readout_type(f) == "pixel"


def test_subset_decodes_every_hit_in_csr_order(tmp_path):
    """
    The viewers draw the subset's decoded hits against hit_labels, entry for
    entry, so they must be the hits file's own CSR entries in its own order.
    """
    from pysupera.readers.pixel_hits import decode_plane_hits
    dst = tmp_path / "pixel_hits.h5"
    extract(str(_P[2]), str(dst), n_events=1, verbose=False)
    with h5py.File(dst, "r") as f, h5py.File(_P[2], "r") as src:
        for vol in (0, 1):
            g = f[f"event_000/volume_{vol}/Pixel"]
            h = decode_plane_hits(src[f"event_000/volume_{vol}/Pixel"])
            np.testing.assert_array_equal(g["hit_py"][:], h["py"])
            np.testing.assert_array_equal(g["hit_pz"][:], h["pz"])
            np.testing.assert_array_equal(g["hit_tick"][:], h["tick"])
            np.testing.assert_allclose(g["hit_charge"][:], h["charge"], rtol=1e-6)


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
    assert st["n_hits"] > 100_000          # the detected image is not small
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
    # The truth: every segment of the event, by the track its group maps to.
    with JaxtpcHDF5Reader(**hit_kwargs(truth_segments=True)) as r:
        r[0]
        flat, group = r.last_truth_segments
    with h5py.File(_P[2], "r") as f:
        ev = f["event_000"]
        g2t = np.concatenate([ev[f"volume_{k}/group_to_track"][:] for k in (0, 1)])
    track = g2t[group]
    order = np.argsort(track, kind="stable")
    tids, starts = np.unique(track[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    truth = {int(t): flat[order[a:b], :3] for t, a, b in zip(tids, starts, ends)}

    got = {}
    for mode in ("true_t0", "nominal"):
        with JaxtpcHDF5Reader(**hit_kwargs(hit_x_from=mode)) as r:
            big = sorted((p for p in r[0] if len(p.point_cloud) > 500),
                         key=lambda p: -len(p.point_cloud))[:10]
            d = []
            for p in big:
                t = truth.get(int(getattr(p, "geant4_id", -1)))
                if t is None or len(t) < 20:
                    continue
                dist, _ = cKDTree(t).query(p.point_cloud[:, :3], k=1)
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


# ---------------------------------------------------------------------------
# The run metadata the viewers read
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# The sensor image decides which pixels exist
# ---------------------------------------------------------------------------

def _sensor_pixel_count(event_key="event_000"):
    """Pixels JAXTPC wrote to the sensor file, counted from its attributes."""
    with h5py.File(_sensor(), "r") as f:
        ev = f[event_key]
        return sum(int(ev[v][p].attrs.get("n_pixels", 0))
                   for v in ev for p in ev[v])


def test_every_sensor_pixel_is_labelled_and_nothing_else_is(hit_reader):
    """
    The labels are for a model whose input is the sensor image, so the cloud
    has to cover that image exactly: no sensor pixel without a hit on it, and
    no hit off it.  The hits file holds about twice as many pixels -- every
    group's share, whether or not the sum passed threshold -- so the mask
    has real work to do.
    """
    hit_reader[0]
    st = hit_reader.last_hit_stats
    assert st["n_sensor_pixels"] == _sensor_pixel_count()
    assert st["n_sensor_unlabelled"] == 0
    assert st["n_off_sensor"] > 0
    assert st["n_hits"] + st["n_off_sensor"] == st["n_decoded"]


def test_hit_mode_needs_the_sensor_file():
    from pysupera.readers import JaxtpcHDF5Reader
    with pytest.raises(ValueError, match="jaxtpc_sensor_path"):
        JaxtpcHDF5Reader(**hit_kwargs(sensor_path=None))


def test_a_sensor_file_from_another_run_is_refused(tmp_path):
    """A mask from another run would keep whatever happens to overlap."""
    import shutil
    from pysupera.readers import JaxtpcHDF5Reader
    other = tmp_path / "other_sensor.h5"
    shutil.copy(_sensor(), other)
    with h5py.File(other, "a") as f:
        f["config"].attrs["batch_timestamp"] = 1
    with pytest.raises(ValueError, match="different JAXTPC runs"):
        JaxtpcHDF5Reader(**hit_kwargs(sensor_path=str(other)))


@pytest.mark.skipif(not (_WRAPPED_DIR / "hits").is_dir(),
                    reason=f"no pre-fix batch at {_WRAPPED_DIR}")
def test_a_batch_with_wrapped_group_sizes_is_refused():
    """
    The batch as JAXTPC first wrote it: uint8 group_sizes, so each group of
    more than 255 entries shifted every later group on its plane.  Decoding
    it would put about 8% of the sensor pixels' labels on the wrong pixels.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    from pysupera.readers.pixel_hits import HitsFileError
    old = {d: str(sorted((_WRAPPED_DIR / d).rglob("*.h5"))[0])
           for d in ("step", "hits", "sensor")}
    kw = hit_kwargs(seg_path=old["step"], inst_path=old["hits"],
                    sensor_path=old["sensor"])
    with JaxtpcHDF5Reader(**kw) as r:
        with pytest.raises(HitsFileError, match="x 256 lost"):
            r[0]


def test_every_segment_arrives_with_its_step_values_and_group():
    """
    The truth cloud is built downstream from these, by group ownership, so
    every segment has to arrive -- in either mode -- with a group inside the
    event's group space and its EDepSim step's values: the seg file's own
    (rounded) position, time and dX must agree.
    """
    from pysupera.readers import JaxtpcHDF5Reader
    edep, step, hits = (str(x) for x in _P)
    with JaxtpcHDF5Reader(**hit_kwargs(truth_segments=True)) as r:
        r[0]
        flat, group = r.last_truth_segments
        segs = r._load_seg_volumes("event_000")
        n_groups = r.last_n_groups
    seg = np.concatenate([s_["positions_mm"][:s_["n_actual"]] for s_ in segs
                          if s_["n_actual"]])
    assert flat.shape == (len(seg), 9) and len(group) == len(seg)
    assert group.min() >= 0
    if n_groups is not None:
        assert group.max() < n_groups
    np.testing.assert_allclose(flat[:, :3], seg, atol=0.16)      # 0.3 mm grid
    t0 = np.concatenate([s_["t0_us"][:s_["n_actual"]] for s_ in segs if s_["n_actual"]])
    np.testing.assert_allclose(flat[:, 3], t0, atol=0.6)          # float16 us
    dx = np.concatenate([s_["dx"][:s_["n_actual"]] for s_ in segs if s_["n_actual"]])
    np.testing.assert_allclose(flat[:, 5] * 10.0, dx, rtol=2e-3, atol=1e-4)  # cm vs mm (float16)
    assert (flat[:, 4] >= 0).all()
    # direction and momentum: theta in [0, pi], phi in [-pi, pi], |p| > 0
    assert (flat[:, 6] >= 0).all() and (flat[:, 6] <= np.pi + 1e-3).all()
    assert (np.abs(flat[:, 7]) <= np.pi + 1e-3).all() and (flat[:, 8] > 0).all()


def test_without_truth_segments_the_reader_reads_none(hit_reader):
    hit_reader[0]
    assert hit_reader.last_truth_segments is None



def _run_to(tmp_path, reader, edep, step, hits, *extra):
    """Run the pipeline from this tree on one event; the output path."""
    import subprocess, sys
    out = tmp_path / f"{reader}.h5"
    code = ("import sys; from pysupera._run import main; "
            "sys.argv[0] = 'run_pysupera'; main()")
    r = subprocess.run(
        [sys.executable, "-c", code, f"reader={reader}",
         f"io.input_path={edep}", f"io.output_path={out}",
         f"reader.jaxtpc_seg_path={step}", f"reader.jaxtpc_inst_path={hits}",
         "max_events=1", "progress=false", f"hydra.run.dir={tmp_path}", *extra],
        cwd=str(_REPO), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return str(out)


def _check_labels_against_the_hits_file(out, hits):
    """Shared: one label per CSR entry, naming real rows, instance consistent."""
    from pysupera.io_v3 import read_events_v3, check_ranges
    assert check_ranges(out) == []
    with read_events_v3(out) as st:
        v = st[0]
        lab = st.hit_labels(0)
        assert st.point_columns == ("x", "y", "z", "t", "dE", "dX",
                                    "theta", "phi", "p")
    # every voxel carries a real direction and momentum (its earliest step)
    assert (v.points[:, 8] > 0).all()
    assert (v.points[:, 6] >= 0).all() and (v.points[:, 6] <= np.pi + 1e-3).all()
    c = v.columns
    frag_rows = {int(i) for i, f in zip(c["id"], c["frag_id"]) if f == i}
    inst_rows = {int(i) for i, j in zip(c["id"], c["inst_id"]) if j == i}
    frag_inst = {int(i): int(x) for i, x in zip(c["id"], c["frag_inst_id"])}
    with h5py.File(hits, "r") as f:
        for (vol, pl), d in lab.items():
            g = f[f"event_000/volume_{vol}/{d['source']}"]
            assert len(d["fragment_id"]) == int(g["group_sizes"][:].astype(np.int64).sum())
            fr, ins = d["fragment_id"], d["instance_id"]
            t = fr >= 0
            assert set(np.unique(fr[t]).tolist()) <= frag_rows
            assert set(np.unique(ins[ins >= 0]).tolist()) <= inst_rows
            assert all(frag_inst[int(a)] == int(b) for a, b in
                       set(zip(fr[t].tolist(), ins[t].tolist())))
    return v, lab


def test_pixel_hit_labels_follow_the_sensor_image(tmp_path):
    """
    Pixel: every CSR entry gets a label, and exactly the hits whose pixel the
    sensor file lacks are untraced (-1, on_sensor False) -- checked here
    against the sensor file itself, not the reader's own mask.
    """
    from pysupera.readers.pixel_hits import (decode_plane_hits,
                                             decode_sensor_plane, pixel_key,
                                             on_sensor)
    edep, step, hits = (str(x) for x in _P)
    out = _run_to(tmp_path, "jaxtpc_pixel", edep, step, hits,
                  f"reader.jaxtpc_sensor_path={_sensor()}")
    v, lab = _check_labels_against_the_hits_file(out, hits)
    assert set(lab) == {(0, 0), (1, 0)}
    with h5py.File(hits, "r") as fh, h5py.File(_sensor(), "r") as fs:
        for (vol, _), d in lab.items():
            h = decode_plane_hits(fh[f"event_000/volume_{vol}/Pixel"])
            s = decode_sensor_plane(fs[f"event_000/volume_{vol}/Pixel"])
            on = on_sensor(np.unique(pixel_key(s["py"], s["pz"], s["tick"])),
                           h["py"], h["pz"], h["tick"])
            np.testing.assert_array_equal(d["on_sensor"], on)
            np.testing.assert_array_equal(d["fragment_id"] >= 0, on)


@pytest.mark.skipif(_wire() is None, reason="no JAXTPC wire batch")
def test_wire_hit_labels_are_the_group_owners(tmp_path):
    """Wire: every plane's hits labelled by the fragment owning their group."""
    edep, step, hits = _wire()
    out = _run_to(tmp_path, "jaxtpc_wire", edep, step, hits)
    v, lab = _check_labels_against_the_hits_file(out, hits)
    from pysupera.io_v3 import read_events_v3
    with read_events_v3(out) as st:
        fo, le, voff = st.group_owners(0)
    with h5py.File(hits, "r") as f:
        for (vol, pl), d in lab.items():
            assert "on_sensor" not in d
            g = f[f"event_000/volume_{vol}/{d['source']}"]
            gid = (np.repeat(g["group_ids"][:].astype(np.int64),
                             g["group_sizes"][:].astype(np.int64)) + int(voff[vol]))
            np.testing.assert_array_equal(d["fragment_id"], fo[gid])



@pytest.mark.skipif(_wire() is None, reason="no JAXTPC wire batch")
def test_deposit_points_and_voxels_describe_the_same_fragments(tmp_path):
    """
    For truth deposits -- the wire readout -- store=points writes each
    deposit and store=voxels the proximity-grid voxels.  Only what a point
    is may differ: the same fragments, with the same energy each.
    """
    import subprocess, sys
    from pysupera.io_v3 import read_events_v3, check_ranges
    edep, step, hits = _wire()
    code = ("import sys; from pysupera._run import main; "
            "sys.argv[0] = 'run_pysupera'; main()")
    energy, rows = {}, {}
    for store in ("points", "voxels"):
        out = tmp_path / f"{store}.h5"
        r = subprocess.run(
            [sys.executable, "-c", code, "reader=jaxtpc_wire",
             f"io.input_path={edep}", f"io.output_path={out}",
             f"reader.jaxtpc_seg_path={step}", f"reader.jaxtpc_inst_path={hits}",
             "max_events=1", f"particle.voxelize.store={store}",
             "progress=false", f"hydra.run.dir={tmp_path}"],
            cwd=str(_REPO), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert check_ranges(str(out)) == []
        with read_events_v3(str(out)) as st:
            v = st[0]
            f = v.group_of_points("frag")
            ids, inv = np.unique(f, return_inverse=True)
            energy[store] = dict(zip(ids.tolist(), np.bincount(
                inv, weights=v.points[:, 4].astype(np.float64)).tolist()))
            rows[store] = len(v.points)
    assert rows["points"] > rows["voxels"]
    assert set(energy["points"]) == set(energy["voxels"])
    for k, e in energy["points"].items():
        assert e == pytest.approx(energy["voxels"][k], rel=1e-5)
