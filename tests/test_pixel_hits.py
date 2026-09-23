"""
Detected pixel hits as a point cloud (``point_source=hits``).

In the ordinary JAXTPC path a point is a Geant4 deposit the readout happened
to detect, so the geometry is truth geometry.  Here a point is a fired pixel,
and ``(py, pz, tick)`` has to be turned back into ``(x, y, z)``.  These tests
cover the three things that can silently go wrong in that conversion -- the
CSR expansion, the coordinate transform, and the geometry calibration that
supplies its constants -- plus the per-particle layout that follows.
"""

import numpy as np
import pytest

from pysupera.readers.pixel_hits import (
    ENERGY_FROM,
    PixelGeometryError,
    VolumePixelGeometry,
    X_FROM,
    build_hit_point_cloud,
    calibrate_volume,
    window_truncation,
    decode_plane_hits,
    group_reductions,
)
from pysupera.utils import PointFeature


# ---------------------------------------------------------------------------
# A plane standing in for an h5py group
# ---------------------------------------------------------------------------

class FakePlane(dict):
    """Enough of an h5py group for decode_plane_hits: __contains__ and [] ."""
    name = "/fake/Pixel"


def make_plane(sizes, centres, deltas, peaks, frac, pad=0, charge_key="charges_i16"):
    """
    A CSR pixel plane.

    *deltas* is a list of (dpy, dpz, dt) per entry and *frac* the stored
    per-entry charge fraction.  *pad* appends unused trailing entries, which
    is what JAXTPC actually writes -- the delta arrays run to a chunk
    boundary past ``group_sizes.sum()``.
    """
    n = sum(sizes)
    d = np.asarray(deltas, dtype=np.int8).reshape(n, 3)
    tail = np.zeros((pad, 3), dtype=np.int8)
    d = np.concatenate([d, tail])
    fdt = np.uint16 if charge_key.endswith("u16") else np.int16
    f = np.concatenate([np.asarray(frac, dtype=fdt), np.zeros(pad, dtype=fdt)])
    c = np.asarray(centres, dtype=np.int16).reshape(len(sizes), 3)
    return FakePlane({
        "group_ids":    np.arange(len(sizes), dtype=np.int32) * 2,   # not 0..G
        "group_sizes":  np.asarray(sizes, dtype=np.uint8),
        "center_py":    c[:, 0], "center_pz": c[:, 1], "center_times": c[:, 2],
        "peak_charges": np.asarray(peaks, dtype=np.float32),
        "delta_py":     d[:, 0], "delta_pz": d[:, 1], "delta_times": d[:, 2],
        charge_key:     f,
    })


# ---------------------------------------------------------------------------
# CSR expansion
# ---------------------------------------------------------------------------

def test_decode_expands_centres_plus_deltas():
    plane = make_plane(
        sizes=[2, 1],
        centres=[[10, 20, 30], [40, 50, 60]],
        deltas=[[0, 0, 0], [1, -2, 3], [-1, 1, 0]],
        peaks=[100.0, 50.0],
        frac=[32767, 16383, -32767])
    got = decode_plane_hits(plane)
    np.testing.assert_array_equal(got["py"], [10, 11, 39])
    np.testing.assert_array_equal(got["pz"], [20, 18, 51])
    np.testing.assert_array_equal(got["tick"], [30, 33, 60])
    np.testing.assert_array_equal(got["group"], [0, 0, 2])
    np.testing.assert_allclose(got["charge"], [100.0, 50.0, -50.0], rtol=1e-3)


def test_decode_ignores_the_padding_tail():
    """
    JAXTPC pads the delta arrays past ``group_sizes.sum()``.  Reading to the
    end of the array instead would append zero-delta ghosts sitting exactly
    on the last group's peak -- points that look plausible and are not there.
    """
    plane = make_plane(sizes=[2], centres=[[10, 20, 30]],
                       deltas=[[0, 0, 0], [1, 1, 1]], peaks=[10.0],
                       frac=[32767, 32767], pad=64)
    got = decode_plane_hits(plane)
    assert len(got["py"]) == 2


def test_decode_accepts_the_legacy_unsigned_charge():
    plane = make_plane(sizes=[1], centres=[[1, 2, 3]], deltas=[[0, 0, 0]],
                       peaks=[10.0], frac=[65535], charge_key="charges_u16")
    np.testing.assert_allclose(decode_plane_hits(plane)["charge"], [10.0],
                               rtol=1e-4)


def test_decode_without_a_charge_array_names_the_subset_tool():
    plane = make_plane(sizes=[1], centres=[[1, 2, 3]], deltas=[[0, 0, 0]],
                       peaks=[1.0], frac=[1])
    del plane["charges_i16"]
    with pytest.raises(KeyError, match="hits-subset"):
        decode_plane_hits(plane)


def test_decode_reports_groups_at_the_uint8_cap():
    plane = make_plane(sizes=[255], centres=[[0, 0, 0]],
                       deltas=[[0, 0, 0]] * 255, peaks=[1.0], frac=[1] * 255)
    assert decode_plane_hits(plane)["n_capped_groups"] == 1


# ---------------------------------------------------------------------------
# The coordinate transform
# ---------------------------------------------------------------------------

GEOM = VolumePixelGeometry(y_min_mm=-2160.0, z_min_mm=-2160.0,
                           x_min_mm=-2160.0, x_max_mm=0.0,
                           drift_direction=-1, pitch_mm=4.32,
                           drift_velocity_mm_us=1.6, time_step_us=0.5)

MIRRORED = VolumePixelGeometry(y_min_mm=-2160.0, z_min_mm=-2160.0,
                               x_min_mm=0.0, x_max_mm=2160.0,
                               drift_direction=1, pitch_mm=4.32,
                               drift_velocity_mm_us=1.6, time_step_us=0.5)


def test_mm_per_tick_is_the_period_times_the_velocity():
    assert GEOM.mm_per_tick == pytest.approx(0.8)
    assert GEOM.x_anode_mm == -2160.0 and GEOM.x_cathode_mm == 0.0
    assert MIRRORED.x_anode_mm == 2160.0 and MIRRORED.x_cathode_mm == 0.0


def test_to_xyz_uses_the_pixel_centre():
    """Pixel 0 sits half a pitch inside the boundary, not on it."""
    x, y, z = GEOM.to_xyz([0], [0], [0])
    assert y[0] == pytest.approx(-2160.0 + 2.16)
    assert z[0] == pytest.approx(-2160.0 + 2.16)


def test_to_xyz_drifts_away_from_the_anode():
    x, _, _ = GEOM.to_xyz([0], [0], [100])
    # drift_direction -1 puts the anode at x_min, so x grows with tick
    assert x[0] == pytest.approx(-2160.0 + 80.0)

    x2, _, _ = MIRRORED.to_xyz([0], [0], [100])
    assert x2[0] == pytest.approx(2160.0 - 80.0)


def test_reference_tick_moves_the_origin_of_the_drift_coordinate():
    """
    The reference tick is where Geant4 t=0 at zero drift lands.  A run with a
    pre-window has it non-zero, and every x must move with it -- otherwise
    the anode is reported in the wrong place and the whole cloud is offset.
    """
    from dataclasses import replace
    shifted = replace(GEOM, reference_tick=100.0)
    x0, _, _ = GEOM.to_xyz([0], [0], [100])
    x1, _, _ = shifted.to_xyz([0], [0], [100])
    assert x1[0] == pytest.approx(GEOM.x_anode_mm)      # zero elapsed time
    assert x0[0] - x1[0] == pytest.approx(80.0)


def test_outside_flags_points_past_the_volume_faces():
    # tick 3000 is past the cathode: 3000 * 0.8 = 2400 mm > the 2160 mm drift
    x, _, _ = GEOM.to_xyz([0], [0], [3000])
    assert GEOM.outside(x).all()
    x, _, _ = GEOM.to_xyz([0], [0], [1000])
    assert not GEOM.outside(x).any()


# ---------------------------------------------------------------------------
# Geometry calibration
# ---------------------------------------------------------------------------

def _synth_groups(geom, n=400, dt=0.5, seed=0):
    """Groups placed on a known geometry, with the truth x the fit must find."""
    rng = np.random.default_rng(seed)
    py = rng.integers(0, 1000, n)
    pz = rng.integers(0, 1000, n)
    # a handful of distinct interaction times, as in a real event
    t0 = rng.choice(np.array([0.0, 120.0, 400.0, 875.5]), n)
    tick = rng.integers(0, 2700, n).astype(float)
    x, y, z = geom.to_xyz(py, pz, tick - t0 / dt)
    return py, pz, tick, t0, np.stack([x, y, z], axis=1)


def test_calibration_recovers_a_known_geometry():
    py, pz, tick, t0, truth = _synth_groups(GEOM)
    got, report = calibrate_volume(py, pz, tick, t0, truth,
                                   x_range=(-2160.0, 0.0))
    assert got.pitch_mm == pytest.approx(GEOM.pitch_mm, rel=1e-6)
    assert got.mm_per_tick == pytest.approx(GEOM.mm_per_tick, rel=1e-6)
    assert got.drift_direction == GEOM.drift_direction
    assert got.x_anode_mm == pytest.approx(GEOM.x_anode_mm, abs=1e-3)
    assert got.y_min_mm == pytest.approx(GEOM.y_min_mm, abs=1e-3)
    # dt and the drift velocity fall out of the two x slopes together
    assert report["time_step_us"] == pytest.approx(0.5, rel=1e-6)
    assert report["drift_velocity_mm_us"] == pytest.approx(1.6, rel=1e-6)


def test_calibration_recovers_the_mirrored_volume():
    """
    The two volumes of a cathode-in-the-middle detector drift opposite ways.
    Getting the sign from the fit rather than from a config is the whole
    reason this function exists, so it has to work both ways round.
    """
    py, pz, tick, t0, truth = _synth_groups(MIRRORED, seed=1)
    got, _ = calibrate_volume(py, pz, tick, t0, truth, x_range=(0.0, 2160.0))
    assert got.drift_direction == 1
    assert got.x_anode_mm == pytest.approx(2160.0, abs=1e-3)


def test_calibration_rejects_a_mismatch():
    """Hits and deposits from different runs must not yield a quiet answer."""
    py, pz, tick, t0, truth = _synth_groups(GEOM, seed=2)
    rng = np.random.default_rng(3)
    truth = truth + rng.normal(0, 400.0, truth.shape)
    with pytest.raises(PixelGeometryError, match="does not reproduce"):
        calibrate_volume(py, pz, tick, t0, truth)


def test_calibration_needs_enough_groups():
    py, pz, tick, t0, truth = _synth_groups(GEOM, n=4, seed=4)
    with pytest.raises(PixelGeometryError, match="too few"):
        calibrate_volume(py, pz, tick, t0, truth)


# ---------------------------------------------------------------------------
# Per-group truth reductions
# ---------------------------------------------------------------------------

def test_group_reductions():
    d2g = np.array([0, 0, 1, 1, 1, 3])
    red = group_reductions(d2g, 4, t0_us=np.array([1.0, 3.0, 5.0, 5.0, 5.0, 9.0]),
                           de=np.array([1.0, 1.0, 2.0, 2.0, 2.0, 4.0]))
    np.testing.assert_allclose(red["t0"], [2.0, 5.0, 0.0, 9.0])
    np.testing.assert_allclose(red["de"], [2.0, 6.0, 0.0, 4.0])


# ---------------------------------------------------------------------------
# Per-particle layout
# ---------------------------------------------------------------------------

def _one_volume(n_per_group=(2, 2), tracks=(7, 9), t0=(0.0, 100.0),
                de=(1.0, 2.0)):
    hits = {
        "py":     np.array([0, 1, 2, 3], dtype=np.int32),
        "pz":     np.array([0, 0, 0, 0], dtype=np.int32),
        "tick":   np.array([10, 10, 20, 20], dtype=np.int32),
        "charge": np.array([3.0, 1.0, 2.0, 2.0], dtype=np.float32),
        "group":  np.repeat([0, 1], n_per_group).astype(np.int32),
        "n_capped_groups": 0,
    }
    return {"hits": hits, "geom": GEOM,
            "group_to_track": np.array(tracks, dtype=np.int64),
            "group_offset": 100,
            "group_t0": np.array(t0, dtype=np.float32),
            "group_de": np.array(de, dtype=np.float32)}


def test_offsets_follow_the_edepsim_particle_order():
    """
    The offsets index the EDepSim particle list, not the hit order -- if they
    drifted, every particle would get another particle's points and nothing
    would raise.
    """
    tid = np.array([9, 5, 7])          # deliberately not sorted
    flat, offsets, group, shift, stats = build_hit_point_cloud([_one_volume()], tid)
    assert stats["n_attached"] == 4 and stats["n_unmatched"] == 0
    # track 9 is group 1 (2 hits), track 5 has none, track 7 is group 0
    np.testing.assert_array_equal(offsets, [[0, 2], [2, 2], [2, 4]])
    np.testing.assert_array_equal(group[:2], [101, 101])   # group_offset applied
    np.testing.assert_array_equal(group[2:], [100, 100])


def test_hits_of_absent_tracks_are_dropped_and_counted():
    flat, offsets, group, shift, stats = build_hit_point_cloud(
        [_one_volume()], np.array([7]))
    assert stats["n_hits"] == 4
    assert stats["n_attached"] == 2
    assert stats["n_unmatched"] == 2
    assert len(flat) == 2


def test_true_t0_shifts_only_the_group_whose_t0_is_non_zero():
    tid = np.array([7, 9])
    naive, _, _, _, _ = build_hit_point_cloud([_one_volume()], tid,
                                           x_from="nominal")
    corr, _, _, _, _ = build_hit_point_cloud([_one_volume()], tid,
                                          x_from="true_t0",
                                          time_step_us=0.5)
    # group 0 has t0 = 0, so it does not move; group 1 has t0 = 100 us
    np.testing.assert_allclose(corr[:2, PointFeature.x], naive[:2, PointFeature.x])
    shift = corr[2:, PointFeature.x] - naive[2:, PointFeature.x]
    # drift_direction -1: removing t0 lowers the effective tick, so x drops
    np.testing.assert_allclose(shift, -(100.0 / 0.5) * GEOM.mm_per_tick, rtol=1e-5)


def test_energy_charge_versus_true_de():
    tid = np.array([7, 9])
    q, _, _, _, _ = build_hit_point_cloud([_one_volume()], tid, energy="charge")
    np.testing.assert_allclose(q[:, PointFeature.energy], [3.0, 1.0, 2.0, 2.0])

    d, _, _, _, _ = build_hit_point_cloud([_one_volume()], tid, energy="true_de")
    # group 0: dE 1.0 split 3:1 by charge; group 1: dE 2.0 split evenly
    np.testing.assert_allclose(d[:, PointFeature.energy], [0.75, 0.25, 1.0, 1.0])
    # and the apportionment conserves each group's energy exactly
    assert d[:2, PointFeature.energy].sum() == pytest.approx(1.0)
    assert d[2:, PointFeature.energy].sum() == pytest.approx(2.0)


def test_true_de_shares_by_magnitude_not_signed_charge():
    """
    The response is bipolar.  Sharing a group's true dE by signed charge
    would give the undershoot pixels negative deposited energy, which the
    deposit-mode column this one is meant to be comparable with can never
    hold -- and the positive pixels more than the group actually deposited.
    """
    vol = _one_volume()
    vol['hits']['charge'] = np.array([3.0, -1.0, 2.0, 2.0], dtype=np.float32)
    d, _, _, _, _ = build_hit_point_cloud([vol], np.array([7, 9]),
                                       energy="true_de")
    e = d[:, PointFeature.energy]
    assert (e >= 0).all()
    np.testing.assert_allclose(e[:2], [0.75, 0.25])
    assert e[:2].sum() == pytest.approx(1.0)


def test_dx_is_zero_because_a_hit_has_no_path_length():
    flat, _, _, _, _ = build_hit_point_cloud([_one_volume()], np.array([7, 9]))
    assert np.all(flat[:, PointFeature.dx] == 0)


def test_rejects_unknown_conventions():
    with pytest.raises(ValueError, match="x_from"):
        build_hit_point_cloud([], np.array([1]), x_from="nope")
    with pytest.raises(ValueError, match="energy"):
        build_hit_point_cloud([], np.array([1]), energy="nope")
    assert X_FROM == ("nominal", "true_t0")
    assert ENERGY_FROM == ("charge", "true_de")


def test_empty_event_gives_empty_slices():
    flat, offsets, group, shift, stats = build_hit_point_cloud([], np.array([1, 2, 3]))
    assert flat.shape == (0, 7)
    assert offsets.shape == (3, 2) and not offsets.any()
    assert stats["n_hits"] == 0


# ---------------------------------------------------------------------------
# The readout window
# ---------------------------------------------------------------------------

def test_window_truncation_counts_what_t0_pushes_off_the_end():
    """
    The window on the batch this was written against is exactly one full
    drift, so any positive t0 walks the cathode-side end of an interaction
    past the last tick, where it is never recorded.  That loss scales with
    t0, so it biases a comparison against deposit mode rather than merely
    shrinking it -- it has to be counted, not inferred.
    """
    # 2700 ticks * 0.8 mm = 2160 mm: the window is exactly the drift length
    n_steps = 2701
    x = np.array([-2160.0, -1080.0, -1.0, -1080.0, -1080.0])
    t0 = np.array([0.0, 0.0, 0.0, 900.0, -2000.0])
    before, after, total = window_truncation(x, t0, GEOM, n_steps)
    assert total == 5
    # at t0=0 everything inside the volume fits, by construction: the
    # deepest deposit drifts 2160 mm = 1350 us = exactly 2700 ticks
    # x=-1080 drifts 675 us; +900 us of t0 is 3150 ticks, past the end
    assert after == 1
    # a large negative t0 arrives before the window opened
    assert before == 1


def test_window_truncation_is_empty_when_nothing_is_late():
    x = np.array([-2160.0, -1080.0])
    t0 = np.zeros(2)
    assert window_truncation(x, t0, GEOM, 2701) == (0, 0, 2)


# ---------------------------------------------------------------------------
# Out-of-volume points
# ---------------------------------------------------------------------------

def test_out_of_volume_points_are_counted_not_clipped():
    """
    A nominal x is an inference, so one landing past the cathode is the
    honest answer; clipping would invent a wall the reconstruction lacks.
    """
    vol = _one_volume()
    # 3000 ticks * 0.8 = 2400 mm from the anode, past the 2160 mm cathode
    vol['hits']['tick'] = np.array([10, 10, 3000, 3000], dtype=np.int32)
    flat, _, _, _, stats = build_hit_point_cloud([vol], np.array([7, 9]))
    assert stats['n_outside_volume'] == 2
    # kept, and still outside
    assert (flat[:, PointFeature.x] > GEOM.x_max_mm).sum() == 2


# ---------------------------------------------------------------------------
# true_x_shift
# ---------------------------------------------------------------------------

def test_shift_recovers_the_true_x_from_the_nominal_one():
    """
    The contract the column exists for: true_x = x + true_x_shift, whichever
    drift convention produced x.
    """
    tid = np.array([7, 9])
    nom, _, _, shift, _ = build_hit_point_cloud(
        [_one_volume()], tid, x_from="nominal", time_step_us=0.5)
    tru, _, _, _, _ = build_hit_point_cloud(
        [_one_volume()], tid, x_from="true_t0", time_step_us=0.5)
    np.testing.assert_allclose(nom[:, PointFeature.x] + shift,
                               tru[:, PointFeature.x], atol=1e-3)


def test_shift_is_zero_when_x_is_already_true():
    """
    Under true_t0 the stored x *is* the true one, so the relation still
    holds with a zero shift -- which is what lets a consumer apply it
    unconditionally instead of branching on the convention.
    """
    _, _, _, shift, _ = build_hit_point_cloud(
        [_one_volume()], np.array([7, 9]), x_from="true_t0")
    assert not shift.any()


def test_shift_follows_the_drift_direction():
    """
    shift = drift_direction * t0 * v, so the two volumes of a shared-cathode
    detector shift opposite ways for the same interaction time.  A track
    crossing the cathode therefore needs this per point, not per particle.
    """
    v0 = _one_volume()
    v1 = _one_volume()
    v1["geom"] = MIRRORED
    tid = np.array([7, 9])
    _, _, _, s0, _ = build_hit_point_cloud([v0], tid, time_step_us=0.5)
    _, _, _, s1, _ = build_hit_point_cloud([v1], tid, time_step_us=0.5)
    np.testing.assert_allclose(s0, -s1)
    # group 1 has t0 = 100 us; -1 * 100 * 1.6 = -160 mm
    np.testing.assert_allclose(s0[2:], -160.0, atol=1e-3)
    np.testing.assert_allclose(s0[:2], 0.0, atol=1e-6)   # group 0 has t0 = 0
