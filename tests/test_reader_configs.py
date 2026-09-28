"""
The reader config group: one file per kind of input.

``pysupera/conf/reader/`` holds three configs -- plain EDepSim, JAXTPC wire
and JAXTPC pixel -- and the two JAXTPC ones inherit the first so the shared
dataset keys are written once.  These tests hold that arrangement together:
that each one composes, that it selects the reader it claims to, and that
choosing a JAXTPC config without its input paths fails loudly.

That last one is not hypothetical.  ``cfg.reader.get(key, None)`` returns
``None`` for an unfilled mandatory (``???``) value rather than raising, and
``key in cfg`` is ``False`` for one -- so the obvious spellings of the guard
both let a ``reader=jaxtpc_pixel`` run fall through to plain EDepSim and
produce truth steps with nothing said about it.
"""

import pytest
from omegaconf import OmegaConf

from pysupera.config import build_reader, load_cfg

pytest.importorskip("h5py")

#: Every config in the reader group, and the reader class it must produce.
READERS = ("edepsim_h5", "jaxtpc_wire", "jaxtpc_pixel")

#: Keys the base defines and every variant must therefore inherit.
SHARED_KEYS = ("format", "particle_key", "vertex_key", "step_key",
               "ass_key", "electron_energy_threshold")


def _cfg(name, *extra):
    return load_cfg([f"reader={name}", "io.input_path=in.h5",
                     "io.output_path=out.h5", *extra])


@pytest.mark.parametrize("name", READERS)
def test_every_reader_config_composes(name):
    cfg = _cfg(name)
    assert cfg.reader.format == "edepsim_h5"


@pytest.mark.parametrize("name", READERS)
def test_variants_inherit_the_shared_keys(name):
    """The point of the within-group defaults: no duplicated dataset keys."""
    cfg = _cfg(name)
    for key in SHARED_KEYS:
        assert key in cfg.reader, f"{name} lost {key}"


@pytest.mark.parametrize("name", ("jaxtpc_wire", "jaxtpc_pixel"))
def test_jaxtpc_configs_demand_their_input_paths(name):
    cfg = _cfg(name)
    for key in ("jaxtpc_seg_path", "jaxtpc_inst_path"):
        assert OmegaConf.is_missing(cfg.reader, key)


def test_the_hit_reader_also_demands_the_sensor_file():
    """Its pixels are the ones that become points."""
    assert OmegaConf.is_missing(_cfg("jaxtpc_pixel").reader, "jaxtpc_sensor_path")


@pytest.mark.parametrize("name", ("jaxtpc_wire", "jaxtpc_pixel"))
def test_a_jaxtpc_config_without_paths_is_refused(name):
    """
    Not silently downgraded to EDepSim.  Asking for JAXTPC input and being
    handed truth steps is the kind of mistake that is only noticed in a plot
    weeks later.
    """
    with pytest.raises(ValueError, match="JAXTPC input"):
        build_reader(_cfg(name))


def test_plain_edepsim_does_not_demand_them():
    cfg = _cfg("edepsim_h5")
    for key in ("jaxtpc_seg_path", "jaxtpc_inst_path"):
        assert not OmegaConf.is_missing(cfg.reader, key)


def test_only_the_pixel_config_carries_the_hit_options():
    """
    A wire readout has no 3-D image to convert, so the options that only
    mean something for pixel input stay out of its config rather than
    sitting there as nulls that look like they might work.
    """
    pixel = _cfg("jaxtpc_pixel").reader
    wire = _cfg("jaxtpc_wire").reader
    for key in ("point_source", "hit_x_from", "hit_energy",
                "hit_reference_tick", "pixel_pitch_mm",
                "pixel_drift_direction", "jaxtpc_sensor_path"):
        # `in` is False for a mandatory (???) value, which the sensor path is
        assert key in pixel or OmegaConf.is_missing(pixel, key), \
            f"pixel config lost {key}"
        assert key not in wire, f"wire config should not offer {key}"


def test_pixel_defaults_are_the_documented_ones():
    r = _cfg("jaxtpc_pixel").reader
    assert r.point_source == "hits"          # what the sensors recorded
    assert r.hit_x_from == "nominal"         # the detector's own inference
    assert r.hit_energy == "charge"
    assert r.hit_reference_tick == 0
    assert r.pixel_geometry_from_fit is False


def test_pixel_geometry_defaults_to_the_reference_detector():
    """
    The pitch and drift direction are in no JAXTPC output file, so they are
    configured -- and defaulted to the cubic_pixel detector rather than left
    null, so a run needs only paths.  Defaulting them is only safe because
    a wrong value cannot pass quietly: the calibration applies them as
    stated and raises when they do not reproduce the truth deposits.
    """
    r = _cfg("jaxtpc_pixel").reader
    assert r.pixel_pitch_mm == 4.32                      # 0.432 cm
    assert list(r.pixel_drift_direction) == [-1, 1]      # shared cathode
    assert r.pixel_geometry_from_fit is False            # stated, then checked


# ---------------------------------------------------------------------------
# Settings that come with a reader (conf/readout/)
# ---------------------------------------------------------------------------

def test_the_pixel_reader_brings_the_hit_mode_settings():
    """Hits need values outside the reader group; choosing the reader is enough."""
    cfg = _cfg("jaxtpc_pixel")
    assert cfg.particle.min_pc_size == 63
    assert cfg.particle.voxelize.store == "voxels"          # 3 mm truth voxels
    assert cfg.distance_threshold > 4.32 * 2 ** 0.5
    assert cfg.check_group_ownership is False


@pytest.mark.parametrize("name", ("edepsim_h5", "jaxtpc_wire"))
def test_every_other_reader_keeps_the_truth_defaults(name):
    cfg = _cfg(name)
    assert cfg.particle.min_pc_size == 5
    assert cfg.particle.voxelize.store == "voxels"
    assert cfg.distance_threshold == 5.2
    assert cfg.check_group_ownership is True


def test_switching_the_pixel_reader_to_deposits_is_refused():
    """A pixel batch is read as its hits; there is no deposit mode for it."""
    from pysupera.config import check_readout_settings
    with pytest.raises(ValueError, match="not supported"):
        check_readout_settings(_cfg("jaxtpc_pixel", "reader.point_source=deposits"))
    check_readout_settings(_cfg("jaxtpc_pixel"))                 # fine


def test_a_flag_still_beats_the_reader_settings():
    cfg = _cfg("jaxtpc_pixel", "particle.min_pc_size=40", "distance_threshold=7")
    assert cfg.particle.min_pc_size == 40 and cfg.distance_threshold == 7


# ---------------------------------------------------------------------------
# preset=cubic_pixel_hits
# ---------------------------------------------------------------------------
# A preset bundles the values that are only correct together and that cut
# across config groups -- reader geometry, particle.min_pc_size, and two
# top-level checks -- so a sensor-hit run needs only file paths.


def _preset(*extra):
    return load_cfg(["preset=cubic_pixel_hits", "io.input_path=in.h5",
                     "io.output_path=out.h5", "reader.jaxtpc_seg_path=s.h5",
                     "reader.jaxtpc_inst_path=i.h5", *extra])


def test_preset_selects_the_pixel_reader_and_hit_mode():
    cfg = _preset()
    assert cfg.reader.format == "edepsim_h5"
    assert cfg.reader.point_source == "hits"
    assert "pixel_pitch_mm" in cfg.reader        # the pixel reader, not wire


def test_preset_fills_in_everything_a_sensor_hit_run_needs():
    """The point of it: nothing but paths should be needed on the command line."""
    cfg = _preset()
    assert cfg.reader.pixel_pitch_mm == 4.32
    assert list(cfg.reader.pixel_drift_direction) == [-1, 1]
    # The sensor image sets the threshold; a per-share cut would leave
    # sensor pixels unlabelled.
    assert cfg.reader.hit_charge_threshold == 0
    assert cfg.particle.min_pc_size == 63
    # points/flat: the true deposits behind the hits, 3 mm voxels
    assert cfg.particle.voxelize.store == "voxels"
    # must clear the pixel diagonal, 4.32 * sqrt(2) = 6.11 mm
    assert cfg.distance_threshold > 4.32 * 2 ** 0.5
    # a group's hits straddle fragment splits in hit mode, so the invariant
    # the groups/ table rests on cannot be enforced
    assert cfg.check_group_ownership is False


def test_a_command_line_flag_still_beats_the_preset():
    cfg = _preset("particle.min_pc_size=42", "reader.hit_charge_threshold=3")
    assert cfg.particle.min_pc_size == 42
    assert cfg.reader.hit_charge_threshold == 3


def test_without_the_preset_nothing_changes():
    """
    The preset is opt-in.  A plain run must keep the truth-tuned defaults,
    or every existing wire workflow moves under it.
    """
    cfg = load_cfg(["io.input_path=in.h5", "io.output_path=out.h5"])
    assert cfg.particle.min_pc_size == 5
    assert cfg.check_group_ownership is True
    assert cfg.reader.format == "edepsim_h5"
    assert "point_source" not in cfg.reader     # the plain EDepSim reader
