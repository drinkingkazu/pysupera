"""
Tests for the panoptic training pipeline.

The interesting behaviour is voxel merging: a voxel touched by several
instances yields one row, its energy summed and its labels taken from the
winning contribution.  Those rules are what these pin down.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pysupera.torchdata import (build_event, collate_events, to_torch,
                                PysuperaEvents, make_dataloader, stage_batch,
                                SEM_PRIORITY, NO_LABEL)
from pysupera.utils import SemanticType as ST


def _split(b):
    """Where the LE run starts: the 6th field, defaulting to 'no LE'."""
    m = b[5] if len(b) > 5 else b[4]
    return b[4] if m < 0 else m


class FakeView:
    """Minimal stand-in for EventView: instance blocks over a point array."""

    def __init__(self, points, blocks, interactions=None):
        # blocks: (inst_id, sem_type, interaction_id, start, end[, split])
        # split defaults to end, i.e. the block carries no absorbed LE points.
        self.points = np.asarray(points, dtype=np.float32)
        n = len(blocks)
        self.columns = {
            "id":              np.array([b[0] for b in blocks], np.int32),
            "inst_id":         np.array([b[0] for b in blocks], np.int32),
            "inst_sem_type":   np.array([b[1] for b in blocks], np.int8),
            "interaction_id":  np.array([b[2] for b in blocks], np.int32),
            # start/end bound the non-LE run, le_start/le_end the LE run;
            # they are adjacent, so end == le_start.
            "inst_pc_start":    np.array([b[3] for b in blocks], np.int64),
            "inst_pc_end":      np.array([_split(b) for b in blocks], np.int64),
            "inst_pc_le_start": np.array([_split(b) for b in blocks], np.int64),
            "inst_pc_le_end":   np.array([b[4] for b in blocks], np.int64),
        }
        self.is_instance = np.ones(n, bool)
        self.interactions = interactions or {}
        self.event = 0


def pt(x, y, z, t, e):
    return [x, y, z, t, e, 0.0]


class TestVoxelMerging:

    def test_shared_voxel_becomes_one_row(self):
        # two instances deposit in the same cell
        v = FakeView([pt(0, 0, 0, 1.0, 3.0), pt(0, 0, 0, 2.0, 4.0)],
                     [(10, ST.kShower.value, 0, 0, 1),
                      (20, ST.kShower.value, 0, 1, 2)])
        ev = build_event(v)
        assert len(ev["points"]) == 1

    def test_energy_is_summed_not_replaced(self):
        v = FakeView([pt(0, 0, 0, 1.0, 3.0), pt(0, 0, 0, 2.0, 4.0)],
                     [(10, ST.kShower.value, 0, 0, 1),
                      (20, ST.kShower.value, 0, 1, 2)])
        ev = build_event(v)
        assert ev["points"][0, 3] == pytest.approx(7.0)

    def test_distinct_voxels_are_kept(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0), pt(1, 0, 0, 1.0, 2.0)],
                     [(10, ST.kTrack.value, 0, 0, 2)])
        ev = build_event(v)
        assert len(ev["points"]) == 2
        assert ev["points"][:, 3].sum() == pytest.approx(3.0)

    def test_total_energy_is_conserved(self):
        rng = np.random.default_rng(0)
        xyz = rng.integers(0, 3, size=(50, 3)).astype(np.float32)
        pts = np.column_stack([xyz, rng.random(50), rng.random(50),
                               np.zeros(50)]).astype(np.float32)
        v = FakeView(pts, [(1, ST.kShower.value, 0, 0, 50)])
        ev = build_event(v)
        assert ev["points"][:, 3].sum() == pytest.approx(pts[:, 4].sum(), rel=1e-5)


class TestSemanticPriority:

    @pytest.mark.parametrize("loser", [ST.kShower, ST.kDelta,
                                       ST.kMichel, ST.kLEScatter])
    def test_track_wins_every_contest(self, loser):
        # the track deposits later, so only priority can decide it
        v = FakeView([pt(0, 0, 0, 9.0, 1.0), pt(0, 0, 0, 1.0, 1.0)],
                     [(10, ST.kTrack.value, 0, 0, 1),
                      (20, loser.value,     1, 1, 2)])
        ev = build_event(v)
        assert ev["voxel_sem"][0] == ST.kTrack.value
        assert ev["voxel_instance"][0] == 10
        assert ev["voxel_interaction"][0] == 0

    def test_full_priority_order(self):
        order = [ST.kTrack, ST.kShower, ST.kDelta, ST.kMichel, ST.kLEScatter]
        assert [SEM_PRIORITY[s.value] for s in order] == [0, 1, 2, 3, 4]
        assert SEM_PRIORITY[ST.kUnknown.value] > SEM_PRIORITY[ST.kLEScatter.value]

    def test_shower_beats_le(self):
        v = FakeView([pt(0, 0, 0, 5.0, 1.0), pt(0, 0, 0, 1.0, 1.0)],
                     [(10, ST.kShower.value,     0, 0, 1),
                      (20, ST.kLEScatter.value,  0, 1, 2)])
        ev = build_event(v)
        assert ev["voxel_sem"][0] == ST.kShower.value
        assert ev["voxel_instance"][0] == 10


class TestTimeTieBreak:

    def test_earlier_time_wins_within_a_class(self):
        v = FakeView([pt(0, 0, 0, 9.0, 1.0), pt(0, 0, 0, 2.0, 1.0)],
                     [(10, ST.kShower.value, 0, 0, 1),
                      (20, ST.kShower.value, 1, 1, 2)])
        ev = build_event(v)
        assert ev["voxel_instance"][0] == 20        # t=2 beats t=9
        assert ev["voxel_interaction"][0] == 1

    def test_time_does_not_override_semantic_priority(self):
        # LE arrives first but must still lose to the shower
        v = FakeView([pt(0, 0, 0, 0.0, 1.0), pt(0, 0, 0, 8.0, 1.0)],
                     [(10, ST.kLEScatter.value, 0, 0, 1),
                      (20, ST.kShower.value,    1, 1, 2)])
        ev = build_event(v)
        assert ev["voxel_sem"][0] == ST.kShower.value
        assert ev["voxel_instance"][0] == 20


class TestLowEnergyScatters:
    """
    LE-ness is positional: points in the ``inst_pc_le_*`` run are deposits the
    instance absorbed, and must be labelled LE regardless of the host's class.
    These pass ``include_le=True``; the default is covered below.
    """

    def test_points_after_the_split_are_le(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0), pt(1, 0, 0, 1.0, 1.0),
                      pt(2, 0, 0, 1.0, 1.0)],
                     [(10, ST.kTrack.value, 0, 0, 3, 2)])
        ev = build_event(v, include_le=True)
        by_x = {p[0]: s for p, s in zip(ev["points"], ev["voxel_sem"])}
        assert by_x[0.0] == ST.kTrack.value
        assert by_x[1.0] == ST.kTrack.value
        assert by_x[2.0] == ST.kLEScatter.value

    def test_le_point_keeps_its_host_instance(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0), pt(1, 0, 0, 1.0, 1.0)],
                     [(77, ST.kShower.value, 4, 0, 2, 1)])
        ev = build_event(v, include_le=True)
        assert set(ev["voxel_instance"].tolist()) == {77}
        assert set(ev["voxel_interaction"].tolist()) == {4}
        assert sorted(ev["voxel_sem"].tolist()) == sorted(
            [ST.kShower.value, ST.kLEScatter.value])

    def test_le_loses_a_contested_voxel_to_its_host(self):
        # same cell reached by the shower's own point and an absorbed LE one,
        # the LE arriving first so only priority can decide it
        v = FakeView([pt(0, 0, 0, 9.0, 1.0), pt(0, 0, 0, 0.0, 1.0)],
                     [(10, ST.kShower.value, 0, 0, 2, 1)])
        ev = build_event(v, include_le=True)
        assert len(ev["points"]) == 1
        assert ev["voxel_sem"][0] == ST.kShower.value

    def test_le_loses_to_another_instance_too(self):
        v = FakeView([pt(0, 0, 0, 0.0, 1.0), pt(0, 0, 0, 9.0, 1.0)],
                     [(10, ST.kShower.value, 0, 0, 1, 0),   # pure LE block
                      (20, ST.kTrack.value,  1, 1, 2)])
        ev = build_event(v, include_le=True)
        assert ev["voxel_sem"][0] == ST.kTrack.value
        assert ev["voxel_instance"][0] == 20

    def test_no_le_run_means_no_le_labels(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0), pt(1, 0, 0, 1.0, 1.0)],
                     [(10, ST.kTrack.value, 0, 0, 2, -1)])
        ev = build_event(v)
        assert set(ev["voxel_sem"].tolist()) == {ST.kTrack.value}


class TestIncludeLeFlag:
    """
    The flag governs the voxel data as a whole: an excluded LE point
    contributes neither a row nor its energy, and the labels shrink with it.
    """

    def _view(self):
        # one shower voxel, one LE voxel elsewhere, both in the same block
        return FakeView([pt(0, 0, 0, 1.0, 3.0), pt(5, 0, 0, 2.0, 4.0)],
                        [(10, ST.kShower.value, 0, 0, 2, 1)])

    def test_le_is_dropped_by_default(self):
        ev = build_event(self._view())
        assert len(ev["points"]) == 1
        assert ev["points"][0, 0] == 0.0
        assert ST.kLEScatter.value not in ev["voxel_sem"].tolist()

    def test_dropped_le_energy_is_not_carried_over(self):
        ev = build_event(self._view())
        assert ev["points"][:, 3].sum() == pytest.approx(3.0)

    def test_include_le_keeps_both(self):
        ev = build_event(self._view(), include_le=True)
        assert len(ev["points"]) == 2
        assert ev["points"][:, 3].sum() == pytest.approx(7.0)

    def test_labels_stay_aligned_when_le_is_dropped(self):
        ev = build_event(self._view())
        n = len(ev["points"])
        for k in ("voxel_sem", "voxel_instance", "voxel_interaction"):
            assert len(ev[k]) == n

    def test_a_shared_voxel_loses_only_the_le_energy(self):
        # shower and absorbed LE in the same cell: the row survives, but
        # without the LE contribution
        v = FakeView([pt(0, 0, 0, 1.0, 3.0), pt(0, 0, 0, 2.0, 4.0)],
                     [(10, ST.kShower.value, 0, 0, 2, 1)])
        assert build_event(v)["points"][0, 3] == pytest.approx(3.0)
        assert build_event(v, include_le=True)["points"][0, 3] == pytest.approx(7.0)

    def test_an_all_le_event_becomes_empty(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0)],
                     [(10, ST.kShower.value, 0, 0, 1, 0)])
        ev = build_event(v)
        assert ev["points"].shape == (0, 4)
        assert len(ev["voxel_sem"]) == 0


class TestLabelShapes:

    def test_labels_align_with_points(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0), pt(1, 0, 0, 1.0, 1.0),
                      pt(1, 0, 0, 2.0, 1.0)],
                     [(10, ST.kTrack.value, 3, 0, 3)])
        ev = build_event(v)
        n = len(ev["points"])
        assert n == 2
        for k in ("voxel_sem", "voxel_instance", "voxel_interaction"):
            assert len(ev[k]) == n
        assert set(ev["voxel_interaction"].tolist()) == {3}

    def test_empty_event(self):
        v = FakeView(np.zeros((0, 6), np.float32), [])
        ev = build_event(v)
        assert ev["points"].shape == (0, 4)
        assert len(ev["voxel_sem"]) == 0

    def test_object_tables_present(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0)],
                     [(10, ST.kTrack.value, 0, 0, 1)],
                     interactions={"vertex_id": np.array([7]),
                                   "x": np.array([1.5])})
        ev = build_event(v)
        assert ev["instances"]["id"].tolist() == [10]
        assert ev["interactions"]["vertex_id"].tolist() == [7]
        # an interaction's id is its row index, which is what the voxel
        # label holds -- there is no separate id column
        assert "id" not in ev["interactions"]


class TestCollate:

    def test_batch_is_a_list_of_events(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0)], [(1, ST.kTrack.value, 0, 0, 1)])
        b = collate_events([build_event(v), build_event(v)])
        assert b["batch_size"] == 2
        assert len(b["events"]) == 2
        assert b["n_points"] == [1, 1]

    def test_per_event_timings_are_summed(self):
        a = {"points": np.zeros((1, 4), np.float32),
             "profile": {"read_s": 0.1, "preprocess_s": 0.2}}
        b = {"points": np.zeros((2, 4), np.float32),
             "profile": {"read_s": 0.3, "preprocess_s": 0.4}}
        p = collate_events([a, b])["profile"]
        assert p["read_s"] == pytest.approx(0.4)
        assert p["preprocess_s"] == pytest.approx(0.6)

    def test_profile_is_zero_when_events_carry_none(self):
        e = {"points": np.zeros((1, 4), np.float32)}
        p = collate_events([e])["profile"]
        assert p["read_s"] == 0.0 and p["preprocess_s"] == 0.0

    def test_payload_counts_the_arrays_shipped(self):
        e = {"points": np.zeros((100, 4), np.float32),          # 1600 B
             "voxel_sem": np.zeros(100, np.int16),              #  200 B
             "instances": {"id": np.zeros(10, np.int32)},       #   40 B
             "event": 0}
        mb = collate_events([e])["profile"]["payload_mb"]
        assert mb == pytest.approx(1840 / (1024 * 1024))

    def test_stage_batch_reports_a_duration(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0)], [(1, ST.kTrack.value, 0, 0, 1)])
        b = collate_events([build_event(v)])
        b, secs = stage_batch(b, device=None)
        assert secs >= 0.0
        assert isinstance(b["events"][0]["points"], torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_stage_batch_moves_to_the_gpu(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0)], [(1, ST.kTrack.value, 0, 0, 1)])
        b, _ = stage_batch(collate_events([build_event(v)]), device="cuda")
        ev = b["events"][0]
        assert ev["points"].is_cuda
        assert ev["voxel_sem"].is_cuda

    def test_to_torch_converts_arrays_only(self):
        v = FakeView([pt(0, 0, 0, 1.0, 1.0)], [(1, ST.kTrack.value, 0, 0, 1)])
        t = to_torch(build_event(v))
        assert isinstance(t["points"], torch.Tensor)
        assert t["points"].shape == (1, 4)
        assert isinstance(t["instances"], dict)
