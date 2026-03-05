"""Tests for utility functions: trace_ancestry and SetSemanticType."""
import numpy as np
import pytest

from pysupera.utils import trace_ancestry, SetSemanticType, SemanticType
from tests.conftest import make_particle, PT_PRIMARY, PT_TRACK, PT_DECAY, PT_INVALID


# ============================================================================
# trace_ancestry
# ============================================================================

class TestTraceAncestry:
    # ── basic cases ──────────────────────────────────────────────────────────

    def test_root_particle_chain_length_one(self):
        root = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1)
        chain, children = trace_ancestry(1, [root], print_result=False)
        assert len(chain) == 1
        assert chain[0].id == 1

    def test_root_has_no_children(self):
        root = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1)
        chain, children = trace_ancestry(1, [root], print_result=False)
        assert children == []

    def test_unknown_id_raises_key_error(self):
        p = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1)
        with pytest.raises(KeyError):
            trace_ancestry(99, [p], print_result=False)

    # ── multi-level chain ────────────────────────────────────────────────────

    def test_three_level_chain_root_first(self):
        # root (1) → middle (2) → leaf (3)
        root   = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1)
        middle = make_particle(2, PT_DECAY,   pdg=13, parent_id=1, ancestor_id=1)
        leaf   = make_particle(3, PT_DECAY,   pdg=11, parent_id=2, ancestor_id=1)
        particles = [root, middle, leaf]
        chain, children = trace_ancestry(3, particles, print_result=False)
        assert [p.id for p in chain] == [1, 2, 3]

    def test_children_of_middle_node_listed(self):
        root   = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1)
        middle = make_particle(2, PT_DECAY,   pdg=13, parent_id=1, ancestor_id=1)
        leaf   = make_particle(3, PT_DECAY,   pdg=11, parent_id=2, ancestor_id=1)
        chain, children = trace_ancestry(2, [root, middle, leaf], print_result=False)
        assert len(children) == 1
        assert children[0].id == 3

    def test_leaf_has_no_children(self):
        root = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1)
        leaf = make_particle(2, PT_DECAY,   pdg=11, parent_id=1, ancestor_id=1)
        _, children = trace_ancestry(2, [root, leaf], print_result=False)
        assert children == []

    def test_multiple_children_all_listed(self):
        root  = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1)
        ch1   = make_particle(2, PT_DECAY,   pdg=11, parent_id=1, ancestor_id=1)
        ch2   = make_particle(3, PT_DECAY,   pdg=-11, parent_id=1, ancestor_id=1)
        ch3   = make_particle(4, PT_DECAY,   pdg=13, parent_id=1, ancestor_id=1)
        _, children = trace_ancestry(1, [root, ch1, ch2, ch3], print_result=False)
        child_ids = {c.id for c in children}
        assert child_ids == {2, 3, 4}

    # ── return types ─────────────────────────────────────────────────────────

    def test_returns_tuple_of_two_lists(self):
        p = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1)
        result = trace_ancestry(1, [p], print_result=False)
        assert isinstance(result, tuple)
        chain, children = result
        assert isinstance(chain, list)
        assert isinstance(children, list)

    def test_print_result_false_no_output(self, capsys):
        p = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1)
        trace_ancestry(1, [p], print_result=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_result_true_produces_output(self, capsys):
        p = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1)
        trace_ancestry(1, [p], print_result=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    # ── orphan handling ──────────────────────────────────────────────────────

    def test_orphan_chain_terminates_at_orphan(self):
        # Particle's parent is missing from the list → chain stops at orphan
        orphan = make_particle(2, PT_DECAY, pdg=11, parent_id=99, ancestor_id=99)
        chain, _ = trace_ancestry(2, [orphan], print_result=False)
        assert chain[-1].id == 2

    # ── chain direction ──────────────────────────────────────────────────────

    def test_chain_ordered_root_to_target(self):
        root   = make_particle(1, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1)
        child  = make_particle(2, PT_DECAY,   pdg=11, parent_id=1, ancestor_id=1)
        chain, _ = trace_ancestry(2, [root, child], print_result=False)
        assert chain[0].id == 1
        assert chain[-1].id == 2


# ============================================================================
# SetSemanticType — edge / integration cases not covered in test_semantic_type
# ============================================================================

class TestSetSemanticTypeEdgeCases:
    def test_with_zero_size_point_cloud(self):
        # Empty cloud with default point_cloud_size=-1: 0 < -1 is False → kDelta
        st = SetSemanticType(process_type=6, pdg=11, parent_pdg=0,
                             point_cloud=np.zeros((0, 3)))
        assert st == SemanticType.kDelta

    def test_with_none_point_cloud_raises_or_returns_unknown(self):
        # When point_cloud=None, should either use point_cloud_size or raise
        # At minimum it should not silently return a wrong type
        try:
            st = SetSemanticType(process_type=0, pdg=13, parent_pdg=0,
                                  point_cloud=None)
            # kTrack is always kTrack regardless of cloud size
            assert st == SemanticType.kTrack
        except (TypeError, AttributeError):
            pass  # acceptable — None cloud not valid

    def test_large_cloud_explicit_size_override(self):
        # Explicit point_cloud_size overrides actual array length
        # kDelta with explicit large size → kDelta not kLEScatter
        large_pc = np.ones((100, 3), dtype=np.float32)
        st = SetSemanticType(process_type=6, pdg=11, parent_pdg=0,
                             point_cloud=large_pc, point_cloud_size=100)
        assert st == SemanticType.kDelta

    def test_small_cloud_explicit_size_override(self):
        # kDelta: cloud size (2) < point_cloud_size (100) → kLEScatter
        small_pc = np.ones((2, 3), dtype=np.float32)
        st = SetSemanticType(process_type=6, pdg=11, parent_pdg=0,
                             point_cloud=small_pc, point_cloud_size=100)
        assert st == SemanticType.kLEScatter
