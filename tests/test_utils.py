"""Tests for utility functions: trace_ancestry and SetSemanticType."""
import numpy as np
import pytest

from pysupera.utils import (trace_ancestry, SetSemanticType, SemanticType,
                            resolve_orphans, validate_ancestor_ids)
from tests.conftest import make_particle, PT_PRIMARY, PT_TRACK, PT_DECAY, PT_DELTA, PT_INVALID


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
        st = SetSemanticType(interaction_type=PT_DELTA, pdg=11, parent_pdg=0,
                             point_cloud=np.zeros((0, 3)))
        assert st == SemanticType.kDelta

    def test_with_none_point_cloud_raises_or_returns_unknown(self):
        # When point_cloud=None, should either use point_cloud_size or raise
        # At minimum it should not silently return a wrong type
        try:
            st = SetSemanticType(interaction_type=PT_TRACK, pdg=13, parent_pdg=0,
                                  point_cloud=None)
            # kTrack is always kTrack regardless of cloud size
            assert st == SemanticType.kTrack
        except (TypeError, AttributeError):
            pass  # acceptable — None cloud not valid

    def test_large_cloud_explicit_size_override(self):
        # Explicit point_cloud_size overrides actual array length
        # kDelta with explicit large size → kDelta not kLEScatter
        large_pc = np.ones((100, 3), dtype=np.float32)
        st = SetSemanticType(interaction_type=PT_DELTA, pdg=11, parent_pdg=0,
                             point_cloud=large_pc, point_cloud_size=100)
        assert st == SemanticType.kDelta

    def test_small_cloud_explicit_size_override(self):
        # kDelta: cloud size (2) < point_cloud_size (100) → kLEScatter
        small_pc = np.ones((2, 3), dtype=np.float32)
        st = SetSemanticType(interaction_type=PT_DELTA, pdg=11, parent_pdg=0,
                             point_cloud=small_pc, point_cloud_size=100)
        assert st == SemanticType.kLEScatter


# ============================================================================
# resolve_orphans
# ============================================================================

class TestResolveOrphans:
    """
    Dropping one particle must not scatter its descendants across false
    roots.  When the parent links survive, the ancestor has to follow them.
    """

    def _family(self):
        # 1 (primary) -> 2 -> 3, 4;  all naming 1 as ancestor
        return [make_particle(1, PT_PRIMARY, pdg=22, parent_id=1, ancestor_id=1),
                make_particle(2, PT_TRACK, pdg=11, parent_id=1, ancestor_id=1),
                make_particle(3, PT_TRACK, pdg=22, parent_id=2, ancestor_id=1),
                make_particle(4, PT_TRACK, pdg=22, parent_id=2, ancestor_id=1)]

    def test_intact_family_is_untouched(self):
        fam = self._family()
        resolve_orphans(fam)
        assert [p.ancestor_id for p in fam] == [1, 1, 1, 1]

    def test_dropped_primary_reroots_onto_the_surviving_chain(self):
        # the primary (1) is dropped; 2 survives and becomes the root, so
        # 3 and 4 must name 2 -- not themselves
        fam = [p for p in self._family() if p.id != 1]
        resolve_orphans(fam)
        by = {p.id: p for p in fam}
        assert by[2].parent_id == 2 and by[2].ancestor_id == 2
        assert by[3].ancestor_id == 2
        assert by[4].ancestor_id == 2

    def test_the_repair_satisfies_the_validator(self):
        fam = [p for p in self._family() if p.id != 1]
        resolve_orphans(fam)
        assert validate_ancestor_ids(fam, raise_on_missing=False) in (None, [])

    def test_self_rooting_only_when_nothing_survives(self):
        # 3's parent 2 is gone too, so 3 genuinely has no chain left
        fam = [p for p in self._family() if p.id == 3]
        resolve_orphans(fam)
        assert fam[0].parent_id == 3
        assert fam[0].ancestor_id == 3

    def test_dangling_parent_is_reset(self):
        p = make_particle(7, PT_TRACK, pdg=11, parent_id=99, ancestor_id=99)
        resolve_orphans([p])
        assert p.parent_id == 7 and p.ancestor_id == 7

    def test_a_live_ancestor_is_left_alone(self):
        fam = self._family()
        fam[2].ancestor_id = 2            # unusual but present: not repaired
        resolve_orphans(fam)
        assert fam[2].ancestor_id == 2

    def test_deep_chain_shares_one_root(self):
        n = 50
        ps = [make_particle(1, PT_PRIMARY, pdg=22, parent_id=1, ancestor_id=1)]
        ps += [make_particle(i, PT_TRACK, pdg=11, parent_id=i - 1,
                             ancestor_id=999) for i in range(2, n + 1)]
        resolve_orphans(ps)
        assert {p.ancestor_id for p in ps} == {1}

    def test_a_cycle_terminates(self):
        # 1 <-> 2 with a dangling ancestor: must not spin forever
        a = make_particle(1, PT_TRACK, pdg=11, parent_id=2, ancestor_id=99)
        b = make_particle(2, PT_TRACK, pdg=11, parent_id=1, ancestor_id=99)
        resolve_orphans([a, b])
        assert a.ancestor_id in (1, 2) and b.ancestor_id in (1, 2)

    def test_returns_the_same_list(self):
        fam = self._family()
        assert resolve_orphans(fam) is fam
