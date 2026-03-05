"""Tests for UnionFind (path compression + union by rank)."""
import pytest
from pysupera.proxck import UnionFind


class TestUnionFindBasic:
    def test_fresh_node_is_own_root(self):
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_connects_two_nodes(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_union_returns_true_on_new_merge(self):
        uf = UnionFind(4)
        assert uf.union(0, 1) is True

    def test_union_returns_false_on_already_connected(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        assert uf.union(0, 1) is False
        assert uf.union(1, 0) is False

    def test_union_same_element_is_false(self):
        uf = UnionFind(3)
        assert uf.union(2, 2) is False


class TestUnionFindTransitivity:
    def test_transitive_connection(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_three_way_chain_all_same_root(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        roots = {uf.find(i) for i in range(4)}
        assert len(roots) == 1

    def test_two_separate_components(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)

    def test_merge_two_components(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(1, 2)   # bridge the two components
        roots = {uf.find(i) for i in range(4)}
        assert len(roots) == 1


class TestUnionFindPathCompression:
    def test_find_is_idempotent_after_compression(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        root_first  = uf.find(0)
        root_second = uf.find(0)
        assert root_first == root_second

    def test_n_equals_one(self):
        uf = UnionFind(1)
        assert uf.find(0) == 0

    def test_full_chain_then_find(self):
        # Build a deep chain to exercise path compression
        n = 20
        uf = UnionFind(n)
        for i in range(n - 1):
            uf.union(i, i + 1)
        root = uf.find(0)
        for i in range(n):
            assert uf.find(i) == root
