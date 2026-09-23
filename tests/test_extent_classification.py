"""
Classifying by extent, not by row count.

``min_pc_size`` decides whether a particle is a low-energy scatter, and it
used to count rows.  Row count is a property of the *input sampling*, not of
the particle: Geant4 deposits arrive every 0.3 mm and pixel sensor hits on a
4.32 mm grid, so the same object is 1 row in one and 40 in the other, and a
threshold in rows quietly means something different for each.

Counting occupied voxels measures how much space the object fills, which is
what the question is actually about, and is the same count defragmentation
already splits on.
"""

import numpy as np
import pytest

from pysupera.data import Particle
from pysupera.utils import (InteractionType, SemanticType, SetSemanticType,
                            count_extent)


def cube(n_per_side, pitch, jitter=0, seed=0):
    """A filled cube of points on a regular grid, optionally over-sampled."""
    g = np.arange(n_per_side) * pitch
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    if jitter:
        # the same solid, sampled more finely: more rows, identical extent
        rng = np.random.default_rng(seed)
        P = np.repeat(P, jitter, axis=0) + rng.uniform(0, pitch * 0.4,
                                                       (len(P) * jitter, 3))
    return P.astype(np.float32)


# ---------------------------------------------------------------------------
# count_extent
# ---------------------------------------------------------------------------

def test_extent_counts_cells_not_rows():
    P = cube(4, 3.0)                       # 64 cells on a 3 mm grid
    assert count_extent(P, 3.0) == 64
    assert count_extent(P) == len(P)       # no grid: rows, as before


def test_extent_is_unchanged_by_resampling_the_same_solid():
    """
    The whole point.  Ten times the rows over the same volume must give the
    same answer, or the threshold is measuring the sampling.
    """
    coarse = cube(4, 3.0)
    fine = cube(4, 3.0, jitter=10)
    assert len(fine) == 10 * len(coarse)
    assert count_extent(coarse, 3.0) == count_extent(fine, 3.0) == 64


def test_extent_handles_negative_coordinates():
    """A detector straddling the origin must not fold cells together."""
    P = np.array([[-4.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [4.0, 0, 0]],
                 dtype=np.float32)
    assert count_extent(P, 3.0) == 4


def test_extent_of_an_empty_cloud():
    assert count_extent(np.zeros((0, 3), dtype=np.float32), 3.0) == 0


# ---------------------------------------------------------------------------
# the classifier
# ---------------------------------------------------------------------------

def _sem(P, size, vox):
    # kCompton on an electron: shower when large, LE scatter when small
    return SetSemanticType(InteractionType.kCompton, 11, 22, P,
                           point_cloud_size=size, voxel_size=vox)


def test_same_object_sampled_two_ways_classifies_the_same():
    """
    A 4x4x4 solid is the same object whether it arrives as 64 deposits or
    640 sensor hits.  Counting rows calls one small and the other large;
    counting cells calls both large.
    """
    coarse, fine = cube(4, 3.0), cube(4, 3.0, jitter=10)
    assert _sem(coarse, 100, 3.0) == _sem(fine, 100, 3.0) == SemanticType.kLEScatter
    assert _sem(coarse, 10, 3.0) == _sem(fine, 10, 3.0) == SemanticType.kShower
    # and the old behaviour is what it is being fixed: rows disagree
    assert _sem(coarse, 100, None) != _sem(fine, 100, None)


def test_the_threshold_is_the_cell_count_of_a_track_width_blob():
    """
    The rule the pixel default comes from: a chunk only has a direction if
    it is bigger than one track width across.  A 4x4x4 blob is 64 cells, so
    a threshold of 85 calls it low-energy and 50 does not.
    """
    blob = cube(4, 3.0)
    assert count_extent(blob, 3.0) == 64
    assert _sem(blob, 85, 3.0) == SemanticType.kLEScatter
    assert _sem(blob, 50, 3.0) == SemanticType.kShower


def test_row_shortcut_cannot_change_the_answer():
    """
    Cells are never more numerous than rows, so a cloud already under the
    threshold in rows is under it in cells -- which is what lets the
    classifier skip the unique on the common small case.  It must not change
    any answer.
    """
    rng = np.random.default_rng(3)
    for n in (1, 3, 7, 40, 200):
        P = rng.uniform(0, 12, (n, 3)).astype(np.float32)
        for thr in (2, 5, 20, 85):
            fast = _sem(P, thr, 3.0)
            slow = (SemanticType.kLEScatter
                    if count_extent(P, 3.0) < thr else SemanticType.kShower)
            assert fast == slow, (n, thr)


# ---------------------------------------------------------------------------
# reaching it through Particle
# ---------------------------------------------------------------------------

def test_particle_classifies_by_extent_when_given_a_voxel_size():
    fine = cube(4, 3.0, jitter=10)         # 640 rows, 64 cells
    rows = Particle(id=0, parent_id=0, ancestor_id=0, pdg=11, parent_pdg=22,
                    interaction_id=0,
                    interaction_type=InteractionType.kCompton,
                    point_cloud=fine, min_pc_size=100)
    cells = Particle(id=0, parent_id=0, ancestor_id=0, pdg=11, parent_pdg=22,
                     interaction_id=0,
                     interaction_type=InteractionType.kCompton,
                     point_cloud=fine, min_pc_size=100, voxel_size=3.0)
    assert rows.sem_type == SemanticType.kShower        # 640 rows > 100
    assert cells.sem_type == SemanticType.kLEScatter    # 64 cells < 100


def test_voxel_size_none_keeps_the_old_row_behaviour():
    """Callers with no voxel grid must be unaffected by any of this."""
    P = cube(3, 3.0)                       # 27 rows, 27 cells
    for thr in (10, 27, 50):
        assert _sem(P, thr, None) == _sem(P, thr, 3.0)
