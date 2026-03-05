"""
Partitioning condition strategies for ``ParticlePartitioner``.

Each public class in this package is a concrete ``PartitionConditionBase``
subclass that can be passed directly to
``ParticlePartitioner.partition`` or ``ParticlePartitioner.partition_combined``.

Available conditions
--------------------
TouchingEMShower
    Merge PDG-22/11 parent-child pairs that share a common ancestor and
    whose point clouds are within the distance threshold.
AbsorbLEScatter
    Absorb each kLEScatter particle into at most one touching non-kLEScatter
    neighbour.
PhotonDecay
    Merge PDG ±11 children of a photon (PDG 22) into the photon's
    partition without a proximity check.  The photon becomes the
    surviving representative.

Examples
--------
>>> from pysupera.conditions import TouchingEMShower, CombineLEScatters, AbsorbLEScatter, PhotonDecay
>>> result = partitioner.partition(TouchingEMShower())
>>> result = partitioner.partition_combined([PhotonDecay(), TouchingEMShower(), CombineLEScatters(), AbsorbLEScatter()])
"""

from .base import PartitionConditionBase
from .touching_em_shower import TouchingEMShower
from .absorb_le_scatter import AbsorbLEScatter
from .combine_le_scatters import CombineLEScatters
from .photon_decay import PhotonDecay

__all__ = [
    "PartitionConditionBase",
    "TouchingEMShower",
    "AbsorbLEScatter",
    "CombineLEScatters",
    "PhotonDecay",
]

