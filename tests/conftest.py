"""
Shared fixtures and helpers for the pysupera test suite.

process_type mapping (0-based raw int → InteractionType):
  0  kTrack          → sem kTrack
  1  kNeutron        → sem kLEScatter
  2  kNucleus        → sem kTrack (large) / kLEScatter (small)
  3  kPhoton         → sem kShower (pdg=11/22, large) / kLEScatter (small)
  4  kPrimary        → sem kShower (pdg=11/22) / kTrack (other)
  5  kCompton        → sem kShower (pdg=11/22, large) / kLEScatter (small)
  6  kDelta          → sem kDelta (large) / kLEScatter (small)
  7  kConversion     → sem kShower (pdg=11/22, large) / kLEScatter (small)
  8  kIonization     → sem kLEScatter
  9  kPhotoElectron  → sem kLEScatter
  10 kDecay          → sem kMichel / kShower / kTrack
  11 kOtherShower    → sem kShower (pdg=11/22, large) / kLEScatter (small)
  12 kInvalidProcess → sem kUnknown
"""
import numpy as np
import pytest
from pysupera.data import Particle

# ── Symbolic aliases for process_type raw ints ──────────────────────────────
PT_TRACK        = 0
PT_NEUTRON      = 1
PT_NUCLEUS      = 2
PT_PHOTON       = 3
PT_PRIMARY      = 4
PT_COMPTON      = 5
PT_DELTA        = 6
PT_CONVERSION   = 7
PT_IONIZATION   = 8
PT_PHOTO_ELEC   = 9
PT_DECAY        = 10
PT_OTHER_SHOWER = 11
PT_INVALID      = 12


def make_particle(
    pid: int,
    process_type: int,
    pdg: int = 11,
    parent_pdg: int = 0,
    parent_id: int | None = None,
    ancestor_id: int | None = None,
    pc: np.ndarray | None = None,
    n_pts: int = 10,
    offset: tuple = (0.0, 0.0, 0.0),
) -> Particle:
    if parent_id is None:
        parent_id = pid
    if ancestor_id is None:
        ancestor_id = pid
    if pc is None:
        rng = np.random.default_rng(pid)
        xyz = rng.uniform(0, 0.1, size=(n_pts, 3)).astype(np.float32)
        xyz += np.array(offset, dtype=np.float32)
        pc = xyz
    return Particle(
        id           = pid,
        parent_id    = parent_id,
        ancestor_id  = ancestor_id,
        pdg          = pdg,
        parent_pdg   = parent_pdg,
        process_type = process_type,
        point_cloud  = pc,
    )


def cloud(pts, dtype=np.float32):
    """Create a point-cloud array from a list of (x, y, z) tuples."""
    return np.array(pts, dtype=dtype)


def make_partitioner(particles, D=5.0, backend='cpu-single'):
    """Construct a pysupera ParticlePartitioner for test use."""
    from pysupera.partitioner import ParticlePartitioner
    return ParticlePartitioner(
        particles          = particles,
        distance_threshold = D,
        backend            = backend,
        enable_diagnostics = False,
    )
