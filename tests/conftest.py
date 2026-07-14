"""
Shared fixtures and helpers for the pysupera test suite.

interaction_type mapping (InteractionType enum → SemanticType):
  kTrack          → sem kTrack
  kNeutron        → sem kLEScatter
  kNucleus        → sem kTrack (large) / kLEScatter (small)
  kPhoton         → sem kShower (pdg=11/22, large) / kLEScatter (small)
  kPrimary        → sem kShower (pdg=11/22) / kTrack (other)
  kCompton        → sem kShower (pdg=11/22, large) / kLEScatter (small)
  kDelta          → sem kDelta (large) / kLEScatter (small)
  kConversion     → sem kShower (pdg=11/22, large) / kLEScatter (small)
  kIonization     → sem kLEScatter
  kPhotoElectron  → sem kLEScatter
  kDecay          → sem kMichel / kShower / kTrack
  kOtherShower    → sem kShower (pdg=11/22, large) / kLEScatter (small)
  kInvalidProcess → sem kUnknown
"""
import numpy as np
import pytest
from pysupera.data import Particle
from pysupera.utils import InteractionType

# ── Symbolic aliases for interaction_type (InteractionType enum members) ───────
PT_TRACK        = InteractionType.kTrack
PT_NEUTRON      = InteractionType.kNeutron
PT_NUCLEUS      = InteractionType.kNucleus
PT_PHOTON       = InteractionType.kPhoton
PT_PRIMARY      = InteractionType.kPrimary
PT_COMPTON      = InteractionType.kCompton
PT_DELTA        = InteractionType.kDelta
PT_CONVERSION   = InteractionType.kConversion
PT_IONIZATION   = InteractionType.kIonization
PT_PHOTO_ELEC   = InteractionType.kPhotoElectron
PT_DECAY        = InteractionType.kDecay
PT_OTHER_SHOWER = InteractionType.kOtherShower
PT_INVALID      = InteractionType.kInvalidProcess


def make_particle(
    pid: int,
    interaction_type: int,
    pdg: int = 11,
    parent_pdg: int = 0,
    parent_id: int | None = None,
    root_id: int | None = None,
    pc: np.ndarray | None = None,
    n_pts: int = 10,
    offset: tuple = (0.0, 0.0, 0.0),
    interaction_id: int = -1,
) -> Particle:
    if parent_id is None:
        parent_id = pid
    if root_id is None:
        root_id = pid
    if pc is None:
        rng = np.random.default_rng(pid)
        xyz = rng.uniform(0, 0.1, size=(n_pts, 3)).astype(np.float32)
        xyz += np.array(offset, dtype=np.float32)
        pc = xyz
    return Particle(
        id               = pid,
        parent_id        = parent_id,
        root_id          = root_id,
        pdg              = pdg,
        parent_pdg       = parent_pdg,
        interaction_id   = interaction_id,
        interaction_type = interaction_type,
        point_cloud      = pc,
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
