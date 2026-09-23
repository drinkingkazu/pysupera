"""
pysupera.readers
================

File-format readers that translate external simulation file formats into lists
of :class:`~pysupera.data.Particle` objects.

Available readers
-----------------
:class:`EDepSimHDF5Reader`
    Reads EDepSim HDF5 output directly (``particle/geant4``,
    ``pstep/lar_vol``, ``ass/particle_pstep_lar_vol`` datasets).
    Point clouds contain **all** energy-deposit steps in the file.
    Hydra config group: ``reader: edepsim_h5``.

:class:`JaxtpcHDF5Reader`
    Combines three files produced by a JAXTPC simulation batch:

    * **EDepSim HDF5** — particle-level metadata (PDG, track IDs, start
      vertices, interaction type).
    * **JAXTPC seg HDF5** — per-volume 3-D truth deposits (positions in mm,
      dE, dx, t0, charge, …).
    * **JAXTPC inst HDF5** — per-readout-plane correspondence data used to
      determine which segments were visible above the signal threshold.
      Both JAXTPC readout geometries are read by the same code path:
      **wire** (``U``/``V``/``Y`` planes, three 2-D projections) and
      **pixel** (a single ``Pixel`` plane, natively 3-D).  Use
      :func:`read_readout_type` to tell them apart.

    Each particle's point cloud contains **only** the energy-deposit segments
    that survived the JAXTPC readout threshold ("visible" segments).
    Hydra config group: ``reader: edepsim_h5`` with optional
    ``reader.jaxtpc_seg_path`` / ``reader.jaxtpc_inst_path`` flags.

    ``point_source`` chooses what a point is.  ``'deposits'`` (default) is
    the above: truth geometry, with the readout deciding only which deposits
    survive.  ``'hits'`` replaces the cloud with the *detected image* --
    every fired pixel, converted from ``(py, pz, tick)`` back to
    ``(x, y, z)`` -- so the partitioner sees pixelation, diffusion,
    threshold and the drift-time ambiguity.  Pixel readout only; see
    :mod:`pysupera.readers.pixel_hits`.

All readers implement the :class:`EventReaderBase` interface which supports:

- ``len(reader)``                     — number of events
- ``reader[i]``                       — event by index (negative indices ok)
- ``for particles in reader``         — iteration over all events
- ``with SomeReader(...) as reader``  — context-manager usage

Examples
--------
EDepSim HDF5::

    from pysupera.readers import EDepSimHDF5Reader

    with EDepSimHDF5Reader("sim.h5") as reader:
        for event_particles in reader:
            ...

JAXTPC (visibility-filtered)::

    from pysupera.readers import JaxtpcHDF5Reader

    with JaxtpcHDF5Reader(
        edepsim_path="edepsim.h5",
        seg_path="sim_seg_0000.h5",
        inst_path="sim_inst_0000.h5",
    ) as reader:
        for event_particles in reader:
            ...

    # Or from a JAXTPC production directory:
    with JaxtpcHDF5Reader.from_production_dir(
        "/data/dataset_100_wire",
        edepsim_path="edepsim.h5",
    ) as reader:
        ...
"""

from .base import EventReaderBase
from .format_edepsim_h5 import EDepSimHDF5Reader
from .format_jaxtpc import (
    JaxtpcHDF5Reader,
    READOUT_PLANE_NAMES,
    read_readout_type,
)
from .pixel_hits import (
    PixelGeometryError,
    VolumePixelGeometry,
)

__all__ = [
    "EventReaderBase",
    "EDepSimHDF5Reader",
    "JaxtpcHDF5Reader",
    "READOUT_PLANE_NAMES",
    "read_readout_type",
    "PixelGeometryError",
    "VolumePixelGeometry",
]
