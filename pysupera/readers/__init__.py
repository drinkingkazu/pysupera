"""
pysupera.readers
================

File-format readers that produce lists of :class:`~pysupera.data.Particle`
objects from simulation output files.

Available readers
-----------------
:class:`EDepSimHDF5Reader`
    Reads EDepSim HDF5 output (``particle/geant4``, ``pstep/lar_vol``,
    ``ass/particle_pstep_lar_vol`` datasets).

All readers implement the :class:`EventReaderBase` interface which supports:
- ``len(reader)``  — number of events
- ``reader[i]``    — event by index (negative indices supported)
- ``for particles in reader``  — iteration over all events
- ``with EDepSimHDF5Reader(path) as reader``  — context-manager usage

Example
-------
::

    from pysupera.readers import EDepSimHDF5Reader

    with EDepSimHDF5Reader("sim.h5") as reader:
        for event_particles in reader:
            # event_particles is a list[Particle]
            ...
"""

from .base import EventReaderBase
from .format_edepsim_h5 import EDepSimHDF5Reader

__all__ = [
    "EventReaderBase",
    "EDepSimHDF5Reader",
]
