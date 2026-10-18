"""
Several inputs read as one: one output file from many EDepSim files.

``io.input_path`` given as a list (with ``reader.jaxtpc_*_path`` either lists
of the same length or one path shared by all, e.g. a merged JAXTPC file)
builds one reader per input and chains them: event *i* of the chain is event
*i - start(k)* of input *k*.  Everything else a reader exposes -- the
``last_*`` attributes of the event just read, ``point_source``,
``pixel_geometry`` ... -- is that of the input the last event came from.

JAXTPC inputs share one pixel geometry: it is a property of the detector, so
the first input that settles it hands it on, and each volume is checked once
for the whole run rather than once per input.
"""

from __future__ import annotations

import bisect

from .base import EventReaderBase


class ChainReader(EventReaderBase):
    """Events of *readers*, in order, as one reader."""

    def __init__(self, readers, inputs):
        if not readers:
            raise ValueError("ChainReader needs at least one reader")
        self._readers = list(readers)
        #: one dict per input: {'edepsim', 'jaxtpc_step', 'jaxtpc_hits',
        #: 'jaxtpc_sensor'} -> path ("" when not used)
        self.inputs = list(inputs)
        self._starts = [0]
        for r in self._readers:
            self._starts.append(self._starts[-1] + len(r))
        self._k = 0
        #: Input index and event index within it of the event just read.
        self.last_input = None
        self.last_input_event = None

    def __len__(self) -> int:
        return self._starts[-1]

    def locate(self, index):
        """``(input index, event index within it)`` of chain event *index*."""
        n = len(self)
        if index < -n or index >= n:
            raise IndexError(f"Event index {index} out of range for {n} events "
                             f"in {len(self._readers)} input(s).")
        if index < 0:
            index += n
        k = bisect.bisect_right(self._starts, index) - 1
        return k, index - self._starts[k]

    def __getitem__(self, index):
        k, i = self.locate(index)
        if k != self._k:
            share = getattr(self._readers[k], "share_pixel_geometry", None)
            if share is not None:
                share(self._readers[self._k])
            self._k = k
        self.last_input, self.last_input_event = k, i
        return self._readers[k][i]

    def __getattr__(self, name):
        # Only called for what the chain itself does not have: the current
        # input's attributes (last_*, point_source, readout_type, ...).
        if name.startswith("__") or name in ("_readers", "_k"):
            raise AttributeError(name)
        return getattr(self._readers[self._k], name)

    def close(self) -> None:
        for r in self._readers:
            r.close()

    def __repr__(self) -> str:
        return f"ChainReader({len(self._readers)} inputs, n_events={len(self)})"
