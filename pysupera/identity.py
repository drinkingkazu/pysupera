"""
Where each output event came from, and who it is.

Every 4.1 output records, per event, the input it was made from::

    inputs/edepsim       (I,) str   the EDepSim input files, in run order
    inputs/jaxtpc_step   (I,) str   their JAXTPC step / hits / sensor files
    inputs/jaxtpc_hits   (I,) str   ("" when not used); one row per EDepSim
    inputs/jaxtpc_sensor (I,) str   input, so a merged JAXTPC file repeats
    events/input         (n_events,) int32  row of inputs/ the event came from
    events/input_event   (n_events,) int64  its event index in that EDepSim file
    events/jaxtpc_entry  (n_events,) int64  its entry in the JAXTPC files
                                            (entry_NNNNNN), -1 without JAXTPC

and, when every EDepSim input carries a dprod production's event table
(``/dprod/events``), the event's identity, copied from it unchanged::

    events/event_id      (n_events,) int64  unique within the production campaign
    events/job_id        (n_events,) int64  the production job that generated it
    events/job_event     (n_events,) int64  its number within that job

The identity is what pairs an event with the same event in any other file of
the production (EDepSim, JAXTPC sensor/step/hits, other pysupera outputs).
"""

from __future__ import annotations

import os

import numpy as np

#: The per-event identity columns of /dprod/events (and of events/ here).
ID_COLUMNS = ("event_id", "job_id", "job_event")
INPUT_ROLES = ("edepsim", "jaxtpc_step", "jaxtpc_hits", "jaxtpc_sensor")


def read_dprod_events(f, n_events=None):
    """
    The ``/dprod/events`` identity table of an open EDepSim file.

    Returns ``{column: array}`` with one row per event in file order, or None
    if the file has none.  Raises ValueError if its length is not *n_events*.
    """
    g = f.get("dprod/events")
    if g is None:
        return None
    ids = {c: np.asarray(g[c][:], dtype=np.int64) for c in ID_COLUMNS}
    if n_events is not None:
        for c, a in ids.items():
            if len(a) != n_events:
                raise ValueError(
                    f"{f.filename}: /dprod/events/{c} has {len(a)} rows but "
                    f"the file has {n_events} events")
    return ids


class EventSources:
    """Collects, per written event, where it came from; see the module doc."""

    def __init__(self, inputs):
        #: one dict per input: {role: path or ""}
        self.inputs = [{r: str(d.get(r) or "") for r in INPUT_ROLES} for d in inputs]
        self._ids = []
        for d in self.inputs:
            self._ids.append(_identity_of(d["edepsim"]))
        have = [i is not None for i in self._ids]
        if any(have) and not all(have):
            missing = [d["edepsim"] for d, h in zip(self.inputs, have) if not h]
            raise ValueError(
                "some inputs carry a production event table (/dprod/events) "
                "and some do not, so the output could not name all its events: "
                + ", ".join(missing))
        self.has_ids = all(have) and bool(have)
        self._rows = []

    def append(self, input_index, input_event, jaxtpc_entry=-1):
        self._rows.append((int(input_index), int(input_event),
                           -1 if jaxtpc_entry is None else int(jaxtpc_entry)))

    def __len__(self):
        return len(self._rows)

    def identity(self, i):
        """{event_id, job_id, job_event} of written event *i*, or None."""
        if not self.has_ids:
            return None
        k, e, _ = self._rows[i]
        return {c: int(self._ids[k][c][e]) for c in ID_COLUMNS}

    def write(self, f, n_events):
        """Write inputs/ and the per-event columns to the open output *f*."""
        import h5py
        if len(self._rows) != n_events:
            raise RuntimeError(f"recorded the source of {len(self._rows)} events "
                               f"but the file has {n_events}")
        g = f.require_group("inputs")
        for r in INPUT_ROLES:
            g.create_dataset(r, data=np.array([d[r] for d in self.inputs],
                                              dtype=h5py.string_dtype()))
        rows = np.array(self._rows, dtype=np.int64).reshape(-1, 3)
        ev = f["events"]
        ev.create_dataset("input", data=rows[:, 0].astype(np.int32))
        ev.create_dataset("input_event", data=rows[:, 1])
        ev.create_dataset("jaxtpc_entry", data=rows[:, 2])
        if self.has_ids:
            for c in ID_COLUMNS:
                ev.create_dataset(c, data=np.array(
                    [self._ids[k][c][e] for k, e, _ in self._rows], dtype=np.int64))


def _identity_of(path):
    import h5py
    if not path or not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        return read_dprod_events(f)
