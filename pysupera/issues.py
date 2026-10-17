"""
pysupera.issues
===============

Problems a run notices but does not stop for, recorded in the output file.

A warning printed among thousands of lines of a production log is not seen.
So every check that lets a run go on records what it found here, under a
stable integer code, and :func:`write_issues` stores the record in the
output as the ``issues/`` group:

``issues/event``           int64   event index, -1 for the whole file
``issues/code``            int32   an :class:`Issue` value
``issues/geant4_trackid``  int64   the Geant4 track concerned, -1 for none
``issues/message``         str     what was found, in words
``issues/codes``           the code table: ``code``, ``name``, ``description``

The group is always written, so an empty one says the checks ran and found
nothing.  The root attribute ``n_issues`` holds the total, and
``issues.attrs["n_<name>"]`` the count per code.  Production tools (dprod)
read the group to flag a job that finished with issues.

Codes are never renumbered or reused: a summary built across many files
depends on them.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Issue(IntEnum):
    """What a recorded issue is.  New codes are appended, never renumbered."""

    #: The primary table and ``parent_track_id == -1`` disagree about which
    #: particles are primaries.  The table is used.
    PRIMARY_TABLE_MISMATCH = 1
    #: The input has no primary table, so the particles without a parent
    #: were taken for the primaries.
    NO_PRIMARY_TABLE = 2
    #: Every particle of the event has an empty point cloud, and
    #: ``allow_empty_image`` let the run go on.
    EMPTY_IMAGE = 3
    #: None of the event's deposits arrived inside the readout window; the
    #: event was written empty.
    OUTSIDE_READOUT_WINDOW = 4


#: One line per code, stored in the file next to the codes.
DESCRIPTION = {
    Issue.PRIMARY_TABLE_MISMATCH:
        "primary table and parent_track_id == -1 disagree; the table is used",
    Issue.NO_PRIMARY_TABLE:
        "no primary table in the input; particles without a parent are "
        "taken for the primaries",
    Issue.EMPTY_IMAGE:
        "every particle has an empty point cloud (allow_empty_image)",
    Issue.OUTSIDE_READOUT_WINDOW:
        "no deposit arrived inside the readout window; event written empty",
}


class IssueLog:
    """The issues of one run, in the order they were found."""

    def __init__(self):
        self.event: list[int] = []
        self.code: list[int] = []
        self.geant4_trackid: list[int] = []
        self.message: list[str] = []

    def __len__(self):
        return len(self.code)

    def record(self, code: Issue, message: str, event: int = -1,
               geant4_trackid: int = -1) -> None:
        """Record one issue.  *event* -1 means the whole file."""
        self.event.append(int(event))
        self.code.append(int(Issue(code)))
        self.geant4_trackid.append(int(geant4_trackid))
        self.message.append(str(message))

    def counts(self) -> dict:
        """``{Issue: n}`` for every code recorded at least once."""
        out: dict = {}
        for c in self.code:
            out[Issue(c)] = out.get(Issue(c), 0) + 1
        return out

    def report(self, max_lines: int = 5) -> str:
        """The end-of-run summary: a count per code and the first messages."""
        if not len(self):
            return "[run] Issues: none"
        lines = [f"[run] Issues: {len(self)} recorded in the output's "
                 f"issues/ group"]
        for code, n in sorted(self.counts().items()):
            lines.append(f"  {int(code):>3} {code.name:<24} {n:>8,}   "
                         f"{DESCRIPTION[code]}")
            shown = [k for k, c in enumerate(self.code) if c == code][:max_lines]
            for k in shown:
                ev = "file" if self.event[k] < 0 else f"event {self.event[k]}"
                lines.append(f"        {ev}: {self.message[k]}")
            if n > len(shown):
                lines.append(f"        ... {n - len(shown)} more")
        return "\n".join(lines)


def write_issues(h5file, log: IssueLog) -> None:
    """Store *log* as the ``issues/`` group of the open file *h5file*."""
    import h5py
    g = h5file.require_group("issues")
    for name in list(g.keys()):
        del g[name]
    s = h5py.string_dtype()
    g.create_dataset("event", data=np.asarray(log.event, dtype=np.int64))
    g.create_dataset("code", data=np.asarray(log.code, dtype=np.int32))
    g.create_dataset("geant4_trackid",
                     data=np.asarray(log.geant4_trackid, dtype=np.int64))
    g.create_dataset("message", data=np.array(log.message, dtype=object),
                     dtype=s)
    t = g.create_group("codes")
    t.create_dataset("code", data=np.array([int(c) for c in Issue],
                                           dtype=np.int32))
    t.create_dataset("name", data=np.array([c.name for c in Issue],
                                           dtype=object), dtype=s)
    t.create_dataset("description", data=np.array(
        [DESCRIPTION[c] for c in Issue], dtype=object), dtype=s)
    counts = log.counts()
    for c in Issue:
        g.attrs[f"n_{c.name.lower()}"] = int(counts.get(c, 0))
    h5file.attrs["n_issues"] = int(len(log))


def read_issues(h5file) -> dict | None:
    """
    The ``issues/`` group of an open file as plain arrays, or None when the
    file predates it.  Keys: ``event``, ``code``, ``geant4_trackid``,
    ``message`` (list of str), ``names`` (``{code: name}``).
    """
    if "issues" not in h5file:
        return None
    g = h5file["issues"]
    dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
    return dict(
        event=g["event"][()], code=g["code"][()],
        geant4_trackid=g["geant4_trackid"][()], message=dec(g["message"][()]),
        names=dict(zip((int(c) for c in g["codes/code"][()]),
                       dec(g["codes/name"][()]))),
    )
