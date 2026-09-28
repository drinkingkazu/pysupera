"""
Per-hit labels, aligned with the JAXTPC hits file.

The model's input is the JAXTPC hits file itself, so its targets are one
label per hit, in the hits file's own order: for every readout plane of every
volume, entry *k* labels CSR entry *k* of that plane.  A loader reads the
hits and the labels with the same index -- no reordering, no join.

Each entry gets the pysupera fragment and instance it belongs to, as the
representative particle ids that head them in the output's ``particles``
table, and whether its particle is a low-energy scatter.  Where a hit cannot
be traced to a particle the ids are -1.

How a hit is traced depends on the readout:

* **Wire** (``owner``): a hit belongs to its JAXTPC group, and the group to
  the particle holding its deposits -- exactly, since a group's deposits are
  never split.
* **Pixel** (``row_particle`` + ``hit_index``): the hits are what was
  partitioned, so each hit's particle is read directly -- exact per hit.  Hits
  off the sensor image were never partitioned: their ids are -1, and that is
  checked (:func:`check_pixel_labels`) rather than assumed.
"""

from __future__ import annotations

import numpy as np

from .utils import SemanticType as _ST

_LE = _ST.kLEScatter


def _lookups(particles, frag_groups, inst_groups):
    """Per particle id: its fragment rep, instance rep and LE flag (arrays)."""
    ids = [int(p.id) for p in particles]
    n = (max(ids) + 1) if ids else 0
    frag = np.full(n + 1, -1, dtype=np.int32)      # index n: "no particle"
    inst = np.full(n + 1, -1, dtype=np.int32)
    le = np.zeros(n + 1, dtype=bool)
    for g in frag_groups:
        frag[np.asarray(g.member_ids, dtype=np.int64)] = int(g.rep_id)
    for g in inst_groups:
        inst[np.asarray(g.member_ids, dtype=np.int64)] = int(g.rep_id)
    for p in particles:
        le[int(p.id)] = p.sem_type is _LE
    return frag, inst, le, n


def build_hit_labels(planes, particles, frag_groups, inst_groups, *,
                     owner=None, row_particle=None, hit_index=None):
    """
    Label every CSR entry of every plane.

    Parameters
    ----------
    planes : list of dict
        The reader's ``last_planes``: per (volume, plane) the entries'
        ``group`` and, for pixel, ``on_sensor``.
    particles, frag_groups, inst_groups
        The event's particles and the fragment / written-instance groups
        (see :func:`pysupera.layout.snapshot_groups`).
    owner : ndarray, optional
        Wire: owning particle id per event-global group
        (:func:`pysupera.provenance.build_group_owners`).
    row_particle, hit_index : ndarray, optional
        Pixel: per input hit row, its particle id and its position in the
        event's hits CSR (volume by volume in file order).

    Returns
    -------
    list of dict
        Per plane, in *planes* order: ``volume``, ``plane``, ``source``,
        ``fragment_id``, ``instance_id`` (int32, -1 = untraced), ``is_le``
        (bool) and, for pixel, ``on_sensor`` (bool).
    """
    frag, inst, le, none = _lookups(particles, frag_groups, inst_groups)
    out = []
    if row_particle is not None:
        # Pixel: scatter each partitioned hit's particle into CSR space.
        total = sum(len(pl["group"]) for pl in planes)
        part = np.full(total, none, dtype=np.int64)
        rp = np.asarray(row_particle, dtype=np.int64)
        hi = np.asarray(hit_index, dtype=np.int64)
        ok = rp >= 0
        part[hi[ok]] = rp[ok]
        start = 0
        for pl in planes:
            n = len(pl["group"])
            p = part[start:start + n]
            start += n
            out.append(_entry(pl, p, frag, inst, le))
        return out
    own = np.asarray(owner if owner is not None else [], dtype=np.int64)
    for pl in planes:
        g = pl["group"]
        p = np.full(len(g), none, dtype=np.int64)
        ok = (g >= 0) & (g < len(own))
        p[ok] = own[g[ok]]
        p[p < 0] = none
        out.append(_entry(pl, p, frag, inst, le))
    return out


def _entry(pl, part, frag, inst, le):
    d = {"volume": int(pl["volume"]), "plane": int(pl["plane"]),
         "source": str(pl["source"]),
         "fragment_id": frag[part], "instance_id": inst[part],
         "is_le": le[part]}
    if pl.get("on_sensor") is not None:
        d["on_sensor"] = np.asarray(pl["on_sensor"], dtype=bool)
    return d


def check_pixel_labels(labels):
    """
    Raise unless, on every pixel plane, exactly the hits off the sensor image
    are untraced.

    A hit off the sensor has no sensor pixel to be traced from, so it must
    carry -1; a hit on one was partitioned, so it must carry a fragment.
    Either failing means the labels and the model's input disagree about
    which pixels exist.
    """
    for d in labels:
        if "on_sensor" not in d:
            continue
        traced = d["fragment_id"] >= 0
        on = d["on_sensor"]
        stray = int((traced & ~on).sum())
        lost = int((~traced & on).sum())
        if stray or lost:
            raise RuntimeError(
                f"hit_labels volume{d['volume']}/plane{d['plane']}: {stray} "
                f"hits off the sensor image carry a label and {lost} on it "
                f"carry none.  With the sensor mask every on-sensor hit is "
                f"partitioned and no other is -- a nonzero "
                f"reader.hit_charge_threshold, or hits of tracks missing from "
                f"the EDepSim file, would break that.")
