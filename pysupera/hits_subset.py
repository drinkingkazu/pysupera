"""
Extract a viewer-sized slice of a JAXTPC hits file.

``vis_hits.html`` reads only each readout plane's per-group *centres* --
never the per-tick CSR arrays (``charges_i16``, ``delta_times``,
``delta_wires`` / ``delta_py`` / ``delta_pz``), which are the bulk of the
file.  A handful of events therefore comes to about a megabyte against the
original gigabyte.

Which centres those are depends on the readout JAXTPC simulated:

==========  =============  ==============================================
readout     plane group    centre arrays
==========  =============  ==============================================
``wire``    ``U/V/Y``      ``center_wires``, ``center_times``
``pixel``   ``Pixel``      ``center_py``, ``center_pz``, ``center_times``
==========  =============  ==============================================

Planes are found structurally -- any subgroup holding ``group_ids`` is one --
rather than by name, so a readout this module has never heard of still comes
through with whatever centres it stores.  ``config``'s attributes are copied
across as well, because ``readout_type`` is what tells the viewer which kind
of picture it is about to draw.

The output is gzip, because the browser reads HDF5 through h5wasm, which
supports gzip only.

    pysupera-hits-subset sim_wire_hits_0000.h5 hits_small.h5 --events 3
"""

from __future__ import annotations

import argparse
import os

#: Read by the viewer for every readout.
_COMMON = ("group_ids", "center_times", "peak_charges")

#: Readout-specific hit centres.  A plane keeps whichever of these it has;
#: listing both costs nothing and removes the need to identify the geometry
#: before deciding what to copy.
_CENTERS = ("center_wires", "center_py", "center_pz")

#: What the viewer reads.  Anything else is left behind.
KEEP = _COMMON + _CENTERS


def _plane_names(vol_group) -> list[str]:
    """Subgroups of one volume that are readout planes, in file order."""
    import h5py
    return [k for k in vol_group.keys()
            if isinstance(vol_group[k], h5py.Group)
            and "group_ids" in vol_group[k]]


def extract(src: str, dst: str, n_events: int = 3, level: int = 4,
            verbose: bool = True) -> dict:
    """Copy *n_events* worth of the plane arrays from *src* into *dst*."""
    import h5py
    try:
        import hdf5plugin       # noqa: F401  (registers Blosc/LZ4 for reading)
    except ImportError:
        pass

    n_ds = 0
    planes: set[str] = set()
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        # Carry the run metadata over verbatim.  Only readout_type is read by
        # the viewer, but the group is a few attributes and a subset that
        # cannot say where it came from is a subset nobody trusts.
        if "config" in fin:
            cfg = fout.create_group("config")
            cfg.attrs.update(fin["config"].attrs)

        events = sorted(k for k in fin.keys() if k.startswith("event_"))
        if n_events > 0:
            events = events[:n_events]
        for ek in events:
            for vol in sorted(k for k in fin[ek].keys()
                              if k.startswith("volume_")):
                vg = fin[f"{ek}/{vol}"]
                for plane in _plane_names(vg):
                    planes.add(plane)
                    for name in KEEP:
                        if name not in vg[plane]:
                            continue
                        fout.create_dataset(
                            f"{ek}/{vol}/{plane}/{name}",
                            data=vg[plane][name][:],
                            compression="gzip", compression_opts=level,
                        )
                        n_ds += 1
    out = {"events": len(events), "datasets": n_ds,
           "planes": sorted(planes),
           "readout": _readout_of(src),
           "size_before": os.path.getsize(src),
           "size_after": os.path.getsize(dst)}
    if verbose:
        print(f"[hits-subset] {src} -> {dst}: {len(events)} event(s), "
              f"{out['readout']} readout, plane(s) "
              f"{', '.join(out['planes']) or '(none)'}, "
              f"{n_ds} dataset(s), "
              f"{out['size_before'] / 1048576:.1f} MiB -> "
              f"{out['size_after'] / 1048576:.2f} MiB")
    return out


def _readout_of(path: str) -> str:
    """``'wire'`` or ``'pixel'`` for a hits file, without opening it twice."""
    import h5py
    from .readers.format_jaxtpc import read_readout_type
    with h5py.File(path, "r") as f:
        return read_readout_type(f)


def main() -> None:
    """Entry point for ``pysupera-hits-subset``."""
    p = argparse.ArgumentParser(
        prog="pysupera-hits-subset",
        description="Extract a gzip, viewer-sized slice of a JAXTPC hits file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("src", help="JAXTPC hits (inst) HDF5 file, wire or pixel")
    p.add_argument("dst", help="Output file, gzip")
    p.add_argument("-n", "--events", type=int, default=3,
                   help="Number of events to copy; -1 for all")
    p.add_argument("-l", "--level", type=int, default=4, help="gzip level (1-9)")
    a = p.parse_args()
    if not 1 <= a.level <= 9:
        p.error("--level must be between 1 and 9")
    extract(a.src, a.dst, a.events, a.level)
