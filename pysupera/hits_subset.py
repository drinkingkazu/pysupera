"""
Extract a viewer-sized slice of a JAXTPC hits file.

``vis_hits.html`` reads four arrays per readout plane -- ``center_wires``,
``center_times``, ``peak_charges`` and ``group_ids`` -- and nothing else.  The
per-tick arrays (``charges_u16``, ``delta_times``, ``delta_wires``) are the
bulk of the file and are never touched, so a handful of events comes to about
a megabyte against the original gigabyte.

The output is gzip, because the browser reads HDF5 through h5wasm, which
supports gzip only.

    pysupera-hits-subset sim_wire_hits_0000.h5 hits_small.h5 --events 3
"""

from __future__ import annotations

import argparse
import os

#: What the viewer reads.  Anything else is left behind.
KEEP = ("center_wires", "center_times", "peak_charges", "group_ids")


def extract(src: str, dst: str, n_events: int = 3, level: int = 4,
            verbose: bool = True) -> dict:
    """Copy *n_events* worth of the plane arrays from *src* into *dst*."""
    import h5py
    try:
        import hdf5plugin       # noqa: F401  (registers Blosc/LZ4 for reading)
    except ImportError:
        pass

    n_ds = 0
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        events = sorted(k for k in fin.keys() if k.startswith("event_"))
        if n_events > 0:
            events = events[:n_events]
        for ek in events:
            for vol in sorted(k for k in fin[ek].keys()
                              if k.startswith("volume_")):
                vg = fin[f"{ek}/{vol}"]
                for plane in ("U", "V", "Y"):
                    if plane not in vg:
                        continue
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
           "size_before": os.path.getsize(src),
           "size_after": os.path.getsize(dst)}
    if verbose:
        print(f"[hits-subset] {src} -> {dst}: {len(events)} event(s), "
              f"{n_ds} dataset(s), "
              f"{out['size_before'] / 1048576:.1f} MiB -> "
              f"{out['size_after'] / 1048576:.2f} MiB")
    return out


def main() -> None:
    """Entry point for ``pysupera-hits-subset``."""
    p = argparse.ArgumentParser(
        prog="pysupera-hits-subset",
        description="Extract a gzip, viewer-sized slice of a JAXTPC hits file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("src", help="JAXTPC hits (inst) HDF5 file")
    p.add_argument("dst", help="Output file, gzip")
    p.add_argument("-n", "--events", type=int, default=3,
                   help="Number of events to copy; -1 for all")
    p.add_argument("-l", "--level", type=int, default=4, help="gzip level (1-9)")
    a = p.parse_args()
    if not 1 <= a.level <= 9:
        p.error("--level must be between 1 and 9")
    extract(a.src, a.dst, a.events, a.level)
