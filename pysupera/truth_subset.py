"""
Extract the truth deposits a browser needs to show pysupera labels on them.

``vis_truth.html`` draws the Geant4 energy deposits beside the reconstructed
point cloud, painting each deposit with the label of whatever pysupera object
claimed it.  The bridge is the *group*: a deposit knows its group
(``deposit_to_group``, in the JAXTPC hits file) and the pysupera output's
``groups/`` table says which fragment owns each group.

So the viewer needs three things that live in two different JAXTPC files --
the deposit positions and energies from the *step* file, and
``deposit_to_group`` from the *hits* file -- and it cannot read either of
them.  Both are written with Blosc/LZ4, and the browser reads HDF5 through
h5wasm, which decodes gzip only.  This writes the handful of arrays that
matter into one gzip file, which also takes 22 MB + 44 MB down to under a
megabyte per event.

    pysupera-truth-subset step.h5 hits.h5 truth_small.h5 --events 3

Positions stay in JAXTPC's own uint16-plus-attributes form rather than being
expanded to mm: it is the same three-line decode on the far side and less
than half the bytes.
"""

from __future__ import annotations

import argparse
import os

#: Per-volume step-file datasets the viewer reads, and the dtype to store.
#: ``de`` is true deposited energy in MeV and ``charge`` the ionisation
#: electrons that survived recombination -- two different quantities, and the
#: viewer shows both, so a deposit can be thresholded in whichever the
#: question is asked in.  float16 on disk is converted because h5wasm has no
#: reliable half-float support.
_STEP_KEEP = {"positions": None, "de": "float32", "charge": "float32"}

#: Attributes needed to turn ``positions`` back into mm.
_STEP_ATTRS = ("n_actual", "pos_step_mm",
               "pos_origin_x", "pos_origin_y", "pos_origin_z")

#: Per-volume hits-file dataset mapping a deposit to its group.  Older JAXTPC
#: output spells it ``segment_to_group``.
_D2G_KEYS = ("deposit_to_group", "segment_to_group")


def extract(step_path: str, hits_path: str, dst: str, n_events: int = 3,
            level: int = 4, verbose: bool = True) -> dict:
    """Copy *n_events* worth of truth deposits and their group ids into *dst*."""
    import h5py
    import numpy as np
    try:
        import hdf5plugin       # noqa: F401  (registers Blosc/LZ4 for reading)
    except ImportError:
        pass

    n_ds = n_dep = 0
    with h5py.File(step_path, "r") as fs, h5py.File(hits_path, "r") as fh, \
            h5py.File(dst, "w") as out:
        if "config" in fh:
            cfg = out.create_group("config")
            cfg.attrs.update(fh["config"].attrs)

        events = sorted(k for k in fs.keys() if k.startswith("event_"))
        if n_events > 0:
            events = events[:n_events]
        for ek in events:
            if ek not in fh:
                raise KeyError(
                    f"{ek} is in the step file but not the hits file "
                    f"{hits_path!r}; the two must come from the same run.")
            for vol in sorted(k for k in fs[ek].keys()
                              if k.startswith("volume_")):
                sv, hv = fs[f"{ek}/{vol}"], fh.get(f"{ek}/{vol}")
                n = int(sv.attrs.get("n_actual", 0) or 0)
                g = out.create_group(f"{ek}/{vol}")
                for a in _STEP_ATTRS:
                    if a in sv.attrs:
                        g.attrs[a] = sv.attrs[a]
                if not n:
                    continue
                n_dep += n
                for name, dt in _STEP_KEEP.items():
                    if name not in sv:
                        continue
                    # Only the first n_actual rows are real; the rest is the
                    # pad JAXTPC writes to a fixed shape.
                    d = sv[name][:n]
                    g.create_dataset(name, data=d if dt is None else d.astype(dt),
                                     compression="gzip", compression_opts=level)
                    n_ds += 1
                key = next((k for k in _D2G_KEYS if hv is not None and k in hv), None)
                if key is None:
                    raise KeyError(
                        f"{ek}/{vol} in the hits file has none of "
                        f"{list(_D2G_KEYS)}, so its deposits cannot be tied to "
                        f"the groups the pysupera labels hang off.")
                g.create_dataset("deposit_to_group",
                                 data=hv[key][:n].astype(np.int32),
                                 compression="gzip", compression_opts=level)
                n_ds += 1

    out_info = {"events": len(events), "datasets": n_ds, "deposits": n_dep,
                "size_step": os.path.getsize(step_path),
                "size_hits": os.path.getsize(hits_path),
                "size_after": os.path.getsize(dst)}
    if verbose:
        print(f"[truth-subset] {step_path} + {hits_path} -> {dst}: "
              f"{len(events)} event(s), {n_dep:,} deposit(s), "
              f"{n_ds} dataset(s), "
              f"{(out_info['size_step'] + out_info['size_hits']) / 1048576:.1f} MiB "
              f"-> {out_info['size_after'] / 1048576:.2f} MiB")
    return out_info


def main() -> None:
    """Entry point for ``pysupera-truth-subset``."""
    p = argparse.ArgumentParser(
        prog="pysupera-truth-subset",
        description="Extract a gzip, viewer-sized slice of the JAXTPC truth "
                    "deposits and their group ids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("step", help="JAXTPC step/seg HDF5 file (truth deposits)")
    p.add_argument("hits", help="JAXTPC hits/inst HDF5 file (deposit_to_group)")
    p.add_argument("dst", help="Output file, gzip")
    p.add_argument("-n", "--events", type=int, default=3,
                   help="Number of events to copy; -1 for all")
    p.add_argument("-l", "--level", type=int, default=4, help="gzip level (1-9)")
    a = p.parse_args()
    if not 1 <= a.level <= 9:
        p.error("--level must be between 1 and 9")
    extract(a.step, a.hits, a.dst, a.events, a.level)
