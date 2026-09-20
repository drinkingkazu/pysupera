#!/usr/bin/env python3
"""
Benchmark HDF5 read speed across two or more files holding the same data.

Intended for comparing compression filters -- e.g. an LZ4-compressed pysupera
output against the gzip copy used by the WebGL viewer -- but it works on any set
of HDF5 files with matching layout.

Stand-alone: needs only h5py, numpy, and (for LZ4/Blosc files) hdf5plugin.

Examples
--------
    h5_read_bench.py aho.h5 aho_vis.h5
    h5_read_bench.py aho.h5 aho_vis.h5 --mode slices --repeat 7
    h5_read_bench.py a.h5 b.h5 --mode both --repeat 5 --slices 200

Modes
-----
full    Read every dataset in full, once per repeat.  Measures raw
        decompression throughput.
slices  Read many random row-ranges from the largest 2-D datasets.  Closer to
        how an event viewer or training loop actually touches the file, where
        chunk-boundary and seek costs matter.
both    Run each mode in turn (default).

Cache handling
--------------
Repeats are interleaved across files (file A, file B, file A, ...) so no file
gets a systematically warmer page cache.  The FIRST repeat of each file is
reported separately as "cold" -- it includes whatever I/O the OS had not yet
cached -- and the remaining repeats are summarised by their median as "warm".
Compare warm numbers to judge decompression cost, and cold numbers to judge
first-open latency.  For a genuinely cold measurement you need to drop the page
cache between runs, which requires root:

    sync && echo 3 > /proc/sys/vm/drop_caches
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
import h5py

try:
    import hdf5plugin  # noqa: F401  registers LZ4 / Blosc filters
except ImportError:
    pass


_FILTER_NAMES = {'32004': 'lz4', '32001': 'blosc'}


def describe_filter(ds: h5py.Dataset) -> str:
    for fid, name in _FILTER_NAMES.items():
        if fid in dict(ds._filters or {}):
            return name
    return str(ds.compression) if ds.compression else 'none'


def survey(path: str) -> dict:
    """Collect dataset inventory and filter mix for *path*."""
    datasets: list[str] = []
    logical = 0
    filters: dict[str, int] = {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            nonlocal logical
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)
                logical += obj.size * obj.dtype.itemsize
                key = describe_filter(obj)
                filters[key] = filters.get(key, 0) + 1
        f.visititems(visit)
    return {'datasets': datasets, 'logical': logical, 'filters': filters,
            'size': os.path.getsize(path)}


def read_full(path: str) -> tuple[float, int]:
    """Read every dataset fully.  Returns (seconds, bytes touched)."""
    total = 0
    t0 = time.perf_counter()
    with h5py.File(path, 'r') as f:
        def visit(_name, obj):
            nonlocal total
            if isinstance(obj, h5py.Dataset):
                if obj.ndim == 0:
                    obj[()]
                    total += obj.dtype.itemsize
                elif obj.size:
                    arr = obj[...]
                    total += arr.nbytes
        f.visititems(visit)
    return time.perf_counter() - t0, total


def pick_slice_targets(path: str, min_rows: int) -> list[str]:
    """Names of 2-D datasets big enough to slice, largest first."""
    found: list[tuple[int, str]] = []
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 1 \
                    and obj.shape[0] >= min_rows:
                found.append((obj.size * obj.dtype.itemsize, name))
        f.visititems(visit)
    found.sort(reverse=True)
    return [name for _, name in found]


def read_slices(path: str, targets: list[str], n_slices: int,
                rows: int, seed: int) -> tuple[float, int]:
    """Read *n_slices* random row-ranges spread over *targets*."""
    rng = np.random.default_rng(seed)
    total = 0
    t0 = time.perf_counter()
    with h5py.File(path, 'r') as f:
        usable = [t for t in targets if t in f]
        if not usable:
            return 0.0, 0
        for i in range(n_slices):
            name = usable[i % len(usable)]
            ds = f[name]
            n = ds.shape[0]
            span = min(rows, n)
            start = int(rng.integers(0, max(1, n - span + 1)))
            arr = ds[start:start + span]
            total += arr.nbytes
    return time.perf_counter() - t0, total


def fmt_rate(nbytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "     n/a"
    return f"{nbytes / 2**20 / seconds:8.1f}"


def run_mode(mode: str, paths: list[str], repeat: int, args) -> dict:
    """Interleave *repeat* passes of *mode* across *paths*."""
    timings: dict[str, list[float]] = {p: [] for p in paths}
    volumes: dict[str, int] = {p: 0 for p in paths}

    targets = None
    if mode == 'slices':
        targets = pick_slice_targets(paths[0], args.rows)
        if not targets:
            print(f"  (no dataset with >= {args.rows} rows; skipping slices)")
            return {}

    for r in range(repeat):
        for path in paths:
            if mode == 'full':
                dt, nb = read_full(path)
            else:
                # Same seed every pass so all files read identical ranges.
                dt, nb = read_slices(path, targets, args.slices,
                                     args.rows, args.seed)
            timings[path].append(dt)
            volumes[path] = nb

    print(f"\n  {'file':<28} {'cold s':>9} {'warm s':>9} "
          f"{'warm MiB/s':>11} {'vs first':>9}")
    print(f"  {'-' * 70}")

    baseline = None
    results = {}
    for path in paths:
        ts = timings[path]
        cold = ts[0]
        warm = statistics.median(ts[1:]) if len(ts) > 1 else ts[0]
        rate = fmt_rate(volumes[path], warm)
        if baseline is None:
            baseline = warm
            rel = "   1.00x"
        else:
            rel = f"{warm / baseline:7.2f}x" if baseline > 0 else "     n/a"
        print(f"  {os.path.basename(path):<28} {cold:9.3f} {warm:9.3f} "
              f"{rate:>11} {rel:>9}")
        results[path] = {'cold': cold, 'warm': warm, 'bytes': volumes[path]}

    if repeat < 3:
        print("  (note: --repeat < 3 gives a weak median; try 5+)")
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare HDF5 read speed across files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('files', nargs='+', help="two or more HDF5 files")
    p.add_argument('--mode', default='both', choices=('full', 'slices', 'both'))
    p.add_argument('--repeat', type=int, default=5,
                   help="passes per file; first is reported as cold")
    p.add_argument('--slices', type=int, default=300,
                   help="number of random row-ranges read in slices mode")
    p.add_argument('--rows', type=int, default=4096,
                   help="rows per slice, and minimum size to be a slice target")
    p.add_argument('--seed', type=int, default=1234,
                   help="RNG seed; identical ranges are read from every file")
    args = p.parse_args()

    if len(args.files) < 2:
        sys.exit("error: give at least two files to compare")
    for path in args.files:
        if not os.path.exists(path):
            sys.exit(f"error: no such file: {path}")

    print("=" * 72)
    print("HDF5 read benchmark")
    print("=" * 72)
    infos = {}
    for path in args.files:
        info = survey(path)
        infos[path] = info
        mix = ', '.join(f"{k}x{v}" for k, v in sorted(info['filters'].items()))
        print(f"  {os.path.basename(path):<28} "
              f"{info['size'] / 2**20:8.2f} MiB on disk  "
              f"{info['logical'] / 2**20:8.2f} MiB logical  "
              f"[{mix}]")

    shapes = {tuple(sorted(i['datasets'])) for i in infos.values()}
    if len(shapes) > 1:
        print("\n  WARNING: files do not contain the same dataset names; "
              "the comparison may not be meaningful.")

    modes = ('full', 'slices') if args.mode == 'both' else (args.mode,)
    for mode in modes:
        print(f"\n{'-' * 72}\nmode: {mode}"
              + (f"  ({args.slices} slices x {args.rows} rows)"
                 if mode == 'slices' else "")
              + f"  repeat={args.repeat}")
        run_mode(mode, args.files, args.repeat, args)

    print(f"\n{'=' * 72}")
    print("Lower warm-s is faster.  Compare 'MiB/s' for decompression "
          "throughput;\nsee the module docstring for how to get a truly "
          "cold-cache measurement.")


if __name__ == '__main__':
    main()
