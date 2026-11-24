"""
Format 3 inputs: step, mctruth, hits and presupera files.

Format 3 splits what :class:`~.format_jaxtpc.JaxtpcHDF5Reader` reads from the
EDepSim file and the JAXTPC step/hits files:

  step       Geant4 steps and their groups (flat per event), shared by readouts
  mctruth    particles, primaries, vertices and per-step truth, same step rows
  hits       per plane the cells of each group (group 2 ids, per volume)
  presupera  per volume group_to_track and group2_to_group1

The classes here present them the way the reader expects its files, one event at
a time; the labelling code does not know the difference.  This module is the
only place that knows the format.
"""

from __future__ import annotations

import numpy as np


def is_format3(hits_file):
    cfg = hits_file.get('config')
    return cfg is not None and int(cfg.attrs.get('format_version', 0)) >= 3


def decode_cells(ds):
    """Cells (row differences) to mm: (cumsum(k) + cell_offset) * scale_factor."""
    k = np.cumsum(ds[()].astype(np.int64), axis=0)
    return (k + float(ds.attrs['cell_offset'])) * float(ds.attrs['scale_factor'])


def entry(i):
    return f'entry_{int(i):06d}'


def refine_groups(group1, volume, n_volumes):
    """Group 2: new at every change of group 1 or volume in row order, 1, 2, ... in
    each volume; 0 outside all volumes.  Also group2 -> group1 per volume."""
    start = np.ones(len(group1), dtype=bool)
    start[1:] = (group1[1:] != group1[:-1]) | (volume[1:] != volume[:-1])
    group2 = np.zeros(len(group1), dtype=np.int32)
    maps = []
    for v in range(n_volumes):
        m = volume == v
        group2[m] = np.cumsum(start[m])
        maps.append(group1[m & start])
    return group2, maps


class _PerEvent:
    """Looks like a per-event vlen dataset."""

    def __init__(self, n, read):
        self._n, self._read = n, read

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._read(int(i))


class TruthFile:
    """mctruth + step, read as an EDepSim file."""

    _STEP_DTYPE = [(c, 'f4') for c in ('x', 'y', 'z', 't', 'theta', 'phi', 'p',
                                       'de', 'dx')] + \
                  [(c, 'i4') for c in ('track_id', 'pdg', 'ancestor_track_id')]

    def __init__(self, truth_path, step_path):
        import h5py
        self._t = h5py.File(truth_path, 'r')
        self._s = h5py.File(step_path, 'r')
        self.filename = self._t.filename
        a, b = self._t['event/geant4'][()], self._s['event/geant4'][()]
        if not np.array_equal(a, b):
            raise ValueError(f"{truth_path} and {step_path} do not hold the same events")
        self._n = len(a)
        self._cache = (None, None, None)

    def close(self):
        self._t.close()
        self._s.close()

    def step_columns(self, i):
        """(steps as an EDepSim step table, group 1 per row) of event *i*."""
        if self._cache[0] == i:
            return self._cache[1], self._cache[2]
        g = self._s[entry(i)]
        out = np.zeros(int(g['t'].shape[0]), dtype=self._STEP_DTYPE)
        pos = decode_cells(g['positions']).astype(np.float32)
        for k, c in enumerate('xyz'):
            out[c] = pos[:, k]
        for c in ('t', 'theta', 'phi', 'p', 'de', 'dx', 'track_id'):
            out[c] = g[c][()]
        part = self._t[entry(i)]['particle'][()]
        out['pdg'] = part['pdg'][out['track_id']]
        out['ancestor_track_id'] = part['ancestor_track_id'][out['track_id']]
        runs = g['group_run_lengths'][()].astype(np.int64)
        group1 = np.repeat(np.arange(1, len(runs) + 1, dtype=np.int32), runs)
        self._cache = (i, out, group1)
        return out, group1

    def __getitem__(self, key):
        table = key.split('/')[0]
        if table in ('particle', 'primary', 'vertex'):
            return _PerEvent(self._n, lambda i: self._t[entry(i)][table][()])
        if table == 'pstep':
            return _PerEvent(self._n, lambda i: self.step_columns(i)[0])
        return self._t[key]

    def __contains__(self, key):
        return (key.split('/')[0] in ('particle', 'primary', 'vertex', 'pstep')
                or key in self._t)

    def get(self, key, default=None):
        return self[key] if key in self else default


class _Entry:
    """One event of the hits file: identity attributes, volumes made on demand."""

    def __init__(self, owner, key):
        self._owner, self._key = owner, key
        self.attrs = owner._hits[key].attrs
        self.name = f'/{key}'

    def _group(self):
        return self._owner._materialise(self._key)

    def keys(self):
        return self._group().keys()

    def __iter__(self):
        return iter(self._group())

    def __contains__(self, k):
        return k in self._group()

    def __getitem__(self, k):
        return self._group()[k]

    def get(self, k, default=None):
        return self._group().get(k, default)


class HitsFile:
    """hits + presupera (+ step, for the groups), read as a JAXTPC hits file.

    The event being read is laid out in an in-memory HDF5 file with the names the
    reader uses; group ids and coordinates are copied unchanged.
    """

    _PIXEL = (('group_center_py', 'center_py'), ('group_center_pz', 'center_pz'),
              ('group_center_time', 'center_times'), ('offset_py', 'delta_py'),
              ('offset_pz', 'delta_pz'), ('offset_time', 'delta_times'))

    def __init__(self, hits_path, presupera_path, truth):
        import h5py
        if not presupera_path:
            raise ValueError(f"{hits_path} is format 3 and needs "
                             f"reader.jaxtpc_presupera_path")
        self._hits = h5py.File(hits_path, 'r')
        self._pre = h5py.File(presupera_path, 'r')
        self._truth = truth
        self.filename = self._hits.filename
        self._mem = None
        self._now = None
        #: entry -> event index in the mctruth/step files; set by the reader
        self.event_of = {}

    def close(self):
        self._hits.close()
        self._pre.close()
        if self._mem is not None:
            self._mem.close()

    def keys(self):
        return self._hits.keys()

    def __contains__(self, k):
        return k in self._hits

    def __getitem__(self, k):
        return _Entry(self, k) if k.startswith('entry_') and '/' not in k else self._hits[k]

    def get(self, k, default=None):
        return self[k] if k in self else default

    def volumes(self, key):
        """Per volume (rows of the event's steps, group 2 of those rows)."""
        steps, group1 = self._truth.step_columns(self.event_of[key])
        ranges = self._hits['config/volume_ranges'][()].astype(np.float64) / 10.0
        cm = np.c_[steps['x'], steps['y'], steps['z']].astype(np.float32) / np.float32(10.0)
        vol = np.full(len(cm), -1, dtype=np.int32)
        for v, r in enumerate(ranges):
            m = np.ones(len(cm), dtype=bool)
            for a in range(3):
                m &= (cm[:, a] >= r[a, 0]) & (cm[:, a] < r[a, 1])
            vol[m] = v
        group2, maps = refine_groups(group1, vol, len(ranges))
        out = []
        for v in range(len(ranges)):
            rows = np.flatnonzero(vol == v)
            p = self._pre[key][f'volume_{v}']
            if len(rows) != int(p.attrs['n_steps']):
                raise ValueError(f"{key}/volume_{v}: {len(rows)} steps, the simulation "
                                 f"had {int(p.attrs['n_steps'])}")
            if 'deposit_to_group' in p:              # the simulation regrouped
                g = p['deposit_to_group'][()]
            else:
                if not np.array_equal(maps[v], p['group2_to_group1'][()]):
                    raise ValueError(f"{key}/volume_{v}: groups rebuilt from the step "
                                     f"file differ from group2_to_group1")
                g = group2[rows]
            out.append((rows, g))
        return out

    def _materialise(self, key):
        import h5py
        if self._now == key:
            return self._mem[key]
        if self._mem is not None:
            self._mem.close()
        self._mem = h5py.File(f'format3-event-{id(self)}', 'w', driver='core', backing_store=False)
        src, pre = self._hits[key], self._pre[key]
        for a in ('job_id', 'job_event', 'source_index', 'source_event_idx'):
            if a in src.attrs and src.attrs[a] != pre.attrs.get(a):
                raise ValueError(f"{key} is not the same event in the hits and "
                                 f"presupera files ({a})")
        pixel = self._hits['config'].attrs['readout_type'] in ('pixel', b'pixel')
        ev = self._mem.create_group(key)
        for v, (rows, g) in enumerate(self.volumes(key)):
            name = f'volume_{v}'
            vg = ev.create_group(name)
            vg.attrs['n_actual'] = len(rows)
            # group ids start at 1: row 0 is unused, as in the older files
            vg['group_to_track'] = np.r_[np.int32(0), pre[name]['group_to_track'][()]]
            vg['deposit_to_group'] = g
            if name not in src:
                continue
            for plane in src[name]:
                p = src[name][plane]
                q = vg.create_group(plane)
                for c in ('group_ids', 'group_sizes'):
                    q[c] = p[c][()]
                if pixel:
                    for new, old in self._PIXEL:
                        q[old] = p[new][()]
                    q['charge'] = (p['charge'][()] *
                                   float(p['charge'].attrs['scale_factor'])).astype(np.float32)
        self._now = key
        return ev
