"""Format 3: group refinement and the label layout."""
import types

import h5py
import numpy as np

from pysupera.io_v3 import COLUMNS, INTER_COLUMNS, LabelWriter, _DTYPE, _INTER_STR, _VEC4
from pysupera.readers.format3 import refine_groups


def test_refine_groups():
    g1 = np.array([1, 1, 1, 1, 2, 2], np.int32)
    vol = np.array([0, 0, 1, 0, 0, -1])
    g2, maps = refine_groups(g1, vol, 2)
    assert g2.tolist() == [1, 1, 1, 2, 3, 0]
    assert [m.tolist() for m in maps] == [[1, 1, 2], [1]]


def test_label_layout(tmp_path):
    n = 3
    cols = {c: (np.arange(n * 4, dtype=float).reshape(n, 4) if c in _VEC4
                else np.arange(n).astype(_DTYPE[c])) for c in COLUMNS}
    inter = {c: ([b'CC'] if c in _INTER_STR else np.zeros(1)) for c in INTER_COLUMNS}
    p = str(tmp_path / 'l.h5')
    with LabelWriter(p) as w:
        w.append_event(types.SimpleNamespace(columns=cols, interactions=inter, rows=range(n)))
        w.append_hit_labels([])
        # two volumes with 2 and 3 groups, each with its unused slot 0
        w.append_group_owners(np.array([-1, 5, 6, -1, 7, -1, 8]), np.array([0, 1, 0, 0, 0, 0, 1]), [0, 3])
    with h5py.File(p) as f:
        e = f['entry_000000']
        assert int(f['n_events'][()]) == 1 and 'points' not in f and 'hit_labels' not in f
        assert not any('pc_' in c for c in e['particles'])
        assert e['particles/start_pos'].dtype == np.float32
        assert e['volume_0/fragment_id'][()].tolist() == [5, 6]
        assert e['volume_1/fragment_id'][()].tolist() == [7, -1, 8]
        assert e['volume_1/is_le'][()].tolist() == [False, False, True]
        assert e['interactions/reaction'][0] == b'CC'


def test_two_hits_files_open_at_once():
    # a run over several inputs keeps every reader open
    import pysupera.readers.format3 as F
    a, b = (F.HitsFile.__new__(F.HitsFile) for _ in range(2))
    for h in (a, b):
        h._now = h._mem = None
        h._hits = h._pre = {'entry_000000': types.SimpleNamespace(attrs={})}
        h._hits = dict(h._hits, config=types.SimpleNamespace(attrs={'readout_type': 'wire'}))
        h.volumes = lambda key: []
    for h in (a, b, a):
        assert 'entry_000000' in h._materialise('entry_000000').name
    a._mem.close()
    b._mem.close()
