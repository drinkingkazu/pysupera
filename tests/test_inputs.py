"""
Several inputs into one output, and each event's source and identity.

* reader_inputs: io.input_path as a list, with reader.jaxtpc_*_path lists or
  one shared path (a merged JAXTPC file)
* ChainReader: events of several readers as one, with the input of each
* JaxtpcHDF5Reader._map_entries / _entry_key: pairing EDepSim events with
  JAXTPC format-2 entries (entry_NNNNNN), by dprod ids or by input file name
* EventSources: events/input, input_event, jaxtpc_entry and the identity
  copied from /dprod/events
"""
import h5py
import numpy as np
import pytest

from omegaconf import OmegaConf

from pysupera.config import reader_inputs
from pysupera.identity import EventSources, read_dprod_events
from pysupera.readers.chain import ChainReader
from pysupera.readers.format_jaxtpc import JaxtpcHDF5Reader


# ── reader_inputs ──────────────────────────────────────────────────────────────

def load_cfg(overrides):
    """io + reader only, as the overrides give them.  (pysupera.config.load_cfg
    would also configure() pysupera-wide defaults, which other tests rely on.)"""
    cfg = OmegaConf.create({"io": {}, "reader": {}})
    for ov in overrides:
        if ov.startswith("reader="):
            continue
        k, v = ov.split("=", 1)
        if v.startswith("["):
            v = [x.strip().strip("'") for x in v[1:-1].split(",")]
        OmegaConf.update(cfg, k, v)
    return cfg


def test_single_input_unchanged():
    cfg = load_cfg(["io.input_path=/d/e.h5", "io.output_path=/d/o.h5"])
    assert reader_inputs(cfg) == [{"edepsim": "/d/e.h5", "jaxtpc_step": "",
                                   "jaxtpc_hits": "", "jaxtpc_sensor": ""}]


def test_list_with_shared_jaxtpc_files():
    cfg = load_cfg(["reader=jaxtpc_wire", "io.input_path=[/d/a.h5,/d/b.h5]",
                    "io.output_path=/d/o.h5", "reader.jaxtpc_seg_path=/j/step.h5",
                    "reader.jaxtpc_inst_path=/j/hits.h5"])
    got = reader_inputs(cfg)
    assert [d["edepsim"] for d in got] == ["/d/a.h5", "/d/b.h5"]
    assert {d["jaxtpc_hits"] for d in got} == {"/j/hits.h5"}
    assert {d["jaxtpc_step"] for d in got} == {"/j/step.h5"}


def test_list_with_one_jaxtpc_file_per_input():
    cfg = load_cfg(["reader=jaxtpc_wire", "io.input_path=['/d/a.h5','/d/b.h5']",
                    "io.output_path=/d/o.h5",
                    "reader.jaxtpc_seg_path=['/j/s0.h5','/j/s1.h5']",
                    "reader.jaxtpc_inst_path=['/j/h0.h5','/j/h1.h5']"])
    got = reader_inputs(cfg)
    assert [(d["jaxtpc_step"], d["jaxtpc_hits"]) for d in got] == [
        ("/j/s0.h5", "/j/h0.h5"), ("/j/s1.h5", "/j/h1.h5")]


def test_list_lengths_must_agree():
    cfg = load_cfg(["reader=jaxtpc_wire", "io.input_path=[/d/a.h5,/d/b.h5]",
                    "io.output_path=/d/o.h5", "reader.jaxtpc_seg_path=[/j/s0.h5]",
                    "reader.jaxtpc_inst_path=/j/h.h5"])
    with pytest.raises(ValueError, match="lists 1 file"):
        reader_inputs(cfg)


# ── ChainReader ────────────────────────────────────────────────────────────────

class _Fake:
    def __init__(self, n, tag):
        self.n, self.tag, self.closed = n, tag, False
        self.last_thing = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.last_thing = (self.tag, i)
        return [self.tag, i]

    def close(self):
        self.closed = True


def test_chain_reads_inputs_in_order():
    a, b = _Fake(2, "a"), _Fake(3, "b")
    ch = ChainReader([a, b], [{"edepsim": "a"}, {"edepsim": "b"}])
    assert len(ch) == 5
    got = [ev for ev in ch]
    assert got == [["a", 0], ["a", 1], ["b", 0], ["b", 1], ["b", 2]]
    assert (ch.last_input, ch.last_input_event) == (1, 2)
    assert ch.last_thing == ("b", 2)          # delegated to the current input
    assert ch.locate(-1) == (1, 2) and ch.locate(2) == (1, 0)
    with pytest.raises(IndexError):
        ch[5]
    ch.close()
    assert a.closed and b.closed


def test_chain_hands_on_the_pixel_geometry():
    class _Px(_Fake):
        def share_pixel_geometry(self, other):
            self.got_from = other.tag
    a, b = _Px(1, "a"), _Px(1, "b")
    ch = ChainReader([a, b], [{}, {}])
    list(ch)
    assert b.got_from == "a"


# ── JAXTPC entries ─────────────────────────────────────────────────────────────

def _edepsim(path, job=None, n=3):
    with h5py.File(path, "w") as f:
        if job is not None:
            e = f.create_group("dprod/events")
            e["job_id"] = np.full(n, job, dtype=np.int64)
            e["job_event"] = np.arange(n, dtype=np.int64)
            e["event_id"] = job * 10 + np.arange(n, dtype=np.int64)
    return path


def _jaxtpc(path, sources, entries):
    with h5py.File(path, "w") as f:
        c = f.create_group("config")
        c.attrs["format_version"] = 2
        c.create_dataset("source_files", data=np.array(sources, dtype=h5py.string_dtype()))
        for i, a in enumerate(entries):
            g = f.create_group(f"entry_{i:06d}")
            for k, v in a.items():
                g.attrs[k] = v
    return path


def _reader(edep, hits, n=3, seg=None):
    """A JaxtpcHDF5Reader with only what entry pairing uses."""
    r = object.__new__(JaxtpcHDF5Reader)
    r._edepsim_path, r._inst_path = edep, hits
    r._edepsim_file, r._inst_file = h5py.File(edep, "r"), h5py.File(hits, "r")
    r._seg_file = h5py.File(seg or hits, "r")
    r._sensor_file = None
    r._n_events = n
    r._entries = r._map_entries()
    return r


def test_merged_file_paired_by_dprod_ids(tmp_path):
    e7 = _edepsim(str(tmp_path / "e7.h5"), job=7)
    e8 = _edepsim(str(tmp_path / "e8.h5"), job=8)
    # entries in input order: job 7 events 0-2, then job 8
    ents = [dict(source_index=k, source_event_idx=i, job_id=j, job_event=i, event_id=j * 10 + i)
            for k, j in enumerate((7, 8)) for i in range(3)]
    h = _jaxtpc(str(tmp_path / "hits.h5"), [e7, e8], ents)
    r = _reader(e8, h)
    assert r._entries == ["entry_000003", "entry_000004", "entry_000005"]
    assert r._entry_key(1) == "entry_000004" and r.last_jaxtpc_entry == 4
    assert read_dprod_events(r._edepsim_file)["event_id"].tolist() == [80, 81, 82]


def test_paired_by_file_name_without_dprod(tmp_path):
    e0 = _edepsim(str(tmp_path / "edepsim_a.h5"))
    e1 = _edepsim(str(tmp_path / "edepsim_b.h5"))
    ents = [dict(source_index=k, source_event_idx=i, event_id=i) for k in (0, 1) for i in range(3)]
    h = _jaxtpc(str(tmp_path / "hits.h5"), ["/elsewhere/edepsim_a.h5", "/elsewhere/edepsim_b.h5"], ents)
    assert _reader(e1, h)._entries == ["entry_000003", "entry_000004", "entry_000005"]
    assert _reader(e0, h)._entries == ["entry_000000", "entry_000001", "entry_000002"]


def test_missing_entry_and_mismatched_files(tmp_path):
    e7 = _edepsim(str(tmp_path / "e7.h5"), job=7)
    ents = [dict(source_index=0, source_event_idx=i, job_id=7, job_event=i) for i in range(2)]
    h = _jaxtpc(str(tmp_path / "hits.h5"), [e7], ents)
    r = _reader(e7, h)
    with pytest.raises(KeyError, match="has no entry"):
        r._entry_key(2)
    s = _jaxtpc(str(tmp_path / "step.h5"), [e7],
                [dict(source_index=0, source_event_idx=i, job_id=7, job_event=1 - i) for i in range(2)])
    with pytest.raises(ValueError, match="not the same event"):
        _reader(e7, h, seg=s)._entry_key(0)


def test_format1_files_keep_event_names(tmp_path):
    e = _edepsim(str(tmp_path / "e.h5"))
    with h5py.File(tmp_path / "hits.h5", "w") as f:
        f.create_group("event_000")
    r = _reader(e, str(tmp_path / "hits.h5"))
    assert r._entries is None and r._entry_key(0) == "event_000"


# ── EventSources ───────────────────────────────────────────────────────────────

def test_event_sources_written(tmp_path):
    e7 = _edepsim(str(tmp_path / "e7.h5"), job=7, n=2)
    e8 = _edepsim(str(tmp_path / "e8.h5"), job=8, n=2)
    src = EventSources([{"edepsim": e7, "jaxtpc_hits": "/j/h.h5"}, {"edepsim": e8, "jaxtpc_hits": "/j/h.h5"}])
    assert src.has_ids
    for k, i, ent in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)):
        src.append(k, i, ent)
    assert src.identity(2) == {"event_id": 80, "job_id": 8, "job_event": 0}
    with h5py.File(tmp_path / "out.h5", "w") as f:
        f.create_group("events")
        src.write(f, 4)
    with h5py.File(tmp_path / "out.h5", "r") as f:
        assert f["events/event_id"][:].tolist() == [70, 71, 80, 81]
        assert f["events/job_event"][:].tolist() == [0, 1, 0, 1]
        assert f["events/input"][:].tolist() == [0, 0, 1, 1]
        assert f["events/jaxtpc_entry"][:].tolist() == [0, 1, 2, 3]
        assert [x.decode() for x in f["inputs/edepsim"][:]] == [e7, e8]
        assert [x.decode() for x in f["inputs/jaxtpc_step"][:]] == ["", ""]


def test_event_sources_without_ids_and_mixed(tmp_path):
    plain = _edepsim(str(tmp_path / "p.h5"))
    src = EventSources([{"edepsim": plain}])
    src.append(0, 0)
    assert not src.has_ids and src.identity(0) is None
    with h5py.File(tmp_path / "out.h5", "w") as f:
        f.create_group("events")
        src.write(f, 1)
        assert "event_id" not in f["events"] and f["events/jaxtpc_entry"][0] == -1
    with pytest.raises(ValueError, match="some do not"):
        EventSources([{"edepsim": plain}, {"edepsim": _edepsim(str(tmp_path / "d.h5"), job=1)}])
