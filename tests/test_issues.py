"""
What a run notices but does not stop for is recorded in the output's
issues/ group, under stable codes, so production tools can find it.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from pysupera.issues import (DESCRIPTION, Issue, IssueLog, read_issues,
                             write_issues)
from tests.test_primaries import write_edepsim

h5py = pytest.importorskip("h5py")


def test_codes_are_fixed():
    """A summary across files depends on these; never renumber them."""
    assert {c.name: int(c) for c in Issue} == {
        "PRIMARY_TABLE_MISMATCH": 1, "NO_PRIMARY_TABLE": 2,
        "EMPTY_IMAGE": 3, "OUTSIDE_READOUT_WINDOW": 4}
    assert set(DESCRIPTION) == set(Issue)


def test_round_trip(tmp_path):
    log = IssueLog()
    log.record(Issue.PRIMARY_TABLE_MISMATCH, "track 2 ...", event=3,
               geant4_trackid=2)
    log.record(Issue.NO_PRIMARY_TABLE, "no table")
    with h5py.File(tmp_path / "f.h5", "w") as f:
        write_issues(f, log)
    with h5py.File(tmp_path / "f.h5", "r") as f:
        got = read_issues(f)
        assert f.attrs["n_issues"] == 2
        assert f["issues"].attrs["n_primary_table_mismatch"] == 1
        assert f["issues"].attrs["n_empty_image"] == 0
    assert got["event"].tolist() == [3, -1]
    assert got["code"].tolist() == [1, 2]
    assert got["geant4_trackid"].tolist() == [2, -1]
    assert got["message"] == ["track 2 ...", "no table"]
    assert got["names"][3] == "EMPTY_IMAGE"


def test_an_empty_log_is_still_written(tmp_path):
    """An empty group says the checks ran and found nothing."""
    with h5py.File(tmp_path / "f.h5", "w") as f:
        write_issues(f, IssueLog())
    with h5py.File(tmp_path / "f.h5", "r") as f:
        assert f.attrs["n_issues"] == 0
        assert len(read_issues(f)["code"]) == 0


def test_report_counts_and_shows_the_first_messages():
    log = IssueLog()
    for k in range(7):
        log.record(Issue.EMPTY_IMAGE, f"empty {k}", event=k)
    text = log.report(max_lines=2)
    assert "EMPTY_IMAGE" in text and "7" in text
    assert "event 1: empty 1" in text and "empty 2" not in text
    assert "... 5 more" in text
    assert IssueLog().report() == "[run] Issues: none"


def _run_pysupera():
    exe = (shutil.which("run_pysupera")
           or str(Path.home() / ".local/bin/run_pysupera"))
    if not Path(exe).exists():
        pytest.skip("run_pysupera not found")
    return exe


@pytest.mark.parametrize("prim, steps, want", [
    ([1, 2], True, {}),
    ([1], True, {Issue.PRIMARY_TABLE_MISMATCH: [2]}),
    (None, True, {Issue.NO_PRIMARY_TABLE: [-1]}),
    ([1, 2], False, {Issue.EMPTY_IMAGE: [-1]}),
])
def test_a_run_records_what_it_found(tmp_path, prim, steps, want):
    write_edepsim(tmp_path / "in.h5", prim_ids=prim, steps=steps)
    out = tmp_path / "out.h5"
    r = subprocess.run([_run_pysupera(), f"io.input_path={tmp_path / 'in.h5'}",
                        f"io.output_path={out}", "progress=false",
                        "allow_empty_image=true",
                        f"hydra.run.dir={tmp_path / 'hydra'}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    from pysupera.io_v3 import EventStoreV3
    with EventStoreV3(str(out)) as s:
        got = s.issues()
    seen = {}
    for c, t in zip(got["code"], got["geant4_trackid"]):
        seen.setdefault(Issue(int(c)), []).append(int(t))
    assert seen == want
    assert "[run] Issues:" in r.stdout
