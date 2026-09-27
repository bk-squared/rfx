"""The showcase record writer: the commit is read or the run stops, and a record
that names a number without its unit, threshold, witness or file hash is refused.

``scripts/showcase/_record.py`` writes the ``result.json`` beside every showcase
job's arrays (issue 1332).  A record whose commit reads ``unknown`` looks
complete and traces to nothing (the vessl-jobs provenance rule), so the writer
has no fallback value; these tests hold that, and the schema checks, without
running any solver.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "showcase"))
import _record  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "HOME": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin"})


def _head_from_disk(repo: Path) -> str:
    """HEAD read from the ref files themselves, without asking git."""
    head = (repo / ".git" / "HEAD").read_text().strip()
    assert head.startswith("ref: "), head
    return (repo / ".git" / head[5:]).read_text().strip()


def test_the_commit_is_read_from_the_tree_and_never_invented(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    # a repository with no commit has no HEAD to read
    with pytest.raises(RuntimeError, match="cannot read the commit"):
        _record.repo_sha(repo)
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "one")
    assert _record.repo_sha(repo) == _head_from_disk(repo)
    # a directory that is not a checkout (a git-archive export) has none either
    plain = tmp_path / "exported"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="cannot read the commit"):
        _record.repo_sha(plain)


def _good_record(directory: Path) -> tuple[dict, list[str]]:
    (directory / "curve.npz").write_bytes(b"not really an npz, but bytes to hash")
    rec = {
        "schema": _record.SCHEMA, "id": "t", "question": "q",
        "source": {"repo_sha": "0" * 40, "rfx_version": "1", "jax_version": "1",
                   "jaxlib_version": "1", "numpy_version": "1", "device_kind": "cpu",
                   "precision": "float32"},
        "run": {"platform": "test", "preset": None, "run_id": None, "wall_s": {}},
        "model": {},
        "claims": [
            _record.claim("f", 2.3e9, "Hz", "curve.npz"),
            _record.claim("rel", 0.01, "1", "curve.npz", threshold=0.05, rule="brief"),
        ],
        "derived": [_record.derived("cost", 10.0, "s", "2 x N x t", {"N": 1, "t": 5.0})],
        "out_of_scope": [],
    }
    return rec, ["curve.npz"]


def test_a_complete_record_is_written_and_validates(tmp_path):
    rec, files = _good_record(tmp_path)
    path = _record.write_result(tmp_path, rec, files)
    back = json.loads(path.read_text())
    _record.validate(back, tmp_path)
    assert back["files"]["curve.npz"] == _record.sha256_file(tmp_path / "curve.npz")
    assert back["claims"][1]["passed"] is True
    assert back["run"]["run_id"] is None


@pytest.mark.parametrize("defect, expected", [
    (lambda r: r["source"].__setitem__("repo_sha", "unknown"), "not a 40-hex commit"),
    (lambda r: r["source"].pop("device_kind"), "source lacks"),
    (lambda r: r["run"].__setitem__("run_id", "UNSET-see-log-filename"), "neither null nor"),
    (lambda r: r["claims"][0].pop("unit"), "lacks 'unit'"),
    (lambda r: r["claims"][1].__setitem__("threshold", {"op": "<=", "value": 0.05}),
     "threshold must be"),
    (lambda r: r["claims"][1].__setitem__("passed", False), "'passed' does not follow"),
    (lambda r: r["claims"][0].__setitem__("witness", "elsewhere.json"), "is not listed in files"),
    (lambda r: r["derived"][0].__setitem__("formula", " "), "empty formula"),
])
def test_a_defective_record_is_refused(tmp_path, defect, expected):
    rec, files = _good_record(tmp_path)
    defect(rec)
    with pytest.raises(ValueError, match=expected):
        _record.write_result(tmp_path, rec, files)
    assert not (tmp_path / "result.json").exists()


def test_a_file_changed_after_the_record_is_refused(tmp_path):
    rec, files = _good_record(tmp_path)
    path = _record.write_result(tmp_path, rec, files)
    (tmp_path / "curve.npz").write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        _record.validate(json.loads(path.read_text()), tmp_path)


def test_the_run_id_comes_from_the_submitter_file(tmp_path):
    rec, files = _good_record(tmp_path)
    _record.write_result(tmp_path, rec, files)
    with pytest.raises(FileNotFoundError):
        _record.fill_run_id(tmp_path)
    (tmp_path / "run_id.txt").write_text("369367265999\n")
    assert _record.fill_run_id(tmp_path) == "369367265999"
    back = json.loads((tmp_path / "result.json").read_text())
    assert back["run"]["run_id"] == "369367265999"
    _record.validate(back, tmp_path)


# A judged number that misses its bar must read as a miss, whatever the
# operator, and a record that says otherwise must be refused.  Each row: the
# operator, the bar, a value that fails it, a value that passes it (the
# equality rows tell "<" from "<=" and ">" from ">=").
_BARS = [
    ("<=", 0.05, 0.1, 0.05),
    ("<", 0.05, 0.05, 0.04),
    (">=", 2.0, 1.0, 2.0),
    (">", 2.0, 2.0, 3.0),
    ("==", 2.0, 3.0, 2.0),
]


@pytest.mark.parametrize("op, bar, failing, passing", _BARS)
def test_a_judged_claim_reads_its_own_verdict(op, bar, failing, passing):
    miss = _record.claim("q", failing, "1", "curve.npz", threshold=bar, rule="r", op=op)
    hit = _record.claim("q", passing, "1", "curve.npz", threshold=bar, rule="r", op=op)
    assert miss["passed"] is False, (op, bar, failing)
    assert hit["passed"] is True, (op, bar, passing)


@pytest.mark.parametrize("op, bar, failing, passing", _BARS)
def test_a_record_claiming_a_pass_it_did_not_earn_is_refused(tmp_path, op, bar, failing, passing):
    rec, files = _good_record(tmp_path)
    rec["claims"][1] = _record.claim("q", failing, "1", "curve.npz", threshold=bar,
                                     rule="r", op=op)
    path = _record.write_result(tmp_path, rec, files)   # an honest miss is a valid record
    written = json.loads(path.read_text())
    assert written["claims"][1]["passed"] is False
    written["claims"][1]["passed"] = True                # the same miss, reported as a pass
    with pytest.raises(ValueError, match="'passed' does not follow"):
        _record.validate(written, tmp_path)


def test_a_value_of_0_1_against_a_0_05_bar_fails(tmp_path):
    c = _record.claim("|AD - FD| / |FD|", 0.1, "1", "curve.npz", threshold=0.05, rule="brief")
    assert c["passed"] is False
    rec, files = _good_record(tmp_path)
    rec["claims"][1] = dict(c, passed=True)
    with pytest.raises(ValueError, match="'passed' does not follow"):
        _record.write_result(tmp_path, rec, files)


def test_the_run_id_is_never_overwritten_with_another(tmp_path):
    rec, files = _good_record(tmp_path)
    _record.write_result(tmp_path, rec, files)
    (tmp_path / "run_id.txt").write_text("369367265111\n")
    _record.fill_run_id(tmp_path)
    (tmp_path / "run_id.txt").write_text("369367265222\n")
    with pytest.raises(ValueError, match="already names run 369367265111"):
        _record.fill_run_id(tmp_path)
    back = json.loads((tmp_path / "result.json").read_text())
    assert back["run"]["run_id"] == "369367265111"
