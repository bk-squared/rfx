"""The ``result.json`` every showcase script writes (schema ``rfx-showcase-result/1``).

One file per job, beside the arrays it describes.  It says which code made the
numbers, what each number is, what it is held against, and where the witness
file is.  The fields:

* ``id``, ``question`` -- which showcase item and what it measures.
* ``source`` -- the repository commit (``git rev-parse HEAD`` in the tree the
  job ran; no placeholder: a tree that is not a git checkout stops the job),
  the rfx version, jax / jaxlib / numpy versions, the device kind and the
  field precision.
* ``run`` -- platform, preset and wall times.  ``run_id`` stays ``null`` here:
  VESSL does not export the id into the pod, so the submitter writes it from
  ``run_id.txt`` afterwards (``python scripts/showcase/_record.py fill-run-id
  DIR``).
* ``model`` -- the structure, mesh, record length and anything else needed to
  rebuild the run.
* ``claims`` -- one entry per number: ``quantity``, ``value``, ``unit``,
  ``threshold`` (``{"op": "<=", "value": ..., "rule": ...}``) or the string
  ``"reported"``, and ``witness`` (a file in this directory).
* ``derived`` -- numbers computed from other stored numbers, each with its
  ``formula`` and ``inputs``.
* ``out_of_scope`` -- what this record does not measure.
* ``files`` -- sha256 of every file the record names.

Only numpy and the standard library are imported, so the unit tests and the
submitter can load it without jax.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "rfx-showcase-result/1"
TOP_KEYS = ("schema", "id", "question", "source", "run", "model", "claims",
            "derived", "out_of_scope", "files")
SOURCE_KEYS = ("repo_sha", "rfx_version", "jax_version", "jaxlib_version",
               "numpy_version", "device_kind", "precision")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_OPS = ("<=", ">=", "<", ">", "==")


def repo_sha(repo_dir) -> str:
    """The commit checked out at ``repo_dir``, from ``git rev-parse HEAD``.

    Raises ``RuntimeError`` when ``repo_dir`` is not a git checkout, has no
    commit, or git prints anything but a 40-hex sha.  There is deliberately no
    fallback value: a record without its commit cannot be traced to the code
    that produced it.
    """
    proc = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    sha = proc.stdout.strip()
    if proc.returncode != 0 or not _SHA_RE.match(sha):
        raise RuntimeError(
            f"cannot read the commit of {repo_dir}: git rev-parse HEAD exited "
            f"{proc.returncode} with {sha!r} / {proc.stderr.strip()!r}")
    return sha


def tree_is_clean(repo_dir) -> bool:
    """True when ``git status --porcelain`` lists nothing (untracked included)."""
    proc = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain"],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip() == ""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_block(repo_dir, precision: str) -> dict:
    """``source``: the commit and the numerical stack this process runs."""
    import numpy as np

    sha = repo_sha(repo_dir)
    out = {"repo_sha": sha, "tree_clean": tree_is_clean(repo_dir),
           "numpy_version": np.__version__, "precision": precision}
    try:
        import jax
        import jaxlib
        import rfx
    except ImportError as exc:  # the record is about rfx; without it there is none
        raise RuntimeError(f"cannot import the numerical stack: {exc}") from exc
    dev = jax.devices()[0]
    out.update(rfx_version=getattr(rfx, "__version__", None),
               rfx_path=str(Path(rfx.__file__).resolve()),
               jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
               backend=jax.default_backend(), device_kind=dev.device_kind,
               n_devices=len(jax.devices()),
               x64=bool(jax.config.read("jax_enable_x64")),
               python=sys.version.split()[0])
    return out


def claim(quantity: str, value, unit: str, witness: str, *, threshold=None,
          rule: str | None = None, op: str = "<=", note: str | None = None) -> dict:
    """One number.  ``threshold=None`` makes it ``"reported"``."""
    c = {"quantity": quantity, "value": value, "unit": unit, "witness": witness}
    if threshold is None:
        c["threshold"] = "reported"
    else:
        c["threshold"] = {"op": op, "value": threshold, "rule": rule}
        c["passed"] = _compare(value, op, threshold)
    if note:
        c["note"] = note
    return c


def derived(quantity: str, value, unit: str, formula: str, inputs: dict) -> dict:
    return {"quantity": quantity, "value": value, "unit": unit,
            "formula": formula, "inputs": inputs}


def _compare(value, op, threshold) -> bool:
    v, t = float(value), float(threshold)
    return {"<=": v <= t, ">=": v >= t, "<": v < t, ">": v > t, "==": v == t}[op]


def _finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(rec: dict, directory=None) -> None:
    """Raise ``ValueError`` naming the first thing wrong with ``rec``.

    With ``directory`` every file in ``files`` must exist there with the
    recorded sha256, and every witness must be one of those files.
    """
    missing = [k for k in TOP_KEYS if k not in rec]
    if missing:
        raise ValueError(f"missing top-level keys {missing}")
    if rec["schema"] != SCHEMA:
        raise ValueError(f"schema {rec['schema']!r}, expected {SCHEMA!r}")
    src = rec["source"]
    missing = [k for k in SOURCE_KEYS if k not in src]
    if missing:
        raise ValueError(f"source lacks {missing}")
    if not isinstance(src["repo_sha"], str) or not _SHA_RE.match(src["repo_sha"]):
        raise ValueError(f"source.repo_sha {src['repo_sha']!r} is not a 40-hex commit")
    run = rec["run"]
    if "run_id" not in run:
        raise ValueError("run.run_id is absent (it is null until the submitter fills it)")
    rid = run["run_id"]
    if rid is not None and not (isinstance(rid, str) and rid.isdigit()):
        raise ValueError(f"run.run_id {rid!r} is neither null nor a VESSL run id")
    if not isinstance(rec["claims"], list) or not rec["claims"]:
        raise ValueError("claims must be a non-empty list")
    files = rec["files"]
    for i, c in enumerate(rec["claims"]):
        for k in ("quantity", "value", "unit", "threshold", "witness"):
            if k not in c:
                raise ValueError(f"claim {i} ({c.get('quantity')!r}) lacks {k!r}")
        if not isinstance(c["unit"], str):
            raise ValueError(f"claim {i} ({c['quantity']!r}): unit must be a string "
                             "(\"1\" for a dimensionless number)")
        t = c["threshold"]
        if t != "reported":
            if not isinstance(t, dict) or t.get("op") not in _OPS \
                    or not _finite_number(t.get("value")) or not t.get("rule"):
                raise ValueError(f"claim {i} ({c['quantity']!r}): threshold must be "
                                 "'reported' or {op, value, rule}")
            if not _finite_number(c["value"]):
                raise ValueError(f"claim {i} ({c['quantity']!r}): a judged value must be "
                                 f"a finite number, got {c['value']!r}")
            if c.get("passed") != _compare(c["value"], t["op"], t["value"]):
                raise ValueError(f"claim {i} ({c['quantity']!r}): 'passed' does not "
                                 "follow from value, op and threshold")
        if c["witness"] not in files:
            raise ValueError(f"claim {i} ({c['quantity']!r}): witness {c['witness']!r} "
                             "is not listed in files")
    for i, d in enumerate(rec["derived"]):
        for k in ("quantity", "value", "unit", "formula", "inputs"):
            if k not in d:
                raise ValueError(f"derived {i} ({d.get('quantity')!r}) lacks {k!r}")
        if not str(d["formula"]).strip():
            raise ValueError(f"derived {i} ({d['quantity']!r}) has an empty formula")
    if directory is not None:
        directory = Path(directory)
        for name, digest in files.items():
            p = directory / name
            if not p.is_file():
                raise ValueError(f"file {name!r} is listed but absent")
            if sha256_file(p) != digest:
                raise ValueError(f"file {name!r} does not match its recorded sha256")


# Written by the job's shell around the script (tee, return codes, the exit
# summary) or by the submitter afterwards: still changing when the record is
# written, so never hashed into it.  The archive manifest hashes them.
_NOT_DATA_SUFFIXES = (".log", ".rc", ".tmp")
_NOT_DATA_NAMES = ("result.json", "run_id.txt", "summary.txt")


def data_files(directory) -> list[str]:
    """The files of ``directory`` a record may name: everything but logs,
    return codes, the exit summary, ``run_id.txt`` and ``result.json``."""
    return sorted(p.name for p in Path(directory).iterdir()
                  if p.is_file() and p.name not in _NOT_DATA_NAMES
                  and not p.name.endswith(_NOT_DATA_SUFFIXES))


def write_result(directory, rec: dict, file_names) -> Path:
    """Hash ``file_names`` (relative to ``directory``), validate, write
    ``result.json``, and return its path."""
    directory = Path(directory)
    rec = dict(rec)
    rec.setdefault("schema", SCHEMA)
    rec["files"] = {n: sha256_file(directory / n) for n in sorted(set(file_names))}
    validate(rec, directory)
    path = directory / "result.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=False, default=_jsonable) + "\n")
    tmp.replace(path)
    return path


def _jsonable(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def fill_run_id(directory) -> str:
    """Copy the id from ``run_id.txt`` (written by ``scripts/vessl_submit.sh``)
    into ``result.json``.  Run by the submitter, never inside the job."""
    directory = Path(directory)
    rid = (directory / "run_id.txt").read_text().strip()
    if not rid.isdigit():
        raise ValueError(f"run_id.txt holds {rid!r}, not a VESSL run id")
    path = directory / "result.json"
    rec = json.loads(path.read_text())
    if rec["run"].get("run_id") not in (None, rid):
        raise ValueError(f"result.json already names run {rec['run']['run_id']}, "
                         f"run_id.txt says {rid}")
    rec["run"]["run_id"] = rid
    rec["run"]["run_id_source"] = "run_id.txt (scripts/vessl_submit.sh)"
    validate(rec, directory)
    path.write_text(json.dumps(rec, indent=1) + "\n")
    return rid


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fill-run-id", help="copy run_id.txt into result.json")
    f.add_argument("directory")
    v = sub.add_parser("validate", help="validate result.json and its file hashes")
    v.add_argument("directory")
    a = ap.parse_args(argv)
    if a.cmd == "fill-run-id":
        print(fill_run_id(a.directory))
    else:
        d = Path(a.directory)
        validate(json.loads((d / "result.json").read_text()), d)
        print(f"{d / 'result.json'}: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
