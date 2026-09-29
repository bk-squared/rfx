"""``examples/index.json`` stays true to the examples it describes (#1271).

The index is the public, machine-readable list of the official examples: for
each one, the public function that builds its model without solving it,
whether running it solves, and the files it leaves behind. It is read by
programs that run the examples unmodified, so a stale entry hands them a
wrong answer rather than a missing one. Every claim in it that can be read off
the code is checked here:

* every example file under ``examples/`` is listed, and nothing else is;
* the builder names are the public builders of
  ``tests/_example_fidelity_lib.py::CLASSIFICATION`` (the one hand-maintained
  record of them), in the same order; a script whose builders are all private
  lists none and names them in its reason;
* each named builder exists, takes exactly the listed required arguments, and
  returns a ``Simulation`` without solving, called build-only through the
  fidelity lib with the same variants the snapshot gate uses;
* ``runs_solve``/``solve_via`` agree with the solve calls in the source;
* every listed output's file name is written literally in the source that
  names it, and the call listed as writing it appears in the script.

What is NOT checked: the ``contains`` and ``units`` prose. Those are read off
the code by a person; tests/contracts/test_tutorial_examples.py runs the
far-field tutorial and checks the columns of the cut it writes.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _example_fidelity_lib as lib  # noqa: E402

INDEX_PATH = lib.REPO_ROOT / "examples" / "index.json"
SCHEMA = "rfx-example-index/1"
EXAMPLE_SUFFIXES = (".py", ".yaml", ".yml")
KINDS = {".py": "script", ".yaml": "config", ".yml": "config"}
CONFIG_BUILDER = "rfx.config.simulation_from_yaml"
_PLACEHOLDER = re.compile(r"\{[^{}]+\}")


def _load_index() -> dict:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _entries() -> list[dict]:
    return _load_index()["examples"]


def discover_examples() -> list[str]:
    """Every example file: a script or a YAML config anywhere under examples/."""
    root = lib.REPO_ROOT / "examples"
    return sorted(
        p.relative_to(lib.REPO_ROOT).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in EXAMPLE_SUFFIXES
        and "__pycache__" not in p.parts
    )


def _resolve(dotted: str):
    """Import ``a.b.c`` as module ``a.b`` attribute ``c`` (or deeper)."""
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(f"cannot resolve {dotted!r}")


def _required_args(fn) -> list[str]:
    return [
        name for name, par in inspect.signature(fn).parameters.items()
        if par.default is inspect.Parameter.empty
        and par.kind not in (par.VAR_POSITIONAL, par.VAR_KEYWORD)
    ]


def _returns_label(builder: lib.Builder) -> str:
    if builder.result_index is None:
        return "Simulation"
    return f"tuple; Simulation at index {builder.result_index}"


def _called_names(relpath: str) -> set[str]:
    """Every name called in the script, as ``f(...)`` or ``x.f(...)``."""
    tree = ast.parse((lib.REPO_ROOT / relpath).read_text(), filename=relpath)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def test_index_is_well_formed() -> None:
    index = _load_index()
    assert index.get("schema") == SCHEMA, (
        f"examples/index.json schema is {index.get('schema')!r}, expected "
        f"{SCHEMA!r} -- a format change bumps the version and this test")
    assert str(index.get("about", "")).strip(), "the index needs its 'about' line"
    for entry in index["examples"]:
        where = entry.get("path", "<no path>")
        for key in ("path", "kind", "builders", "runs_solve", "outputs"):
            assert key in entry, f"{where}: entry lacks {key!r}"
        assert entry["kind"] == KINDS.get(Path(entry["path"]).suffix), (
            f"{where}: kind {entry['kind']!r} does not match the file type")
        assert isinstance(entry["runs_solve"], bool), f"{where}: runs_solve is not a bool"
        for builder in entry["builders"]:
            assert set(builder) == {"name", "required_args", "returns"}, (
                f"{where}: builder fields are {sorted(builder)}")
        if entry["builders"]:
            assert "no_builder_reason" not in entry, (
                f"{where}: lists builders AND a reason for having none")
        else:
            assert str(entry.get("no_builder_reason", "")).strip(), (
                f"{where}: no builder listed and no no_builder_reason given")
        for out in entry["outputs"]:
            for key in ("path", "format", "written_by", "contains", "units"):
                assert key in out, f"{where}: output {out.get('path')!r} lacks {key!r}"
            assert isinstance(out["units"], dict), (
                f"{where}: output {out['path']!r} units must be a mapping")


def test_every_example_is_listed_and_nothing_else() -> None:
    listed = [entry["path"] for entry in _entries()]
    duplicates = sorted({p for p in listed if listed.count(p) > 1})
    assert not duplicates, f"examples/index.json lists these twice: {duplicates}"
    on_disk = set(discover_examples())
    unlisted = sorted(on_disk - set(listed))
    stale = sorted(set(listed) - on_disk)
    assert not unlisted, (
        f"example file(s) missing from examples/index.json: {unlisted} -- "
        "add an entry (builder, runs_solve, outputs)")
    assert not stale, (
        f"examples/index.json lists file(s) that are not on disk: {stale}")


def test_builders_are_the_classification_table_builders() -> None:
    """One hand-maintained builder record: CLASSIFICATION. The index repeats
    its public builder names and must not drift from it."""
    problems = []
    for entry in _entries():
        path = entry["path"]
        listed = [(b["name"], b["returns"]) for b in entry["builders"]]
        if entry["kind"] == "config":
            expected = [(CONFIG_BUILDER, "Simulation")]
            if listed != expected:
                problems.append(f"{path}: index {listed} != {expected}")
            continue
        cls = lib.CLASSIFICATION.get(path)
        if cls is None:
            problems.append(f"{path}: no CLASSIFICATION entry")
            continue
        public = [(b.fn, _returns_label(b)) for b in cls.builders
                  if not b.fn.startswith("_")]
        private = [b.fn for b in cls.builders if b.fn.startswith("_")]
        if listed != public:
            problems.append(
                f"{path}: index builders {listed} != CLASSIFICATION public "
                f"builders {public} ({cls.kind})")
        if not public:
            reason = entry.get("no_builder_reason", "")
            missing = [fn for fn in private if fn not in reason]
            if missing:
                problems.append(
                    f"{path}: no_builder_reason does not name the private "
                    f"builder(s) {missing}")
    assert not problems, "\n".join(problems)


def _builder_cases() -> list[tuple[str, str]]:
    return [(entry["path"], b["name"]) for entry in _entries()
            for b in entry["builders"]]


@pytest.mark.parametrize("path,name", _builder_cases(),
                         ids=[f"{p}::{n}" for p, n in _builder_cases()])
def test_named_builder_returns_a_simulation_without_solving(path: str, name: str) -> None:
    from rfx import Simulation

    entry = next(e for e in _entries() if e["path"] == path)
    listed = next(b for b in entry["builders"] if b["name"] == name)
    if entry["kind"] == "config":
        fn = _resolve(name)
        assert _required_args(fn) == listed["required_args"], (
            f"{name} requires {_required_args(fn)}, index says "
            f"{listed['required_args']}")
        with lib.build_only():
            sims = [fn(lib.REPO_ROOT / path)]
    else:
        builder = next(b for b in lib.CLASSIFICATION[path].builders if b.fn == name)
        try:
            module = lib.load_module(path)
            fn = getattr(module, name)
            assert _required_args(fn) == listed["required_args"], (
                f"{path}::{name} requires {_required_args(fn)}, index says "
                f"{listed['required_args']}")
            sims = [lib.call_builder(path, module, builder, variant)
                    for variant in builder.variants]
        except lib.MissingOptionalDependency as exc:
            pytest.skip(str(exc))
    for sim in sims:
        assert isinstance(sim, Simulation), (
            f"{path}::{name} returned {type(sim).__name__}, not a Simulation")


def test_runs_solve_agrees_with_the_source() -> None:
    problems = []
    for entry in _entries():
        path, runs, via = entry["path"], entry["runs_solve"], entry.get("solve_via")
        if entry["kind"] == "config":
            if not runs or not via or "rfx run" not in via:
                problems.append(f"{path}: a config solves through 'rfx run'; "
                                f"runs_solve={runs}, solve_via={via!r}")
            continue
        seen = lib.has_any_solve_call(path)
        if seen and not runs:
            problems.append(f"{path}: the source calls a solve method but "
                            "runs_solve is false")
        if seen and via:
            problems.append(f"{path}: solve_via given although the source "
                            "calls a Simulation solve method itself")
        if runs and not seen:
            if not via:
                problems.append(f"{path}: runs_solve is true, the source calls "
                                "no Simulation solve method, and solve_via "
                                "does not say what solves")
            else:
                head = via.split()[0].rstrip(",;:")
                if head.split(".")[-1] not in _called_names(path):
                    problems.append(f"{path}: solve_via names {head!r}, which "
                                    "the script never calls")
        if not runs and via:
            problems.append(f"{path}: solve_via given but runs_solve is false")
    assert not problems, "\n".join(problems)


def test_every_listed_output_is_named_in_the_source() -> None:
    problems = []
    for entry in _entries():
        path = entry["path"]
        script_src = (lib.REPO_ROOT / path).read_text(encoding="utf-8")
        for out in entry["outputs"]:
            basename = out["path"].rsplit("/", 1)[-1]
            fragments = [f for f in _PLACEHOLDER.split(basename) if f]
            named_by = out.get("named_by")
            if named_by and named_by.startswith("rfx."):
                where, src = named_by, inspect.getsource(_resolve(named_by))
            else:
                where, src = path, script_src
            for fragment in fragments:
                if fragment not in src:
                    problems.append(
                        f"{path}: output {out['path']!r} -- {fragment!r} is "
                        f"not written in {where}")
            if not fragments and not named_by:
                problems.append(
                    f"{path}: output {out['path']!r} has no literal name; "
                    "say what names it in named_by")
            if entry["kind"] == "script":
                call = out["written_by"].split()[-1].split(".")[-1]
                if call not in _called_names(path):
                    problems.append(
                        f"{path}: output {out['path']!r} is written by "
                        f"{out['written_by']!r}, which the script never calls")
    assert not problems, "\n".join(problems)
