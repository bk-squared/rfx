"""scripts/dev/progress_numbers.py reads the commit it is given, never the working tree."""

import importlib.util
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("progress_numbers", REPO / "scripts/dev/progress_numbers.py")
numbers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(numbers)

FIVE = ("rfx/api/_execute.py", "rfx/api/__init__.py", "rfx/simulation.py", "rfx/nonuniform.py",
        "rfx/sources/waveguide_port.py")
LAYERS = ("rfx/model", "rfx/stepping", "rfx/measurement", "rfx/boundaries")


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "core.hooksPath=/dev/null", *args],
                   check=True, capture_output=True)


def test_numbers_come_from_the_commit_not_the_working_tree(tmp_path, capsys):
    for path in FIVE:
        write(tmp_path, path, "")
    for package in LAYERS:
        write(tmp_path, package + "/__init__.py", "")
    write(tmp_path, numbers.CHECKER, (REPO / numbers.CHECKER).read_text())
    write(tmp_path, "rfx/simulation.py", "def kept():\n    return np.zeros(1)\n")
    write(tmp_path, "rfx/stepping/reader.py", "from rfx.simulation import kept\n")
    write(tmp_path, "rfx/api/__init__.py", "class Simulation:\n"
          "    def add_port(self):\n        return self._build_grid()\n"
          "    def compute(self):\n        x = 1\n        return x\n")
    write(tmp_path, numbers.CONTRACT, "KNOWN_ROWS = (\n"
          "    ('P-a', 'uniform', 'L2', 'D1', 'why'),\n    ('P-a', 'uniform', 'L2', 'D2', 'why'),\n"
          "    ('B-b', 'graded', 'L1', 'D1', 'why'),\n)\n"
          "TWO_GUARD_REFUSALS = {('P-c', 'uniform'): 'why'}\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "tree")
    # The working tree now disagrees with the commit on every number.
    write(tmp_path, "rfx/simulation.py", "")
    write(tmp_path, "rfx/api/__init__.py", "")
    write(tmp_path, "rfx/stepping/reader.py", "")
    write(tmp_path, numbers.CONTRACT, "KNOWN_ROWS = ()\nTWO_GUARD_REFUSALS = {}\nKNOWN_DIRECTIONS = {}\n")
    write(tmp_path, numbers.CHECKER, "raise SystemExit('the working-tree checker ran')\n")

    assert numbers.main(["HEAD", "--repo", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "  rfx/simulation.py: 1 (layer units 1 of 1, imports 0, declarations 0)" in out
    assert "  rfx/api/__init__.py: 2 (layer units 0 of 2, imports 0, declarations 2)" in out
    assert "  five files total: 3" in out
    assert "  layer imports of a five-file definition: 1" in out
    assert "reverse imports from layers to callers: R=1 " in out
    assert ("  known-failing cells: 3\n  KNOWN_ROWS: 3 entries on 2 cells\n"
            "  TWO_GUARD_REFUSALS: 1 entries on 1 cells\n"
            "  KNOWN_DIRECTIONS (narrows KNOWN_ROWS, marks no cell): absent\n") in out


def test_five_file_numbers_at_head_equal_the_committed_baseline():
    """CI keeps scripts/ci/layer_baseline.json equal to the checker's count, so at HEAD the two agree."""
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    committed = json.loads(subprocess.run(
        ["git", "-C", str(REPO), "show", f"{head}:scripts/ci/layer_baseline.json"],
        check=True, capture_output=True, text=True).stdout)
    got = numbers.layer_numbers(REPO, head)
    for path, rules in committed["files"].items():
        assert got["files"][path]["imports"] == len(rules["I"]), path
        assert got["files"][path]["units"] == len(set().union(*(rules[r] for r in "SMAG"))), path
    assert got["reverse"] == sum(len(r["R"]) for r in committed.get("reverse", {}).values())
    assert got["outside"] == {p: {r: len(v[r]) for r in ("S", "M")} for p, v in committed["outside"].items()}


def test_tables_built_from_unpacked_comprehensions_are_expanded_and_names_are_not_counted(tmp_path, capsys):
    write(tmp_path, numbers.CONTRACT, "KNOWN_ROWS = (\n    *[(d, 'uniform', 'L2', 'D1', 'why') for d in ('P-a', 'P-b')],\n)\n"
          "TWO_GUARD_REFUSALS = {\n    **{('P-coax', p): 'why' for p in ('uniform', 'graded')},\n"
          "    **{(d, 'multi-device-graded'): 'why' for d in ('P-msl',)},\n    ('R-rlc', 'uniform'): 'why',\n}\n"
          "KNOWN_DIRECTIONS = {key: ('a',) for key in KNOWN_ROWS}\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "tables")
    found = numbers.contract_numbers(tmp_path, "HEAD")
    assert found["KNOWN_ROWS"] == {"entries": 2, "cells": 2}
    assert found["TWO_GUARD_REFUSALS"] == {"entries": 4, "cells": 4}
    assert "unread" in found["KNOWN_DIRECTIONS"]
    write(tmp_path, numbers.CONTRACT, "TWO_GUARD_REFUSALS = {(d, 'uniform'): 'why' for d in DECLARATIONS}\n")
    git(tmp_path, "commit", "-q", "-am", "a table read through a name")
    found = numbers.contract_numbers(tmp_path, "HEAD")
    assert "unread" in found["TWO_GUARD_REFUSALS"]
    numbers.report("sha", "header", numbers.layer_numbers(REPO, "HEAD"), found)
    assert "  known-failing cells: not counted at this commit\n" in capsys.readouterr().out
