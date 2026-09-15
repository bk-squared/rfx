"""Failure injections for the build-only audit; all writes target tmp_path."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _example_fidelity_lib as lib  # noqa: E402


@pytest.fixture
def example_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "REPO_ROOT", tmp_path)
    crossval = tmp_path / "validation" / "crossval"
    crossval.mkdir(parents=True)
    evidence = crossval / "retained.json"
    evidence.write_bytes(b'{"meep_version": "retained measurement"}\n')
    return tmp_path, crossval, evidence


@pytest.mark.parametrize("relpath", [
    p for p, entry in lib.CLASSIFICATION.items()
    if entry.kind == "module_level_solve"
])
def test_module_level_solvers_are_refused_before_import(relpath, monkeypatch):
    def unexpected_import(*args):
        pytest.fail("unsafe script reached the import machinery")

    monkeypatch.setattr(lib.importlib.util, "spec_from_file_location", unexpected_import)
    with pytest.raises(lib.UnsafeExampleExecution, match="main guard"):
        lib.load_module(relpath)


@pytest.mark.parametrize("operation", ["write", "append", "os_open", "unlink", "replace"])
def test_evidence_mutation_fails_before_bytes_change(example_tree, operation):
    root, _, evidence = example_tree
    before = evidence.read_bytes()
    replacement = root / "replacement.json"
    replacement.write_text("bad measurement")
    with lib.build_only():
        with pytest.raises(lib.UnsafeExampleExecution, match="crossval mutation"):
            if operation == "write":
                evidence.write_text("bad measurement")
            elif operation == "append":
                with evidence.open("a") as stream:
                    stream.write("bad measurement")
            elif operation == "os_open":
                fd = os.open(evidence, os.O_WRONLY | os.O_TRUNC)
                os.close(fd)
            elif operation == "unlink":
                evidence.unlink()
            else:
                replacement.replace(evidence)
        assert evidence.read_bytes() == before
    # No permanent permission/mocking changes after the context exits.
    evidence.write_bytes(before)


def test_main_guard_does_not_hide_import_time_writer(example_tree):
    _, crossval, evidence = example_tree
    before = evidence.read_bytes()
    (crossval / "writer.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('retained.json').write_text('bad')\n"
        "if __name__ == '__main__':\n    pass\n")
    with pytest.raises(lib.UnsafeExampleExecution, match="crossval mutation"):
        lib.load_module("validation/crossval/writer.py")
    assert evidence.read_bytes() == before


@pytest.mark.parametrize("method", ["run", "forward", "compute_waveguide_s_matrix"])
def test_indirect_solve_is_refused_even_when_ast_cannot_see_it(example_tree, method):
    _, crossval, _ = example_tree
    # getattr deliberately bypasses the AST's .run()/forward() heuristic.
    (crossval / "solver.py").write_text(
        "from rfx import Simulation\n"
        f"getattr(Simulation, {method!r})(None)\n"
        "if __name__ == '__main__':\n    pass\n")
    with pytest.raises(lib.UnsafeExampleExecution, match=f"Simulation.{method}"):
        lib.load_module("validation/crossval/solver.py")


def test_nested_scope_does_not_release_outer_protection(example_tree):
    root, _, evidence = example_tree
    with lib.build_only():
        with lib.build_only():
            assert evidence.read_bytes()
        with pytest.raises(lib.UnsafeExampleExecution):
            evidence.write_text("bad")
        # Build-only does not forbid unrelated scratch outputs.
        (root / "scratch.txt").write_text("scratch")


def test_contract_teardown_detects_child_process_write(example_tree):
    # Exercise the actual module fixture against a temporary evidence tree.
    # Child processes do not inherit Python audit hooks; the hash check must
    # still report the change, without "repairing" the user's file afterward.
    from tests.contracts.test_example_fidelity_contract import _preserve_crossval_evidence

    _, _, evidence = example_tree
    fixture = _preserve_crossval_evidence.__wrapped__()
    next(fixture)
    try:
        subprocess.run(
            [sys.executable, "-c", "import pathlib,sys; "
             "pathlib.Path(sys.argv[1]).write_text('child output')", str(evidence)],
            check=True,
        )
    finally:
        with pytest.raises(AssertionError, match="retained.json"):
            next(fixture)
    assert evidence.read_text() == "child output"
