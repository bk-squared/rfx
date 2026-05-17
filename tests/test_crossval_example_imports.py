"""Import smoke tests for committed crossval example scripts.

A committed example that imports a module absent from the repository fails
with ``ModuleNotFoundError`` the moment a user runs it.  This guards the
Stage-2 disjoint prototype example, which previously imported a gate script
(``scripts.stage2_disjoint_full_physics_gate``) that is not bundled.
"""

from __future__ import annotations

import importlib.util
import pathlib

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples" / "crossval"


def _load_example(name: str):
    """Execute an example module top level (does not run ``main``)."""
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"_crossval_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disjoint_prototype_example_imports_cleanly():
    """examples/crossval/12_subgrid_disjoint_prototype.py imports without error."""
    module = _load_example("12_subgrid_disjoint_prototype.py")
    assert hasattr(module, "main")
