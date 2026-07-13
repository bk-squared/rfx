"""Import smoke tests for committed research subgrid examples.

A committed example that imports a module absent from the repository fails
with ``ModuleNotFoundError`` the moment a user runs it.  This guards the
Stage-2 disjoint prototype example, which previously imported a gate script
(``scripts.stage2_disjoint_full_physics_gate``) that is not bundled.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Final

RESEARCH_SUBGRID: Final = (
    pathlib.Path(__file__).resolve().parent.parent / "validation" / "research" / "subgrid"
)


def _load_example(path: pathlib.Path) -> ModuleType:
    """Execute an example module top level (does not run ``main``)."""
    spec = importlib.util.spec_from_file_location(f"_research_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_disjoint_prototype_example_imports_cleanly() -> None:
    module = _load_example(RESEARCH_SUBGRID / "12_subgrid_disjoint_prototype.py")
    assert hasattr(module, "main")


def test_material_validation_example_imports_cleanly() -> None:
    module = _load_example(RESEARCH_SUBGRID / "13_subgrid_material_validation.py")
    assert hasattr(module, "main")
    assert hasattr(module, "run_example")
