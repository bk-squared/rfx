"""The removed solver refuses declarations and cannot re-enter through imports."""

import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

import rfx
from rfx.interop import design_to_dict, simulation_from_design
from rfx.model.solver_admission import ADI_REMOVED


NAMES = ("ADIState2D", "ADIState3D", "thomas_solve", "adi_step_2d",
         "run_adi_2d", "adi_step_3d", "run_adi_3d")
DELETED = ("rfx.adi", "rfx.stepping.adi", "rfx.runners.adi",
           "rfx.model.adi_admission", "rfx._adi_notice")


def declaration(**kwargs):
    return rfx.Simulation(freq_max=1e9, domain=(.01, .01, .01), dx=.001,
                          boundary="pec", cpml_layers=0, **kwargs)


@pytest.mark.parametrize("graded", [False, True])
def test_constructor_refuses_adi(graded):
    options = {"dz_profile": np.full(10, .001)} if graded else {}
    with pytest.raises(ValueError) as caught:
        declaration(solver="adi", **options)
    assert str(caught.value) == ADI_REMOVED


@pytest.mark.parametrize("factor", [2.0, 0, False, "unused"])
def test_factor_refuses_any_value(factor):
    with pytest.raises(ValueError) as caught:
        declaration(adi_cfl_factor=factor)
    assert str(caught.value) == ADI_REMOVED


@pytest.mark.parametrize("name", NAMES)
def test_removed_attribute(name):
    with pytest.raises(AttributeError) as caught:
        getattr(rfx, name)
    assert ADI_REMOVED in str(caught.value)


def test_old_yee_document_ignores_factor():
    document = design_to_dict(declaration())
    assert "adi_cfl_factor" not in document["solver"]
    document["solver"]["adi_cfl_factor"] = 5.0
    rebuilt = simulation_from_design(document)
    del document["solver"]["adi_cfl_factor"]
    assert design_to_dict(rebuilt) == document
    assert not hasattr(rebuilt, "_adi_cfl_factor")


def test_old_adi_document_refuses():
    document = design_to_dict(declaration())
    document["solver"].update(solver="adi", adi_cfl_factor=2.0)
    with pytest.raises(ValueError) as caught:
        simulation_from_design(document)
    assert ADI_REMOVED in str(caught.value)


def test_deleted_modules_have_no_importers():
    root = Path(rfx.__file__).parent
    for module in DELETED:
        assert not (root.parent / (module.replace(".", "/") + ".py")).exists()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        package = ".".join(path.relative_to(root.parent).parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                imports = [base, *(base + "." + alias.name for alias in node.names)]
            else:
                continue
            assert not set(imports).intersection(DELETED), (path, node.lineno, imports)


def test_removed_module_cannot_import(monkeypatch):
    # Editable installs may otherwise fall back to a different checkout.
    monkeypatch.setattr(sys, "meta_path", [finder for finder in sys.meta_path
                                         if not finder.__module__.startswith("__editable__")])
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rfx.adi")
