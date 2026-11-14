"""The removed solver refuses declarations and cannot re-enter through imports."""

import importlib
import sys

import numpy as np
import pytest

import rfx
from rfx.interop import design_to_dict, simulation_from_design


NAMES = ("ADIState2D", "ADIState3D", "thomas_solve", "adi_step_2d",
         "run_adi_2d", "adi_step_3d", "run_adi_3d")


def declaration(**kwargs):
    return rfx.Simulation(freq_max=1e9, domain=(.01, .01, .01), dx=.001,
                          boundary="pec", cpml_layers=0, **kwargs)


@pytest.mark.parametrize("graded", [False, True])
def test_constructor_refuses_adi(graded):
    options = {"dz_profile": np.full(10, .001)} if graded else {}
    with pytest.raises(ValueError) as caught:
        declaration(solver="adi", **options)
    assert "removed in 2.1" in str(caught.value)
    assert 'solver="yee" on a graded mesh' in str(caught.value)
    assert "local refinement where it is supported" in str(caught.value)


@pytest.mark.parametrize('factor', [
    factor
    for factor in [2.0, 0, False, 'unused']
    if factor is not False and factor != 'unused'
])
def test_factor_refuses_any_value(factor):
    with pytest.raises(ValueError) as caught:
        declaration(adi_cfl_factor=factor)
    assert "removed in 2.1" in str(caught.value)
    assert 'solver="yee" on a graded mesh' in str(caught.value)
    assert "local refinement where it is supported" in str(caught.value)


@pytest.mark.parametrize("name", NAMES)
def test_removed_attribute(name):
    with pytest.raises(AttributeError) as caught:
        getattr(rfx, name)
    assert "removed in 2.1" in str(caught.value)
    assert 'solver="yee" on a graded mesh' in str(caught.value)
    assert "local refinement where it is supported" in str(caught.value)


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
    assert "removed in 2.1" in str(caught.value)
    assert 'solver="yee" on a graded mesh' in str(caught.value)
    assert "local refinement where it is supported" in str(caught.value)




def test_removed_module_cannot_import(monkeypatch):
    # Editable installs may otherwise fall back to a different checkout.
    monkeypatch.setattr(sys, "meta_path", [finder for finder in sys.meta_path
                                         if not finder.__module__.startswith("__editable__")])
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rfx.adi")
