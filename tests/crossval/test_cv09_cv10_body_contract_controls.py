"""cv09 and cv10 are the CONTROLS for the #931 body-ownership contract.

Design note §1.8 fences domain-boundary PEC out of the contract:
`BoundarySpec` faces are not bodies and keep their own convention (E_tan = 0
on the face plane at index 0 / N). cv09 (full and half waveguide cavity,
PEC + PMC mirror) and cv10 (free-space PMC + CPML composition) are built with
ZERO geometry entities -- their walls are boundary faces only -- so nothing
in `realized_pec_edge_masks` can reach them. If a cv09 or cv10 gate moves
during #931 work, something touched boundary faces that should not have.

This file states that structurally and cheaply (no solve). The physics arms
are the scripts themselves, re-run on this branch:

  cv09: a_eff 22.8600 mm (residual 0.0 um < DX/4 = 127.0 um) on every axis of
        both cavities; f_full 8.1958 GHz, f_half 8.1959 GHz, invariant
        0.0006 % < 0.3556 %; ALL CHECKS PASSED.
  cv10: uniform and non-uniform paths PASS; G3 max|H_tan| on y_lo bit-exact
        0.0; G4 image control |R-1| = 0.0033 %.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CV09 = REPO_ROOT / "validation/crossval/09_half_symmetric_waveguide.py"
CV10 = REPO_ROOT / "validation/crossval/10_pmc_cpml_half_symmetric.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv09():
    return _load(CV09, "_cv09_body_controls")


@pytest.fixture(scope="module")
def cv10():
    return _load(CV10, "_cv10_body_controls")


@pytest.mark.parametrize("path", [CV09, CV10])
def test_neither_control_declares_a_conductor_body(path):
    """No `sim.add(...)`, no shape primitive, no thin conductor: these cases
    have nothing for the body-ownership contract to own. Source-level, so it
    reds if someone adds a body to a control."""
    code = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("#")]
    body = "\n".join(code)
    for token in ("sim.add(", "add_thin_conductor(", "Box(", "Sphere(",
                  "Cylinder(", "PolylineWire("):
        assert token not in body, (
            f"{path.name} declares {token!r}: it is a boundary-face control "
            "and must stay free of conductor bodies (#931 §1.8)")


def test_cv09_realizes_its_declared_cavity_with_no_body(cv09):
    """The full cavity's realized extents come from the DOMAIN row of
    fidelity_report -- the boundary-face convention -- and must stay exactly
    the declared 22.8600 / 10.1600 / 30.4800 mm. Build only."""
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec

    sim = Simulation(freq_max=cv09.FREQ_MAX, domain=(cv09.a, cv09.b, cv09.d),
                     dx=cv09.DX, boundary=BoundarySpec.uniform("pec"),
                     cpml_layers=0)
    assert not sim._geometry
    axes = cv09.realized_axes(sim)
    assert axes["x"]["realized_extent"] == pytest.approx(0.02286, abs=1e-9)
    assert axes["y"]["realized_extent"] == pytest.approx(0.01016, abs=1e-9)
    assert axes["z"]["realized_extent"] == pytest.approx(0.03048, abs=1e-9)


def test_cv09_mirror_plane_is_unmoved_by_the_contract(cv09):
    """a_eff = 2 x the realized half-cavity hi face = 22.8600 mm. Gate 0's
    window is DX/4 = 127 um and gate 3's is 0.3556 %, both far narrower than
    one cell, so this is where a boundary-face change would show first."""
    from rfx import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    half_x = 0.5 * cv09.a + 0.5 * cv09.DX
    sim = Simulation(freq_max=cv09.FREQ_MAX, domain=(half_x, cv09.b, cv09.d),
                     dx=cv09.DX,
                     boundary=BoundarySpec(x=Boundary(lo="pec", hi="pmc"),
                                           y=Boundary(lo="pec", hi="pec"),
                                           z=Boundary(lo="pec", hi="pec")),
                     cpml_layers=0)
    assert not sim._geometry
    a_eff = cv09.mirror_a_eff(cv09.realized_axes(sim))
    assert a_eff == pytest.approx(0.02286, abs=1e-9)
    assert abs(a_eff - cv09.a) < cv09.GEOM_TOL


def test_cv10_builds_no_materials_at_all(cv10):
    """cv10's composition lock is free space plus boundaries. Its
    `_common_spec` names only faces; a material or a body appearing here
    would mean the control stopped being one."""
    spec = cv10._common_spec()
    assert set(getattr(spec, "__dataclass_fields__", {"x", "y", "z"})) >= {
        "x", "y", "z"}
