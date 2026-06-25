"""Sentinel: volumetric PEC scatterers do NOT reflect on the NU waveguide path.

Discovered 2026-06-25 while scoping a non-uniform external-E4 iris comparison.
An identical inductive iris (a symmetric pair of PEC fins, ``sigma=1e7`` > the
1e6 PEC threshold, blocking ~60% of the WR-90 width) reflects strongly on the
UNIFORM Yee path (``|S11|`` ~ 0.78-0.95) but is effectively ABSENT on the
non-uniform (graded-``dy``) path (``|S11|`` ~ 0, ``|S21|`` ~ 1 -- the iris
neither reflects nor blocks transmission). Dielectric Boxes DO rasterize +
reflect on the NU path (``tests/test_waveguide_nu_flux.py`` and the rung-1
broad-E5 Airy envelope), so the gap is PEC-scatterer-specific: the
geometry-derived ``pec_mask`` from ``rasterize_geometry`` on non-uniform coords
is not taking effect in the NU scan (root cause not yet isolated -- R2-STOP, no
blind fix).

Consequence: the NU waveguide lane is currently DIELECTRIC-ONLY for scatterers;
metal obstacles (irises, posts, septa) silently vanish. Documented in
``docs/guides/support_matrix.{json,md}``.

The uniform witness PASSES (confirms the iris geometry/material is a valid
strong reflector). The NU sentinel is ``xfail(strict=True)``: when the
pec_mask-on-NU path is fixed it will XPASS and trip CI, forcing this doc + the
support-matrix note to be promoted to a hard gate.
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.auto_config import smooth_grading
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

_A, _B, _FMAX = 0.02286, 0.01016, 12e9
_FREQS = jnp.linspace(8.2e9, 12.4e9, 5)
_NP = 20


def _graded_dy(ratio: float = 2.0, base: float = 0.75e-3):
    n = int(round(_A / base))
    x = np.linspace(-1, 1, n)
    w = 1.0 + (ratio - 1.0) * np.abs(x)
    return smooth_grading(w / w.sum() * _A, max_ratio=1.3)


def _iris_sim(*, nonuniform: bool):
    """WR-90 with a symmetric inductive PEC iris (~60% width blocked)."""
    dx = 1.5e-3
    nx = int(round(0.100 / dx))
    kw = {"dy_profile": _graded_dy()} if nonuniform else {}
    sim = Simulation(
        freq_max=_FMAX, domain=(nx * dx, _A, _B), dx=dx,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8, **kw,
    )
    sim.add_material("metal", eps_r=1.0, sigma=1e7)  # sigma > 1e6 -> PEC
    xc = 0.5 * nx * dx
    fin = 0.30 * _A
    sim.add(Box((xc - 1e-3, 0.0, 0.0), (xc + 1e-3, fin, _B)), material="metal")
    sim.add(Box((xc - 1e-3, _A - fin, 0.0), (xc + 1e-3, _A, _B)), material="metal")
    for x0, d, nm in ((0.015, "+x", "left"), (nx * dx - 0.015, "-x", "right")):
        sim.add_waveguide_port(
            x0, direction=d, mode=(1, 0), mode_type="TE",
            freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
            reference_plane=(0.020 if d == "+x" else nx * dx - 0.020), name=nm,
        )
    return sim


def _iris_s11_max(*, nonuniform: bool) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _iris_sim(nonuniform=nonuniform).compute_waveguide_s_matrix(
            num_periods=_NP, normalize="flux",
        )
    return float(np.abs(np.asarray(res.s_params)[0, 0, :]).max())


@pytest.mark.slow
def test_uniform_pec_iris_reflects():
    """Witness: the inductive iris reflects strongly on the UNIFORM path
    (confirms the geometry/material is a valid strong reflector)."""
    s11 = _iris_s11_max(nonuniform=False)
    assert s11 > 0.5, (
        f"uniform PEC iris |S11|max={s11:.3f} — expected strong reflection; "
        "the witness fixture is broken, not the NU path"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Volumetric PEC scatterers do not reflect on the NU waveguide path: "
        "the same iris that gives |S11|~0.9 on the uniform path gives |S11|~0 "
        "on the graded-dy path (pec_mask from rasterize_geometry not effective "
        "in the NU scan; dielectric Boxes DO work). NU is dielectric-only for "
        "scatterers. XPASS => the PEC-on-NU path was fixed: promote this to a "
        "hard gate and update docs/guides/support_matrix.* + nu known-limits."
    ),
)
def test_nonuniform_pec_iris_reflects():
    """Sentinel (xfail-strict): the NU PEC iris SHOULD reflect like the uniform
    one. It does not today (|S11|~0) — documenting the dielectric-only limit."""
    s11 = _iris_s11_max(nonuniform=True)
    assert s11 > 0.2, f"NU PEC iris |S11|max={s11:.3f} <= 0.2 — iris not reflecting"
