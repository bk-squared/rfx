"""Gate: volumetric PEC scatterers reflect on the NU waveguide path (RESOLVED).

Discovered 2026-06-25 while scoping a non-uniform external-E4 iris comparison:
an identical inductive iris (a symmetric pair of PEC fins, ``sigma=1e7`` > the
1e6 PEC threshold, blocking ~60% of the WR-90 width) appeared to reflect on the
UNIFORM Yee path (``|S11|`` ~ 0.78-2.1) but to be ABSENT on the non-uniform
(graded-``dy``) path (``|S11|`` ~ 0, ``|S21|`` ~ 1). The first framing blamed
the rasterized ``pec_mask`` "not taking effect in the NU scan / NU is
dielectric-only for scatterers" -- THAT FRAMING WAS WRONG.

ROOT CAUSE (verified experimentally 2026-06-25, fixed same day): the interior
PEC IS applied to the NU device run. The NU two-run S-matrix *vacuum reference*
run, however, retained the device's interior ``pec_mask`` -- the vacuum
override (``run_nonuniform_path(eps_override=vacuum_eps,
sigma_override=vacuum_sigma)``) replaced only eps_r/sigma and never neutralized
``pec_mask``. So the reference was physically identical to the device, the two
flux DFTs were bit-identical, ``(device - reference) = 0``, and ``S11 = 0`` for
ANY PEC reflector regardless of the extractor formula. The uniform path never
had this bug -- it builds the reference with ``dielectric_shapes=[]`` plus
boundary-only PEC (see ``rfx/api/_sparams.py``, which warns inline that
applying the interior ``pec_mask`` to the reference makes it identical to the
device and forces ``S11=0``).

FIX: ``run_nonuniform_path(..., strip_interior_pec=True)`` drops the
interior-geometry ``pec_mask`` from the NU vacuum reference while KEEPING the
boundary y/z guide walls (those are enforced via ``pec_faces`` / ``apply_pec``,
not ``pec_mask``). After the fix the NU iris recovers to ``|S11|`` ~ 1.4-1.6 (a
full NU short to ``|S11|`` ~ 0.6-2.1), matching the uniform reflector class, and
the empty NU guide still reads ``|S11|`` ~ 0 (no spurious reflection, boundary
walls intact). Documented in ``docs/guides/support_matrix.{json,md}``.

Both tests are now hard gates: the uniform witness confirms the iris fixture is
a valid strong reflector, and the NU gate confirms the PEC iris reflects on the
graded-``dy`` path within the uniform reflector class.
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
def test_nonuniform_pec_iris_reflects():
    """Gate (RESOLVED 2026-06-25): the NU PEC iris reflects like the uniform one.

    Root cause was the NU two-run S-matrix vacuum reference retaining the
    device's interior PEC mask (vacuum override replaced only eps/sigma, never
    pec_mask) → device and reference DFTs bit-identical → S11=0 for any
    reflector. Fixed by ``run_nonuniform_path(..., strip_interior_pec=True)`` on
    the reference (drops interior PEC, keeps the boundary guide walls). The
    iris now recovers to |S11| ~ 1.4-1.6 on the graded-dy path, in the same
    strong-reflector class as the uniform witness."""
    s11_nu = _iris_s11_max(nonuniform=True)
    assert s11_nu > 0.2, (
        f"NU PEC iris |S11|max={s11_nu:.3f} <= 0.2 — iris not reflecting; the "
        "NU vacuum-reference interior-PEC-strip fix has regressed"
    )
    # Same strong-reflector class as the uniform witness (both are the
    # identical iris geometry); the NU iris must land within a factor of the
    # uniform max, not collapse toward 0.
    s11_uni = _iris_s11_max(nonuniform=False)
    assert s11_nu > 0.4 * s11_uni, (
        f"NU PEC iris |S11|max={s11_nu:.3f} not in the uniform reflector class "
        f"(uniform |S11|max={s11_uni:.3f}) — NU reflection is weak/wrong"
    )
