"""End-to-end: on a uniform mesh the NU solve reduces to the uniform solve.

The missing anchor behind #565 (and behind how #562's F1 shipped past CI): every
committed non-uniform test gates a FREQUENCY, an S-parameter (reference-run
normalized), or a smoother's own output. None ran a solve on both builders and
compared the fields, so a half-cell displacement of every smoothed voxel was
invisible to the suite.

#562 made that comparison possible: a uniform mesh expressed as profiles now
produces the same grid, so the two paths must produce the same physics.

They do — to 2.7e-5 relative over the whole time series, 9.5e-6 with subpixel
smoothing on — but ONLY after accounting for a normalization difference that is
NOT a bug and IS worth pinning, because it is what made the divergence in #565
look like an instability:

    the same ``add_source(position, component)`` call means a FIELD INCREMENT on
    the uniform path and a CURRENT IN AMPERES on the non-uniform path

so the NU probe trace is the uniform one times ``dt / (eps0 * dV)`` — about
2.15e8 at dx = 1 mm. The NU convention is deliberate (see
``rfx.nonuniform.make_current_source``: resolution-independent injected power,
Meep's approach); the uniform path's is the legacy one. This test pins the
RELATIONSHIP rather than either side, so changing either normalization fails
here and says why.

What #565 reported as "NU + subpixel_smoothing diverges wildly on a trivial
closed box (uniform max|E| 0.0078 vs NU ~1.5e6)" is this factor. Measured here:
correlation 1.00000000, peak at the same step, tail amplitude equal to head
amplitude — no growth anywhere, on either builder.

Scope, stated rather than implied: this pins the UNIFORM-PROFILE reduction. A
genuinely graded mesh computes ``dV`` from the forward cell rather than the
node's dual volume (pre-existing, unchanged by #562), so the factor there is not
expected to be this exact expression and is not covered here.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.core.yee import EPS_0
from rfx.geometry.csg import Box
from rfx.grid import C0

NA, NB, NZ = 45, 39, 4
STEPS = 1200

# Deliberately NOT marked slow: the whole point is that this runs in the default
# lane. An anchor the fast suite deselects would have missed #562's F1 exactly
# the way the existing suite did. Measured 6.5 s for all three cases.


def _f110(dx: float) -> float:
    return (C0 / 2) * np.sqrt((1 / (NA * dx)) ** 2 + (1 / (NB * dx)) ** 2)


def _expected_factor(boundary: str, dt: float, dx: float) -> float:
    """The documented cross-path amplitude ratio, per boundary condition.

    The NU path divides by the cell volume; the uniform path never does, and
    what it adds depends on the boundary (`rfx/runners/uniform.py`): a closed
    PEC domain takes a raw field add, an open domain goes through
    `make_j_source` and is already `Cb = dt/eps` normalized. So the ratio is
    `dt/(eps0*dV)` closed and `1/dV` open — a factor `dt/eps0` = 4.64x apart at
    dx = 1 mm. The first version of this test covered only PEC and the
    `add_source` docstring stated the PEC factor unconditionally (#570 review,
    finding 9); the open case is rfx's more common one.
    """
    dV = dx ** 3
    return (dt / (EPS_0 * dV)) if boundary == "pec" else (1.0 / dV)


def _run(dx: float, *, nonuniform: bool, smoothing: bool, boundary: str = "pec"):
    f0 = _f110(dx)
    profiles = (dict(dx_profile=np.full(NA, dx), dy_profile=np.full(NB, dx))
                if nonuniform else {})
    sim = Simulation(freq_max=2.5 * f0, domain=(NA * dx, NB * dx, NZ * dx),
                     dx=dx, boundary=boundary,
                     cpml_layers=(0 if boundary == "pec" else 8), **profiles)
    if smoothing:
        sim.add_material("slab", eps_r=4.0)
        # interface deliberately inside a voxel so the smoother engages
        sim.add(Box((0.0, 0.0, 0.3 * dx), (NA * dx, NB * dx, 2.0 * dx)),
                material="slab")
    # z = 3*dx, NOT NZ*dx/2: the latter is EXACTLY the slab's upper face, so the
    # factor prediction would ride on Box's exclusive-upper-bound convention and
    # a future inclusive rule would red this test for an unrelated reason
    # (#570 review, finding 10). 3*dx is unambiguously vacuum.
    sim.add_source((NA * dx / 3, NB * dx / 3, 3.0 * dx), "ez",
                   waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((2 * NA * dx / 3, 2 * NB * dx / 3, 3.0 * dx), "ez")
    result = sim.run(n_steps=STEPS, compute_s_params=False,
                     skip_preflight=True, subpixel_smoothing=smoothing)
    return np.asarray(result.time_series, dtype=float).ravel(), float(result.dt)


_CASES = [
    pytest.param(False, "pec", id="plain-pec"),
    pytest.param(True, "pec", id="subpixel-pec"),
    pytest.param(False, "cpml", id="plain-cpml"),
    pytest.param(
        True, "cpml", id="subpixel-cpml",
        marks=pytest.mark.xfail(strict=True, reason=(
            "NU + subpixel_smoothing does NOT reduce to the uniform path on an "
            "OPEN boundary (#582). Measured, and record-length "
            "independent at 600/1200/2400 steps: amplitude factor off by "
            "0.1804 %, waveform residual 1.978e-02, 1-corr 2.8e-04 — against "
            "~1e-5 residual for the other three boundary x smoothing "
            "combinations. strict=True so this XPASSes and hard-fails the suite "
            "the day it is fixed, rather than silently passing.")),
    ),
]


@pytest.mark.parametrize("smoothing,boundary", _CASES)
def test_nu_solve_reduces_to_uniform_solve_up_to_source_normalization(
        smoothing, boundary):
    """Same mesh, same geometry, both builders: same physics, one known factor.

    Run for BOTH boundary conditions, because the uniform path's source
    normalization differs between them and the factor therefore does too.
    """
    dx = 1e-3
    uni, dt_u = _run(dx, nonuniform=False, smoothing=smoothing, boundary=boundary)
    nu, dt_n = _run(dx, nonuniform=True, smoothing=smoothing, boundary=boundary)

    assert dt_u == pytest.approx(dt_n, rel=1e-12), (dt_u, dt_n)
    assert np.abs(uni).max() > 1e-6, "uniform leg produced no field"

    # the two traces are the SAME waveform: identical shape, identical timing.
    # Threshold accommodates both boundaries — measured deviation from 1 is
    # <1e-10 (pec) and 3.7e-9 (cpml, whose absorber implementations differ in
    # detail between the two paths); 1e-7 leaves ~27x margin on the worse case.
    assert np.corrcoef(uni, nu)[0, 1] > 1 - 1e-7
    assert int(np.argmax(np.abs(nu))) == int(np.argmax(np.abs(uni)))

    # neither leg grows: a diverging run would break this, and #565 read this
    # amplitude ratio as divergence
    for tag, trace in (("uniform", uni), ("nonuniform", nu)):
        head = np.abs(trace[:len(trace) // 4]).max()
        tail = np.abs(trace[3 * len(trace) // 4:]).max()
        assert tail <= 5.0 * max(head, 1e-30), (
            f"{tag} leg grows: head {head:.4g} -> tail {tail:.4g}")

    # and the amplitude ratio is exactly the source-normalization factor
    scale = float(np.dot(uni, nu) / np.dot(uni, uni))
    predicted = _expected_factor(boundary, dt_u, dx)
    assert scale == pytest.approx(predicted, rel=1e-5), (
        f"cross-path amplitude ratio {scale:.6g} does not match the documented "
        f"{boundary} factor {predicted:.6g} — a source normalization changed; "
        f"see rfx.nonuniform.make_current_source (current in amperes, divided "
        f"by cell volume) versus rfx/runners/uniform.py's boundary-dependent "
        f"choice of make_source (raw add) vs make_j_source (Cb-normalized)")

    residual = float(np.abs(nu - scale * uni).max() / np.abs(nu).max())
    # measured: 1.5e-5 (pec), 8.2e-6 (pec + smoothing), 1.1e-4 (cpml).
    # cpml + smoothing measures 2.0e-2 and is xfail(strict) above, not covered
    # by this bound — see _CASES.
    assert residual < 3e-4, (
        f"after removing the known source-normalization factor the two paths "
        f"still differ by {residual:.3e} of full scale — the NU solve is not "
        f"reducing to the uniform solve on an identical mesh")


def test_source_normalization_factor_is_cell_size_dependent_as_documented():
    """The factor is `dt/(eps0*dV)`, so it MUST move with dx — a constant would
    mean something else is producing the offset.

    Two cell sizes an octave apart; `dt` follows the Courant condition, so the
    factor changes by 4x while the physics stays the same problem scaled.
    """
    ratios = {}
    for dx in (1e-3, 0.5e-3):
        uni, dt = _run(dx, nonuniform=False, smoothing=False)
        nu, _ = _run(dx, nonuniform=True, smoothing=False)
        scale = float(np.dot(uni, nu) / np.dot(uni, uni))
        ratios[dx] = scale / _expected_factor("pec", dt, dx)
        assert ratios[dx] == pytest.approx(1.0, rel=1e-5), (dx, ratios[dx])
    # the raw factors differ by 4x (dt halves, dV falls 8x) — assert the
    # comparison is not vacuous
    assert len(set(round(v, 6) for v in ratios.values())) == 1
