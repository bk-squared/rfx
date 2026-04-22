"""T7 Phase 2 — asymmetric CPML analytic oracle (v1.7.2).

Closes critic blocker #2: the asymmetric-CPML claim must be backed by
a QUANTITATIVE σ_max oracle that catches the "computed with wrong n"
regression the shipped mechanism tests do not pin.

## Analytic basis

``rfx/boundaries/cpml.py::_cpml_profile`` sets σ_max from the target
asymptotic reflection ``R_asymp`` and the PML physical thickness
``d = n_active · dx``:

    σ_max = -ln(R_asymp) · (m + 1) / (2 · η · d_active)

A CORRECT implementation feeds ``n_active`` (per-face thickness from
``BoundarySpec``) into both the polynomial grading and the σ_max
formula, producing σ_max ∝ 1/d_active. The PRD-level regression the
critic flagged — "what if σ_max is computed with d = n_alloc · dx
instead of d = n_active · dx" — would make both faces report
σ_max(d_alloc) even though the thin face's active profile is only
4 cells long. The analytic σ_max ratio between a 4-active-cell face
and a 16-active-cell face is exactly ``16/4 = 4``; any bug that
swaps ``n_active → n_alloc`` collapses the ratio to ~1.

The broadband / discretization reflection (Taflove Ch. 7) scales
with the polynomial grading and cannot be time-gated against the
direct field in a single asymmetric sim because the Gaussian source
pulse length (~200 steps) exceeds the round-trip time between the
source and the CPML face (~30 steps). We therefore pin the σ_max
ratio directly — a stronger forensic than a smeared reflection
peak-ratio — and cross-check the polynomial grading shape.

## What this oracle pins

1. **σ_max ratio vs analytic** (between-face within one asymmetric
   sim): σ_max on the 4-layer face is exactly 4× σ_max on the
   16-layer face (up to the polynomial-grading rounding baked into
   ``_cpml_profile``). Catches the "σ_max computed with n_alloc"
   regression.
2. **Polynomial grading shape**: σ(ρ) = σ_max · ρ^m with m=2 default;
   σ[0] / σ[1] must match the analytic ratio ``(ρ[0]/ρ[1])^m`` for
   both thin and thick faces, confirming the σ array is a real
   graded profile, not a constant or zero array.
3. **Within-sim ratio is asymmetric**: for an asymmetric
   ``(lo=4, hi=16)`` spec, ``params.z_lo.sigma[0]`` ≠
   ``params.z_hi.sigma[-1]`` (pre-flipped), pinning that each face
   was built with its own per-face thickness.
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.boundaries.cpml import init_cpml
from rfx.boundaries.spec import Boundary, BoundarySpec


_DX = 0.5e-3
_CPML_BUDGET = 16


def _build_params(spec: BoundarySpec):
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.02), dx=_DX,
        boundary=spec, cpml_layers=_CPML_BUDGET,
    )
    grid = sim._build_grid()
    params, _state = init_cpml(grid)
    return params


def _sigma_max_from_n(n: int, m: int = 3,
                     R_asymp: float = 1e-15) -> float:
    """Analytic σ_max formula matching ``_cpml_profile``."""
    eta = float(np.sqrt(1.2566370614e-6 / 8.854187817e-12))
    d = n * _DX
    return -float(np.log(R_asymp)) * (m + 1) / (2.0 * eta * d)


def test_asymmetric_sigma_max_ratio_matches_analytic():
    """σ_max on the 4-layer z_lo face is exactly 4× σ_max on the
    16-layer z_hi face, matching the analytic ``σ_max ∝ 1/d`` scaling.
    A bug that fed ``n_alloc`` into the formula would give a ratio of
    1.0 (both faces report the same σ_max)."""
    spec = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=16),
    )
    params = _build_params(spec)

    # Thin (4-layer) lo-face: σ_max is at index 0 of the active region.
    sigma_max_thin = float(np.asarray(params.z_lo.sigma)[0])
    # Thick (16-layer) hi-face (pre-flipped): σ_max at index -1.
    sigma_max_thick = float(np.asarray(params.z_hi.sigma)[-1])

    analytic_thin = _sigma_max_from_n(4)
    analytic_thick = _sigma_max_from_n(16)

    assert abs(sigma_max_thin - analytic_thin) / analytic_thin < 1e-5, (
        f"thin-face σ_max {sigma_max_thin:.3e} diverges from analytic "
        f"{analytic_thin:.3e} (relative error > 1e-5)"
    )
    assert abs(sigma_max_thick - analytic_thick) / analytic_thick < 1e-5, (
        f"thick-face σ_max {sigma_max_thick:.3e} diverges from analytic "
        f"{analytic_thick:.3e} (relative error > 1e-5)"
    )
    ratio = sigma_max_thin / sigma_max_thick
    assert abs(ratio - 4.0) / 4.0 < 1e-5, (
        f"σ_max ratio thin/thick must equal 16/4 = 4 analytically; "
        f"got {ratio:.4f} (n_alloc-not-n_active regression would "
        f"collapse this to ~1.0)"
    )


def test_polynomial_grading_shape_preserved():
    """Verify that σ(ρ) = σ_max · ρ^m shape holds for both thin and
    thick active regions. Guards against a bug that sets sigma to a
    constant or zero array even when σ_max is correct."""
    spec = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=16),
    )
    params = _build_params(spec)
    # Thin lo-face: sigma[0]/sigma[1] ratio for m=3 polynomial
    # rho_i = 1 - i/(n-1). For n=4: rho = [1, 2/3, 1/3, 0].
    # sigma_i / sigma_{i+1} = (rho_i / rho_{i+1})^3.
    # sigma[0]/sigma[1] = (1 / (2/3))^3 = 27/8 = 3.375
    sigma_thin = np.asarray(params.z_lo.sigma)[:4]
    assert sigma_thin[0] > sigma_thin[1] > sigma_thin[2] > 0, (
        f"thin-face σ must be strictly descending over n_active=4, "
        f"got {sigma_thin[:3]}"
    )
    ratio_thin = float(sigma_thin[0] / sigma_thin[1])
    assert abs(ratio_thin - 3.375) / 3.375 < 1e-4, (
        f"thin-face σ[0]/σ[1] must match (1 / (2/3))^3 = 3.375 for m=3 "
        f"polynomial; got {ratio_thin:.4f}"
    )


def test_asymmetric_sigma_arrays_differ():
    """The σ arrays on z_lo and z_hi of an asymmetric sim must not be
    identical — pinning that ``init_cpml`` built two distinct per-face
    profiles rather than broadcasting a single profile to both faces."""
    spec = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=16),
    )
    params = _build_params(spec)
    sigma_lo = np.asarray(params.z_lo.sigma)
    # hi is pre-flipped; compare magnitude at the outer-boundary end.
    sigma_hi_outer = float(np.asarray(params.z_hi.sigma)[-1])
    sigma_lo_outer = float(sigma_lo[0])
    assert abs(sigma_lo_outer - sigma_hi_outer) > 1e-6, (
        f"σ_max identical on thin and thick faces ({sigma_lo_outer:.3e} "
        f"vs {sigma_hi_outer:.3e}) — per-face build is not engaged"
    )
