"""Identifiability vs measurement bandwidth — the core Stage 2B result.

Demonstrates that whether ``eps_inf`` and a Debye ``delta_eps`` can be
separated depends on the *richness* of the S11 measurement (the KOH /
branch-ambiguity concern from the calibration literature): they trade off over
a narrow band but are separated by a broadband sweep.

Physics
-------
At a single frequency the S11 sensitivity to ``eps_inf`` and to ``delta_eps``
differ only by the complex Debye factor ``1/(1 + j w tau)``, which rotates the
(Re,Im) sensitivity by ``theta = arctan(w tau)``.  For a *tight* band well
below the relaxation (``w tau << 1``, ``theta -> 0``) the two sensitivities are
nearly collinear, so the Fisher information is ill-conditioned
(``cond ~ 4 / theta^2``).  A *wide* band that sweeps up toward ``w tau ~ 1``
samples a range of rotation angles and de-collinearizes the two parameters,
lowering the condition number.

The fixture is a thin, non-resonant dielectric slab (electrical length
<< lambda across the band) so that the Debye dispersion — not a fixture
half-wave resonance — dominates the frequency structure, making the effect
clean and monotonic.

Assertion: the wide-band Fisher condition number is SUBSTANTIALLY lower than
the narrow-band one (both printed).
"""
from __future__ import annotations

import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.materials.debye import DebyePole
from rfx.differentiable_material_fit import _poles_to_params
from rfx.calibration_identifiability import (
    s11_residual_fn,
    s11_jacobian,
    fisher_information,
    identifiability_report,
)


FREQ_MAX = 20e9
DX = 1.0e-3
DOMAIN = (0.010, 0.004, 0.004)   # 10 x 4 x 4 cells
NUM_PERIODS = 16.0

BASE_EPS_INF = 2.0
DELTA_EPS = 1.5
# Relaxation near 24 GHz (above the band top) so a tight low-frequency band
# sits at w tau << 1 (collinear) while the wide band sweeps up toward w tau ~ 1.
F_RELAX = 24e9
TAU = 1.0 / (2 * np.pi * F_RELAX)

NOISE_STD = 0.01  # i.i.d. Gaussian on Re/Im S11 (does not affect condition #)

# Narrow: tight (+-1%) band well below relaxation.  Wide: full 6-18 GHz sweep.
NARROW_FREQS = np.linspace(6e9 * 0.99, 6e9 * 1.01, 3)
WIDE_FREQS = np.linspace(6e9, 18e9, 13)


def _make_sim(eps_inf, debye_poles, lorentz_poles):
    """Thin (2 mm) non-resonant dielectric slab, one lumped ez port, PEC box."""
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX, boundary="pec")
    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles or None)
    sim.add(Box((0.004, 0.0, 0.0), (0.002, 0.004, 0.004)), material="dut")
    sim.add_port(
        position=(0.002, 0.002, 0.002),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=FREQ_MAX / 2, bandwidth=0.9, amplitude=1.0),
    )
    return sim


def _condition_number(freqs):
    """Fisher condition number for {eps_inf, delta_eps} at the given band.

    Builds the real S11 Jacobian for a 1-Debye model and restricts it to the
    ``eps_inf`` and ``delta_eps`` columns (``tau`` held known/fixed), so the
    2x2 Fisher information isolates the eps_inf/delta_eps trade-off.
    """
    poles = [DebyePole(delta_eps=DELTA_EPS, tau=TAU)]
    params = _poles_to_params(BASE_EPS_INF, poles, [])
    residual = s11_residual_fn(_make_sim, freqs, n_debye=1, n_lorentz=0,
                               num_periods=NUM_PERIODS)
    J = np.asarray(s11_jacobian(residual, params))
    J_sub = J[:, [0, 1]]  # columns: log(eps_inf), log(delta_eps)
    F = fisher_information(J_sub, NOISE_STD)
    report = identifiability_report(
        F, ["eps_inf", "delta_eps"], J=J_sub, noise_std=NOISE_STD)
    return report


def test_bandwidth_improves_identifiability():
    narrow = _condition_number(NARROW_FREQS)
    wide = _condition_number(WIDE_FREQS)

    cond_narrow = narrow.condition_number
    cond_wide = wide.condition_number
    ratio = cond_narrow / cond_wide

    print("\n[identifiability-bandwidth] {eps_inf, delta_eps}, tau fixed")
    print(f"[identifiability-bandwidth] narrow band  {NARROW_FREQS[0]/1e9:.2f}-{NARROW_FREQS[-1]/1e9:.2f} GHz "
          f"({len(NARROW_FREQS)} pts): cond = {cond_narrow:.2f}  eig = {narrow.eigenvalues}")
    print(f"[identifiability-bandwidth] wide band    {WIDE_FREQS[0]/1e9:.2f}-{WIDE_FREQS[-1]/1e9:.2f} GHz "
          f"({len(WIDE_FREQS)} pts): cond = {cond_wide:.2f}  eig = {wide.eigenvalues}")
    print(f"[identifiability-bandwidth] narrow/wide condition-number ratio = {ratio:.2f}x")

    # Both bands are individually well-conditioned enough to be identifiable,
    # but the wide band is SUBSTANTIALLY better conditioned.
    assert np.isfinite(cond_narrow) and np.isfinite(cond_wide)
    assert cond_wide < cond_narrow, (
        f"wide band not better conditioned: wide={cond_wide:.2f} narrow={cond_narrow:.2f}")
    assert ratio > 3.0, (
        f"bandwidth improvement not substantial: ratio {ratio:.2f}x "
        f"(narrow {cond_narrow:.2f}, wide {cond_wide:.2f})")
