"""The coax line carries TEM at the phase constant its fill implies.

A homogeneously filled PEC-bounded coax carries TEM at ``beta =
omega sqrt(eps)/c`` whatever the staircase does to the cross-section — the
staircase moves ``Z_TEM``, not ``beta``. This lane did not: with the conductors
written as ``sigma = PEC_SIGMA`` the fitted phase constant sat 8-18 % high and
up to 15 % of the column power went missing, both shrinking with the mesh in a
way that looked like ordinary under-resolution and was not.

``rfx/core/yee.py`` applies ``materials.sigma`` per NODE to the three E
components co-indexed with that node, so a conductor cell damped only its three
plus-side edges while the edges entering it from the minus side stayed live on
the neighbouring dielectric node. Realizing the same cells through
``realized_pec_edge_masks`` — walls on BOTH faces, the lattice ownership
contract — is what this file pins.

Marked ``slow_physics``: the thru is a two-drive FDTD and the two Z0 legs are
one-port solves, so this is minutes on CPU rather than the fast lane's seconds.

Two sets of before-numbers exist and they are NOT interchangeable, so each is
labelled with the board it was measured on:

BEFORE, on THESE fixtures, from ``scripts/diagnostics/coax_conductor_mutation.py``
(the sigma realization put back with every helper call left in place):
  thru      beta 16.62 % (unwrapped S21 phase) / 24.78 % (pencil) from
            omega sqrt(eps_r)/c; column power [0.92406, 1.03789]
  25 ohm    Z0 57.176 ohm      100 ohm   Z0 57.539 ohm
BEFORE, on the DIAGNOSTIC's 60 mm board at 4 annulus cells, from
``scripts/diagnostics/coax_shell_seal_diagnostic.py`` arm 0 (branch
``meas/coax-chain-battery``):
  thru      beta/beta_analytic 1.1867 (pencil) / 1.1795 (S21 phase),
            max column power 0.8786
  25 ohm    Z0 46.169 ohm      100 ohm   Z0 48.697 ohm

FIXTURE UNDER REVIEW. These boards realize 3.789 annulus cells and this file
asserts a 1 % bar on that single mesh, which the v2 accuracy bar does not allow
("a comparison made on a mesh that has not been shown to converge says nothing
either way"). ``scripts/diagnostics/coax_conductor_oracle_ladder.py`` runs these
same four comparisons across 3.789/4/6/9 annulus cells on this board and on the
diagnostic's 60 mm one; at the time of writing the 60 mm board is inside every
bar at 3.789 cells (beta 0.01 % / 0.30 %, column power [0.98153, 0.99342]) and
this board is outside all of them at every rung. The bars below are unchanged
and left asserting pending that decision.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.coaxial_port import (
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    coaxial_tem_characteristic_impedance,
)
from rfx.sources.sources import GaussianPulse

C0 = 299792458.0

# The smallest two-feed line that settles: the board the committed AD gate uses
# (tests/unit/autodiff/test_coax_two_port_ad.py), whose own concrete twin reads
# settling_db [-61.36, -63.88] at this step count.
THRU_DOMAIN = (0.008, 0.008, 0.012)
THRU_STEPS = 600
THRU_PROBES = dict(probe_count=3, probe_start_cells=4, probe_spacing_cells=2)
# A band, not a single bin: the phase-slope estimate needs neighbouring bins and
# a one-bin gate cannot see a trend.
FREQS = np.linspace(6.0e9, 12.0e9, 13)

# The one-port board the committed one-port AD gate uses.
LOAD_DOMAIN = (0.008, 0.008, 0.020)
LOAD_STEPS = 1500
LOAD_PROBES = 9
LOADS_OHM = (25.0, 100.0)

BETA_FRAC = 0.01          # the v2 bar's 1 % on a frequency-like quantity
COLUMN_POWER_MAX = 1.02   # the v2 bar's passivity number
Z0_FRAC = 0.01


def _sim(domain):
    sim = Simulation(domain=domain, freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port((domain[0] / 2.0, domain[1] / 2.0, domain[2] / 2.0),
                         face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


def _beta_from_s21_phase(freqs, s21, l12):
    """``beta`` from the slope of the unwrapped ``S21`` phase over the distance
    between the reference planes.

    On a matched thru ``S21 ~ exp(-gamma L12)``, so ``angle(S21) = -beta L12``
    modulo 2 pi. The absolute branch is unknown; its SLOPE is not, as long as
    the phase turns less than pi between neighbouring bins. This never touches
    the matrix-pencil fit the extractor uses, so it is a second reading of the
    same line and not a restatement of the first.
    """
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    ph = np.unwrap(np.angle(np.asarray(s21)))
    steps = np.abs(np.diff(ph))
    assert steps.max() < math.pi, (
        f"the S21 phase turns {steps.max():.3f} rad between neighbouring bins, "
        "past the unwrapping limit; the band is sampled too coarsely for this "
        "estimate to mean anything")
    slope = np.polyfit(omega, ph, 1)[0]
    v_p = -float(l12) / slope
    return omega / v_p, v_p


@pytest.fixture(scope="module")
def thru():
    sim = _sim(THRU_DOMAIN)
    res = sim.compute_coaxial_two_port(n_steps=THRU_STEPS, freqs=FREQS, **THRU_PROBES)
    return res


@pytest.mark.slow_physics
def test_the_thru_carries_tem_at_the_phase_constant_its_fill_implies(thru):
    freqs = np.asarray(thru.freqs, dtype=float)
    s21 = np.asarray(thru.s_params)[1, 0, :]
    planes = np.asarray(thru.reference_planes, dtype=float)
    l12 = float(abs(planes[0] - planes[1]))
    beta_analytic = 2.0 * np.pi * freqs * math.sqrt(float(PTFE_EPS_R)) / C0

    beta_phase, v_p = _beta_from_s21_phase(freqs, s21, l12)
    worst = float(np.max(np.abs(beta_phase / beta_analytic - 1.0)))
    assert worst <= BETA_FRAC, (
        f"the phase constant from the unwrapped S21 phase is {worst*100:.2f} % "
        f"from omega*sqrt({float(PTFE_EPS_R)})/c (phase velocity {v_p:.6e} m/s, "
        f"analytic {C0/math.sqrt(float(PTFE_EPS_R)):.6e}); a homogeneously "
        "filled PEC-bounded line has no other phase constant to carry")

    # The extractor's own fit, which is a different estimator on the same field.
    gamma = np.asarray(thru.gamma)
    beta_fit = np.imag(gamma)
    beta_fit = beta_fit.mean(axis=tuple(range(beta_fit.ndim - 1)))
    worst_fit = float(np.max(np.abs(beta_fit / beta_analytic - 1.0)))
    assert worst_fit <= BETA_FRAC, (
        f"the matrix-pencil phase constant is {worst_fit*100:.2f} % from the "
        "analytic one")


@pytest.mark.slow_physics
def test_the_thru_keeps_its_power(thru):
    S = np.asarray(thru.s_params)
    col = np.sum(np.abs(S) ** 2, axis=0)
    assert float(col.max()) <= COLUMN_POWER_MAX, (
        f"max column power {float(col.max()):.5f} exceeds {COLUMN_POWER_MAX} on "
        "a passive line")
    # The missing-power symptom was a DEFICIT, which a passivity bound alone
    # cannot see: 0.8786 before. Both sides are bounded here.
    assert float(col.min()) >= 1.0 - (COLUMN_POWER_MAX - 1.0), (
        f"min column power {float(col.min()):.5f} — the through line is losing "
        "power that neither reflects nor transmits")


@pytest.mark.slow_physics
@pytest.mark.parametrize("load_ohm", LOADS_OHM)
def test_a_resistive_load_reads_the_declared_characteristic_impedance(load_ohm):
    """``Z0 = R (1 - Gamma) / (1 + Gamma)`` from a known load. It is the second
    line constant: a realization that fixed ``beta`` by moving the geometry
    would show up here."""
    res = _sim(LOAD_DOMAIN).compute_coaxial_line_reflection(
        termination="matched", dut_impedance=load_ohm, n_steps=LOAD_STEPS,
        freqs=FREQS, probe_count=LOAD_PROBES)
    z0 = np.asarray(res.z0_numerical_ohm)
    finite = np.isfinite(np.real(z0))
    assert finite.any(), "the lane returned no finite Z0"
    measured = float(np.median(np.real(z0)[finite]))
    declared = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS, SMA_OUTER_RADIUS, float(PTFE_EPS_R))
    frac = abs(measured - declared) / declared
    assert frac <= Z0_FRAC, (
        f"the {load_ohm:g} ohm load reads Z0 = {measured:.3f} ohm against the "
        f"analytic {declared:.3f} on the declared radii, {frac*100:.2f} %")
