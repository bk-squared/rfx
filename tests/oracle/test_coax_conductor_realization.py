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

**Shape, and why.** The comparison is a mesh LADDER, not a single mesh: the v2
accuracy bar says a comparison on a mesh that has not been shown to converge
"says nothing either way — not a pass, not a defect". So the physics runs on
VESSL and lands in a committed record, and the tests here are:

* a resistor-calibration check against the declared annulus, using the
  stored load reflection;
* one ``slow_physics`` LIVE test that solves the cheapest rung so the weekly
  lane still exercises the solver rather than only the fixture.

The structural half — that the lane hands ``rfx.simulation.run`` the conductors
as ``pec_edge_masks`` with no ``PEC_SIGMA`` left in ``materials.sigma`` — is in
``tests/unit/sparams/test_coax_conductor_geometry.py``, needs no FDTD, and runs
on every PR.

**The board matters as much as the mesh, so the record is the LONG board.** An
earlier version of this file asserted a 1 % bar on the committed AD gate's
8x8x12 mm board with three probe planes. That board realizes 3.789 annulus
cells and gets WORSE under refinement — 4.60 -> 1.51 -> 13.58 % on the phase
estimate — because ``rfx/sparams/coax.py`` places probe planes by CELL INDEX,
so refining dx pulls them together until three planes span 0.17 rad. An
instrument that cannot resolve what it is asked is not entitled to a verdict in
either direction, so it carries no bar here and its numbers live in the
diagnostic record instead.

BEFORE, on the same fixtures, from
``scripts/diagnostics/coax_conductor_mutation.py`` — the sigma realization put
back with every helper call left in place — is quoted in that script and in
``docs/design_notes/coax_conductor_realization.md``. The two sets of
before-numbers are NOT interchangeable and each is labelled with its board.
"""
from __future__ import annotations

import json
import math
import pathlib

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

RECORD = (pathlib.Path(__file__).resolve().parents[1] / "fixtures"
          / "coax_conductor_realization" / "long_board_ladder.json")

BETA_FRAC = 0.01          # the v2 bar's 1 % on a frequency-like quantity
COLUMN_POWER_MAX = 1.02   # the pre-declared passivity gate: an UPPER bound
# A loss bound, not a mirror of the line above. A lossless PTFE line between
# matched ports must not lose more than 0.22 dB; the v2 magnitude bar is 2 dB,
# which on this quantity catches nothing, and a mirrored 0.98 would be a
# tolerance nobody declared. What the line actually loses is printed per rung.
COLUMN_POWER_MIN = 0.95
Z0_FRAC = 0.01
GATE_RUNG = 3.789288121451007


def _record() -> dict:
    if not RECORD.exists():
        pytest.skip(f"no committed record at {RECORD}; produce it with "
                    "scripts/diagnostics/coax_conductor_oracle_ladder.py "
                    "--assemble")
    return json.loads(RECORD.read_text())


def _load_cases(rec: dict) -> list[dict]:
    return [c for c in rec["cases"] if c["board"] == "load_long"]


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


def _declared_z() -> float:
    return coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS, SMA_OUTER_RADIUS, float(PTFE_EPS_R))


# ---------------------------------------------------------------------------
# Resistor calibration against the declared annulus.
# ---------------------------------------------------------------------------


def test_the_resistor_calibration_agrees_with_the_declared_annulus():
    """``Z0 = R (1 - Gamma) / (1 + Gamma)``, recomputed from the stored Gamma.

    **This does not measure the realized line, and the name says so.** ``R`` is
    the DECLARED load and the annular resistor's conductivity is built from
    ``ln(shell_inner / a)``, so the load's realized resistance and the line's
    impedance carry the same discrete geometric factor and it cancels. The
    blind review showed this directly: re-rasterizing the coax half a cell
    off-node moved the realized line 3.7 % (40.36 -> 38.88 ohm by an
    electrostatic solve on the edge masks) and moved this number 0.002 %.
    (Reviewer's measurement; the 40.36 is reproduced by
    ``scripts/diagnostics/coax_realized_impedance_static.py``, the 38.88 is
    not.)

    What it DOES pin is worth keeping: that the resistor's sigma calibration and
    the declared annulus agree, which is a real invariant and is exactly what
    the wall's inner face feeds, because ``shell_inner`` enters that formula.

    This test itself cannot demonstrate that -- it replays a frozen ``Gamma``
    and nothing in the record changes when the geometry does. The demonstration
    needs a live solve with the pre-fix wall placement put back, where the same
    quantity reads 40.78 ohm against a declared 48.59. That is the blind
    reviewer's measurement (VESSL 369367263452), not reproduced by anything in
    this repository: ``scripts/diagnostics/coax_conductor_mutation.py`` revives
    the per-node conductor, not the wall placement.

    The realized cross-section's own impedance is measured by
    ``tests/unit/sparams/test_coax_conductor_geometry.py``'s convergence check,
    which shares nothing with this path.
    """
    rec = _record()
    cases = _load_cases(rec)
    if not cases:
        pytest.skip("the record carries no one-port rungs")
    declared = _declared_z()
    rows = []
    for c in sorted(cases, key=lambda c: (c["measured"]["load_ohm"],
                                          c["rung_annulus_cells"])):
        m = c["measured"]
        gamma = (np.asarray(m["s11_real"], dtype=float)
                 + 1j * np.asarray(m["s11_imag"], dtype=float))
        R = float(m["load_ohm"])
        z0 = R * (1.0 - gamma) / (1.0 + gamma)
        finite = np.isfinite(np.real(z0))
        measured = float(np.median(np.real(z0)[finite]))
        frac = abs(measured - declared) / declared
        rows.append((R, c["rung_annulus_cells"], measured, frac))
    trend = "  ".join(f"{R:g}ohm r{r:.4g}: {z:.3f} ({f*100:.2f} %)"
                      for R, r, z, f in rows)
    for R, rung, measured, frac in rows:
        assert frac <= Z0_FRAC, (
            f"the {R:g} ohm load at {rung:.4g} annulus cells reads Z0 = "
            f"{measured:.3f} ohm against the analytic {declared:.3f} on the "
            f"declared radii, {frac*100:.2f} %. Ladder: {trend}")


# ---------------------------------------------------------------------------
# One live solve, so the weekly lane exercises the solver and not the fixture.
# ---------------------------------------------------------------------------

LIVE_DOMAIN = (0.008, 0.008, 0.060)
LIVE_PROBES = dict(probe_count=12, probe_start_cells=8, probe_spacing_cells=4)
LIVE_FREQS = np.linspace(4.0e9, 12.0e9, 81)
LIVE_RUNG = GATE_RUNG          # the cheapest rung on the long board
LIVE_STEPS = 5000              # 12 one-way traversals at this cell size


@pytest.mark.slow_physics
def test_the_live_thru_reproduces_the_record_s_cheapest_rung():
    """Solve the long thru once and assert the same bars the replay does.

    The replay above cannot catch a solver regression, because it reads stored
    numbers. This can, and it is the cheapest rung so the weekly lane pays for
    one board rather than four.
    """
    a, b = float(SMA_PIN_RADIUS), float(SMA_OUTER_RADIUS)
    dx = (b - a) / LIVE_RUNG
    sim = Simulation(freq_max=40.0e9, domain=LIVE_DOMAIN, boundary="cpml",
                     cpml_layers=16, dx=dx)
    sim.add_coaxial_port((LIVE_DOMAIN[0] / 2.0, LIVE_DOMAIN[1] / 2.0,
                          LIVE_DOMAIN[2] / 2.0), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    res = sim.compute_coaxial_two_port(n_steps=LIVE_STEPS, freqs=LIVE_FREQS,
                                       **LIVE_PROBES)
    assert res.status == "passed", (
        f"the live thru did not settle: {res.status}, {res.settling_db}")
    assert abs(float(res.annulus_cells) - LIVE_RUNG) < 1e-6, (
        f"realized {float(res.annulus_cells)} annulus cells, declared "
        f"{LIVE_RUNG}")

    freqs = np.asarray(res.freqs, dtype=float)
    S = np.asarray(res.s_params)
    planes = np.asarray(res.reference_planes, dtype=float)
    beta_analytic = 2.0 * np.pi * freqs * math.sqrt(float(PTFE_EPS_R)) / C0
    beta_phase, v_p = _beta_from_s21_phase(freqs, S[1, 0, :],
                                           float(abs(planes[0] - planes[1])))
    worst = float(np.max(np.abs(beta_phase / beta_analytic - 1.0)))
    assert worst <= BETA_FRAC, (
        f"live: the phase constant is {worst*100:.2f} % from "
        f"omega*sqrt({float(PTFE_EPS_R)})/c (phase velocity {v_p:.6e} m/s)")

    gamma = np.asarray(res.gamma)
    beta_fit = np.imag(gamma)
    beta_fit = beta_fit.mean(axis=tuple(range(beta_fit.ndim - 1)))
    worst_fit = float(np.max(np.abs(beta_fit / beta_analytic - 1.0)))
    assert worst_fit <= BETA_FRAC, (
        f"live: the matrix-pencil phase constant is {worst_fit*100:.2f} % from "
        "the analytic one")

    col = np.sum(np.abs(S) ** 2, axis=0)
    # The same two bounds the replay uses, and for the same reasons: 1.02 above
    # is the pre-declared passivity gate, 0.95 below is a loss bound saying a
    # lossless line loses under 0.22 dB. Not a mirrored pair.
    assert float(col.max()) <= COLUMN_POWER_MAX, (
        f"live: the power sum reaches {float(col.max()):.5f}, above "
        f"{COLUMN_POWER_MAX}: a passive line cannot return more than it was given")
    assert float(col.min()) >= COLUMN_POWER_MIN, (
        f"live: the power sum falls to {float(col.min()):.5f}, a lossless line "
        f"losing more than {-10*math.log10(COLUMN_POWER_MIN):.2f} dB")
