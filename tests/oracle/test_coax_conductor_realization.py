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

* a FAST REPLAY that re-derives every comparison from the COMPLEX S stored in
  that record — not from the scalars the producer already computed — and
  asserts the bar at every rung, with the trend stated;
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

# The claims rung and the rungs the bar is asserted at. 3.789 is the cell size
# the committed one-port fixtures realize; it is asserted too, and it passes.
CLAIMS_RUNG = 9.0
GATE_RUNG = 3.789288121451007

# |S11| below this on a matched thru is a deep null: the extractor's RELATIVE
# error there is unbounded, so those bins are reported and not asserted. The
# quantity that does bind everywhere is the column power, which is asserted at
# both ends below.
DEEP_NULL_FLOOR = 0.02    # -34 dB


def _record() -> dict:
    if not RECORD.exists():
        pytest.skip(f"no committed record at {RECORD}; produce it with "
                    "scripts/diagnostics/coax_conductor_oracle_ladder.py "
                    "--assemble")
    return json.loads(RECORD.read_text())


def _thru_cases(rec: dict) -> list[dict]:
    return [c for c in rec["cases"] if c["board"] == "thru_long"]


def _load_cases(rec: dict) -> list[dict]:
    return [c for c in rec["cases"] if c["board"] == "load_long"]


def _S(measured: dict) -> np.ndarray:
    return (np.asarray(measured["s_params_real"], dtype=float)
            + 1j * np.asarray(measured["s_params_imag"], dtype=float))


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
# Fast replay: re-derive every comparison from the stored complex S.
# ---------------------------------------------------------------------------

def test_the_record_is_one_tree_and_covers_the_ladder():
    rec = _record()
    assert rec["schema"] == "rfx.coax_conductor_long_board_record"
    rungs = sorted({c["rung_annulus_cells"] for c in _thru_cases(rec)})
    assert len(rungs) >= 3, (
        f"the thru ladder has {len(rungs)} rungs {rungs}; a trend needs at "
        "least two refinements past the starting mesh")
    assert any(abs(r - CLAIMS_RUNG) < 1e-9 for r in rungs), (
        f"the claims rung {CLAIMS_RUNG} is missing from {rungs}")
    for c in rec["cases"]:
        assert c["commit"] == rec["commit"], (
            f"{c['board']} r{c['rung_annulus_cells']} came from "
            f"{c['commit'][:8]}, the record says {rec['commit'][:8]}")
        m = c["measured"]
        assert abs(m["annulus_cells"] - c["rung_annulus_cells"]) < 1e-6, (
            f"{c['board']} declares rung {c['rung_annulus_cells']} and "
            f"realizes {m['annulus_cells']}")
        assert m["status"] == "passed", (
            f"{c['board']} r{c['rung_annulus_cells']} did not settle: "
            f"status {m['status']}, settling {m.get('settling_db')}")


def test_the_thru_carries_tem_at_the_phase_constant_its_fill_implies():
    """Both estimators, every rung, re-derived from the stored complex S."""
    rec = _record()
    rows = []
    for c in sorted(_thru_cases(rec), key=lambda c: c["rung_annulus_cells"]):
        m = c["measured"]
        freqs = np.asarray(m["freqs_hz"], dtype=float)
        S = _S(m)
        beta_analytic = 2.0 * np.pi * freqs * math.sqrt(float(PTFE_EPS_R)) / C0
        beta_phase, v_p = _beta_from_s21_phase(
            freqs, S[1, 0, :], m["reference_plane_separation_m"])
        worst = float(np.max(np.abs(beta_phase / beta_analytic - 1.0)))
        # The producer's pencil number is carried, not re-derived: the fit is
        # the extractor's and is not recoverable from S alone. It is a second
        # estimator, so it is asserted, and the phase slope above is the one
        # this test computes itself.
        pencil = float(m["beta_ratio_matrix_pencil_worst"])
        rows.append((c["rung_annulus_cells"], worst, pencil, v_p))

    trend = "  ".join(f"r{r:.4g}: phase {w*100:.2f} % pencil {p*100:.2f} %"
                      for r, w, p, _ in rows)
    for rung, worst, pencil, v_p in rows:
        assert worst <= BETA_FRAC, (
            f"at {rung:.4g} annulus cells the phase constant from the "
            f"unwrapped S21 phase is {worst*100:.2f} % from "
            f"omega*sqrt({float(PTFE_EPS_R)})/c (phase velocity {v_p:.6e} m/s, "
            f"analytic {C0/math.sqrt(float(PTFE_EPS_R)):.6e}); a homogeneously "
            f"filled PEC-bounded line has no other phase constant to carry. "
            f"Ladder: {trend}")
        assert pencil <= BETA_FRAC, (
            f"at {rung:.4g} annulus cells the matrix-pencil phase constant is "
            f"{pencil*100:.2f} % from the analytic one. Ladder: {trend}")


def test_the_thru_keeps_its_power():
    """A lossless thru gives back what it is fed: passivity above, loss below.

    Two DIFFERENT bounds, not one mirrored:

    * **Above, 1.02** — the pre-declared passivity gate. A passive structure
      cannot return more than it was given, so an excess is non-physical and is
      never reported as physics.
    * **Below, 0.95** — a LOSS bound: a lossless PTFE line between matched ports
      must not lose more than 0.22 dB. This is a much weaker statement than the
      one above and is meant to be: the v2 magnitude bar is 2 dB, which on this
      quantity is too loose to catch anything, while a mirrored 0.98 would be a
      tolerance nobody declared. The measured minimum, the bin it falls in and
      the worst |S21| in dB are PRINTED at every rung so the curve is on the
      record rather than compressed into a pass.
    """
    rec = _record()
    rows = []
    for c in sorted(_thru_cases(rec), key=lambda c: c["rung_annulus_cells"]):
        m = c["measured"]
        S = _S(m)
        col = np.sum(np.abs(S) ** 2, axis=0)
        freqs = np.asarray(m["freqs_hz"], dtype=float)
        per_bin = col.min(axis=0)
        k = int(np.argmin(per_bin))
        s21_db = 20.0 * np.log10(np.abs(S[1, 0, :]))
        rows.append((c["rung_annulus_cells"], float(col.min()), float(col.max()),
                     float(freqs[k]) / 1e9, float(s21_db.min()),
                     int((per_bin < COLUMN_POWER_MIN).sum()), per_bin.size))
    for rung, lo, hi, f_lo, s21_lo, n_bad, n_bins in rows:
        print(f"[coax ladder] thru r{rung:.4g}: power sum [{lo:.5f}, {hi:.5f}], "
              f"min at {f_lo:.2f} GHz, worst |S21| {s21_lo:.3f} dB, "
              f"{n_bad}/{n_bins} bins under {COLUMN_POWER_MIN}")
    trend = "  ".join(f"r{r:.4g}: [{lo:.5f}, {hi:.5f}] worst |S21| {s:.3f} dB"
                      for r, lo, hi, _, s, _, _ in rows)
    for rung, lo, hi, f_lo, s21_lo, _, _ in rows:
        assert hi <= COLUMN_POWER_MAX, (
            f"at {rung:.4g} annulus cells the power sum reaches {hi:.5f}, above "
            f"{COLUMN_POWER_MAX}: a passive line cannot return more than it was "
            f"given. Ladder: {trend}")
        assert lo >= COLUMN_POWER_MIN, (
            f"at {rung:.4g} annulus cells the power sum falls to {lo:.5f} at "
            f"{f_lo:.2f} GHz, worst |S21| {s21_lo:.3f} dB — a lossless line "
            f"losing more than {-10*math.log10(COLUMN_POWER_MIN):.2f} dB. "
            f"Ladder: {trend}")


def test_the_loads_read_the_declared_characteristic_impedance():
    """``Z0 = R (1 - Gamma) / (1 + Gamma)``, recomputed from the stored Gamma.

    It is the second line constant: a realization that fixed ``beta`` by moving
    the geometry would show up here.
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


def test_the_reflection_is_reported_where_it_is_a_deep_null():
    """|S11| and |S21| on the thru, under the deep-null rule.

    A matched thru's |S11| runs into bins where it is a null, and the
    extractor's RELATIVE error there is unbounded — so those bins are reported
    and not asserted, which is why this test pins only what is true everywhere:
    passivity per entry, and |S21| not vanishing on a through line. The
    quantity that DOES bind at every bin is the column power, asserted above.
    """
    rec = _record()
    for c in sorted(_thru_cases(rec), key=lambda c: c["rung_annulus_cells"]):
        m = c["measured"]
        S = _S(m)
        s11, s21 = np.abs(S[0, 0, :]), np.abs(S[1, 0, :])
        rung = c["rung_annulus_cells"]
        assert float(s11.max()) <= 1.0 + (COLUMN_POWER_MAX - 1.0), (
            f"at {rung:.4g} cells |S11| reaches {float(s11.max()):.5f} on a "
            "passive line")
        assert float(s21.min()) > DEEP_NULL_FLOOR, (
            f"at {rung:.4g} cells |S21| falls to {float(s21.min()):.5f}; a "
            "through line that stops transmitting is not a reference-plane "
            "question")
        n_null = int((s11 < DEEP_NULL_FLOOR).sum())
        print(f"[coax ladder] thru r{rung:.4g}: |S11| "
              f"[{float(s11.min()):.5f}, {float(s11.max()):.5f}], "
              f"{n_null}/{s11.size} bins below the {DEEP_NULL_FLOOR} null "
              f"floor and reported only; |S21| "
              f"[{float(s21.min()):.5f}, {float(s21.max()):.5f}]")


def test_the_probe_array_span_is_recorded_with_every_rung():
    """The ladder shortens the probe array, and the record says by how much.

    ``rfx/sparams/coax.py`` places probe planes by CELL INDEX, so refining dx
    pulls them towards the DUT. That is in every dx-ladder number in this
    record, so the span is stored per rung rather than left to be rediscovered.
    """
    rec = _record()
    rows = [(c["rung_annulus_cells"],
             c["measured"]["probe_span_rad_at_band_centre"])
            for c in sorted(_thru_cases(rec),
                            key=lambda c: c["rung_annulus_cells"])]
    for rung, span in rows:
        assert span > 1.0, (
            f"at {rung:.4g} annulus cells the whole probe array spans "
            f"{span:.3f} rad at band centre; a propagation constant fitted "
            f"over that little phase is not a measurement of the line. "
            f"Spans: {rows}")


def test_the_short_board_is_recorded_and_carries_no_bar():
    """Why the 8x8x12 mm, 3-probe board is reported and not asserted.

    It is the committed AD gate's board, and the first version of this file
    put a 1 % bar on it. It does not deserve one and this is the measurement
    that says so rather than an opinion: its three probe planes span less than
    a quarter radian at band centre on the finer rungs, which is not enough
    phase to fit a propagation constant to, and its readings get WORSE under
    refinement. The numbers stay in the record so the claim is checkable; the
    bars are on the long board.

    If a future change gives that board a probe array that spans a useful
    fraction of a wavelength, this test fails and someone gets to decide
    whether it has earned a bar.
    """
    rec = _record()
    short = [c for c in rec["cases"] if c["board"] in ("thru_gate", "load_gate")]
    if not short:
        pytest.skip("the record carries no short-board diagnostic cases")
    thru = sorted((c for c in short if c["board"] == "thru_gate"),
                  key=lambda c: c["rung_annulus_cells"])
    for c in thru:
        m = c["measured"]
        span = m["probe_span_rad_at_band_centre"]
        print(f"[coax ladder] SHORT board r{c['rung_annulus_cells']:.4g} "
              f"(reported, not asserted): beta "
              f"{m['beta_ratio_s21_phase_worst']*100:.2f} % / "
              f"{m['beta_ratio_matrix_pencil_worst']*100:.2f} %, power "
              f"[{m['min_column_power']:.5f}, {m['max_column_power']:.5f}], "
              f"probe span {span:.3f} rad")
        assert span < 1.0, (
            f"the short board's probe array now spans {span:.3f} rad at "
            f"{c['rung_annulus_cells']:.4g} annulus cells. It carries no bar "
            "BECAUSE it could not resolve a propagation constant; if that has "
            "changed, decide deliberately whether it should.")


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
    assert float(col.max()) <= COLUMN_POWER_MAX, (
        f"live: max column power {float(col.max()):.5f}")
    assert float(col.min()) >= 1.0 - (COLUMN_POWER_MAX - 1.0), (
        f"live: min column power {float(col.min()):.5f}")
