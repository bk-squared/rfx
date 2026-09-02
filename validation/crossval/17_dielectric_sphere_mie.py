"""Dielectric-sphere monostatic RCS vs exact Mie — ka sweep 0.5-2.5 (campaign item 6).

The material-path twin of the PEC ka-sweep (``16_pec_sphere_mie_ka_sweep.py``):
a lossless eps_r = 2.56 (m = 1.6) sphere, gated against the full
Bohren-Huffman dielectric Mie series RE-DERIVED in this script (with
Rayleigh / m->1 / n_max-convergence / unitarity self-witnesses run before any
gate) and AGAIN independently in the frozen gate test. No external solver —
this script never exits 2. The path under test is the BINARY ``rasterize``
dielectric interface (no sub-cell averaging exists in rfx today): this leg
measures what that staircase actually delivers, it does not certify smooth
interface handling.

Design deltas vs the PEC sweep (derived, not tuned):
  * resolution floor uses the INTERNAL wavelength: RES(ka) = max(24,
    ceil(2*pi*CPR/ka)) with 24 = ceil(15 * m) — lambda_internal/15.
  * single-run extraction. The campaign plan names the two-run subtracted
    path for translucent scatterers, but ``monostatic_rcs`` — the ONLY
    quantity this sweep extracts — is always computed from the raw
    (unsubtracted) run BY CONSTRUCTION (rfx/rcs.py documents this), so
    ``subtract_incident_reference=True`` is a no-op for every number here
    at exactly 2x the compute. The first revision of this script ran with
    the flag on and quoted the sub-vs-unsub identity as a "leakage
    FALSIFIED" witness — that comparison compares the same array with
    itself (the PR #476 review, F1: same tautology class as AD==FD).
    Leakage at the monostatic bin is instead excluded by rcs.py's own
    documented backscatter leakage null (~90 dB), not by any probe in
    this script; the review measured the bistatic sub-vs-unsub difference
    at <= 0.025 dB on this geometry for the record.
  * truncation witness on every gated bin from the start — the dielectric
    sphere stores internal energy; measured shifts are nonetheless <= 0.09 dB
    at 2x steps (700 steps is settled at these Q's).

WHAT IS GATED vs WHAT IS REPORTED (measured 2026-07-27; committed 7-point
clearance scan {15,20,25,30,35,40,45} + three domain realizations — the
anti-aliasing posture adopted after PR #475 F1, applied from day one here)
---------------------------------------------------------------------------
GATED (exit-1 on failure), gate = round-UP(measured clearance-scan envelope
x 1.5) to 0.1 dB (the round-up convention adopted in PR #475; the gate test
recomputes the envelope from the committed data AND pins the hard constants,
AND binds these script constants — the PR #475 D1/D2 lessons applied from
day one):
  * coarse bins ka {0.5, 0.75, 1.0, 1.25} (cpr 6.4): scan envelopes
    2.62/2.80/3.38/4.18 dB -> envelope 4.18 -> gate 6.3 dB. This is a
    REGRESSION LOCK on a cross-method record, tighter than the committed
    PEC E4 coarse-ladder gate (13.9 dB) but deliberately framed as a
    diagnostic envelope, not a converged-accuracy claim.
  * NO fine rung is gated, and that is a measured decision, not an
    omission: at clear=20 the cpr-12.8 rung looked convergent (ka=1.0:
    -3.28 -> +0.08 -> -0.97 across cpr 6.4/9.6/12.8), but its own 7-point
    clearance scan measures envelopes 3.75 dB (ka=0.75, clearance-
    monotonic to +3.75 at 45) and 3.04 dB (ka=1.0) — barely below the
    coarse 4.18, so doubling resolution does NOT buy a domain-robust
    tighter claim. This is the PR #475 F1 aliasing class caught BEFORE
    commit (single-clearance convergence is not convergence); the fine
    scan is committed as the fine_rung_witness so the decision is
    auditable.
REPORTED, NOT GATED (documented-unconverged at this operating point):
  * every bin with ka >= 1.5. Under a domain-size-only change the
    delta swings across 11.7 dB (ka=1.75: +6.67/+3.99/-5.05 at clearance
    20/30/40) and 29.0 dB (ka=2.5: +6.31/+1.32/-22.65) — worse than the
    PEC case. Attribution mirrors item 1: truncation FALSIFIED (<= 0.09 dB
    at 2x steps), incident leakage EXCLUDED at the monostatic bin by
    rcs.py's documented backscatter leakage null (NOT by a probe — see
    the extraction note above), resolution non-monotonic at ka >= 1.5
    (ka=1.75: +6.67/-11.58/+3.53 across cpr 6.4/9.6/12.8); the
    domain-size axis dominates. Gating these bins would be
    tuned-tolerance theater.
HEADROOM NOTE: the coarse envelope is dominated by the fence-edge bin
ka = 1.25 (-4.18 dB at clearance 30) and rises monotonically toward the
fence — scan envelopes 2.62 -> 4.18 across ka 0.5 -> 1.25, and on the
consistent 3-domain SPREAD metric 1.87/3.07/3.95/5.21 dB (gated) jumps to
9.05 dB at the first fenced bin ka = 1.5 and 11.7 dB at ka = 1.75. If
ka = 1.25 ever exceeds its gate, move the coarse fence DOWN to ka <= 1.0
rather than widen the gate.

GEOMETRY-FIDELITY CONVENTION (issue #725, applied 2026-08-27; PR #721
review group C — material-path twin of the same fix in
``16_pec_sphere_mie_ka_sweep.py``, applied here in lockstep): a sphere is
a CURVED body, so no cell size realizes it exactly, and a bounding-box
extent mismatch is NOT load-bearing here — trace the reference chain in
``run_point``: the radius enters ONLY through ``ka`` (the Mie series
argument) and ``pi_a2 = pi * a**2`` (the RCS normalization), never through
extent. The adopted fix is issue #725 option 1: evaluate the dielectric
Mie series at the REALIZED effective radius ``a_eff`` derived from the
occupied-cell count ``N`` of the rasterized sphere (the binary
``rasterize`` dielectric interface makes ``N`` exact — no sub-cell
averaging to blur it), ``a_eff = (3*N*dx**3/(4*pi))**(1/3)``, instead of
the declared radius. This ONE ``pi_a2`` (now a_eff-based) feeds BOTH
``mie_dbsm`` and ``rfx_sigma_over_pi_a2`` in ``run_point``, so the two
normalizations move together by construction (PR #721 review, required
change 1) — geometry here is identical to the PEC sweep at the same
(ka, cpr, clear_cells), since rasterization does not depend on the
material value.
  * Measured a_eff/a at every gated coarse point (cpr=6.4): ka=0.50
    a_eff/a=0.988032 (N=1082, -1.197%), ka=0.75 a_eff/a=0.988032
    (N=1082, -1.197%), ka=1.00 a_eff/a=0.989330 (N=1127, -1.067%),
    ka=1.25 a_eff/a=0.995401 (N=1169, -0.460%). fine-rung WITNESS
    (cpr=12.8, not gated): ka=0.75/1.00 a_eff/a=0.999168 (N=8952,
    -0.083%).
  * Reference-side shift, Mie(ka_eff, a_eff) - Mie(ka, a), at each gated
    point: ka=0.50 -0.3026 dB, ka=0.75 -0.2788 dB, ka=1.00 -0.1913 dB,
    ka=1.25 -0.0173 dB.
  * Envelope consequence, recomputed ARITHMETICALLY from the already-
    committed fixture rows (no FDTD re-run): coarse envelope 4.181 ->
    4.164 dB, gate stays round-up(env x 1.5) = 6.3 dB (no movement here
    — unlike the PEC fine rung, this leg's envelope was already close to
    a 0.1 dB quantum boundary from below). fine-rung witness envelope
    3.745 -> 3.765 dB, still comfortably (> 0.7x) below the coarse
    envelope, so the "no fine rung gated" decision is unchanged. No gate
    constant needs to move in this script; GATE_COARSE_DB below is
    unaffected by this revision.
  * Per-row geometric gate (PR #721 review, required change 8, NEW):
    ``A_EFF_TOL_COARSE`` below asserts ``|a_eff/a - 1| <= 1.5%`` at
    cpr=6.4 (measured worst case 1.197% above) — checks the geometry
    claim SEPARATELY from the dB tolerance so a_eff can never silently
    absorb a real rasterization regression.
  * Centre offset and the falsifier: identical mechanism and per-point
    numbers to the PEC sweep (same rasterization, same mesh) — see
    ``16_pec_sphere_mie_ka_sweep.py``'s GEOMETRY-FIDELITY CONVENTION
    note for the full derivation; not re-run independently here.
  * ``claim_scope`` in the fixture payload below still states the OLD
    (declared-radius) convention. Per PR #721 review required change 3,
    updating it — and the internal-consistency assertions in
    ``tests/crossval/test_rcs_dielectric_sphere_mie_gates.py`` (:254-277, both
    the ``rfx_monostatic_dbsm`` and ``mie_dbsm`` halves) and the
    AST-pinned prose binding (:102-121) — is deferred to the next
    ``--write-fixture`` regeneration, done in lockstep with the fixture
    and artifact JSON so the suite never observes a mismatched pair.

MATERIAL-FIDELITY GATE (issue #812 re-gate, 2026-09-01)
---------------------------------------------------------------------------
The #812 audit measured this case's dB gate passing for a rasterized
permittivity wrong by a factor, and it is right. How wrong is committed, not
restated: validation/crossval/_17_dielectric_results/material_blind_window.json
(summary.blind_window_bracket_eps and its two first-failing neighbours), built
with no FDTD from the committed gated_coarse deltas and the Mie oracle, and
reproducing the live defect runs' pass/fail verdict at round 1's four probed
permittivity. That is not a loose threshold, it is the SENSITIVITY of the
observable: the Mie oracle
moves 9.816 / 10.202 / 9.978 / 7.134 dB per unit RELATIVE permittivity at
ka = 0.50 / 0.75 / 1.00 / 1.25, so 6.3 dB simply IS a factor-wide window in
eps_r. GATE_COARSE_DB is already round-up(envelope x 1.5) and cannot be
tightened under the repo rule; and no dB threshold this case could legally
carry would resolve a few-percent permittivity error. **So the claim is
re-scoped rather than the gate pretended tighter: the dB tier is an RCS
agreement envelope, NOT a permittivity calibration**, and the permittivity is
gated on a channel of its own — the material twin of the per-row
``A_EFF_TOL_COARSE`` geometric gate, which exists for exactly this reason on
the geometry side.
  * G17-A ``EPS_REALIZED_TOL`` — the permittivity read back OUT of the
    rasterized array must be within 0.5% relative of the declared ``EPS_R``.
    Derived from the sensitivity above: half the gate's own reporting quantum
    (0.1 dB, the PR #475 round-up convention) divided by the worst gated-ka
    slope 10.202 dB per unit relative eps = 0.0049 -> 0.005.
  * G17-B ``N_DISTINCT_EPS_EXPECTED`` — the array must hold EXACTLY two
    distinct values. This is the FIRST check anywhere of this case's headline
    scope claim ("the BINARY rasterize dielectric interface — no sub-cell
    interface averaging exists"); sub-cell averaging, a partial fill or a
    smoothed interface each put a third value in the array, and G17-A alone
    is blind to that.
  * Measured on today's code: realized eps_r = float32(2.56) = 2.5599999428,
    2.2e-8 relative (four orders inside G17-A), exactly two distinct values
    at every gated bin.
  * NOT claimed: any permittivity resolution from the dB channel. The dB
    tier's blind window is the committed pass set in
    validation/crossval/_17_dielectric_results/material_blind_window.json
    (summary.pass_runs_eps: [2.0, 4.5] and [5.0, 5.6], with a FAIL island at
    4.6-4.9 on the ka = 1.25 Mie resonance) and stays that way.

NOTE on preflight: this script drives the functional ``compute_rcs`` entry
point, which runs NO preflight. The operating-point guarantees (internal-
wavelength floor, cells-per-radius floor, cell-unit clearance, transit-scaled
steps) are enforced BY CONSTRUCTION inside ``run_point`` (max()/derived
expressions, stronger than asserts — there is deliberately no `assert` to
grep for); there is no "All checks passed" line to quote and this docstring
says so.

Usage:
  python validation/crossval/17_dielectric_sphere_mie.py            # gated set (~5 min CPU)
  python validation/crossval/17_dielectric_sphere_mie.py --full     # + 9-point diagnostic curve
  python validation/crossval/17_dielectric_sphere_mie.py --write-fixture
      # full 3-domain + clearance-scan measurement + witnesses; regenerates
      # validation/crossval/_17_dielectric_results/rfx.json AND
      # tests/fixtures/rcs_dielectric_sphere_mie/fixture.json (~25 min CPU,
      # single-run extraction — see the extraction note above)

Exit codes: 0 = all configured gates passed; 1 = oracle self-check or a gate
failed. Failure prints the sentinel line "SOME CHECKS FAILED".
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.special import spherical_jn, spherical_yn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from tests._gate_policy import gate_from_envelope  # noqa: E402

import rfx  # noqa: E402

_RFX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(rfx.__file__)))
if _RFX_ROOT != _REPO_ROOT:
    raise RuntimeError(
        f"import rfx resolved outside this repo tree ({rfx.__file__}); "
        "refusing to report numbers for a different rfx build."
    )

import jax.numpy as jnp  # noqa: E402
from rfx.grid import Grid, C0  # noqa: E402
from rfx.geometry.csg import Sphere, rasterize  # noqa: E402
from rfx.core.yee import MaterialArrays  # noqa: E402
from rfx.rcs import compute_rcs  # noqa: E402

F0 = 3e9
LAM = C0 / F0
CPML_LAYERS = 8
EPS_R = 2.56
M_IDX = float(np.sqrt(EPS_R))          # 1.6, lossless
BANDWIDTH = 0.5
COARSE_CPR = 6.4
FINE_CPR = 12.8
RES_FLOOR = 24                          # ceil(15 * m): lambda_internal / 15
CLEAR_CELLS_DEFAULT = 20
CLEARANCE_SCAN = [15, 20, 25, 30, 35, 40, 45]

KA_ALL = [round(0.5 + 0.25 * i, 2) for i in range(9)]   # 0.5 .. 2.5
KA_GATED_COARSE = [0.5, 0.75, 1.0, 1.25]
KA_FINE_WITNESS = [0.75, 1.0]   # cpr-12.8 scan committed as WITNESS, not gated

# gate = round-UP(measured clearance-scan envelope x 1.5) to 0.1 dB.
# The gate test hard-pins this AND recomputes the envelope AND binds this
# constant to the fixture record (PR #475 D1/D2 lessons). No silent change.
GATE_COARSE_DB = 6.3    # scan envelope 4.18 dB (ka=1.25, clearance 30)

# Per-row geometric gate on the reference convention itself (issue #725
# required change 8) — see the module docstring's GEOMETRY-FIDELITY
# CONVENTION note. Measured worst case at cpr=6.4 is 1.197%.
A_EFF_TOL_COARSE = 0.015   # 1.5% of declared radius, cpr=6.4

# --- issue #812 re-gate: the MATERIAL twin of A_EFF_TOL_COARSE -------------
# This case exists to make the first cross-method record of the BINARY
# dielectric rasterize path at eps_r = 2.56, and until #812 NOTHING anywhere
# in it looked at the permittivity that was actually rasterized. The dB
# channel cannot do that job: d(sigma_dB)/d(eps/eps) from the Mie oracle is
# 9.816 / 10.202 / 9.978 / 7.134 dB per unit RELATIVE permittivity at
# ka = 0.50 / 0.75 / 1.00 / 1.25, so a 6.3 dB window is ~ a factor-of-two
# window in eps_r: the committed model pass set is [2.0, 4.5] and
# [5.0, 5.6] with a FAIL island at 4.6-4.9 (material_blind_window.json,
# summary.pass_runs_eps / summary.fail_islands_eps); live defect runs at
# 2.0 / 5.5 (pass) and 1.8 / 6.0 (fail) agree with the model's verdicts;
# INSIDE the island the live solver passes at 4.6-4.8 and fails only at 4.9
# (_17_dielectric_results/cv17_permittivity_island.json) -- the model
# over-predicts on the resonance, the live window is wider. See the note.
#
# G17-A: the realized permittivity is gated on its own channel, with the
# window derived from that sensitivity — half the gate's own reporting
# quantum (0.1 dB, the PR #475 round-up convention) divided by the worst
# gated-ka sensitivity 10.202 dB per unit relative eps gives 0.0049, i.e. the
# largest permittivity error that cannot move the recorded dB envelope by as
# much as the case can report. Rounded to:
EPS_REALIZED_TOL = 0.005    # 0.5% relative, cpr=6.4 and cpr=12.8 alike
#
# G17-B: structure, no tolerance. The claim_scope's headline — "the BINARY
# rasterize dielectric interface (no sub-cell interface averaging exists);
# this is a measured envelope of that staircase" — is presently prose that
# nothing checks. A binary rasterization puts EXACTLY two values in the
# array; sub-cell averaging, a partial fill, or a smoothed interface each add
# a third.
N_DISTINCT_EPS_EXPECTED = 2


def check_realized_material(eps_r_array) -> dict:
    """Read the permittivity back OUT of the rasterized grid (G17-A/G17-B).

    Returns the three numbers the material gate is made of. Deliberately
    reads the array the solver will integrate, not the constant that was
    handed to ``rasterize`` — the point is to catch a material path that does
    not deliver what it was asked for.
    """
    a = np.asarray(eps_r_array)
    vals = np.unique(a)
    non_bg = vals[vals != 1.0]
    eps_realized = float(non_bg.max()) if non_bg.size else float("nan")
    return {
        "n_distinct_eps": int(vals.size),
        "n_nonbackground_eps": int(non_bg.size),
        "eps_realized": eps_realized,
        "eps_rel_dev": abs(eps_realized / EPS_R - 1.0),
    }


def material_gate_ok(stats: dict) -> bool:
    """G17-A and G17-B together. False means the run is not about eps_r=2.56."""
    return (stats["n_distinct_eps"] == N_DISTINCT_EPS_EXPECTED
            and stats["n_nonbackground_eps"] == 1
            and stats["eps_rel_dev"] <= EPS_REALIZED_TOL)


# --------------------------------------------------------------------------- #
# Dielectric Mie oracle (Bohren & Huffman), re-derived in-script + witnesses.
# --------------------------------------------------------------------------- #
def _mie_coeffs(m: float, x: float, n_max: int):
    n = np.arange(1, n_max + 1)
    mx = m * x
    jx = spherical_jn(n, x)
    jpx = spherical_jn(n, x, derivative=True)
    yx = spherical_yn(n, x)
    ypx = spherical_yn(n, x, derivative=True)
    jmx = spherical_jn(n, mx)
    jpmx = spherical_jn(n, mx, derivative=True)
    psi_x, psi_px = x * jx, jx + x * jpx
    chi_x, chi_px = -x * yx, -(yx + x * ypx)
    xi_x, xi_px = psi_x - 1j * chi_x, psi_px - 1j * chi_px
    psi_mx, psi_pmx = mx * jmx, jmx + mx * jpmx
    a = (m * psi_mx * psi_px - psi_x * psi_pmx) / (m * psi_mx * xi_px - xi_x * psi_pmx)
    b = (psi_mx * psi_px - m * psi_x * psi_pmx) / (psi_mx * xi_px - m * xi_x * psi_pmx)
    return n, a, b


def mie_backscatter_over_pi_a2(m: float, ka: float, n_max: int | None = None) -> float:
    """sigma/(pi a^2), lossless dielectric sphere backscatter, real m."""
    x = float(ka)
    if n_max is None:
        n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 15
    n, a, b = _mie_coeffs(m, x, n_max)
    s = np.sum((2 * n + 1) * ((-1.0) ** n) * (a - b))
    return float(np.abs(s) ** 2 / x ** 2)


def validate_oracle() -> dict:
    """Self-witnesses; raises on failure (refuse to gate on a broken oracle)."""
    w = {}
    # W1 Rayleigh limit: sigma/pi a^2 -> 4 (ka)^4 ((m^2-1)/(m^2+2))^2
    x = 0.05
    ray = 4 * x ** 4 * ((M_IDX ** 2 - 1) / (M_IDX ** 2 + 2)) ** 2
    w["rayleigh_rel_err"] = abs(mie_backscatter_over_pi_a2(M_IDX, x) / ray - 1)
    assert w["rayleigh_rel_err"] < 5e-3
    # W2 m->1 no-scattering limit
    w["m_to_1_limit"] = mie_backscatter_over_pi_a2(1.001, 1.0)
    assert w["m_to_1_limit"] < 1e-5
    # W3 n_max convergence
    v1 = mie_backscatter_over_pi_a2(M_IDX, 2.5)
    v2 = mie_backscatter_over_pi_a2(M_IDX, 2.5, n_max=60)
    w["convergence_abs_change"] = abs(v1 - v2)
    assert w["convergence_abs_change"] < 1e-12
    # W4 unitarity (lossless): Re(c) == |c|^2 for every coefficient
    uni = 0.0
    for x in (0.5, 1.5, 2.5):
        n, a, b = _mie_coeffs(M_IDX, x, 40)
        uni = max(uni,
                  float(np.max(np.abs(a.real - np.abs(a) ** 2))),
                  float(np.max(np.abs(b.real - np.abs(b) ** 2))))
    w["unitarity_max_dev"] = uni
    assert uni < 1e-10
    return w


def run_point(ka: float, cpr: float, clear_cells: int, steps_mult: float = 1.0):
    """One monostatic point at the derived operating point. Returns a record.

    Reference convention (issue #725, applied 2026-08-27 — see the module
    docstring's GEOMETRY-FIDELITY CONVENTION note): the Mie series below is
    evaluated at the REALIZED effective radius ``a_eff``, derived from the
    occupied-cell count of the rasterized sphere, not the declared
    ``radius``. ``pi_a2`` is the ONE variable both ``mie_dbsm`` and
    ``rfx_sigma_over_pi_a2`` consume, so the two normalizations move
    together by construction.
    """
    radius = ka * LAM / (2 * np.pi)
    res = max(RES_FLOOR, int(np.ceil(2 * np.pi * cpr / ka)))
    dx = LAM / res
    domain = 2 * radius + 2 * clear_cells * dx
    grid = Grid(freq_max=F0 * 1.5, domain=(domain,) * 3, dx=dx,
                cpml_layers=CPML_LAYERS)
    n_steps = int(max(700, np.ceil(2.2 * domain / C0 / grid.dt)) * steps_mult)
    center = (domain / 2,) * 3
    eps_r, sigma = rasterize(grid, [(Sphere(center=center, radius=radius), EPS_R, 0.0)])
    n_occupied = int(np.sum(np.asarray(eps_r) > 1.0))
    material = check_realized_material(eps_r)   # issue #812 G17-A/G17-B
    a_eff = (3 * n_occupied * dx ** 3 / (4 * np.pi)) ** (1.0 / 3.0)
    ka_eff = 2 * np.pi * a_eff / LAM
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma,
                          mu_r=jnp.ones(grid.shape, dtype=jnp.float32))
    t0 = time.time()
    result = compute_rcs(
        grid, mats, n_steps,
        f0=F0, bandwidth=BANDWIDTH, theta_inc=0.0, polarization="ez",
        theta_obs=np.array([np.pi / 2]), phi_obs=np.array([0.0, np.pi]),
        freqs=np.array([F0]),
        boundary="cpml", cpml_layers=CPML_LAYERS,
        # NO subtract_incident_reference: monostatic_rcs is always computed
        # from the raw run by construction (rfx/rcs.py), so the flag is a
        # no-op for this sweep at 2x the compute (PR #476 review, F1).
    )
    wall = time.time() - t0
    pi_a2 = np.pi * a_eff ** 2   # REALIZED effective radius (issue #725 option 1)
    mono = float(result.monostatic_rcs[0])
    mie_over = mie_backscatter_over_pi_a2(M_IDX, ka_eff)
    mie_dbsm = float(10 * np.log10(mie_over * pi_a2))
    return {
        "ka": ka, "cells_per_radius": cpr, "clear_cells": clear_cells,
        "resolution": res, "grid": list(grid.shape), "n_steps": n_steps,
        "a_over_dx": round(radius / dx, 2),
        "n_occupied": n_occupied,
        "n_distinct_eps": material["n_distinct_eps"],
        "eps_realized": round(material["eps_realized"], 9),
        "eps_rel_dev": float(f"{material['eps_rel_dev']:.3e}"),
        "a_eff_over_a": round(a_eff / radius, 6),
        "ka_eff": round(ka_eff, 4),
        "rfx_monostatic_dbsm": round(mono, 4),
        "mie_dbsm": round(mie_dbsm, 4),
        "rfx_sigma_over_pi_a2": round(10 ** (mono / 10.0) / pi_a2, 6),
        "mie_sigma_over_pi_a2": round(mie_over, 6),
        "delta_db": round(mono - mie_dbsm, 3),
        "wall_s": round(wall, 1),
    }


def _fmt(r):
    return (f"  ka={r['ka']:4.2f} cpr={r['cells_per_radius']:4.1f} "
            f"clear={r['clear_cells']:2d} grid={tuple(r['grid'])} "
            f"steps={r['n_steps']:5d}  rfx {r['rfx_monostatic_dbsm']:8.2f} dBsm "
            f"vs Mie {r['mie_dbsm']:8.2f}  delta {r['delta_db']:+6.2f} dB "
            f"a_eff/a={r['a_eff_over_a']:.4f} ({r['wall_s']:.0f}s)")


def main(argv):
    full = "--full" in argv
    write_fixture = "--write-fixture" in argv
    ok = True

    witnesses = validate_oracle()
    print("[oracle] dielectric-Mie self-witnesses PASS:",
          {k: float(f"{v:.3e}") for k, v in witnesses.items()})

    print(f"\n== GATED coarse bins ka {KA_GATED_COARSE} "
          f"(cpr={COARSE_CPR}, gate {GATE_COARSE_DB} dB) ==")
    gated = []
    for ka in KA_GATED_COARSE:
        r = run_point(ka, COARSE_CPR, CLEAR_CELLS_DEFAULT)
        gated.append(r)
        passed = abs(r["delta_db"]) <= GATE_COARSE_DB
        aeff_ok = abs(r["a_eff_over_a"] - 1.0) <= A_EFF_TOL_COARSE
        mat_ok = material_gate_ok({"n_distinct_eps": r["n_distinct_eps"],
                                   "n_nonbackground_eps": 1,
                                   "eps_realized": r["eps_realized"],
                                   "eps_rel_dev": r["eps_rel_dev"]})
        ok &= passed and aeff_ok and mat_ok
        print(_fmt(r) + ("  PASS" if passed else "  FAIL")
              + ("" if aeff_ok else f"  A_EFF_OVER_A FAIL ({r['a_eff_over_a']:.4f})")
              + ("" if mat_ok else
                 f"  MATERIAL FAIL (realized eps_r {r['eps_realized']:.6g}, "
                 f"{r['n_distinct_eps']} distinct values; declared {EPS_R}, "
                 f"tol {EPS_REALIZED_TOL} rel, {N_DISTINCT_EPS_EXPECTED} values)"))

    # No gated fine rung — a measured decision (module docstring): the
    # cpr-12.8 clearance scan (committed below as fine_rung_witness)
    # envelopes 3.75/3.04 dB, barely below the coarse 4.18.

    diagnostic = []
    domains = {}
    if full or write_fixture:
        print("\n== DIAGNOSTIC 9-point coarse curve (REPORTED, ka>=1.5 NOT "
              "gated — see module docstring) ==")
        for ka in KA_ALL:
            r = run_point(ka, COARSE_CPR, CLEAR_CELLS_DEFAULT)
            diagnostic.append(r)
            print(_fmt(r))

    if write_fixture:
        print("\n== fixture mode: domain realizations clear 30/40 + witnesses ==")
        for clear in (30, 40):
            rows = [run_point(ka, COARSE_CPR, clear) for ka in KA_ALL]
            domains[str(clear)] = rows
            print(f"  coarse curve @ clear={clear}: deltas",
                  " ".join(f"{r['delta_db']:+.1f}" for r in rows))

        print("\n== clearance scan (gated bins) + fine-rung witness ==")
        scan = {"clearances": CLEARANCE_SCAN, "coarse": {}}
        for ka in KA_GATED_COARSE:
            rows = [run_point(ka, COARSE_CPR, c) for c in CLEARANCE_SCAN]
            scan["coarse"][str(ka)] = rows
            print(f"  coarse ka={ka}: deltas",
                  " ".join(f"{r['delta_db']:+.2f}" for r in rows))
        fine_witness = {"clearances": CLEARANCE_SCAN,
                        "cells_per_radius": FINE_CPR, "fine": {}}
        for ka in KA_FINE_WITNESS:
            rows = [run_point(ka, FINE_CPR, c) for c in CLEARANCE_SCAN]
            fine_witness["fine"][str(ka)] = rows
            print(f"  fine-WITNESS ka={ka}: deltas",
                  " ".join(f"{r['delta_db']:+.2f}" for r in rows))

        coarse_deltas = [abs(r["delta_db"]) for r in gated]
        coarse_deltas += [abs(r["delta_db"]) for c in ("30", "40")
                          for r in domains[c] if r["ka"] <= max(KA_GATED_COARSE)]
        coarse_deltas += [abs(r["delta_db"]) for ka in KA_GATED_COARSE
                          for r in scan["coarse"][str(ka)]]
        env_coarse = max(coarse_deltas)
        env_fine_witness = max(abs(r["delta_db"])
                               for ka in KA_FINE_WITNESS
                               for r in fine_witness["fine"][str(ka)])
        print(f"\n  measured envelopes: coarse {env_coarse:.3f} dB "
              f"(gate {GATE_COARSE_DB}); fine WITNESS {env_fine_witness:.3f} dB "
              f"(not gated)")
        if not (GATE_COARSE_DB >= env_coarse and
                GATE_COARSE_DB <= gate_from_envelope(env_coarse, quantum=10) + 0.05):
            print("  ENVELOPE/GATE MISMATCH (coarse) — fix the constant")
            ok = False

        trunc = []
        for ka, cpr in [(k, COARSE_CPR) for k in KA_GATED_COARSE]:
            r1 = run_point(ka, cpr, CLEAR_CELLS_DEFAULT, steps_mult=1.0)
            r2 = run_point(ka, cpr, CLEAR_CELLS_DEFAULT, steps_mult=2.0)
            trunc.append({"ka": ka, "cells_per_radius": cpr,
                          "delta_1x_db": r1["delta_db"],
                          "delta_2x_db": r2["delta_db"]})
            print(f"  truncation witness ka={ka} cpr={cpr}: "
                  f"1x {r1['delta_db']:+.2f} -> 2x {r2['delta_db']:+.2f} dB")

        payload = {
            "schema": "rfx.rcs_dielectric_sphere_mie",
            "schema_version": 1,
            "campaign": (
                "cross-solver validation campaign, item 6: dielectric-sphere "
                "exact-Mie ka sweep (material-path twin of item 1; first "
                "cross-method record of the binary dielectric rasterize path)"
            ),
            "claim_scope": (
                "Lossless dielectric sphere (eps_r = 2.56, m = 1.6) "
                "monostatic backscatter RCS vs the exact Bohren-Huffman "
                "Mie series (re-implemented twice: the script's "
                "self-witnessing oracle with Rayleigh / m->1 / convergence "
                "/ unitarity checks, and again in the frozen gate test), "
                "ka 0.5-2.5 at a derived CPU-scale operating point "
                "(internal-wavelength resolution floor RES >= 24 = 15*m; "
                "cells-per-radius 6.4 coarse / 12.8 fine; single-run "
                "extraction — monostatic_rcs is computed from the raw run "
                "by construction (rfx/rcs.py), so the campaign plan's "
                "two-run subtracted path is a no-op for this observable "
                "and is deliberately not paid for (PR #476 review F1); "
                "20-cell canonical clearance; F0 = 3 GHz). The path under test is rfx's BINARY "
                "dielectric rasterize (no sub-cell interface averaging "
                "exists) — this is a measured envelope of that staircase, "
                "not a smooth-interface accuracy claim. GATED: coarse ka "
                "<= 1.25 ONLY (single tier), at gate = round-UP(committed "
                "clearance-scan envelope x 1.5) — the envelope population "
                "(7-point clearance scan plus three domain realizations) "
                "is committed fixture data, the gate test recomputes it, "
                "hard-pins the constant, and binds the script's live "
                "constant. NO fine rung is gated — a measured decision: "
                "the cpr-12.8 clearance scan (committed as "
                "fine_rung_witness) envelopes 3.75 dB (ka=0.75, "
                "clearance-monotonic) and 3.04 dB (ka=1.0), barely below "
                "the coarse 4.18, so doubled resolution does not buy a "
                "domain-robust tighter claim; the clear=20-only apparent "
                "convergence was the PR #475 F1 single-sample aliasing "
                "class, caught before commit. REPORTED, NOT GATED: every "
                "bin with ka >= 1.5 — under a domain-size-"
                "only change the delta spans 11.7 dB (ka=1.75: "
                "+6.67/+3.99/-5.05 at clearance 20/30/40) and 29.0 dB "
                "(ka=2.5: +6.31/+1.32/-22.65), worse than the PEC twin; "
                "truncation is FALSIFIED (<= 0.09 dB at 2x steps), "
                "incident leakage at the monostatic bin is EXCLUDED by "
                "rcs.py's documented backscatter leakage null (~90 dB) — "
                "NOT by a sub-vs-unsub probe, which is a same-array "
                "tautology for this observable (PR #476 review F1) — "
                "resolution is non-monotonic at ka >= 1.5 (ka=1.75: "
                "+6.67/-11.58/+3.53 over cpr 6.4/9.6/12.8) while "
                "appearing convergent only at single clearances inside "
                "the gated region, and the domain-size axis dominates. Non-FDTD corroboration "
                "(Bempp PMCHWT) is an offline follow-up; no Bempp leg is "
                "committed for the dielectric case yet. "
                "MATERIAL FIDELITY (issue #812 re-gate, 2026-09-01): until #812 no gate in this case looked at the permittivity that was actually rasterized, and the dB channel cannot do that job -- the Mie oracle's sensitivity is 9.816/10.202/9.978/7.134 dB per unit RELATIVE permittivity at ka = 0.50/0.75/1.00/1.25, so the 6.3 dB window is a FACTOR-wide window in eps_r: the width of that window is COMMITTED rather than restated here -- validation/crossval/_17_dielectric_results/material_blind_window.json, keys summary.pass_runs_eps (every contiguous PASS run on the declared grid: [2.0, 4.5] and [5.0, 5.6]), summary.fail_islands_eps (the FAIL island 4.6-4.9 between them, where the ka = 1.25 bin sits on a Mie resonance -- found by the round-2 review after the original grid stepped over it; the grid now carries 0.1 steps from 4.0 to 5.6), summary.blind_window_bracket_eps (the run containing the declared 2.56, [2.0, 4.5]), summary.outer_pass_envelope_eps ([2.0, 5.6]), summary.first_failing_eps_below (1.8), summary.first_failing_eps_above (4.6) and summary.blind_window_over_material_gate_x (237.5, over the whole pass set), with the per-permittivity rows under scan[*] -- built with NO FDTD from the committed gated_coarse deltas plus the Mie oracle (scripts/diagnostics/build_cv17_material_blind_window.py) and re-derived from the same committed rows in tests/crossval/test_rcs_dielectric_sphere_mie_gates.py. Live defect runs at round 1's four permittivities (1.8 / 2.0 / 5.5 / 6.0) return the model's pass/fail verdict (design note section 5.1, prose-only magnitudes), but INSIDE the island they do not: the round-2 probe validation/crossval/_17_dielectric_results/cv17_permittivity_island.json (rasterizer delivering eps 4.5 ... 5.0, oracle at 2.56, four gated bins each) agrees with the model at summary.n_verdicts_agree_with_model = 3 of summary.n_runs = 6 -- the live gate PASSES at 4.6 / 4.7 / 4.8 where the model says FAIL (live 4.545 / 5.152 / 6.138 dB vs model 7.134 / 9.039 / 9.787) and FAILS only at 4.9 (6.405 dB). The first-order model's assumption -- discretization error independent of the permittivity -- does not hold on the ka = 1.25 Mie resonance, so the model over-predicts the island; the LIVE blind window is wider than the model's, which strengthens the conclusion this gate exists for and is stated rather than tuned. The 6.3 dB gate is round-up(envelope x 1.5) and cannot be tightened under the repo rule, and no tightening of it could resolve a permittivity error of a few percent anyway; this is stated rather than papered over. The permittivity is therefore gated on its OWN channel, the material twin of the existing per-row a_eff geometric gate: G17-A requires the permittivity read back out of the rasterized array to be within 0.5% relative of the declared 2.56 -- derived as half the gate's own 0.1 dB reporting quantum divided by the worst gated-ka sensitivity 10.202 dB per unit relative eps -- and G17-B requires the array to hold EXACTLY two distinct values (background 1.0 and the declared eps_r), which is the first check anywhere of this case's headline BINARY-rasterize claim; sub-cell averaging, a partial fill or a smoothed interface each add a third value. Measured on today's code the realized permittivity is float32(2.56) = 2.5599999428, i.e. 2.2e-8 relative, four orders inside G17-A, with exactly two distinct values at every gated bin."
            ),
            "config": {
                "f0_hz": F0, "bandwidth": BANDWIDTH, "eps_r": EPS_R,
                "refractive_index": M_IDX,
                "resolution_floor": RES_FLOOR,
                "coarse_cells_per_radius": COARSE_CPR,
                "fine_cells_per_radius": FINE_CPR,
                "clear_cells_canonical": CLEAR_CELLS_DEFAULT,
                "cpml_layers": CPML_LAYERS,
                "polarization": "ez",
                "subtract_incident_reference": False,
                "subtract_note": "monostatic_rcs is always computed from "
                                 "the raw run (rfx/rcs.py); the flag would "
                                 "be a no-op at 2x compute (PR #476 F1)",
            },
            "gates": {
                "coarse_ka": KA_GATED_COARSE, "coarse_gate_db": GATE_COARSE_DB,
                "eps_realized_tol": EPS_REALIZED_TOL,
                "n_distinct_eps_expected": N_DISTINCT_EPS_EXPECTED,
                "coarse_measured_envelope_db": round(env_coarse, 3),
                "fine_rung_witness_envelope_db": round(env_fine_witness, 3),
                "posture": "single gated tier: gate = round-UP(measured "
                           "clearance-scan envelope x 1.5) to 0.1 dB "
                           "(PR #475 convention); NO fine rung is gated "
                           "(measured decision, see fine_rung_witness); "
                           "every bin with ka>=1.5 is reported, never "
                           "gated; if ka=1.25 ever exceeds its gate, move "
                           "the coarse fence to ka<=1.0 rather than widen",
            },
            "gated_coarse": gated,
            "diagnostic_curve_clear20": diagnostic,
            "domain_realizations": domains,
            "clearance_scan": scan,
            "fine_rung_witness": fine_witness,
            "truncation_witness": trunc,
            "provenance": {
                "generated_by": "validation/crossval/17_dielectric_sphere_mie.py --write-fixture",
                "oracle": "in-script Bohren-Huffman dielectric Mie with "
                          "Rayleigh/m->1/convergence/unitarity witnesses "
                          "(re-run and printed above)",
                "no_preflight_note": (
                    "compute_rcs is a functional entry point with NO "
                    "preflight; the operating-point guarantees (internal-"
                    "wavelength floor, cells-per-radius floor, cell-unit "
                    "clearance, transit-scaled steps) are enforced by "
                    "construction inside run_point (max()/derived "
                    "expressions — deliberately no assert to grep for)."
                ),
                "offline_probes_2026_07_27": (
                    "NOT committed as data (recorded here as provenance "
                    "only): resolution ladder cpr 6.4/9.6/12.8 "
                    "non-monotonic at ka {1.75, 2.5} and apparently "
                    "convergent at ka=1.0 only at single clearances; "
                    "domain clearance 20/30/40 spreads 11.7 dB (ka=1.75) "
                    "/ 29.0 dB (ka=2.5). The first revision also quoted a "
                    "sub-vs-unsub identity as a leakage witness — "
                    "RETRACTED as a same-array tautology (monostatic_rcs "
                    "is unsubtracted by construction; PR #476 review F1); "
                    "the review's bistatic sub-vs-unsub measurement "
                    "(<= 0.025 dB on this geometry) is recorded there. "
                    "Committed data witnesses: domain_realizations, "
                    "clearance_scan, fine_rung_witness, "
                    "truncation_witness."
                ),
            },
        }
        art_dir = os.path.join(_SCRIPT_DIR, "_17_dielectric_results")
        os.makedirs(art_dir, exist_ok=True)
        art = os.path.join(art_dir, "rfx.json")
        with open(art, "w") as f:
            json.dump(payload, f, indent=1)
        fix_dir = os.path.join(_REPO_ROOT, "tests", "fixtures",
                               "rcs_dielectric_sphere_mie")
        os.makedirs(fix_dir, exist_ok=True)
        fix = os.path.join(fix_dir, "fixture.json")
        with open(fix, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {art}\nwrote {fix}")

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
