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
    radius = ka * LAM / (2 * np.pi)
    res = max(RES_FLOOR, int(np.ceil(2 * np.pi * cpr / ka)))
    dx = LAM / res
    domain = 2 * radius + 2 * clear_cells * dx
    grid = Grid(freq_max=F0 * 1.5, domain=(domain,) * 3, dx=dx,
                cpml_layers=CPML_LAYERS)
    n_steps = int(max(700, np.ceil(2.2 * domain / C0 / grid.dt)) * steps_mult)
    center = (domain / 2,) * 3
    eps_r, sigma = rasterize(grid, [(Sphere(center=center, radius=radius), EPS_R, 0.0)])
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
    pi_a2 = np.pi * radius ** 2
    mono = float(result.monostatic_rcs[0])
    mie_over = mie_backscatter_over_pi_a2(M_IDX, ka)
    mie_dbsm = float(10 * np.log10(mie_over * pi_a2))
    return {
        "ka": ka, "cells_per_radius": cpr, "clear_cells": clear_cells,
        "resolution": res, "grid": list(grid.shape), "n_steps": n_steps,
        "a_over_dx": round(radius / dx, 2),
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
            f"({r['wall_s']:.0f}s)")


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
        ok &= passed
        print(_fmt(r) + ("  PASS" if passed else "  FAIL"))

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
                "committed for the dielectric case yet."
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
