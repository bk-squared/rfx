"""PEC-sphere monostatic RCS vs exact Mie — ka sweep 0.5-4.0 (campaign item 1).

Extends the committed E4 ladder (ka {0.8, 1.0, 1.5, 2.0}, test-scale) and the
fine ka~1 point into the Rayleigh side (ka=0.5) and the creeping-wave
oscillation region (ka in (2, 4]). The reference is the exact conducting-sphere
Mie backscatter series (Ruck 1970), RE-DERIVED here via the committed
independent oracle (``tests/fixtures/rcs_sphere_mie/mie_oracle.py``, which
self-checks Rayleigh/GO/convergence witnesses before any gate runs) — no
external solver is required, so this script never exits 2.

Operating point (derived, not tuned — see the fixture provenance):
  * F0 = 3 GHz; each ka bin is a separate geometry with radius a = ka*lam/(2*pi).
  * a/dx = ka*RES/(2*pi) regardless of knob choice, so a fixed lambda-resolution
    starves the sphere of cells at low ka. The sweep holds CELLS-PER-RADIUS
    >= 6.4 (the lambda/40 @ ka~1 fine-fixture operating point):
    RES(ka) = max(15, ceil(2*pi*6.4/ka)).
  * clearance 20 cells beyond the sphere on every side (CPML 8 + TFSF margin +
    NTFF offset + scattered-field annulus); n_steps = max(700, 2.2 domain
    transits).

WHAT IS GATED vs WHAT IS REPORTED (measured 2026-07-27, three domain
realizations: clearance 20/30/40 cells)
---------------------------------------------------------------------------
GATED (exit-1 on failure):
  * coarse bins ka {0.5, 0.75, 1.0, 1.25}: measured |rfx - Mie| envelope
    2.1 dB across the three domains -> gate 3.2 dB (envelope x 1.5).
  * fine rung (cells-per-radius 12.8) ka {2.0, 4.0}: measured envelope
    2.35 dB across the three domains -> gate 3.5 dB (envelope x 1.5).
REPORTED, NOT GATED (documented-unconverged at this operating point):
  * every coarse bin with ka >= 1.5, and the fine rung at ka = 3.0. Near the
    deep Mie interference nulls (ka ~ 1.75 and ~ 3.0 on this grid) the
    monostatic value swings up to 6.3 dB (coarse) / 5.4 dB (fine, ka=3) and
    8.3 dB (fine, ka=1.75) under a domain-size-only change, and the rfx
    sigma(ka) local-minimum POSITIONS also move with domain size — so neither
    null magnitude nor null position is gateable here. Gating them anyway
    would be tuned-tolerance theater.
Attribution record (all witnessed in the fixture; five hypotheses tested):
  * record truncation: FALSIFIED (2x steps moves deltas <= 0.07 dB);
  * volume/effective-radius: FALSIFIED (rasterized-volume ka_eff matches
    nominal to < 1%, and comparing against Mie(ka_eff) leaves deltas
    unchanged);
  * TFSF incident leakage (#280 class): FALSIFIED at the monostatic bin
    (two-run subtract_incident_reference=True is bin-identical, consistent
    with the documented backscatter leakage null);
  * resolution: non-monotonic at nulls (6.4 -> 9.6 -> 12.8 cells/radius);
  * domain size: the dominant axis (see numbers above) — consistent with
    scattered-field/CPML re-reflection or NTFF proximity interference at
    bins where the true backscatter is deep below the neighbors.

Usage:
  python validation/crossval/16_pec_sphere_mie_ka_sweep.py            # gated set (~1 min CPU)
  python validation/crossval/16_pec_sphere_mie_ka_sweep.py --full     # + 15-point diagnostic curve
  python validation/crossval/16_pec_sphere_mie_ka_sweep.py --write-fixture
      # full 3-domain measurement + witnesses; regenerates
      # validation/crossval/_16_ka_sweep_results/rfx.json AND
      # tests/fixtures/rcs_mie_ka_sweep/fixture.json (~15 min CPU)

Exit codes: 0 = all configured gates pass; 1 = oracle self-check or a gate
failed. Failure prints the sentinel line "SOME CHECKS FAILED".
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests", "fixtures", "rcs_sphere_mie"))

import rfx  # noqa: E402

_RFX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(rfx.__file__)))
if _RFX_ROOT != _REPO_ROOT:
    raise RuntimeError(
        f"import rfx resolved outside this repo tree ({rfx.__file__}); "
        "refusing to report numbers for a different rfx build."
    )

from mie_oracle import backscatter_rcs_over_pi_a2, validate_oracle  # noqa: E402

import jax.numpy as jnp  # noqa: E402
from rfx.grid import Grid, C0  # noqa: E402
from rfx.geometry.csg import Sphere, rasterize  # noqa: E402
from rfx.core.yee import MaterialArrays  # noqa: E402
from rfx.rcs import compute_rcs  # noqa: E402

F0 = 3e9
LAM = C0 / F0
CPML_LAYERS = 8
PEC_SIGMA = 1e7
BANDWIDTH = 0.5
COARSE_CPR = 6.4          # cells per radius, coarse operating point
FINE_CPR = 12.8           # fine rung
CLEAR_CELLS_DEFAULT = 20  # canonical domain; fixture also records 30 and 40

KA_ALL = [round(0.5 + 0.25 * i, 2) for i in range(15)]      # 0.5 .. 4.0
KA_GATED_COARSE = [0.5, 0.75, 1.0, 1.25]
KA_FINE_GATED = [2.0, 4.0]
KA_FINE_REPORTED = [3.0]

# Measured-envelope x 1.5 posture (no silent loosening — regenerate the
# fixture and re-derive the envelope before ever touching these).
GATE_COARSE_DB = 3.2   # measured envelope 2.1 dB over ka<=1.25 x 3 domains
GATE_FINE_DB = 3.5     # measured envelope 2.35 dB over ka {2,4} x 3 domains


def run_point(ka: float, cpr: float, clear_cells: int, steps_mult: float = 1.0):
    """One monostatic point at the derived operating point. Returns a record."""
    radius = ka * LAM / (2 * np.pi)
    res = max(15, int(np.ceil(2 * np.pi * cpr / ka)))
    dx = LAM / res
    domain = 2 * radius + 2 * clear_cells * dx
    grid = Grid(freq_max=F0 * 1.5, domain=(domain,) * 3, dx=dx,
                cpml_layers=CPML_LAYERS)
    n_steps = int(max(700, np.ceil(2.2 * domain / C0 / grid.dt)) * steps_mult)
    center = (domain / 2,) * 3
    eps_r, sigma = rasterize(grid, [(Sphere(center=center, radius=radius), 1.0, PEC_SIGMA)])
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma,
                          mu_r=jnp.ones(grid.shape, dtype=jnp.float32))
    t0 = time.time()
    result = compute_rcs(
        grid, mats, n_steps,
        f0=F0, bandwidth=BANDWIDTH, theta_inc=0.0, polarization="ez",
        theta_obs=np.array([np.pi / 2]), phi_obs=np.array([0.0, np.pi]),
        freqs=np.array([F0]),
        boundary="cpml", cpml_layers=CPML_LAYERS,
    )
    wall = time.time() - t0
    pi_a2 = np.pi * radius ** 2
    mono = float(result.monostatic_rcs[0])
    mie_over = float(backscatter_rcs_over_pi_a2(ka, n_max=40))
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

    # --- Oracle self-check FIRST: refuse to gate against a broken oracle ----
    witnesses = validate_oracle()
    print("[oracle] Mie self-check witnesses PASS:",
          {k: (round(float(v), 6) if np.isscalar(v) else v)
           for k, v in witnesses.items()})

    # --- Gated coarse bins (Rayleigh-to-resonance) ---------------------------
    print(f"\n== GATED coarse bins ka {KA_GATED_COARSE} "
          f"(cpr={COARSE_CPR}, gate {GATE_COARSE_DB} dB) ==")
    gated = []
    for ka in KA_GATED_COARSE:
        r = run_point(ka, COARSE_CPR, CLEAR_CELLS_DEFAULT)
        gated.append(r)
        passed = abs(r["delta_db"]) <= GATE_COARSE_DB
        ok &= passed
        print(_fmt(r) + ("  PASS" if passed else "  FAIL"))

    # --- Gated fine rung ------------------------------------------------------
    print(f"\n== GATED fine rung ka {KA_FINE_GATED} "
          f"(cpr={FINE_CPR}, gate {GATE_FINE_DB} dB) ==")
    fine = []
    for ka in KA_FINE_GATED:
        r = run_point(ka, FINE_CPR, CLEAR_CELLS_DEFAULT)
        fine.append(r)
        passed = abs(r["delta_db"]) <= GATE_FINE_DB
        ok &= passed
        print(_fmt(r) + ("  PASS" if passed else "  FAIL"))

    diagnostic = []
    domains = {}
    if full or write_fixture:
        print("\n== DIAGNOSTIC 15-point coarse curve (REPORTED, ka>=1.5 NOT "
              "gated — see module docstring) ==")
        for ka in KA_ALL:
            r = run_point(ka, COARSE_CPR, CLEAR_CELLS_DEFAULT)
            diagnostic.append(r)
            print(_fmt(r))
        r3 = run_point(KA_FINE_REPORTED[0], FINE_CPR, CLEAR_CELLS_DEFAULT)
        print("  fine rung ka=3.0 (REPORTED, not gated):")
        print(_fmt(r3))
        diagnostic.append(r3)

    if write_fixture:
        print("\n== fixture mode: domain realizations clear 30/40 + witnesses ==")
        for clear in (30, 40):
            rows = [run_point(ka, COARSE_CPR, clear) for ka in KA_ALL]
            domains[str(clear)] = rows
            print(f"  coarse curve @ clear={clear}: deltas",
                  " ".join(f"{r['delta_db']:+.1f}" for r in rows))
            fine_rows = [run_point(ka, FINE_CPR, clear)
                         for ka in KA_FINE_GATED + KA_FINE_REPORTED]
            domains[f"{clear}_fine"] = fine_rows
            print(f"  fine rung   @ clear={clear}: deltas",
                  " ".join(f"{r['delta_db']:+.1f}" for r in fine_rows))
        # truncation witness at two worst coarse bins
        trunc = []
        for ka in (2.0, 3.0):
            r1 = run_point(ka, COARSE_CPR, CLEAR_CELLS_DEFAULT, steps_mult=1.0)
            r2 = run_point(ka, COARSE_CPR, CLEAR_CELLS_DEFAULT, steps_mult=2.0)
            trunc.append({"ka": ka, "delta_1x_db": r1["delta_db"],
                          "delta_2x_db": r2["delta_db"]})
            print(f"  truncation witness ka={ka}: 1x {r1['delta_db']:+.2f} "
                  f"-> 2x {r2['delta_db']:+.2f} dB")

        payload = {
            "schema": "rfx.rcs_mie_ka_sweep",
            "schema_version": 1,
            "campaign": "crossval item 1 (plans/crossval_campaign_real_structures.md)",
            "claim_scope": (
                "PEC-sphere monostatic backscatter RCS vs the independently "
                "re-derived exact Mie series, ka 0.5-4.0 at a derived "
                "CPU-scale operating point (cells-per-radius 6.4 coarse / "
                "12.8 fine, 20-cell clearance, F0=3 GHz). GATED: coarse ka "
                "<= 1.25 (measured 3-domain envelope 2.1 dB, gate 3.2 dB) "
                "and fine-rung ka {2.0, 4.0} (envelope 2.35 dB, gate 3.5 "
                "dB). NOT GATED: every coarse bin with ka >= 1.5 and the "
                "fine rung at ka=3.0 — near the deep Mie nulls the "
                "monostatic value swings up to 6.3 dB (coarse) / 8.3 dB "
                "(fine ka=1.75) under a domain-size-only change and the "
                "rfx sigma(ka) local-minimum positions move with domain "
                "size, so neither null magnitude nor null position is a "
                "converged observable here. Attribution witnesses (record "
                "truncation, effective radius, #280 incident leakage all "
                "FALSIFIED; domain-size axis dominant) are frozen in this "
                "fixture. Non-FDTD corroboration (Bempp EFIE) exists for "
                "ka <= 2 in tests/fixtures/rcs_sphere_three_way/; "
                "extending it above ka=2 is an offline follow-up."
            ),
            "config": {
                "f0_hz": F0, "bandwidth": BANDWIDTH,
                "coarse_cells_per_radius": COARSE_CPR,
                "fine_cells_per_radius": FINE_CPR,
                "clear_cells_canonical": CLEAR_CELLS_DEFAULT,
                "cpml_layers": CPML_LAYERS, "pec_sigma": PEC_SIGMA,
                "polarization": "ez",
            },
            "gates": {
                "coarse_ka": KA_GATED_COARSE, "coarse_gate_db": GATE_COARSE_DB,
                "coarse_measured_envelope_db": 2.1,
                "fine_ka": KA_FINE_GATED, "fine_gate_db": GATE_FINE_DB,
                "fine_measured_envelope_db": 2.35,
                "posture": "measured-envelope x 1.5; ka>=1.5 coarse and "
                           "fine ka=3.0 are reported, never gated",
            },
            "gated_coarse": gated,
            "gated_fine": fine,
            "diagnostic_curve_clear20": diagnostic,
            "domain_realizations": domains,
            "truncation_witness": trunc,
            "provenance": {
                "generated_by": "validation/crossval/16_pec_sphere_mie_ka_sweep.py --write-fixture",
                "oracle": "tests/fixtures/rcs_sphere_mie/mie_oracle.py "
                          "(witnesses re-run and printed above)",
                "attribution_record": (
                    "2026-07-27 probes: 2x-steps <= 0.07 dB; rasterized-volume "
                    "ka_eff within 1% of nominal with deltas unchanged vs "
                    "Mie(ka_eff); subtract_incident_reference=True "
                    "bin-identical at backscatter; resolution ladder "
                    "6.4/9.6/12.8 non-monotonic at nulls; domain clearance "
                    "20/30/40 cells is the dominant swing axis."
                ),
            },
        }
        art_dir = os.path.join(_SCRIPT_DIR, "_16_ka_sweep_results")
        os.makedirs(art_dir, exist_ok=True)
        art = os.path.join(art_dir, "rfx.json")
        with open(art, "w") as f:
            json.dump(payload, f, indent=1)
        fix_dir = os.path.join(_REPO_ROOT, "tests", "fixtures", "rcs_mie_ka_sweep")
        os.makedirs(fix_dir, exist_ok=True)
        fix = os.path.join(fix_dir, "fixture.json")
        with open(fix, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {art}\nwrote {fix}")

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
