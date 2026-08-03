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

WHAT IS GATED vs WHAT IS REPORTED (measured 2026-07-27; three clearance
realizations 20/30/40 cells PLUS a denser clearance scan after the PR #475
review found the original 3-point sample ALIASED at ka = 4.0)
---------------------------------------------------------------------------
GATED (exit-1 on failure):
  * coarse bins ka {0.5, 0.75, 1.0, 1.25}: |rfx - Mie| envelope over the
    committed clearance scan -> gate = round-UP(envelope x 1.5) to 0.1 dB
    (constants below; the gate test recomputes the envelope from the
    fixture and asserts the relation, so neither number can drift alone).
    ROUNDING CONVENTION CHANGE (deliberate, not drift): the first revision
    rounded the x1.5 product DOWN (3.219 -> 3.2); this revision rounds UP
    (3.267 -> 3.3) so a measured envelope never has margin shaved off by
    rounding. Relative to the first revision this widens the coarse gate
    by 0.1 dB — stated here explicitly per the no-silent-gate-loosening
    rule (PR #475 review, caution 2).
  * fine rung (cells-per-radius 12.8) at ka = 2.0 only: same posture.
  * HEADROOM NOTE (PR #475 review, caution 3): the coarse envelope is
    dominated by the fence-edge bin ka = 1.25 (-2.15/-2.18 dB at
    clearances 30/45), and the domain spread rises monotonically toward
    the fence (0.6 -> 2.5 dB across ka 0.5 -> 1.25, jumping to ~5 dB at
    ka = 1.5). If a future clearance pushes ka = 1.25 past the gate, the
    honest response is to move the coarse fence DOWN to ka <= 1.0, not to
    widen the gate again.
REPORTED, NOT GATED (documented-unconverged at this operating point):
  * every coarse bin with ka >= 1.5, and the fine rung at ka = 3.0 AND
    ka = 4.0. Near the deep Mie interference nulls, under a domain-size-
    only change (metric named explicitly, PR #475 review D3): the
    domain-to-domain SPREAD reaches 8.0 dB at coarse ka=1.75 and 14.5 dB
    at fine ka=3.0 (across the committed clearance scan), the worst
    single-point |delta| is 11.1 dB (coarse ka=3.0, clearance 30) and
    9.3 dB (fine ka=3.0, clearance 35), and the rfx sigma(ka)
    local-minimum POSITIONS also move with domain size — so neither null
    magnitude nor null position is gateable here. ka=4.0 (fine) was GATED in the first
    revision of this script on a 3-clearance sample {20,30,40} that
    happened to hit passing values; the review's denser scan showed it
    fails a 3.5 dB gate at 9 of 13 clearances (max 6.17 dB at clear=26)
    and stays domain-unconverged at cpr 19.2/25.6 — it is fenced for the
    same reason as ka=3.0. Gating it was exactly the tuned-tolerance
    theater this script fences against; the committed clearance_scan
    witness now exists so the aliasing cannot recur silently.
Attribution record (five hypotheses tested; truncation + domain + clearance
scans are committed fixture DATA, the remaining three are recorded as
provenance of the 2026-07-27 offline probes — see F3 note in provenance):
  * record truncation: FALSIFIED on the gated bins (fixture data: 2x steps
    moves every gated-bin delta <= 0.07 dB);
  * volume/effective-radius: FALSIFIED (rasterized-volume ka_eff matches
    nominal to < 1%, and comparing against Mie(ka_eff) leaves deltas
    unchanged) — offline probe, recorded in provenance;
  * TFSF incident leakage (#280 class): EXCLUDED at the monostatic bin by
    rcs.py's documented backscatter leakage null (~90 dB). CORRECTION
    (PR #476 review, F1): the first revision quoted a sub-vs-unsub
    bin-identity as a falsifying probe — that comparison is a same-array
    tautology, because ``monostatic_rcs`` is ALWAYS computed from the raw
    (unsubtracted) run by construction (rfx/rcs.py). The exclusion rests
    on the documented null, not on that retracted probe;
  * resolution: non-monotonic at nulls (6.4 -> 9.6 -> 12.8 cells/radius;
    independently reproduced at cpr 19.2/25.6 in the PR #475 review) —
    offline probe, recorded in provenance;
  * domain size: the dominant axis — committed as domain_realizations +
    clearance_scan fixture data.

NOTE on preflight: this script drives the functional ``compute_rcs`` entry
point, which runs NO preflight (unlike ``Simulation.run()``). The
operating-point asserts above (cells-per-radius floor, clearance in cells,
transit-scaled steps) are the hand-ported equivalents; there is no
"All checks passed" line to quote, and this docstring says so rather than
implying one (PR #475 review, F5).

Usage:
  python validation/crossval/16_pec_sphere_mie_ka_sweep.py            # gated set (~1 min CPU)
  python validation/crossval/16_pec_sphere_mie_ka_sweep.py --full     # + 15-point diagnostic curve
  python validation/crossval/16_pec_sphere_mie_ka_sweep.py --write-fixture
      # full 3-domain measurement + clearance scan + witnesses; regenerates
      # validation/crossval/_16_ka_sweep_results/rfx.json AND
      # tests/fixtures/rcs_mie_ka_sweep/fixture.json (~30 min CPU)

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

from tests._gate_policy import gate_from_envelope  # noqa: E402

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
KA_FINE_GATED = [2.0]
KA_FINE_REPORTED = [3.0, 4.0]   # 4.0 fenced after PR #475 review (F1 aliasing)

# Denser clearance scan for the GATED bins (the 3-point {20,30,40} sample
# aliased at ka=4.0 — PR #475 review F1). The fixture commits this scan so
# the envelope is auditable; the gate test recomputes it from the data.
CLEARANCE_SCAN = [15, 20, 25, 30, 35, 40, 45]

# Measured-envelope x 1.5 posture (no silent loosening — regenerate the
# fixture, re-derive the envelope from the committed clearance_scan, and
# write a root-cause before ever touching these).
GATE_COARSE_DB = 3.3   # = round-up(measured clearance-scan envelope x 1.5)
GATE_FINE_DB = 4.0     # = round-up(measured clearance-scan envelope x 1.5), ka=2.0 only


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
        fine_reported = []
        for ka in KA_FINE_REPORTED:
            r3 = run_point(ka, FINE_CPR, CLEAR_CELLS_DEFAULT)
            print(f"  fine rung ka={ka} (REPORTED, not gated):")
            print(_fmt(r3))
            fine_reported.append(r3)   # own key, NOT appended to the coarse curve (F7)

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

        # Denser clearance scan over the GATED bins (anti-aliasing witness,
        # PR #475 F1) + the fenced fine bins so the aliasing story is data.
        print("\n== clearance scan (gated bins + fenced fine bins) ==")
        scan = {"clearances": CLEARANCE_SCAN, "coarse": {}, "fine": {}}
        for ka in KA_GATED_COARSE:
            rows = [run_point(ka, COARSE_CPR, c) for c in CLEARANCE_SCAN]
            scan["coarse"][str(ka)] = rows
            print(f"  coarse ka={ka}: deltas",
                  " ".join(f"{r['delta_db']:+.2f}" for r in rows))
        for ka in KA_FINE_GATED + KA_FINE_REPORTED:
            rows = [run_point(ka, FINE_CPR, c) for c in CLEARANCE_SCAN]
            scan["fine"][str(ka)] = rows
            print(f"  fine   ka={ka}: deltas",
                  " ".join(f"{r['delta_db']:+.2f}" for r in rows))

        # Measured envelopes over EVERYTHING committed for the gated bins
        # (canonical run + 30/40 realizations + clearance scan). The gates
        # must equal round-up(envelope x 1.5); the gate test re-asserts this
        # from the fixture data (PR #475 F6).
        coarse_deltas = [abs(r["delta_db"]) for r in gated]
        coarse_deltas += [abs(r["delta_db"]) for c in ("30", "40")
                          for r in domains[c] if r["ka"] <= max(KA_GATED_COARSE)]
        coarse_deltas += [abs(r["delta_db"]) for ka in KA_GATED_COARSE
                          for r in scan["coarse"][str(ka)]]
        fine_deltas = [abs(r["delta_db"]) for r in fine]
        fine_deltas += [abs(r["delta_db"]) for c in ("30_fine", "40_fine")
                        for r in domains[c] if r["ka"] in KA_FINE_GATED]
        fine_deltas += [abs(r["delta_db"]) for ka in KA_FINE_GATED
                        for r in scan["fine"][str(ka)]]
        env_coarse = max(coarse_deltas)
        env_fine = max(fine_deltas)
        print(f"\n  measured envelopes: coarse {env_coarse:.3f} dB "
              f"(gate {GATE_COARSE_DB}), fine {env_fine:.3f} dB "
              f"(gate {GATE_FINE_DB})")
        if not (GATE_COARSE_DB >= env_coarse and
                GATE_COARSE_DB <= gate_from_envelope(env_coarse, quantum=10) + 0.05):
            print("  ENVELOPE/GATE MISMATCH (coarse) — fix GATE_COARSE_DB")
            ok = False
        if not (GATE_FINE_DB >= env_fine and
                GATE_FINE_DB <= gate_from_envelope(env_fine, quantum=10) + 0.05):
            print("  ENVELOPE/GATE MISMATCH (fine) — fix GATE_FINE_DB")
            ok = False

        # Truncation witness ON THE GATED BINS (PR #475 F4): 1x vs 2x steps.
        trunc = []
        for ka, cpr in ([(k, COARSE_CPR) for k in KA_GATED_COARSE]
                        + [(k, FINE_CPR) for k in KA_FINE_GATED]):
            r1 = run_point(ka, cpr, CLEAR_CELLS_DEFAULT, steps_mult=1.0)
            r2 = run_point(ka, cpr, CLEAR_CELLS_DEFAULT, steps_mult=2.0)
            trunc.append({"ka": ka, "cells_per_radius": cpr,
                          "delta_1x_db": r1["delta_db"],
                          "delta_2x_db": r2["delta_db"]})
            print(f"  truncation witness ka={ka} cpr={cpr}: "
                  f"1x {r1['delta_db']:+.2f} -> 2x {r2['delta_db']:+.2f} dB")

        payload = {
            "schema": "rfx.rcs_mie_ka_sweep",
            "schema_version": 2,
            "campaign": (
                "cross-solver validation campaign, item 1: PEC-sphere "
                "exact-Mie ka sweep (extends the committed E4 ladder and "
                "fine ka~1 fixtures)"
            ),
            "claim_scope": (
                "PEC-sphere monostatic backscatter RCS vs the exact Mie "
                "series (re-implemented twice against the same Ruck/"
                "Bohren-Huffman convention: the committed self-witnessing "
                "oracle in the script and scipy.special again in the "
                "frozen gates; a four-way convention-independent check is "
                "recorded in the PR #475 review), ka 0.5-4.0 at a derived "
                "CPU-scale operating point (cells-per-radius 6.4 coarse / "
                "12.8 fine, 20-cell canonical clearance, F0=3 GHz). "
                "GATED: coarse ka <= 1.25 and fine-rung ka=2.0 only, at "
                "gate = round-up(measured clearance-scan envelope x 1.5) "
                "— envelopes are recomputed from the committed "
                "clearance_scan/domain_realizations data by the gate "
                "test, so gate and envelope cannot drift apart. NOT "
                "GATED: every coarse bin with ka >= 1.5 and the fine "
                "rung at ka=3.0 and ka=4.0 — near the deep Mie nulls, "
                "under a domain-size-only change, the domain-to-domain "
                "SPREAD reaches 8.0 dB (coarse, at ka=1.75) and 14.5 dB "
                "(fine, at ka=3.0 across the committed clearance scan), "
                "the worst single-point |delta| is 11.1 dB (coarse "
                "ka=3.0, clearance 30) and 9.3 dB (fine ka=3.0, "
                "clearance 35), the rfx sigma(ka) local-minimum "
                "positions move with domain size, and fine ka=4.0 fails "
                "a 3.5 dB gate at 9 of 13 clearances (max 6.17 dB at "
                "clearance 26; the original 3-clearance sample "
                "{20,30,40} ALIASED onto passing values — PR #475 "
                "review F1), so neither null magnitude nor null "
                "position is a converged observable here. Committed witnesses: domain_realizations, "
                "clearance_scan, truncation_witness (gated bins). The "
                "effective-radius and 9.6-rung resolution probes were "
                "offline (2026-07-27) and are recorded as provenance, "
                "not data; incident leakage at the monostatic bin is "
                "EXCLUDED by rcs.py's documented backscatter leakage "
                "null, the formerly-quoted sub-vs-unsub probe having "
                "been RETRACTED as a same-array tautology "
                "(monostatic_rcs is unsubtracted by construction — "
                "PR #476 review F1). Non-FDTD corroboration (Bempp "
                "EFIE) exists for ka <= 2 in "
                "tests/fixtures/rcs_sphere_three_way/; extending it "
                "above ka=2 is an offline follow-up."
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
                "coarse_measured_envelope_db": round(env_coarse, 3),
                "fine_ka": KA_FINE_GATED, "fine_gate_db": GATE_FINE_DB,
                "fine_measured_envelope_db": round(env_fine, 3),
                "posture": "gate = round-UP(measured clearance-scan envelope "
                           "x 1.5) to 0.1 dB — round-up convention adopted "
                           "deliberately in the PR #475 revision (the first "
                           "revision rounded down; up never shaves margin "
                           "off a measured envelope); coarse ka>=1.5 and "
                           "fine ka=3.0/ka=4.0 are reported, never gated; "
                           "if ka=1.25 ever exceeds its gate, move the "
                           "coarse fence to ka<=1.0 rather than widen",
            },
            "gated_coarse": gated,
            "gated_fine": fine,
            "fine_rung_reported": fine_reported,
            "diagnostic_curve_clear20": diagnostic,
            "domain_realizations": domains,
            "clearance_scan": scan,
            "truncation_witness": trunc,
            "provenance": {
                "generated_by": "validation/crossval/16_pec_sphere_mie_ka_sweep.py --write-fixture",
                "oracle": "tests/fixtures/rcs_sphere_mie/mie_oracle.py "
                          "(witnesses re-run and printed above)",
                "no_preflight_note": (
                    "compute_rcs is a functional entry point with NO "
                    "preflight; the operating-point asserts (cells-per-"
                    "radius floor, cell-unit clearance, transit-scaled "
                    "steps) are the hand-ported equivalents."
                ),
                "offline_probes_2026_07_27": (
                    "NOT committed as data (recorded here as provenance "
                    "only): rasterized-volume ka_eff within 1% of nominal "
                    "with deltas unchanged vs Mie(ka_eff) at ka "
                    "{1.75,2,3,4}; 6.4/9.6/12.8 resolution ladder "
                    "non-monotonic at nulls (independently reproduced at "
                    "cpr 19.2/25.6 in the PR #475 review). RETRACTED "
                    "(PR #476 review F1): the sub-vs-unsub bin-identity "
                    "formerly quoted here as a leakage probe is a "
                    "same-array tautology — monostatic_rcs is always "
                    "computed from the raw run by construction "
                    "(rfx/rcs.py); leakage exclusion at backscatter rests "
                    "on rcs.py's documented ~90 dB leakage null. "
                    "Committed data witnesses: domain_realizations, "
                    "clearance_scan, truncation_witness."
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
