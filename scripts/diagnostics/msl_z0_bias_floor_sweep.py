#!/usr/bin/env python3
"""MSL thru |S11| floor vs mesh, post-#511/#507 corrected extractor (issue #487).

PRE-DECLARATION (written before this script was run — R2-tight, one campaign)
-------------------------------------------------------------------------------
Issue #487's original 0.16-0.22 "Yee-staircase floor" envelope was measured on
the pre-#511/#507 extractor (far-port-echo single-ratio assembly + a modal-V
span one Ez edge too long) and is RETIRED — see #487's "Unblocking: #511 and
#507 merged" comment. Two post-fix points already exist:

    aligned  dx=84.67um (h_sub/3): mean|S11|=0.0501, Z0=44.11 ohm
    bisecting dx=80um (h_sub/dx=3.175): mean|S11|=0.1160, Z0=57.58 ohm

and both are consistent with a NEW mechanism: |S11|_floor is close to
|Gamma_implied| = |(Z0_measured - Z0_HJ)/(Z0_measured + Z0_HJ)|, i.e. it is the
mismatch between the rasterized line's own Z0 and the analytic Hammerstad-
Jensen anchor S is normalized against, not a resolution artifact per se
(ratio mean|S11|/|Gamma_implied| = 1.22 and 1.26 on those two points).

This script is the pre-declared re-sweep that (1) checks that mechanism holds
across a wider mesh grid, (2) asks whether refinement lowers the floor WITHIN
an alignment class, and (3) asks whether the h_sub/dx alignment class (mixed-
cell danger zone [0.10, 0.40], MESH-ALIGNMENT RULE) shifts the floor at a
comparable cell count, independent of refinement.

DX GRID (predeclared, fixed before running)
---------------------------------------------
Aligned class (frac(h_sub/dx) == 0, h_sub/dx integer):
    dx = h_sub/3  (84.67 um, 3 substrate cells)
    dx = h_sub/4  (63.50 um, 4 substrate cells)
    dx = h_sub/5  (50.80 um, 5 substrate cells)
    dx = h_sub/6  (42.33 um, 6 substrate cells)

Misaligned / mixed-cell-danger-zone class (frac(h_sub/dx) in [0.10, 0.40]):
    dx = 80 um  (h_sub/dx = 3.175, frac = 0.175 -- the committed gate-test mesh)
    dx = 60 um  (h_sub/dx = 4.233, frac = 0.233 -- almost the SAME dx as the
                 aligned h_sub/4 point (63.5 um) but misaligned, so this pair
                 isolates alignment class from refinement almost exactly)

Both classes share the same fixture as
``thin_conductor_cell_thickness_probe.py`` / ``test_msl_thru_line_passive_gate``
(RO4350B eps_r=3.66, h_sub=254um, W=600um, L=10mm, one-cell PEC trace, ports at
both ends, band 3.0-4.5 GHz, ``compute_msl_s_matrix(n_freqs=30, num_periods=12)``).

EXPECTATIONS (predeclared; falsifiable, not adjusted after the run)
-----------------------------------------------------------------------
(a) floor approx= |Gamma_implied| within ~1.3x, i.e.
    0.77 <= mean|S11|_raw / |Gamma_implied| <= 1.35 at EVERY point.
    If this breaks anywhere, STOP: do not write the Leg-2 advisory, report the
    dump instead -- it would mean the mechanism story from #487's unblocking
    comment is incomplete, not merely under-measured.
(b) WITHIN the aligned class, refinement reduces |Gamma_implied| monotonically
    (or very close to it) as dx: h_sub/3 -> h_sub/4 -> h_sub/5 -> h_sub/6.
(c) At comparable cell count, the misaligned/mixed-cell point reads a LARGER
    |Gamma_implied| (and floor) than the aligned point of similar or even finer
    dx -- alignment class is a lever independent of raw refinement. Primary
    witness: dx=60um (misaligned, n=4.233) vs dx=63.5um (aligned, n=4, i.e.
    coarser dx yet aligned).

Both the enforce_passivity=False (raw) and enforce_passivity=True (default,
passivity-projected) mean|S11| are recorded per point from ONE FDTD run per
mesh point (the projection in ``_project_passive`` is a post-hoc SVD clip of
the already-computed raw S -- no second FDTD run needed). Z0 and beta are
never projected (see ``compute_msl_s_matrix`` docstring), so Gamma_implied
uses the same Z0 in both columns.

RUNTIME
-------
6 mesh points x 1 FDTD run each (n_freqs=30, num_periods=12), matching the
calibration test's scale. ~5-8 min/point on one CPU core (thin-conductor probe
precedent). Settling (source-off transient decay, dB) is quoted per point --
this is a ring-down settling witness per the repo's open-domain rule, not a
truncation artifact.

    python scripts/diagnostics/msl_z0_bias_floor_sweep.py

POST-RUN REVIEW NOTE (2026-08-02, appended after the run; not part of the
pre-declaration)
--------------------------------------------------------------------------
Adversarial review (PR #535) found two things worth recording here AFTER
the fact, WITHOUT touching the pre-declared script body or its committed
JSON above this note -- doing so would destroy the auditable property
that the criteria predate the data (the JSON's verdict block is, and
remains, the AS-RUN one, computed by ``_check_expectations`` exactly as
declared and run):

* The passivity projection's own effect is small across this sweep --
  ``max_passivity_correction`` <= 0.00144 at every point (see the
  committed JSON) -- so "raw" and "default" mean|S11| differ by at most
  that much; the analysis in this script reads the raw column
  throughout.

* ``_check_expectations``'s coded criteria are WEAKER than their own
  prose in two places, found post-run: (b)'s "near-monotone
  non-increasing" check used an ABSOLUTE 0.01 epsilon, which is ~3x the
  finest measured |Gamma_implied| (0.00328, at h_sub/6) and would
  silently pass a real reversal at that end of the sweep; (c)'s check
  compared |Gamma_implied| only, not the |S11| FLOOR its own prose claims
  alignment class shifts. A FUTURE re-run's script should code (b) with a
  RELATIVE tolerance (~2%) instead of an absolute one, and code (c) on
  ``mean_s11_raw`` directly, in addition to ``abs_gamma_implied``.
  Recomputing (NOT re-running -- same committed rows) through those
  tightened criteria changes no conclusion: (a) still breaks at h_sub/6
  (unchanged); (b) is still True under a 2% relative bound (the aligned-
  class Gamma sequence is strictly decreasing, so a tighter bound does
  not flip it); (c) is True under EITHER check (misaligned 60um
  mean_s11_raw=0.06566 > aligned h_sub/4 mean_s11_raw=0.02228, in
  addition to the Gamma comparison already in the artifact). Recorded
  here as prose only.

POST-RUN REVIEW NOTE (2026-08-27, appended after the fact; issue #752 --
same rule as the note above: the committed rows and verdict block above
are NOT touched)
--------------------------------------------------------------------------
Adversarial review found that ``z0_hj_ohm`` above (and the "-7.9%/-3.8%/
-1.2%/+0.7%" vs "+20.2%/+11.0%" comparison it enables) is Hammerstad-
Jensen on the DECLARED 600/254um board at every dx, but the two
misaligned points (80um, 60um) rasterize a substrate 320um/300um thick
(+26%/+18% vs declared -- the half-open rasterizer rule rounds h_sub/dx
UP), while the four aligned points realize h_sub exactly. Comparing
declared-board deviations across those different realized boards reads
as "misalignment makes Z0 extraction worse", but is largely board
rasterization, not extractor bias. A NEW SIBLING script,
``msl_z0_bias_floor_sweep_realized_anchor.py``, reads z0_measured_ohm
back out of this file's own JSON (never re-solved) and scores it against
Hammerstad-Jensen on each point's REALIZED h/W instead
(``sim.fidelity_report()``, no solve): the extractor tracks that
realized-board anchor to within 0.4% at all six points, aligned or not
(sibling artifact: ``msl_z0_bias_floor_sweep_realized_anchor.json``,
same directory). See ``rfx/api/_preflight.py``'s
``_check_msl_port_geometry`` class docstring and checks 2/2b for the
corrected advisory text this drives.

POST-RUN REVIEW NOTE (2026-09-02, audit finding A1; appended after the
fact, pre-declared body/JSON untouched)
--------------------------------------------------------------------------
The "within 0.4% at all six points" reading just above is a PRE-#802
result. The realized-board anchor reused each point's z0_measured_ohm
from THIS file's JSON, solved on the pre-#802 f32 rasterization. Main's
exact-coordinate rasterizer (#802/#834) moves three of the six aligned
points' realized trace width (h_sub/3 677.3->592.7um, h_sub/5
609.6->558.8um, h_sub/6 592.7->635.0um), so both the measured Z0 and the
realized-board HJ anchor on those points change and the 0.4% figure is no
longer a live bound for them. The misaligned pair (dx=80/60um) did not
move (realized W unchanged), so its ~0.2% agreement is still
representative. The user-facing preflight advisories were corrected
(finding A1) to state only the qualitative realized-board claim and to
point here for the OWED re-solve. RE-SOLVE: run this script (6 FDTD
points) on main, then regenerate the anchor with
``msl_z0_bias_floor_sweep_realized_anchor.py``; a single-point check of
the h_sub/3 re-solve (2026-09-02, jax_enable_x64=False, settling
-110.0/-113.1 dB) read Z0=48.16 ohm against HJ(592.7um,254um)=48.27 ohm,
dev -0.23% -- consistent with the 0.4% band still holding on the
post-#802 board, but the full six-point re-solve is what a live bound
requires.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from rfx import Box, Simulation
from rfx.api._sparams import _project_passive
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3
F_MAX = 5e9
GATE_F_LO, GATE_F_HI = 3.0e9, 4.5e9

# --- predeclared dx grid: (label, dx_metres) ---
DX_GRID = [
    ("aligned h_sub/3", H_SUB / 3.0),
    ("aligned h_sub/4", H_SUB / 4.0),
    ("aligned h_sub/5", H_SUB / 5.0),
    ("aligned h_sub/6", H_SUB / 6.0),
    ("misaligned 80um", 80e-6),
    ("misaligned 60um", 60e-6),
]

OUT_DIR = REPO / "scripts" / "diagnostics" / "msl_z0_bias_floor_sweep"


def run_one(label: str, dx: float) -> dict:
    lx = L_LINE + 2 * PORT_MARGIN
    ly = W_TRACE + 2 * (2 * H_SUB + 8 * dx)   # fixed-clearance formula (2026-05-04 sweep)
    lz = H_SUB + 1.5e-3
    sim = Simulation(
        freq_max=F_MAX, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="ro4350b")
    yc = ly / 2.0
    sim.add(Box((0.0, yc - W_TRACE / 2.0, H_SUB),
                (lx, yc + W_TRACE / 2.0, H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, yc, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, yc, 0.0), width=W_TRACE,
                     height=H_SUB, direction="-x", impedance=50.0)

    t0 = time.time()
    # Diagnostics see the RAW extraction (issue #470 rule) -- projection is
    # then applied by hand below from the SAME raw S, no second FDTD run.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(n_freqs=30, num_periods=12,
                                       enforce_passivity=False)
    dt = time.time() - t0

    S_raw = np.asarray(res.S)
    S_proj, correction = _project_passive(S_raw)
    S_proj = np.asarray(S_proj)
    Z0 = np.asarray(res.Z0)
    f = np.asarray(res.freqs)
    m = (f >= GATE_F_LO) & (f <= GATE_F_HI)

    s11_raw = np.abs(S_raw[0, 0, m])
    s11_proj = np.abs(S_proj[0, 0, m])
    s21_raw = np.abs(S_raw[1, 0, m])
    z0_meas = float(Z0[0, m].real.mean())

    z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(W_TRACE, H_SUB, EPS_R)
    gamma_implied = (z0_meas - z0_hj) / (z0_meas + z0_hj)
    mean_s11_raw = float(s11_raw.mean())
    ratio = mean_s11_raw / abs(gamma_implied) if gamma_implied != 0 else float("nan")

    n_z_sub_exact = H_SUB / dx
    n_below = int(n_z_sub_exact)
    frac = n_z_sub_exact - n_below

    settling_db = None if res.settling_db is None else [
        round(float(v), 1) for v in np.asarray(res.settling_db)
    ]

    warn_msgs = [str(w.message) for w in caught]

    return {
        "label": label,
        "dx_um": round(dx * 1e6, 3),
        "n_z_sub_exact": round(n_z_sub_exact, 4),
        "frac": round(frac, 4),
        "mixed_cell_danger_zone": bool(0.10 <= frac <= 0.40),
        "z0_measured_ohm": round(z0_meas, 3),
        "z0_hj_ohm": round(z0_hj, 3),
        "eps_eff_hj": round(eps_eff_hj, 4),
        "gamma_implied": round(gamma_implied, 5),
        "abs_gamma_implied": round(abs(gamma_implied), 5),
        "mean_s11_raw": round(mean_s11_raw, 5),
        "mean_s11_default_projected": round(float(s11_proj.mean()), 5),
        "max_passivity_correction": round(float(np.asarray(correction).max()), 5),
        "mean_s21_raw": round(float(s21_raw.mean()), 5),
        "ratio_floor_over_gamma": round(ratio, 4),
        "mean_s11_raw_db": round(20.0 * np.log10(mean_s11_raw), 2),
        "settling_db": settling_db,
        "wallclock_s": round(dt, 1),
        "preflight_warnings": warn_msgs,
    }


def _check_expectations(rows: list[dict]) -> dict:
    ratios = [r["ratio_floor_over_gamma"] for r in rows]
    exp_a = all(0.77 <= r <= 1.35 for r in ratios)

    aligned = [r for r in rows if r["label"].startswith("aligned")]
    aligned_sorted = sorted(aligned, key=lambda r: -r["dx_um"])  # coarse -> fine
    gammas = [r["abs_gamma_implied"] for r in aligned_sorted]
    # near-monotone non-increasing (allow tiny numerical wiggle)
    exp_b = all(gammas[i + 1] <= gammas[i] + 0.01 for i in range(len(gammas) - 1))

    by_label = {r["label"]: r for r in rows}
    r_60 = by_label.get("misaligned 60um")
    r_h4 = by_label.get("aligned h_sub/4")
    exp_c = None
    if r_60 is not None and r_h4 is not None:
        exp_c = r_60["abs_gamma_implied"] > r_h4["abs_gamma_implied"]

    return {
        "a_floor_matches_gamma_within_1p3x": exp_a,
        "a_ratios": ratios,
        "b_refinement_reduces_gamma_in_aligned_class": exp_b,
        "b_aligned_gammas_coarse_to_fine": gammas,
        "c_misalignment_shifts_floor_at_comparable_cells": exp_c,
        "c_witness_pair": {
            "misaligned_60um_abs_gamma": r_60["abs_gamma_implied"] if r_60 else None,
            "aligned_h_sub_4_abs_gamma": r_h4["abs_gamma_implied"] if r_h4 else None,
        },
    }


def main() -> int:
    rows = []
    for label, dx in DX_GRID:
        row = run_one(label, dx)
        rows.append(row)
        print(
            f"{label:18s} dx={row['dx_um']:7.2f}um n_z_sub={row['n_z_sub_exact']:.3f} "
            f"frac={row['frac']:.3f}{'*' if row['mixed_cell_danger_zone'] else ' '} "
            f"Z0={row['z0_measured_ohm']:6.2f}ohm Gamma={row['gamma_implied']:+.4f} "
            f"|S11|raw={row['mean_s11_raw']:.4f} ({row['mean_s11_raw_db']:+.1f}dB) "
            f"|S11|default={row['mean_s11_default_projected']:.4f} "
            f"ratio={row['ratio_floor_over_gamma']:.3f} "
            f"settling={row['settling_db']} ({row['wallclock_s']}s)",
            flush=True,
        )

    verdict = _check_expectations(rows)
    print("\nEXPECTATION VERDICTS")
    print(f"  (a) floor ~ |Gamma_implied| within 1.3x at every point: "
          f"{verdict['a_floor_matches_gamma_within_1p3x']}  ratios={verdict['a_ratios']}")
    print(f"  (b) refinement reduces |Gamma_implied| within aligned class: "
          f"{verdict['b_refinement_reduces_gamma_in_aligned_class']}  "
          f"gammas(coarse->fine)={verdict['b_aligned_gammas_coarse_to_fine']}")
    print(f"  (c) misalignment shifts floor at comparable cell count: "
          f"{verdict['c_misalignment_shifts_floor_at_comparable_cells']}  "
          f"witness={verdict['c_witness_pair']}")

    if not verdict["a_floor_matches_gamma_within_1p3x"]:
        print(
            "\nSTOP: expectation (a) failed at at least one point -- the "
            "Z0-bias mechanism does NOT fully explain the floor everywhere in "
            "this grid. Do not write the Leg-2 preflight advisory from this "
            "sweep; report the dump above instead."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "msl_z0_bias_floor_sweep.json"
    out_path.write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)
    return 0 if verdict["a_floor_matches_gamma_within_1p3x"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
