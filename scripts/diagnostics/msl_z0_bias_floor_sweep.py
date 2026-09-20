#!/usr/bin/env python3
"""MSL thru |S11| floor vs mesh, post-#511/#507 corrected extractor (issue #487).

CURRENT STATUS (#752): this is a historical experiment, not a current
accuracy benchmark. Its numerical recipe and recorded verdicts below are
retained, but generation is retired: the former command would overwrite
the frozen source JSON after solving a different board with today's code.
Use --show-archive to inspect the original record without a solve or writes.
A new accuracy claim requires newly designed, matched geometry/field records;
neither the historical 0.4% anchor nor this sweep supplies that claim.

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
exact-coordinate rasterizer (#802/#834) moves three of the six points'
realized trace width, so both the measured Z0 and the realized-board HJ
anchor on those points change and the 0.4% figure is no longer a live
bound for them.

THE WIDTHS MOVED TWICE, and both lists above and below are right for
their own date. Re-measured 2026-09-19/20 (issue #752 re-verification;
metadata only, ``sim.fidelity_report(print_report=False)`` on this file's
own ``run_one`` geometry, x64-invariant -- identical under
JAX_ENABLE_X64=0 and 1):

    label             dx (um)   frozen   df819523   main today
                                (pre-#802) (pre-#931) (post-#931)
    aligned h_sub/3    84.667    677.3    592.667     592.667
    aligned h_sub/4    63.500    635.0    635.000     571.500
    aligned h_sub/5    50.800    609.6    558.800     609.600
    aligned h_sub/6    42.333    592.7    635.000     592.667
    misaligned 80um    80.000    560.0    560.000     640.000
    misaligned 60um    60.000    600.0    600.000     600.000

The values on df819523 (2026-09-08) agree with the note's 2026-09-02 list
(h_sub/3 677.3->592.7, h_sub/5 609.6->558.8, h_sub/6 592.7->635.0). That
note was CORRECT. A second move landed at
#931 (485a9b98, 2026-09-10, one geometry->lattice ownership contract for
metal), which is why main today differs from both.

All six realized h_sub reproduce the frozen values exactly
(254/254/254/254/320/300um) and every realized trace face lands on a
node. But h_sub is NOT the whole board: see the conductor note below. The user-facing preflight advisories were
corrected (finding A1) to state only the qualitative realized-board claim
and to point here for the OWED re-solve. RE-SOLVE: run this script (6 FDTD
points) on main, then regenerate the anchor with
``msl_z0_bias_floor_sweep_realized_anchor.py``; a single-point check of
the h_sub/3 re-solve (2026-09-02, jax_enable_x64=False, settling
-110.0/-113.1 dB) read Z0=48.16 ohm against HJ(592.7um,254um)=48.27 ohm,
dev -0.23% -- consistent with the 0.4% band still holding on the
post-#802 board, but the full six-point re-solve is what a live bound
requires.

SECOND PASS (2026-09-02, same audit, findings A1-F1 and A1-F3; again
appended, pre-declared body/JSON untouched)
--------------------------------------------------------------------------
F1 -- the DECLARED-board sequence is stale too. The note above retired
only the realized-board "within 0.4%" reading. The four aligned
DECLARED-board deviations "-7.9%/-3.8%/-1.2%/+0.7%" quoted in the
2026-08-27 note above, and the ">5% below 4 cells" prediction they
supported, come from the SAME six frozen rows and fall to the SAME
argument -- in fact harder, because the declared anchor does NOT follow
the realized W, so a W move shifts the declared-board deviation by the
full amount. Only h_sub/4 (W unmoved at #802) survives as an as-solved
figure. Measured on the post-#802 rasterization, aligned h_sub/3 reads
Z0=48.162 ohm vs the declared-board HJ 47.895 ohm = +0.56%, where the
frozen row says -7.9% and the old advisory predicted ">5%". Those numbers
were therefore removed from ``rfx/api/_preflight.py``'s check-2 message
and class docstring; the check now states the O(dx) convergence order,
its <5% accuracy TARGET, and this re-solve pointer, and quotes no measured
percentage.

F3 -- SUPERSEDED IN PART, see the note at the end of this block. The
misaligned pair's "~0.2% still representative" sentence was
asserted, not measured. Geometry invariance (realized W unchanged at #802)
is necessary but not sufficient: the MSL extractor lane itself moved after
this sweep ran (#698 port metric sizing, #771 N-probe fit span, #791,
#798). It has now been MEASURED rather than argued. Re-solve of
"misaligned 80um" on the post-#802 tree (2026-09-02, jax_enable_x64=False,
CPU, 149 s; ring-down settling -100.3/-101.0 dB, well past the -40 dB
open-domain floor; mean|S11|raw=0.11607 vs the frozen row's 0.11609):
Z0=57.572 ohm vs frozen 57.576 ohm; against HJ(560um,320um)=57.463 ohm
that is +0.190%, vs the frozen row's +0.197%. So the misaligned half of
the anchor IS live-representative, on evidence. The aligned half is still
re-solve-owed, and two measured points do not make a six-point bound.

F3's PREMISE DOES NOT HOLD ON MAIN (re-measured 2026-09-19, issue #752
re-verification). "Realized W unchanged at #802" is false for misaligned
80um: main rasterizes its trace at y 1120.000..1760.000um = 640.000um
(8 cells), not the frozen 560.000um. HJ on the realized board therefore
reads 53.11 ohm there, not 57.46, so pairing the frozen Z0=57.572 ohm
with a live geometry rebuild is exactly the mismatch this whole issue is
about. F3's conclusion -- that the misaligned half is live-representative
-- rested on that premise and does not survive it. Of the misaligned
pair only 60um still has its frozen geometry on main. Nothing in the
frozen JSON changes; what changes is that neither half of the anchor can
be called live without the owed six-point re-solve.

THE OWED RE-SOLVE RAN (2026-09-19, issue #752 re-verification). It does NOT
restore the 0.4% reading -- it refutes it on this tree. New artifact beside
this one, ``msl_z0_bias_floor_sweep_realized_anchor_2026-09-19.json``, with
its own provenance (commit, rfx path, jax version, x64 flag, per-point wall
time and settling); the pre-declared body, its JSON and its verdict block are
untouched and their sha256 was re-checked after the run. Six fresh FDTD points
at this file's committed settings, each scored against Hammerstad-Jensen on
the board THAT SAME BUILD realized:

    label             Z0 meas   HJ(realized)   dev      settling
    aligned h_sub/3    44.179      48.271     -8.48%   -102.3/-100.0
    aligned h_sub/4    45.987      49.391     -6.89%   -102.4/-103.3
    aligned h_sub/5    44.646      47.412     -5.83%    -99.6/-103.1
    aligned h_sub/6    45.799      48.271     -5.12%    -99.6/-103.4
    misaligned 80um    39.016      53.106    -26.53%    -98.0/ -99.7
    misaligned 60um    41.491      53.106    -21.87%    -98.9/-100.5

Every point settled far below the -40 dB floor. The readability numbers --
probe clearance on all twelve ports, and the fit conditioning on the worst
point -- are NOT in the run record above (``run_one`` returns none of those
fields), so they are their own witness beside it,
``msl_z0_bias_floor_sweep_readability_witness_2026-09-20.json``: clearance is
build-only on the identical geometry, and the conditioning is ONE re-solved
point (misaligned 80um: beta_railed 0/60 bins, reliable 9/9 in the gate, Z0
flat over the gate 38.960-39.076 ohm, imag 0.126). On that evidence none of
the repo's reliability gates flags these numbers.

RE-VERIFICATION CONCLUSION (issue #752, written by the leader session
2026-09-20; pasted verbatim, wrapped only):
--------------------------------------------------------------------------
Re-verification of #752 on main (2026-09-19/20). Six points re-solved with
``run_one``'s committed settings (artifact
``msl_z0_bias_floor_sweep_realized_anchor_2026-09-19.json``, commit
15c1d325, jax 0.10.2, x64 off, all points settled at or below -98.0 dB on
both drives (worst stored value -98.0 dB, misaligned 80 um, port 0; values
stored to one decimal)).

Measured against Hammerstad-Jensen on the board each run realized: aligned
h_sub/3 -8.48 %, h_sub/4 -6.89 %, h_sub/5 -5.83 %, h_sub/6 -5.12 %;
misaligned 80 um -26.53 %, 60 um -21.87 %. The frozen artifact's "≤ 0.38 %
on the realized board" does not hold on this tree.

Two things are known to differ from the tree the frozen figure was taken
on, and neither has a full error budget here:
1. The board. #931 (485a9b98, 2026-09-10) changed how a one-cell-thick
   conductor realizes: before it a zero-thickness sheet on one node plane,
   after it a slab with walls on both faces (``rfx/fidelity.py:227-228``);
   on three points the trace's z placement also moved one cell
   (reviewer's builds of df819523 against the PR head: h_sub/6
   296.3-338.7 -> 254.0-296.3 um; 80 um 320-400 -> 240-320; 60 um
   300-360 -> 240-300). One
   measurement of the size of this: at aligned h_sub/3 the same trace
   declared as a zero-thickness Box reads 46.240 ohm (-4.21 %) where the
   one-cell slab reads 44.179 ohm (-8.48 %). At that one point conductor
   thickness accounts for 4.27 of 8.48 percentage points and a -4.21 %
   residual remains.
2. The anchor. ``hammerstad_jensen_z0_eps_eff`` is the simplified quasi-
   static form, not the full HJ-1980 model
   (``docs/guides/msl_geometry_diagnostics.md:79-82``); its error against
   the full model is not measured here.

What this does and does not establish: the split between board change and
residual is measured at one point and is about even there; the other five
points have no split measured. The residual is not attributed to the
extractor and not excluded from it. No pull request is named as a cause.
An earlier version of this PR named #986/#987 from columns (dielectric
height, trace width) that cannot see conductor realization; that reading
is withdrawn.

Owed, not run here: the six points re-solved on node-aligned zero-
thickness foil boards (the geometry ``msl_geometry_diagnostics.md:85-86``
describes), scored against both the repository formula and a full HJ-1980
reference. About 48 min of CPU; under the 2026-09-20 rule it runs on VESSL
(remilab-c0); the committed runner's docstring says how to package it.
Until it runs, #752 stays open with this measurement as its state.

COST OF THE OWED RE-SOLVE (measured, so nobody has to guess again): the
frozen JSON's own ``wallclock_s`` column sums to 5686.6 s = 94.8 min on
the machine that produced it. The two points re-solved on 2026-09-02 ran
at 131.6 s (h_sub/3, frozen 312.5 s) and 149.1 s (misaligned 80um, frozen
375.3 s) -- 0.42x and 0.40x -- so the full six points are ~39 min of CPU
here, i.e. one background job, not a GPU-lane errand.
"""

from __future__ import annotations

import argparse
import hashlib
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
SOURCE_JSON = OUT_DIR / "msl_z0_bias_floor_sweep.json"
_SOURCE_SHA256 = "f56f6b17691613d8782c1d5ce1241c1cd9bc10ef61715b203ed5cd6d4ab18362"


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the frozen historical MSL floor sweep.")
    parser.add_argument("--show-archive", action="store_true",
                        help="verify and print the original measurements and verdicts; no solve")
    args = parser.parse_args(argv)
    if not args.show_archive:
        parser.error(
            "This historical sweep is retired as a current benchmark. Its old "
            "generation path would overwrite the frozen experiment record. "
            "Use --show-archive to inspect that record; new validation needs "
            "matched field/geometry provenance and a separate output location.")
    source_bytes = SOURCE_JSON.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != _SOURCE_SHA256:
        parser.error("The frozen source record changed; restore its recorded bytes before inspection.")
    print(json.dumps({
        "record_kind": "historical_msl_floor_sweep",
        "current_solver_validation": False,
        "new_field_solves": 0,
        "source_json_sha256": _SOURCE_SHA256,
        "historical_record": json.loads(source_bytes),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
