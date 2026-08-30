#!/usr/bin/env python3
"""|Gamma_passive| re-measurement of the MSL thru's passive port region (#524 items 1-2).

REPORT-ONLY. Nothing here changes the solver, the extractor, a gate, a
tolerance or a reference number. It builds the #507 fixture, runs the
production ``compute_msl_s_matrix`` with its raw N-probe dump, and reads the
dump back offline.

WHY
---
Issue #524 carries two orphaned #507 loose ends whose every number is July
2026, pre-#511/#516, from an uncommitted session script: "|a_passive/a_driven|
= 0.07-0.51", the "0.194-vs-0.073" drive asymmetry, the "~30 ohm termination"
inferred against a "38.75 ohm" line Zc that the committed #535 sweep has since
superseded (44.108 ohm on the same mesh), and a "10.53 mm" reflector plane that
lands on no geometric feature of today's grid. None of them has been measured
on current main. This driver is the predeclared re-measurement
(design: scratch ``524_design_dup.json``; adversarial review: ``524_review.json``,
required changes 1-10 applied; see "REVIEW ITEMS" below).

FIXTURE (design C0, VERBATIM)
-----------------------------
``scripts/diagnostics/thin_conductor_cell_thickness_probe.py::run_one(n_cells=2)``:
RO4350B-like eps_r 3.66, h_sub 254 um, w 600 um, L 10 mm, PORT_MARGIN 2 mm,
dx 84.67 um (= h_sub/3), 2-cell PEC trace spanning the FULL lx into both CPMLs,
CPML 8, ports "+x" at 2.0 mm and "-x" at 12.0 mm, R = 50 ohm, band 3.0-4.5 GHz,
``compute_msl_s_matrix(n_freqs=30, num_periods=12, enforce_passivity=False)``.
Grid 183 x 53 x 30 = 290,970 cells. Constants are copied, not imported, so the
fixture cannot drift under this script without a visible diff here.

QUANTITY Q1 (extractor-light)
-----------------------------
For each drive ``d`` and each port ``p`` the dump's ``raw_v[d, p, :n_probes, :]``
holds the N probe voltages of port ``p`` while ``d`` was driven. Per frequency
this script fits, in float64 numpy,

    V_n = alpha * exp(-j*beta*x_n) + gamma * exp(+j*beta*x_n)

over ``p``'s probes with ``x_n`` the PHYSICAL propagation-axis coordinates
(increasing for a positive-going port, decreasing for a negative-going one),
so that ``alpha`` is the wave travelling toward +axis at BOTH ports --
exactly the convention ``extract_msl_nprobe`` is fed (issue #661,
``rfx/sources/msl_port.py`` msl_loop_current docstring). ``beta > 0`` is
anchored at the analytic Hammerstad-Jensen ``beta0`` and bracketed by the
SAME 41-node +/-35 % scan ``_estimate_beta`` uses (``_BETA_SCAN_NODES``,
``_BETA_SCAN_FRAC`` imported from ``rfx.probes.msl_wave_decomp``); the only
difference is that the offline refine is a golden-section search inside the
bracketing nodes instead of one parabolic step (no JAX traceability
constraint here). The beta branch is NEVER chosen by which choice gives
|Gamma| < 1 (review item 6).

WAVE ROLES (stated once, used everywhere)
-----------------------------------------
Let ``s_p`` = +1 for a positive-going port ("+x"/"+y"), -1 for a
negative-going one ("-x"/"-y"). At port ``p``'s probes:

* the wave travelling in ``p``'s own launch direction (away from the feed,
  INTO the structure) is ``alpha`` if ``s_p > 0`` else ``gamma``;
* the wave travelling the opposite way (from the structure TOWARD the feed,
  i.e. INCIDENT ON the port region) is ``gamma`` if ``s_p > 0`` else ``alpha``.

On a drive ``d != p`` the port ``p`` is passive, so

    Gamma_passive(p) = (wave travelling back INTO the structure from p)
                       / (wave INCIDENT on p from the structure)
                     = alpha/gamma   for s_p > 0   (the design's Gamma_1)
                     = gamma/alpha   for s_p < 0   (the design's Gamma_2)

both referenced at ``p``'s probe 0. |Gamma| is position-independent on a
lossless uniform line and any uniform V-scale error cancels in the ratio.

ROLE WITNESS FIRST (review item 6)
----------------------------------
Before any Gamma is assigned, on each drive ``d`` the dominant wave at EVERY
port must be the one travelling in ``d``'s launch direction (alpha for a
positive-going drive, gamma for a negative-going one): band-median
|dominant|/|other| >= ``ROLE_WITNESS_MIN_RATIO`` (= 2) AND > 1 at every
in-band frequency, at BOTH ports. On drive 0 of the design fixture this is
the #661 fact "|alpha| >> |gamma| at both ports". If the witness fails the
script REFUSES to assign Gamma for that drive and says so; it never flips
roles to make the numbers look right.

COMPARATOR Q2 (review item 4)
-----------------------------
Production-style wave split at ``p``'s probe 0 from the same dump:
``a = (V0 + Z_ref*I1)/2``, ``b = (V0 - Z_ref*I1)/2`` with ``raw_i1`` (the
un-normalised loop current the production solve consumes) and ``Z_ref`` =
the band-median fitted |z0| (re-referenced to the measured line), so
``Gamma_Q2 = a/b`` is the same physical ratio by a different route (one probe
+ current, vs five probes). GATE: |band-median|Gamma_Q1| - band-median
|Gamma_Q2|| <= ``COMPARATOR_GATE`` (= 0.02); the per-frequency spread is
REPORTED, not gated. The production multi-drive S diagonal and the
z0_hj-referenced ``|a_passive/a_driven|`` (the July quantity) are reported
alongside for context.

INDEPENDENT REALIZED-TERMINATION WITNESS (review item 3)
--------------------------------------------------------
The prediction is predeclared as realized R = requested R (k = 1). Per run
the script mirrors the runner's port assembly on the built grid
(``compute_msl_mode_profile`` + ``setup_msl_port`` from
``rfx.sources.msl_port``) and evaluates the fold it actually deposited:
``R_fold = 1 / sum(dsigma * ez_w^2 * dual_prop * dual_width * prim_normal)``
(the TEM dissipation for V = 1 V, i.e. the eigenmode-fold integral of
``msl_port.py`` setup_msl_port evaluated on the assembled sigma), plus a
mode-weight-free lumped series-in-normal/parallel-in-width network reading
``R_network`` that equals R only for a uniform-Ez column. If the R-sweep
follows the 1/(2R+Zc) shape but at a level implying k != 1, ``--summarize``
reports the implied k as a FINDING; nothing is adjusted.

PRECISION (review item 5)
-------------------------
The fixture pins no X64, so by default ``raw_v``/``raw_i1`` are complex64
(``_complex_dtype`` in compute_msl_s_matrix; ``extract_msl_nprobe`` casts to
complex64). Default = run the fixture as-is and LABEL every number with the
dump dtype. ``--x64`` enables ``jax_enable_x64`` before the build and re-times
the run. A result JSON records ``precision_mode`` and ``--summarize`` refuses
to mix modes in one comparison. Invoke as
``JAX_ENABLE_X64=0 python3 ...`` (default) or ``JAX_ENABLE_X64=1 python3 ...
--x64``; a mismatch between the env and the flag is an error.

VARIANTS (design B; R applied to BOTH ports so each drive reads the other)
-------------------------------------------------------------------------
C0    shipped fixture, R = 50                (current-main baseline)
C1    R = 1e6                                (shunt removed: stub+CPML control)
C2    R = Zc_meas (C0 fitted |z0| median; pass --zc-ohm or --c0-json)
C3    R = 25
C4    trace CUT at port 2's feed (x_hi just past the snapped feed column so
      PEC still covers i_feed, review item 7), R = 50: drive 1 reads an
      END-terminated port 2 while drive 2 reads the CONTINUING port 1 in the
      same run. If the builder/extractor refuses, the row is reported as
      "not runnable without code" -- nothing is patched to make it run.
C5    C4 with R = Zc_meas                    ("what matched looks like")
SYM   cell-aligned symmetric thru: C0's 167-node interior, feeds at 24 dx / 143 dx
      (24-cell stubs both sides), R = 50     (item 2: snapping vs lane)
ROTY  C0 rotated to +/-y (legal since #661), R = 50   (item 2: direction
      sign vs lo/hi CPML face)

SETTLING WITNESS: the lane's enforced ring-down bar (-40 dB, #662/#664;
``_SETTLING_WITNESS_DB`` in ``rfx/api/_sparams.py``) is quoted per run. A run
above the bar is marked DISCARDED_SETTLING and excluded from every hypothesis
reading (re-run with larger ``--num-periods``; never averaged, never pinned --
review item 9, C1 is the likely case). Preflight warnings are quoted VERBATIM.

PREDICTIONS (evaluated by --summarize, declared here, Zc := C0 fitted |z0|
band-median, Gamma_c := C1's complex Gamma at band centre)
  H_shunt  Gamma = (R||Z_stub - Zc)/(R||Z_stub + Zc), Z_stub = Zc(1+Gc)/(1-Gc);
           for Gc ~ 0 this is Zc/(2R+Zc): monotone decreasing in R, no
           minimum at R = Zc; C4/C5 collapse to |R-Zc|/(R+Zc).
  H_end    |Gamma| = |R-Zc|/(R+Zc): minimum (0) at R = Zc.
  H_cpml   C0-C3 within +/-0.02 of C1 and x_L within 1 mm of a CPML inner
           face, not of a feed plane.
  FALSIFIER: no hypothesis within 0.03 at every available R point ->
           "no single-reflector model"; report and stop. No two-parameter
           fit to four numbers; no post-hoc k fit (k is REPORTED per point).
  Item 2:  |Gamma_1| vs |Gamma_2| in C0; > 0.02 apart -> read SYM (vanishes
           -> snapping) and ROTY (persists with the sign -> lane asymmetry).

STALE-LITERAL ENUMERATION (review item 2)
-----------------------------------------
``--enumerate-stale-literals`` greps the tree for the July literal class
("0.07-0.51", "0.194", "0.073", "30-vs-48", "38.75", "10.53 mm", "~30 ohm")
and prints each site classified as docstring-prose / comment /
test-parametrize-value(keep) / issue-text / code-other. Report only; no edits.

REVIEW ITEMS applied: 1 (raw_3probe_dump_path npz, not return_diagnostics;
z0_hj recomputed, probe x re-derived), 2 (enumeration), 3 (realized-R
witness, k = 1 predeclared), 4 (band-median gate, spread reported),
5 (precision labelled, --x64), 6 (beta anchored, role witness first),
7 (C4 cut at port 2's feed), 8 (the #507 RETRACTION comment, filed as #511,
is the provenance that retires 38.75 / "~30 ohm"; its surviving post-#511
|a2/b2| = 0.187-0.211 is the only uncontaminated July figure and is NOT a
target), 9 (C1 settling budget), 10 (settling cites: _sparams.py
_SETTLING_WITNESS_DB / settling_db_runs on the pure-MSL lane).

USAGE
-----
    cd <repo> && PYTHONPATH=<repo> JAX_ENABLE_X64=0 python3 \\
        scripts/diagnostics/msl_passive_port_reflection.py --variant C0 \\
        --output out/C0.json --dump-dir out/dumps
    ... --variant C2 --c0-json out/C0.json ...
    ... --summarize out/C0.json out/C1.json out/C2.json out/C3.json \\
        out/SYM.json out/ROTY.json --output out/summary.json
    ... --enumerate-stale-literals
    smoke (NOT the measurement):  --variant C0 --num-periods 2 --n-freqs 5 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tokenize
import warnings
from pathlib import Path

# Import the rfx in THIS checkout, not whatever is pip-installed (same guard
# as thin_conductor_cell_thickness_probe.py; issue #511 lesson).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

# ---------------------------------------------------------------------------
# Fixture constants -- VERBATIM copy of
# scripts/diagnostics/thin_conductor_cell_thickness_probe.py (run_one(n_cells=2))
# ---------------------------------------------------------------------------
H_SUB = 254e-6
W_TRACE = 600e-6
DX = 84.67e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3
EPS_R = 3.66
GATE_F_LO, GATE_F_HI = 3.0e9, 4.5e9
FREQ_MAX = 5e9
CPML_LAYERS = 8
N_CELLS_TRACE = 2
R_SHIPPED = 50.0
N_FREQS_DESIGN = 30
NUM_PERIODS_DESIGN = 12.0

# Cell-aligned symmetric variant (design C0'): the shipped fixture's interior
# is 167 nodes (183 - 2*8, hi CPML face at 167 dx = 14.140 mm); feeds at
# 24 dx and 143 dx give 24-cell stubs on both sides.
SYM_NX_INTERIOR_NODES = 167
SYM_FEED_LO_CELLS = 24
SYM_FEED_HI_CELLS = 143

# Predeclared thresholds (report-only; none of these is a physics gate).
ROLE_WITNESS_MIN_RATIO = 2.0     # band-median |dominant|/|other| at BOTH ports
COMPARATOR_GATE = 0.02           # band-median |Gamma_Q1| vs |Gamma_Q2|
HYPOTHESIS_BAND = 0.03           # falsifier band per R point (design)
ITEM2_ASYMMETRY_BAND = 0.02      # |Gamma_1| vs |Gamma_2| (design)
SETTLING_BAR_DB = -40.0          # quoted from rfx.api._sparams._SETTLING_WITNESS_DB

VARIANTS = {
    "C0": {"r_ohm": R_SHIPPED, "geometry": "shipped",
               "what": "current-main baseline of the July numbers"},
    "C1": {"r_ohm": 1.0e6, "geometry": "shipped",
               "what": "shunt removed: stub+CPML reflection Gamma_c alone (control)"},
    "C2": {"r_ohm": None, "geometry": "shipped",
               "what": "R = Zc_meas (C0 fitted |z0| band-median)"},
    "C3": {"r_ohm": 25.0, "geometry": "shipped", "what": "R = 25"},
    "C4": {"r_ohm": R_SHIPPED, "geometry": "cut_at_port2",
               "what": "trace cut at port 2's feed, R = 50 (drive 1 reads an "
                    "END-terminated port 2; drive 2 reads the CONTINUING port 1)"},
    "C5": {"r_ohm": None, "geometry": "cut_at_port2",
               "what": "C4 with R = Zc_meas (the matched-termination picture)"},
    "SYM": {"r_ohm": R_SHIPPED, "geometry": "symmetric",
                "what": "cell-aligned symmetric thru (24-cell stubs both sides)"},
    "ROTY": {"r_ohm": R_SHIPPED, "geometry": "rotated_y",
                 "what": "C0 rotated to +/-y (#661)"},
}

STALE_LITERAL_PATTERNS = {
    "0.07-0.51": re.compile(r"0\.07\s*[-–]\s*0\.51"),
    "0.194": re.compile(r"(?<![\d.])0\.194(?![\d])"),
    "0.073": re.compile(r"(?<![\d.])0\.073(?![\d])"),
    "30-vs-48": re.compile(r"30-vs-48"),
    "38.75": re.compile(r"(?<![\d.])38\.75(?![\d])"),
    "10.53 mm": re.compile(r"10\.53\s*mm"),
    "~30 ohm": re.compile(r"~\s*30\s*ohm"),
}


# ---------------------------------------------------------------------------
# Standing-wave fit (float64 numpy; same bracket protocol as _estimate_beta)
# ---------------------------------------------------------------------------


def _lstsq_alpha_gamma_np(v: np.ndarray, x: np.ndarray, beta: float):
    """Solve V_n = alpha e^{-j beta x_n} + gamma e^{+j beta x_n} by SVD lstsq."""
    a = np.stack([np.exp(-1j * beta * x), np.exp(1j * beta * x)], axis=1)
    sol, _res, _rank, _sv = np.linalg.lstsq(a, v, rcond=None)
    resid = float(np.linalg.norm(v - a @ sol))
    return complex(sol[0]), complex(sol[1]), resid


def fit_two_wave(v, x, beta0, *, n_nodes=None, scan_frac=None,
                 golden_iters=200) -> dict:
    """Fit one frequency's probe voltages to a two-wave standing pattern.

    ``x`` are the probe coordinates along the propagation axis (signed,
    physical); only differences enter and the model is referenced at probe 0,
    so ``alpha``/``gamma`` are probe-0 amplitudes with ``alpha`` the wave
    travelling toward +axis. ``beta0 > 0`` is the analytic anchor. The scan
    window and node count are the production ``_estimate_beta`` values
    (imported), the argmin node is bracketed by its neighbours, and a
    golden-section search refines inside that bracket. ``railed`` follows the
    production rule (raw argmin at a window edge, or refined beta within
    half a node of a limit).
    """
    from rfx.probes.msl_wave_decomp import _BETA_SCAN_FRAC, _BETA_SCAN_NODES
    n_nodes = int(_BETA_SCAN_NODES if n_nodes is None else n_nodes)
    frac = float(_BETA_SCAN_FRAC if scan_frac is None else scan_frac)

    v = np.asarray(v, dtype=np.complex128)
    x = np.asarray(x, dtype=np.float64)
    x = x - x[0]
    beta0 = float(np.real(beta0))
    if not beta0 > 0.0:
        raise ValueError(f"beta0 must be > 0 (got {beta0}); the branch is "
                         "anchored at the analytic HJ beta, never chosen by |Gamma|")
    v_scale = float(np.max(np.abs(v)))
    v_n = v / (v_scale if v_scale > 0.0 else 1.0)

    lo, hi = beta0 * (1.0 - frac), beta0 * (1.0 + frac)
    grid = np.linspace(lo, hi, n_nodes)
    resids = np.array([_lstsq_alpha_gamma_np(v_n, x, b)[2] for b in grid])
    k_raw = int(np.argmin(resids))
    k = int(np.clip(k_raw, 1, n_nodes - 2))
    b_lo, b_hi = float(grid[k - 1]), float(grid[k + 1])

    # Golden-section refine inside the bracketing nodes.
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a_, b_ = b_lo, b_hi
    c_ = b_ - invphi * (b_ - a_)
    d_ = a_ + invphi * (b_ - a_)
    f_c = _lstsq_alpha_gamma_np(v_n, x, c_)[2]
    f_d = _lstsq_alpha_gamma_np(v_n, x, d_)[2]
    for _ in range(golden_iters):
        if abs(b_ - a_) <= 1e-13 * beta0:
            break
        if f_c < f_d:
            b_, d_, f_d = d_, c_, f_c
            c_ = b_ - invphi * (b_ - a_)
            f_c = _lstsq_alpha_gamma_np(v_n, x, c_)[2]
        else:
            a_, c_, f_c = c_, d_, f_d
            d_ = a_ + invphi * (b_ - a_)
            f_d = _lstsq_alpha_gamma_np(v_n, x, d_)[2]
    beta = 0.5 * (a_ + b_)
    half = 0.5 * (grid[1] - grid[0])
    railed = bool(
        (k_raw <= 0) or (k_raw >= n_nodes - 1)
        or (beta <= grid[0] + half) or (beta >= grid[-1] - half)
    )
    alpha, gamma, resid = _lstsq_alpha_gamma_np(v, x, beta)
    return {"alpha": alpha, "gamma": gamma, "beta": float(beta), "beta0": beta0,
                "railed": railed, "residual": resid,
                "residual_rel": resid / (v_scale if v_scale > 0 else 1.0)}


def fit_two_wave_band(v: np.ndarray, x: np.ndarray, beta0: np.ndarray) -> dict:
    """Vectorised-over-frequency wrapper. ``v`` is (n_probes, n_freqs)."""
    v = np.asarray(v)
    n_f = v.shape[1]
    out = {"alpha": np.zeros(n_f, complex), "gamma": np.zeros(n_f, complex),
               "beta": np.zeros(n_f), "beta0": np.zeros(n_f),
               "railed": np.zeros(n_f, bool), "residual_rel": np.zeros(n_f)}
    for fi in range(n_f):
        r = fit_two_wave(v[:, fi], x, float(beta0[fi]))
        for key, arr in out.items():
            arr[fi] = r[key]
    return out


def role_witness(alpha, gamma, drive_sign: float, band: np.ndarray,
                 *, min_ratio: float = ROLE_WITNESS_MIN_RATIO) -> dict:
    """Is the dominant wave at this port the one the DRIVE launched?

    A positive-going drive launches the +axis wave (alpha); a negative-going
    drive launches gamma. Passes iff band-median |dominant|/|other| >=
    ``min_ratio`` and the ratio exceeds 1 at every in-band frequency.
    """
    alpha = np.asarray(alpha); gamma = np.asarray(gamma)
    dom, oth = (alpha, gamma) if drive_sign > 0 else (gamma, alpha)
    ratio = np.abs(dom) / np.maximum(np.abs(oth), 1e-300)
    rb = ratio[band]
    med = float(np.median(rb)) if rb.size else float("nan")
    mn = float(np.min(rb)) if rb.size else float("nan")
    return {"dominant": "alpha" if drive_sign > 0 else "gamma",
                "median_ratio_in_band": med, "min_ratio_in_band": mn,
                "passed": bool(rb.size and med >= min_ratio and mn > 1.0),
                "min_ratio_required": float(min_ratio)}


def passive_gamma(alpha, gamma, port_sign: float) -> np.ndarray:
    """Gamma_passive(p) = (wave back INTO the structure from p)/(wave INCIDENT on p).

    ``alpha`` is the +axis wave at the port's probe 0; ``port_sign`` is the
    port's own launch sign. alpha/gamma for a positive-going port, gamma/alpha
    for a negative-going one (see the module docstring, WAVE ROLES).
    """
    alpha = np.asarray(alpha, complex); gamma = np.asarray(gamma, complex)
    eps = 1e-300
    if port_sign > 0:
        return alpha / np.where(np.abs(gamma) > 0, gamma, eps)
    return gamma / np.where(np.abs(alpha) > 0, alpha, eps)


def assign_passive_gamma(fit_by_port: dict, drive_sign: float,
                         port_signs: dict, driven: int, band: np.ndarray) -> dict:
    """Role witness at EVERY port first; refuse to assign if it fails."""
    witness = {p: role_witness(f["alpha"], f["gamma"], drive_sign, band)
               for p, f in fit_by_port.items()}
    ok = all(w["passed"] for w in witness.values())
    gammas = {}
    if ok:
        for p, f in fit_by_port.items():
            if p == driven:
                continue
            gammas[p] = passive_gamma(f["alpha"], f["gamma"], port_signs[p])
    return {"witness": witness, "assigned": ok, "gamma_passive": gammas,
                "refusal": None if ok else (
                    "role witness failed on this drive (|dominant|/|other| "
                    f"must be >= {ROLE_WITNESS_MIN_RATIO} band-median and > 1 "
                    "at every in-band frequency at BOTH ports); Gamma NOT "
                    "assigned -- no role flip is attempted")}


def reflector_plane_from_phase(gamma_p: np.ndarray, beta: np.ndarray,
                               band: np.ndarray) -> dict:
    """Phase-localise the reflector: arg Gamma = phi0 - 2*beta*d.

    With Gamma_passive referenced at probe 0 and the incident wave arriving
    from the structure, the reflected wave originates on the FEED side of
    probe 0 at a plane a distance ``d`` away, so unwrap(arg Gamma) is linear
    in beta with slope -2d. ``d`` is therefore the distance from probe 0
    TOWARD the feed (probe 0 is the probe nearest the feed); the caller
    places the plane at x_L = x0 - sign_p * d. Both port signs reduce to the
    same formula (derivation in the script's WAVE ROLES section).
    """
    ph = np.unwrap(np.angle(gamma_p[band]))
    b = np.asarray(beta)[band]
    if b.size < 3:
        return {"d_from_probe0_m": None, "note": "fewer than 3 in-band points"}
    slope, intercept = np.polyfit(b, ph, 1)
    resid = float(np.sqrt(np.mean((ph - (slope * b + intercept)) ** 2)))
    return {"d_from_probe0_m": float(-slope / 2.0), "phase_fit_rms_rad": resid,
                "gamma_L_phase_rad": float(intercept)}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def build_fixture(variant: str, r_ohm: float, *, cut_x_hi: float | None = None):
    """Build the Simulation for a variant. Geometry is the design's, verbatim."""
    from rfx import Box, Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    geometry = VARIANTS[variant]["geometry"]
    info: dict = {"variant": variant, "geometry": geometry, "r_ohm_requested": float(r_ohm)}
    lz = H_SUB + 1.5e-3
    lat = W_TRACE + 2 * (2 * H_SUB + 8 * DX)     # the fixture's ly
    if geometry in ("shipped", "cut_at_port2"):
        lx = L_LINE + 2 * PORT_MARGIN
        feeds = (PORT_MARGIN, PORT_MARGIN + L_LINE)
    elif geometry == "symmetric":
        # Same declared lx as C0 -> the SAME 167-node interior (nx = 183, hi
        # CPML face at 167 dx = 14.140 mm); only the feeds move onto 24 dx /
        # 143 dx so both stubs are 24 cells. (Declaring lx = 167*dx exactly
        # rounds up to a 168-node interior and gives 24/25-cell stubs.)
        lx = L_LINE + 2 * PORT_MARGIN
        feeds = (SYM_FEED_LO_CELLS * DX, SYM_FEED_HI_CELLS * DX)
    elif geometry == "rotated_y":
        lx = L_LINE + 2 * PORT_MARGIN
        feeds = (PORT_MARGIN, PORT_MARGIN + L_LINE)
    else:  # pragma: no cover
        raise ValueError(geometry)
    trace_hi = lx if cut_x_hi is None else float(cut_x_hi)
    info.update(lx_m=lx, lat_m=lat, lz_m=lz, feeds_m=list(feeds),
                trace_prop_extent_m=[0.0, trace_hi])

    bspec = BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"))
    if geometry == "rotated_y":
        sim = Simulation(freq_max=FREQ_MAX, domain=(lat, lx, lz), dx=DX,
                         cpml_layers=CPML_LAYERS, boundary=bspec)
        sim.add_material("sub", eps_r=EPS_R)
        sim.add(Box((0.0, 0.0, 0.0), (lat, lx, H_SUB)), material="sub")
        c = lat / 2.0
        sim.add(Box((c - W_TRACE / 2.0, 0.0, H_SUB),
                    (c + W_TRACE / 2.0, trace_hi, H_SUB + N_CELLS_TRACE * DX)),
                material="pec")
        for y, direction in ((feeds[0], "+y"), (feeds[1], "-y")):
            sim.add_msl_port(position=(c, y, 0.0), width=W_TRACE,
                             height=H_SUB, direction=direction, impedance=r_ohm)
        info["directions"] = ["+y", "-y"]
    else:
        sim = Simulation(freq_max=FREQ_MAX, domain=(lx, lat, lz), dx=DX,
                         cpml_layers=CPML_LAYERS, boundary=bspec)
        sim.add_material("sub", eps_r=EPS_R)
        sim.add(Box((0.0, 0.0, 0.0), (lx, lat, H_SUB)), material="sub")
        y_c = lat / 2.0
        sim.add(Box((0.0, y_c - W_TRACE / 2.0, H_SUB),
                    (trace_hi, y_c + W_TRACE / 2.0, H_SUB + N_CELLS_TRACE * DX)),
                material="pec")
        for x, direction in ((feeds[0], "+x"), (feeds[1], "-x")):
            sim.add_msl_port(position=(x, y_c, 0.0), width=W_TRACE,
                             height=H_SUB, direction=direction, impedance=r_ohm)
        info["directions"] = ["+x", "-x"]
    return sim, info


def grid_facts(sim) -> dict:
    """Grid shape, pads, feed indices/coords, CPML inner faces -- for the JSON."""
    from rfx.sources.msl_port import (
        _MSL_AXIS_INDEX,
        _msl_coord_for_index,
        msl_axis_roles,
        msl_cross_section_span,
        msl_port_from_entry,
    )
    grid = sim._build_grid()
    facts = {"shape": [int(grid.nx), int(grid.ny), int(grid.nz)],
                 "n_cells": int(grid.nx * grid.ny * grid.nz), "dx_m": float(grid.dx),
                 "dt_s": float(grid.dt), "ports": []}
    for pe in sim._msl_ports:
        mp = msl_port_from_entry(pe)
        prop, _w, _n, sign = msl_axis_roles(pe.direction)
        span = msl_cross_section_span(grid, mp)
        i_feed = span["i_feed"]
        n_axis = int(getattr(grid, {"x": "nx", "y": "ny", "z": "nz"}[prop]))
        pad = int(getattr(grid, "cpml_layers", CPML_LAYERS) or CPML_LAYERS)
        facts["ports"].append({
            "name": pe.name, "direction": pe.direction, "sign": float(sign),
            "i_feed": int(i_feed), "feed_coord_m": float(_msl_coord_for_index(grid, prop, i_feed)),
            "cpml_inner_faces_m": [float(_msl_coord_for_index(grid, prop, pad)),
                                float(_msl_coord_for_index(grid, prop, n_axis - pad))],
            "domain_prop_m": float(sim._domain[_MSL_AXIS_INDEX[prop]]),
        })
    return grid, facts


def resolve_cut_x_hi(r_ohm: float, variant: str) -> tuple[float, dict]:
    """C4/C5: trace x_hi just past port 2's snapped feed column (review item 7).

    Builds the shipped geometry once to learn the snapped feed index, then
    tries candidate x_hi values and keeps the first whose PEC mask covers
    the feed column and NOT the next one. Reports the check; if no candidate
    satisfies it the caller reports 'not runnable without code'.
    """
    from rfx.sources.msl_port import _msl_coord_for_index
    sim0, _ = build_fixture("C0", r_ohm)
    grid0, facts0 = grid_facts(sim0)
    p2 = facts0["ports"][1]
    i_feed = p2["i_feed"]
    x_feed = float(_msl_coord_for_index(grid0, "x", i_feed))
    x_next = float(_msl_coord_for_index(grid0, "x", i_feed + 1))
    k_tr = round(H_SUB / DX)
    j_c = int(grid0.ny // 2)
    checks = []
    for frac in (0.5, 0.75, 0.25, 1.0, 0.1):
        x_hi = x_feed + frac * (x_next - x_feed)
        sim, _ = build_fixture(variant, r_ohm, cut_x_hi=x_hi)
        grid = sim._build_grid()
        pec = np.asarray(sim._assemble_materials(grid)[3])
        at_feed = bool(pec[i_feed, j_c, k_tr:k_tr + N_CELLS_TRACE].any())
        at_next = bool(pec[i_feed + 1, j_c, k_tr:k_tr + N_CELLS_TRACE].any())
        checks.append({"x_hi_m": x_hi, "pec_at_feed_column": at_feed,
                           "pec_at_next_column": at_next})
        if at_feed and not at_next:
            return x_hi, {"i_feed_port2": i_feed, "x_feed_m": x_feed,
                              "x_next_m": x_next, "chosen_x_hi_m": x_hi, "checks": checks}
    return float("nan"), {"i_feed_port2": i_feed, "x_feed_m": x_feed,
                              "x_next_m": x_next, "chosen_x_hi_m": None, "checks": checks}


# ---------------------------------------------------------------------------
# Independent realized-termination witness (review item 3)
# ---------------------------------------------------------------------------


def realized_termination_witness(sim, grid) -> list[dict]:
    """Evaluate the termination the built grid realizes, per port.

    Mirrors the runner's port assembly (rfx/runners/uniform.py: mode_profile
    via compute_msl_mode_profile for the default 'laplace' mode, then
    setup_msl_port) on the assembled materials, and integrates the deposited
    sigma two ways. PREDICTION (predeclared): k_fold = R_fold/R_requested = 1.
    R_network is the mode-weight-free lumped reading (series in the normal
    axis, parallel across width) and equals R only for a uniform-Ez column;
    it is reported, with no prediction attached, as the second witness.
    """
    from rfx.sources.msl_port import (
        _axis_cell_size,
        _axis_dual_size,
        compute_msl_mode_profile,
        msl_cell,
        msl_cross_section_span,
        msl_port_from_entry,
        setup_msl_port,
    )
    materials = sim._assemble_materials(grid)[0]
    rows = []
    for pe in sim._msl_ports:
        mp = msl_port_from_entry(pe)
        span = msl_cross_section_span(grid, mp)
        k_mid = (span["n_lo"] + span["n_hi"]) // 2
        eps_cell = msl_cell(pe.direction, span["i_feed"], span["w_centre"], k_mid)
        eps_r_sub = (float(pe.eps_r_sub) if pe.eps_r_sub is not None
                     else float(np.asarray(materials.eps_r[eps_cell])))
        mode = getattr(pe, "mode", "uniform")
        mode_profile = None
        if mode in ("eigenmode", "laplace"):
            mode_profile = compute_msl_mode_profile(grid, mp, eps_r_sub)
        sigma_before = np.asarray(materials.sigma, dtype=np.float64)
        materials = setup_msl_port(grid, mp, materials, mode_profile=mode_profile)
        dsigma = np.asarray(materials.sigma, dtype=np.float64) - sigma_before
        ax_p, ax_w, ax_n = span["prop_axis"], span["width_axis"], span["normal_axis"]
        ip, iw, inr = span["prop_idx"], span["width_idx"], span["normal_idx"]
        loaded = np.argwhere(dsigma > 0.0)
        row = {"name": pe.name, "direction": pe.direction, "mode": mode,
                   "eps_r_sub_used": eps_r_sub, "r_requested_ohm": float(pe.impedance),
                   "n_loaded_cells": int(loaded.shape[0]),
                   "sigma_sum_S_per_m": float(dsigma.sum())}
        # (a) mode-weighted TEM dissipation for V = 1 V -> R_fold.
        p_diss = 0.0
        if mode_profile is not None:
            ez = np.asarray(mode_profile["ez_profile"], dtype=np.float64)
            j0, k0 = int(mode_profile["j_grid_lo"]), int(mode_profile["k_grid_lo"])
            for cell in loaded:
                i, j, k = (int(c) for c in cell)
                jl, kl = cell[iw] - j0, cell[inr] - k0
                if not (0 <= jl < ez.shape[0] and 0 <= kl < ez.shape[1]):
                    continue
                p_diss += (dsigma[i, j, k] * ez[jl, kl] ** 2
                           * _axis_dual_size(grid, ax_p, cell[ip])
                           * _axis_dual_size(grid, ax_w, cell[iw])
                           * _axis_cell_size(grid, ax_n, cell[inr]))
            row["r_fold_ohm"] = float(1.0 / p_diss) if p_diss > 0 else None
            row["k_fold"] = (float(row["r_fold_ohm"] / pe.impedance)
                             if p_diss > 0 else None)
            row["r_fold_note"] = (
                "1/sum(dsigma*ez_w^2*dual_prop*dual_width*prim_normal) over the "
                "cells the fold loaded, ez_w normalised to 1 V at the trace "
                "centre -- the setup_msl_port eigenmode integral evaluated on "
                "the ASSEMBLED sigma; shares the mode weights with the fold, so "
                "it is independent of the requested R only through what the "
                "assembly actually deposited")
        else:
            row["r_fold_ohm"] = None
            row["k_fold"] = None
        # (b) mode-weight-free lumped network over the loaded cells.
        cols: dict = {}
        for cell in loaded:
            i, j, k = (int(c) for c in cell)
            g = (dsigma[i, j, k] * _axis_dual_size(grid, ax_p, cell[ip])
                 * _axis_dual_size(grid, ax_w, cell[iw])
                 / _axis_cell_size(grid, ax_n, cell[inr]))
            cols.setdefault(int(cell[iw]), []).append(g)
        g_total = sum(1.0 / sum(1.0 / g for g in gs) for gs in cols.values() if gs)
        row["r_network_ohm"] = float(1.0 / g_total) if g_total > 0 else None
        row["r_network_note"] = ("series-in-normal / parallel-in-width lumped "
                                 "reading of the deposited sigma; equals R only "
                                 "for a uniform-Ez column (no prediction attached)")
        row["prediction"] = "realized R = requested R, i.e. k_fold = 1 (predeclared)"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Dump analysis
# ---------------------------------------------------------------------------


def _band_median(arr, band):
    a = np.asarray(arr)[band]
    return float(np.median(a)) if a.size else float("nan")


def _c2j(z) -> list:
    z = np.asarray(z)
    return [[float(np.real(v)), float(np.imag(v))] for v in z.reshape(-1)]


def analyze_dump(npz_path: Path, sim, grid, res, band_lo: float, band_hi: float) -> dict:
    """Read the v3 npz (raw_3probe_dump_path contract) and form Q1/Q2 per port."""
    from rfx.core.yee import EPS_0, MU_0
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.sources.msl_port import (
        msl_axis_roles,
        msl_cell,
        msl_cross_section_span,
        msl_port_from_entry,
        msl_probe_x_coords_n,
    )
    c0 = 1.0 / float(np.sqrt(MU_0 * EPS_0))

    with np.load(npz_path, allow_pickle=True) as z:
        meta = json.loads(str(z["metadata_json"]))
        freqs = np.asarray(z["freqs_hz"], dtype=np.float64)
        raw_v = np.asarray(z["raw_v"])
        raw_i1 = np.asarray(z["raw_i1"])
        raw_z0 = np.asarray(z["raw_z0"])
        prod_s = np.asarray(z["production_smatrix"])
        prod_z0 = np.asarray(z["production_z0"])
    dtype_label = str(raw_v.dtype)
    band = (freqs >= band_lo) & (freqs <= band_hi)
    n_ports = raw_v.shape[1]
    pdefs = meta["port_definitions"]
    materials = sim._assemble_materials(grid)[0]

    ports = []
    for p, (pe, pd) in enumerate(zip(sim._msl_ports, pdefs)):
        mp = msl_port_from_entry(pe)
        _prop, _w, _n, sign = msl_axis_roles(pe.direction)
        span = msl_cross_section_span(grid, mp)
        k_mid = (span["n_lo"] + span["n_hi"]) // 2
        eps_cell = msl_cell(pe.direction, span["i_feed"], span["w_centre"], k_mid)
        eps_r_ref = (float(pe.eps_r_sub) if pe.eps_r_sub is not None
                     else float(np.asarray(materials.eps_r[eps_cell])))
        z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(pe.width, pe.height, eps_r_ref)
        beta0 = 2.0 * np.pi * freqs * float(np.sqrt(eps_eff_hj)) / c0
        xs = np.asarray(msl_probe_x_coords_n(
            grid, mp, n_probes=int(pd["n_probes"]),
            n_offset_cells=int(pd["n_probe_offset"]),
            n_spacing_cells=int(pd["n_probe_spacing"])), dtype=np.float64)
        ports.append({"index": p, "name": pe.name, "direction": pe.direction, "sign": float(sign),
                          "n_probes": int(pd["n_probes"]),
                          "n_probe_offset": int(pd["n_probe_offset"]),
                          "n_probe_spacing": int(pd["n_probe_spacing"]),
                          "probe_coords_m": [float(v) for v in xs],
                          "z0_hj_ohm": float(z0_hj), "eps_eff_hj": float(eps_eff_hj),
                          "eps_r_ref": eps_r_ref, "beta0": beta0, "xs": xs,
                          "impedance_ohm": float(pd["impedance_ohm"])})
    port_signs = {p: ports[p]["sign"] for p in range(n_ports)}

    drives = []
    fits_all: dict = {}
    for d in range(n_ports):
        fit_by_port = {}
        for p in range(n_ports):
            n_p = ports[p]["n_probes"]
            v = raw_v[d, p, :n_p, :].astype(np.complex128)
            fit_by_port[p] = fit_two_wave_band(v, ports[p]["xs"], ports[p]["beta0"])
        fits_all[d] = fit_by_port
        assign = assign_passive_gamma(fit_by_port, port_signs[d], port_signs, d, band)
        drive_row = {"driven": d, "driven_name": ports[d]["name"],
                         "drive_sign": port_signs[d], "role_witness": assign["witness"],
                         "gamma_assigned": assign["assigned"], "refusal": assign["refusal"],
                         "ports": {}}
        # production a/b at every port on this drive (z0_hj reference).
        a_hj = {p: 0.5 * (raw_v[d, p, 0, :].astype(complex)
                          + ports[p]["z0_hj_ohm"] * raw_i1[d, p, :].astype(complex))
                for p in range(n_ports)}
        b_hj = {p: 0.5 * (raw_v[d, p, 0, :].astype(complex)
                          - ports[p]["z0_hj_ohm"] * raw_i1[d, p, :].astype(complex))
                for p in range(n_ports)}
        for p in range(n_ports):
            f = fit_by_port[p]
            z0_mine = (f["alpha"] - f["gamma"]) / np.where(
                np.abs(raw_i1[d, p, :]) > 0, raw_i1[d, p, :].astype(complex), 1e-300
            ) * port_signs[p]
            prow = {
                "port": p, "name": ports[p]["name"], "is_passive": (p != d),
                "beta_fit_over_beta0_in_band": [float(v) for v in
                                             (f["beta"] / f["beta0"])[band]],
                "beta_fit_over_beta0_band_median": _band_median(f["beta"] / f["beta0"], band),
                "beta_railed_any_in_band": bool(np.any(f["railed"][band])),
                "fit_residual_rel_band_max": float(np.max(f["residual_rel"][band]))
                if band.any() else None,
                "abs_z0_fit_float64_band_median_ohm": _band_median(np.abs(z0_mine), band),
                "abs_z0_production_raw_z0_band_median_ohm": _band_median(
                    np.abs(raw_z0[d, p, :]), band),
                "abs_alpha_over_gamma_in_band": [float(v) for v in
                                              (np.abs(f["alpha"]) /
                                               np.maximum(np.abs(f["gamma"]), 1e-300))[band]],
            }
            if p != d:
                prow["abs_a_passive_over_a_driven_z0hj_band_median"] = _band_median(
                    np.abs(a_hj[p]) / np.maximum(np.abs(a_hj[d]), 1e-300), band)
                prow["abs_a_passive_over_a_driven_z0hj_in_band"] = [
                    float(v) for v in (np.abs(a_hj[p]) / np.maximum(np.abs(a_hj[d]), 1e-300))[band]]
                prow["abs_a_over_b_z0hj_band_median(production wave split at z0_hj)"] = \
                    _band_median(np.abs(a_hj[p]) / np.maximum(np.abs(b_hj[p]), 1e-300), band)
            if p != d and assign["assigned"]:
                g1 = assign["gamma_passive"][p]
                zref = prow["abs_z0_fit_float64_band_median_ohm"]
                i_p = raw_i1[d, p, :].astype(complex)
                v0 = raw_v[d, p, 0, :].astype(complex)
                a_ref = 0.5 * (v0 + zref * i_p)
                b_ref = 0.5 * (v0 - zref * i_p)
                g2 = a_ref / np.where(np.abs(b_ref) > 0, b_ref, 1e-300)
                g2_hj = a_hj[p] / np.where(np.abs(b_hj[p]) > 0, b_hj[p], 1e-300)
                m1, m2 = _band_median(np.abs(g1), band), _band_median(np.abs(g2), band)
                dev = np.abs(np.abs(g1) - np.abs(g2))[band]
                prow.update(
                    gamma_passive_Q1_fit={
                        "abs_band_median": m1,
                        "abs_in_band": [float(v) for v in np.abs(g1)[band]],
                        "abs_band_max": float(np.max(np.abs(g1)[band])) if band.any() else None,
                        "abs_band_min": float(np.min(np.abs(g1)[band])) if band.any() else None,
                        "complex_at_band_centre": _c2j(
                            g1[band][len(np.flatnonzero(band)) // 2])[0] if band.any() else None,
                        "role": ("alpha/gamma" if port_signs[p] > 0 else "gamma/alpha"),
                    },
                    gamma_passive_Q2_ab_split={
                        "z_ref_ohm": zref, "abs_band_median": m2,
                        "abs_in_band": [float(v) for v in np.abs(g2)[band]],
                        "abs_band_median_z0hj_reference": _band_median(np.abs(g2_hj), band),
                    },
                    comparator={
                        "gate": "band-median |Gamma_Q1| vs |Gamma_Q2(Z_ref=fitted |z0|)| within 0.02",
                        "band_median_abs_difference": abs(m1 - m2),
                        "gate_value": COMPARATOR_GATE,
                        "passed": bool(abs(m1 - m2) <= COMPARATOR_GATE),
                        "per_frequency_max_abs_difference_REPORTED_NOT_GATED": float(np.max(dev))
                        if dev.size else None,
                    },
                    reflector_plane=reflector_plane_from_phase(g1, f["beta"], band),
                    production_S_diagonal_abs_band_median=_band_median(np.abs(prod_s[p, p, :]), band),
                )
                rp = prow["reflector_plane"]
                if rp.get("d_from_probe0_m") is not None:
                    x0 = ports[p]["probe_coords_m"][0]
                    # The reflected wave originates on the FEED side of probe 0
                    # (the wave incident from the structure passes probe 0 first);
                    # a plane at distance d towards the feed sits at x0 - sign*d.
                    x_l = x0 - port_signs[p] * rp["d_from_probe0_m"]
                    rp["x_L_m"] = float(x_l)
                    rp["feed_plane_m"] = float(sim._msl_ports[p].position[
                        {"x": 0, "y": 1}[ports[p]["direction"][1]]])
                    rp["distance_x_L_to_feed_m"] = float(abs(x_l - rp["feed_plane_m"]))
            drive_row["ports"][p] = prow
        drives.append(drive_row)

    # Zc_meas for the driver (design: C0 fitted |z0| band-median), from the
    # own-drive float64 refit at both ports.
    own = [drives[d]["ports"][d]["abs_z0_fit_float64_band_median_ohm"] for d in range(n_ports)]
    return {
        "dump_path": str(npz_path), "dump_schema": meta.get("schema"),
        "dump_schema_version": meta.get("schema_version"),
        "production_smatrix_assembly": meta.get("production_smatrix_assembly"),
        "dump_dtype": dtype_label,
        "freqs_hz": [float(v) for v in freqs], "band_hz": [band_lo, band_hi],
        "n_in_band": int(band.sum()),
        "ports": [{k: v for k, v in pr.items() if k not in ("beta0", "xs")} for pr in ports],
        "zc_meas_ohm_fitted_abs_z0_band_median_own_drive": float(np.median(own)),
        "zc_meas_per_port_ohm": [float(v) for v in own],
        "production_z0_abs_band_median_ohm": [_band_median(np.abs(prod_z0[p, :]), band)
                                           for p in range(n_ports)],
        "drives": drives,
    }


# ---------------------------------------------------------------------------
# One variant run
# ---------------------------------------------------------------------------


def run_variant(args) -> dict:
    import jax

    variant = args.variant
    spec = VARIANTS[variant]
    r_ohm = args.r_ohm
    if r_ohm is None:
        r_ohm = spec["r_ohm"]
    if r_ohm is None:  # C2 / C5 need Zc_meas
        if args.zc_ohm is not None:
            r_ohm = float(args.zc_ohm)
            zc_source = "--zc-ohm"
        elif args.c0_json is not None:
            c0 = json.loads(Path(args.c0_json).read_text(encoding="utf-8"))
            r_ohm = float(c0["analysis"]["zc_meas_ohm_fitted_abs_z0_band_median_own_drive"])
            zc_source = f"--c0-json {args.c0_json}"
            if c0.get("precision_mode") != ("x64" if args.x64 else "x32"):
                raise SystemExit("refusing to take Zc_meas from a C0 run of a different "
                                 "precision mode (never mix within one comparison)")
        else:
            raise SystemExit(f"{variant} needs R = Zc_meas: pass --zc-ohm or --c0-json")
    else:
        zc_source = None

    out: dict = {
        "script": "scripts/diagnostics/msl_passive_port_reflection.py",
        "issue": "#524 items 1-2 (orphaned #507 loose ends) re-measurement",
        "report_only": True, "variant": variant, "variant_what": spec["what"],
        "r_ohm_requested": float(r_ohm), "zc_source": zc_source,
        "precision_mode": "x64" if jax.config.x64_enabled else "x32",
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "settings": {"n_freqs": int(args.n_freqs), "num_periods": float(args.num_periods),
                      "enforce_passivity": False, "band_hz": [GATE_F_LO, GATE_F_HI]},
        "design_settings": {"n_freqs": N_FREQS_DESIGN, "num_periods": NUM_PERIODS_DESIGN},
        "measurement_settings_are_reduced": bool(
            args.smoke or args.n_freqs != N_FREQS_DESIGN
            or float(args.num_periods) != NUM_PERIODS_DESIGN),
        "command": " ".join(sys.argv),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if out["measurement_settings_are_reduced"]:
        out["NOT_THE_MEASUREMENT"] = (
            "reduced settings (num_periods/n_freqs differ from the design's "
            "12/30, or --smoke): this run proves the pipeline executes; its "
            "numbers are NOT the #524 re-measurement and must not be quoted as such")

    cut_x_hi = None
    if spec["geometry"] == "cut_at_port2":
        cut_x_hi, cut_info = resolve_cut_x_hi(r_ohm, variant)
        out["c4_cut_check"] = cut_info
        if not np.isfinite(cut_x_hi):
            out["status"] = "NOT_RUNNABLE_WITHOUT_CODE"
            out["finding"] = ("no trace x_hi candidate keeps PEC on port 2's feed "
                              "column while ending the trace before the next column; "
                              "the end-terminated variant is not runnable without a "
                              "code change, which this driver does not make")
            return out

    sim, info = build_fixture(variant, r_ohm, cut_x_hi=cut_x_hi)
    grid, facts = grid_facts(sim)
    out["fixture"] = info
    out["grid"] = facts

    # Independent realized-termination witness on the built grid.
    out["realized_termination_witness"] = realized_termination_witness(sim, grid)

    dump_dir = Path(args.dump_dir) if args.dump_dir else Path(args.output).parent / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    npz_path = dump_dir / f"{variant}_R{r_ohm:g}_{out['precision_mode']}.npz"

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sim.compute_msl_s_matrix(
                n_freqs=int(args.n_freqs), num_periods=float(args.num_periods),
                enforce_passivity=False, raw_3probe_dump_path=str(npz_path),
            )
    except Exception as exc:  # noqa: BLE001 -- ANY builder/extractor refusal is
        # reported as NOT_RUNNABLE_WITHOUT_CODE (design rule); nothing is patched
        out["status"] = "NOT_RUNNABLE_WITHOUT_CODE"
        out["finding"] = f"{type(exc).__name__}: {exc}"
        out["wallclock_s"] = time.perf_counter() - t0
        return out
    out["wallclock_s"] = time.perf_counter() - t0
    out["preflight_warnings_verbatim"] = [
        f"[{w.category.__name__}] {w.message}" for w in caught]

    settling = (None if res.settling_db is None
                else [float(v) for v in np.asarray(res.settling_db)])
    out["production_result"] = {
        "settling_db_per_drive": settling, "settling_bar_db": SETTLING_BAR_DB,
        "assembly": res.assembly,
        "cond_a_max": None if res.cond_a is None else float(np.max(np.asarray(res.cond_a))),
        "beta_railed_any": None if res.beta_railed is None else bool(np.any(np.asarray(res.beta_railed))),
        "S_dtype": str(np.asarray(res.S).dtype),
    }
    discarded = bool(settling is not None and any(
        (not np.isfinite(v)) or v > SETTLING_BAR_DB for v in settling))
    out["status"] = "DISCARDED_SETTLING" if discarded else "OK"
    if discarded:
        out["settling_note"] = (
            f"settling_db {settling} is above the {SETTLING_BAR_DB:.0f} dB bar: this "
            "run is DISCARDED from every hypothesis reading; re-run with a larger "
            "--num-periods (never averaged, never pinned)")
    if res.assembly != "multi_drive_solve":
        out["assembly_note"] = ("production assembly is not 'multi_drive_solve'; "
                                "the production S diagonal reported here carries the "
                                "single-ratio echo (#507) and is context only")

    out["analysis"] = analyze_dump(npz_path, sim, grid, res, GATE_F_LO, GATE_F_HI)
    out["dtype_label"] = (f"all dump-derived numbers: {out['analysis']['dump_dtype']} "
                          f"inputs, float64 offline fit; precision_mode={out['precision_mode']}")
    return out


def print_table(out: dict) -> None:
    print(f"\n=== {out['variant']}  R_req={out['r_ohm_requested']:g} ohm  "
          f"status={out.get('status')}  precision={out['precision_mode']}  "
          f"wall={out.get('wallclock_s', float('nan')):.1f}s ===")
    if out.get("NOT_THE_MEASUREMENT"):
        print("  ** REDUCED SETTINGS -- NOT THE MEASUREMENT **")
    if "finding" in out:
        print(f"  finding: {out['finding']}")
    for w in out.get("realized_termination_witness", []):
        print(f"  realized-R witness {w['name']} ({w['direction']}): R_fold="
              f"{w['r_fold_ohm']}  k_fold={w['k_fold']}  R_network={w['r_network_ohm']}"
              f"  loaded_cells={w['n_loaded_cells']}")
    pr = out.get("production_result")
    if pr:
        print(f"  settling_db={pr['settling_db_per_drive']}  assembly={pr['assembly']}  "
              f"cond(A)max={pr['cond_a_max']}")
    an = out.get("analysis")
    if not an:
        return
    print(f"  dump dtype={an['dump_dtype']}  in-band points={an['n_in_band']}  "
          f"Zc_meas(|z0| fit, own drive)={an['zc_meas_ohm_fitted_abs_z0_band_median_own_drive']:.3f}")
    print("  drive  port  role-witness(med,min)   |Gamma|Q1 med [min,max]   |Gamma|Q2 med  "
          "comp.dev  perfreq.max  beta/beta0 med  |a_p/a_d|(z0hj)  x_L-feed[mm]")
    for dr in an["drives"]:
        for p, prow in dr["ports"].items():
            w = dr["role_witness"][int(p)]
            tag = f"  {dr['driven']:>5}  {prow['name']:>5}  " \
                  f"{w['median_ratio_in_band']:8.3f},{w['min_ratio_in_band']:8.3f}  "
            if not prow["is_passive"]:
                print(tag + f"(driven)  beta/beta0={prow['beta_fit_over_beta0_band_median']:.4f}"
                      f"  |z0|fit={prow['abs_z0_fit_float64_band_median_ohm']:.2f}")
                continue
            if "gamma_passive_Q1_fit" not in prow:
                print(tag + f"REFUSED: {dr['refusal']}")
                continue
            q1, q2, cmp_ = prow["gamma_passive_Q1_fit"], prow["gamma_passive_Q2_ab_split"], prow["comparator"]
            rp = prow["reflector_plane"]
            xl = rp.get("distance_x_L_to_feed_m")
            print(tag + f"{q1['abs_band_median']:.4f} [{q1['abs_band_min']:.4f},{q1['abs_band_max']:.4f}]"
                  f"   {q2['abs_band_median']:.4f}   {cmp_['band_median_abs_difference']:.4f}"
                  f"{'*' if not cmp_['passed'] else ' '}  "
                  f"{cmp_['per_frequency_max_abs_difference_REPORTED_NOT_GATED']:.4f}   "
                  f"{prow['beta_fit_over_beta0_band_median']:.4f}      "
                  f"{prow['abs_a_passive_over_a_driven_z0hj_band_median']:.4f}      "
                  f"{'n/a' if xl is None else f'{1e3 * xl:.2f}'}")
    print("  (* = comparator band-median gate 0.02 NOT met; per-frequency max is reported, not gated)")


# ---------------------------------------------------------------------------
# Summary across runs (hypotheses, falsifier, implied k, item 2)
# ---------------------------------------------------------------------------


def _load_runs(paths):
    runs = {}
    for pth in paths:
        r = json.loads(Path(pth).read_text(encoding="utf-8"))
        runs[r["variant"]] = r
    modes = {r["precision_mode"] for r in runs.values()}
    if len(modes) > 1:
        raise SystemExit(f"refusing to mix precision modes in one comparison: {modes}")
    reduced = [v for v, r in runs.items() if r.get("measurement_settings_are_reduced")]
    return runs, reduced


def _passive_abs_gamma(run: dict) -> dict:
    """{port_index: band-median |Gamma_Q1|} read from the OTHER drive."""
    out = {}
    if run.get("status") != "OK":
        return out
    for dr in run["analysis"]["drives"]:
        for p, prow in dr["ports"].items():
            if prow["is_passive"] and "gamma_passive_Q1_fit" in prow:
                out[int(p)] = {
                    "abs": prow["gamma_passive_Q1_fit"]["abs_band_median"],
                    "complex_centre": prow["gamma_passive_Q1_fit"]["complex_at_band_centre"],
                    "comparator_passed": prow["comparator"]["passed"],
                    "x_L_to_feed_m": prow["reflector_plane"].get("distance_x_L_to_feed_m"),
                    "x_L_m": prow["reflector_plane"].get("x_L_m")}
    return out


def summarize(paths, output: Path | None) -> dict:
    runs, reduced = _load_runs(paths)
    summ: dict = {"report_only": True, "runs": sorted(runs),
                      "precision_mode": next(iter(runs.values()))["precision_mode"],
                      "reduced_settings_runs": reduced}
    if reduced:
        summ["NOT_THE_MEASUREMENT"] = ("one or more inputs ran at reduced settings; "
                                       "this summary is a pipeline check, not the result")
    if "C0" not in runs or runs["C0"].get("status") != "OK":
        summ["verdict"] = "C0 missing or not OK: no Zc reference, nothing decidable"
        summ["statuses"] = {v: r.get("status") for v, r in runs.items()}
        print(json.dumps(summ, indent=2, default=str))
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summ, indent=2), encoding="utf-8")
        return summ
    zc = runs["C0"]["analysis"]["zc_meas_ohm_fitted_abs_z0_band_median_own_drive"]
    summ["zc_ohm_from_C0"] = zc
    g = {v: _passive_abs_gamma(r) for v, r in runs.items()}
    summ["gamma_passive_abs_band_median_by_variant"] = {
        v: {str(p): d["abs"] for p, d in gp.items()} for v, gp in g.items()}
    summ["comparator_all_passed"] = all(d["comparator_passed"]
                                        for gp in g.values() for d in gp.values())
    if not summ["comparator_all_passed"]:
        summ["comparator_note"] = ("Q1-vs-Q2 comparator gate failed on at least one port; "
                                   "design rule: the comparator is the defect, stop before "
                                   "reading physics")

    # Realized-R witness roll-up.
    summ["realized_termination_witness"] = {
        v: [{"name": w["name"], "r_requested": w["r_requested_ohm"], "k_fold": w["k_fold"],
                 "r_network": w["r_network_ohm"]} for w in r.get("realized_termination_witness", [])]
        for v, r in runs.items()}

    # R-sweep points (shipped geometry, both ports).
    gc = None
    if g.get("C1"):
        gc = {p: complex(*d["complex_centre"]) for p, d in g["C1"].items()}
    points = []
    for v in ("C0", "C1", "C2", "C3"):
        if v in runs and runs[v].get("status") == "OK":
            r_req = runs[v]["r_ohm_requested"]
            for p, d in g[v].items():
                gm = d["abs"]
                pred_shunt = zc / (2 * r_req + zc)
                pred_shunt_stub = None
                if gc is not None and p in gc and abs(1 - gc[p]) > 1e-9:
                    z_stub = zc * (1 + gc[p]) / (1 - gc[p])
                    z_par = 1.0 / (1.0 / r_req + 1.0 / z_stub)
                    pred_shunt_stub = float(abs((z_par - zc) / (z_par + zc)))
                pred_end = abs(r_req - zc) / (r_req + zc)
                k_shunt = (zc / gm - zc) / (2 * r_req) if gm > 0 else None
                k_end = [float(zc * (1 + gm) / (1 - gm) / r_req),
                         float(zc * (1 - gm) / (1 + gm) / r_req)] if gm < 1 else None
                points.append({"variant": v, "port": p, "r_requested": r_req, "abs_gamma": gm,
                                   "H_shunt": pred_shunt, "H_shunt_with_C1_stub": pred_shunt_stub,
                                   "H_end": pred_end,
                                   "implied_k_H_shunt": k_shunt, "implied_k_H_end_branches": k_end,
                                   "x_L_to_feed_m": d["x_L_to_feed_m"]})
    summ["r_sweep_points"] = points

    def _fits(key):
        devs = [abs(pt["abs_gamma"] - pt[key]) for pt in points if pt.get(key) is not None]
        return (len(devs) >= 3 and max(devs) <= HYPOTHESIS_BAND), (max(devs) if devs else None), len(devs)

    hyp = {}
    for key in ("H_shunt", "H_shunt_with_C1_stub", "H_end"):
        ok, mx, n = _fits(key)
        hyp[key] = {"fits_within_band": ok, "max_abs_dev": mx, "n_points": n, "band": HYPOTHESIS_BAND}
    if g.get("C1"):
        c1 = {p: d["abs"] for p, d in g["C1"].items()}
        near_c1 = all(abs(pt["abs_gamma"] - c1.get(pt["port"], np.nan)) <= 0.02
                      for pt in points if pt["variant"] != "C1")
        cpml_near = []
        for v in ("C0", "C1", "C2", "C3"):
            if v in runs and runs[v].get("status") == "OK":
                for p, d in g[v].items():
                    faces = runs[v]["grid"]["ports"][p]["cpml_inner_faces_m"]
                    xl = d["x_L_m"]
                    cpml_near.append(bool(xl is not None and
                                          min(abs(xl - faces[0]), abs(xl - faces[1])) < 1e-3
                                          and (d["x_L_to_feed_m"] or 0) >= 1e-3))
        hyp["H_cpml"] = {"all_R_points_within_0p02_of_C1": near_c1,
                             "x_L_within_1mm_of_a_CPML_face_not_feed": all(cpml_near) if cpml_near else None,
                             "fits": bool(near_c1 and cpml_near and all(cpml_near))}
    summ["hypotheses"] = hyp
    fitting = [k for k, h in hyp.items() if h.get("fits_within_band") or h.get("fits")]
    if len(points) < 3:
        summ["verdict"] = "fewer than 3 R-sweep points: not decidable"
    elif not fitting:
        summ["verdict"] = ("NO SINGLE-REFLECTOR MODEL fits all R points within 0.03 "
                           "(predeclared falsifier) -- report and stop; no two-parameter fit")
    else:
        summ["verdict"] = f"hypotheses within band: {fitting}"
    # Implied-k finding (reported, never fitted or applied).
    ks = [pt["implied_k_H_shunt"] for pt in points if pt["implied_k_H_shunt"] is not None
          and pt["variant"] != "C1"]
    if ks:
        summ["implied_k_H_shunt_per_point"] = ks
        spread = max(ks) - min(ks)
        summ["implied_k_finding"] = (
            f"H_shunt implied k per R point = {[f'{k:.3f}' for k in ks]} (spread {spread:.3f}); "
            + ("consistent shape at k != 1 -> FINDING pointing at the realized termination "
               "(compare k_fold above), not a confirmation of H_shunt"
               if spread <= 0.1 and abs(np.median(ks) - 1) > 0.1 else
               "k ~ 1 within the points' spread" if abs(np.median(ks) - 1) <= 0.1 else
               "no consistent k across R points -> shape does not follow 1/(2R+Zc)"))
    # Item 2.
    c0g = g["C0"]
    if len(c0g) == 2:
        d12 = abs(c0g[0]["abs"] - c0g[1]["abs"])
        item2 = {"C0_abs_gamma_port1": c0g[0]["abs"], "C0_abs_gamma_port2": c0g[1]["abs"],
                     "C0_difference": d12, "band": ITEM2_ASYMMETRY_BAND,
                     "asymmetric": bool(d12 > ITEM2_ASYMMETRY_BAND)}
        for v in ("SYM", "ROTY"):
            if v in g and len(g[v]) == 2:
                item2[f"{v}_abs_gamma"] = [g[v][0]["abs"], g[v][1]["abs"]]
                item2[f"{v}_difference"] = abs(g[v][0]["abs"] - g[v][1]["abs"])
        if item2["asymmetric"]:
            item2["reading"] = (
                "C0 ports differ by > 0.02: if SYM_difference <= 0.02 the asymmetry is "
                "feed/CPML snapping (fixture); if it persists in SYM and follows the "
                "direction sign in ROTY it is a '+'/'-' lane asymmetry -> its own defect issue")
        else:
            item2["reading"] = "C0 ports agree within 0.02: item 2 collapses into item 1"
        summ["item2_drive_asymmetry"] = item2
    # C4/C5 within-run discriminator.
    for v in ("C4", "C5"):
        if v in runs:
            summ[f"{v}"] = {"status": runs[v].get("status"), "finding": runs[v].get("finding"),
                                "gamma_passive": summ["gamma_passive_abs_band_median_by_variant"].get(v),
                                "reading": ("drive 1 reads END-terminated port 2 (H_end predicts "
                                         f"|R-Zc|/(R+Zc) = {abs(runs[v]['r_ohm_requested'] - zc) / (runs[v]['r_ohm_requested'] + zc):.3f}); "
                                         "drive 2 reads the CONTINUING port 1 (H_shunt predicts "
                                         f"{zc / (2 * runs[v]['r_ohm_requested'] + zc):.3f})")}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("verdict", "hypotheses", "implied_k_finding",
                                            "item2_drive_asymmetry", "comparator_all_passed")
                      if k in summ}, indent=2, default=str))
    return summ


# ---------------------------------------------------------------------------
# Stale-literal enumeration (review item 2) -- report only
# ---------------------------------------------------------------------------

_SKIP_DIRS = {".git", ".omx", ".omc", "__pycache__", "node_modules", ".pytest_cache",
              ".venv", "venv", "build", "dist"}
_TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".cff", ".toml"}


def _py_line_classes(path: Path) -> dict:
    """Map line number -> token class for a .py file via tokenize."""
    classes: dict = {}
    try:
        with tokenize.open(path) as fh:
            for tok in tokenize.generate_tokens(fh.readline):
                if tok.type == tokenize.COMMENT:
                    for ln in range(tok.start[0], tok.end[0] + 1):
                        classes.setdefault(ln, "comment")
                elif tok.type == tokenize.STRING:
                    for ln in range(tok.start[0], tok.end[0] + 1):
                        classes.setdefault(ln, "docstring-prose")
                elif tok.type == tokenize.NUMBER:
                    for ln in range(tok.start[0], tok.end[0] + 1):
                        classes.setdefault(ln, "number")
    except (tokenize.TokenError, SyntaxError, UnicodeDecodeError):
        pass
    return classes


def enumerate_stale_literals(root: Path) -> list[dict]:
    self_path = Path(__file__).resolve()
    hits = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.resolve() == self_path:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        py_classes = _py_line_classes(path) if path.suffix == ".py" else {}
        rel = path.relative_to(root)
        for ln, line in enumerate(lines, start=1):
            for label, rx in STALE_LITERAL_PATTERNS.items():
                if not rx.search(line):
                    continue
                if path.suffix == ".py":
                    cls = py_classes.get(ln, "code-other")
                    if cls == "number":
                        cls = ("test-parametrize-value(keep)" if rel.parts[0] == "tests"
                               else "code-literal")
                else:
                    cls = "issue-text"
                hits.append({"path": str(rel), "line": ln, "literal": label,
                                 "classification": cls, "text": line.strip()[:160]})
    return hits


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=sorted(VARIANTS), help="which configuration to run")
    ap.add_argument("--output", type=str, help="result JSON path")
    ap.add_argument("--dump-dir", type=str, default=None,
                    help="where the raw_3probe_dump_path npz goes (default: <output dir>/dumps)")
    ap.add_argument("--r-ohm", type=float, default=None,
                    help="override the requested termination R on BOTH ports")
    ap.add_argument("--zc-ohm", type=float, default=None, help="Zc_meas for C2/C5")
    ap.add_argument("--c0-json", type=str, default=None,
                    help="C0 result JSON to take Zc_meas from (C2/C5)")
    ap.add_argument("--n-freqs", type=int, default=N_FREQS_DESIGN)
    ap.add_argument("--num-periods", type=float, default=NUM_PERIODS_DESIGN)
    ap.add_argument("--x64", action="store_true",
                    help="enable jax_enable_x64 before the build (requires JAX_ENABLE_X64=1 in env)")
    ap.add_argument("--smoke", action="store_true",
                    help="mark the run as a pipeline smoke (NOT the measurement)")
    ap.add_argument("--summarize", nargs="+", metavar="JSON",
                    help="aggregate result JSONs: hypotheses, falsifier, implied k, item 2")
    ap.add_argument("--enumerate-stale-literals", action="store_true",
                    help="grep the tree for the July literal class and classify each site")
    ap.add_argument("--root", type=str, default=str(REPO), help="tree root for enumeration")
    args = ap.parse_args(argv)

    if args.enumerate_stale_literals:
        hits = enumerate_stale_literals(Path(args.root))
        print(f"{'path:line':<72} {'literal':<10} {'class':<30} text")
        for h in hits:
            print(f"{h['path'] + ':' + str(h['line']):<72} {h['literal']:<10} "
                  f"{h['classification']:<30} {h['text']}")
        print(f"\n{len(hits)} site(s). Report only -- no edits made. Classes: docstring-prose / "
              "comment / test-parametrize-value(keep) / issue-text / code-other / code-literal.")
        if args.output:
            Path(args.output).write_text(json.dumps(hits, indent=2), encoding="utf-8")
        return 0

    if args.summarize:
        summarize(args.summarize, Path(args.output) if args.output else None)
        return 0

    if not args.variant or not args.output:
        ap.error("--variant and --output are required for a run")

    env_x64 = os.environ.get("JAX_ENABLE_X64")
    if args.x64 and env_x64 != "1":
        ap.error("--x64 requires JAX_ENABLE_X64=1 in the environment (state the precision "
                 "explicitly; never mix within one comparison)")
    if not args.x64 and env_x64 == "1":
        ap.error("JAX_ENABLE_X64=1 is set but --x64 was not passed; pass --x64 or run with "
                 "JAX_ENABLE_X64=0")
    import jax
    if args.x64:
        jax.config.update("jax_enable_x64", True)

    out = run_variant(args)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print_table(out)
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
