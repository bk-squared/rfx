#!/usr/bin/env python3
"""Cross-validation 14: rectangular PEC cavity eigenfrequencies vs the EXACT
Pozar analytic oracle (Tier-1 comparator, fully local, CPU-only).

WHAT THIS VALIDATES
-------------------
rfx's resonance / eigenfrequency pipeline end-to-end through the PUBLIC
``Simulation`` API:

    Simulation(boundary="pec")  ->  add_source (broadband soft dipoles)
                                ->  add_vector_probe (records Ex,Ey,Ez,...)
                                ->  run()  ->  rfx.harminv (Matrix-Pencil)

against the closed-form resonances of an air-filled rectangular PEC cavity
(Pozar, *Microwave Engineering*, cavity-resonator section):

    f_mnl = (c / 2) * sqrt( (m/a)^2 + (n/b)^2 + (l/d)^2 )

The oracle is RE-DERIVED in this file (``pozar_cavity_freq`` below); it does
NOT import any producer-side mode helper. This is an EXACT oracle (no meshing,
no external solver), so the gate is tight: a Yee closed-cavity eigenfrequency
with exact wall registration should match to well under 1%.

A SECOND in-file oracle (``yee_cavity_freq``) gives the EXACT eigenfrequency of
the Yee scheme itself in the same box, closed form and with no free parameter.
Pozar is the physics the case claims; the Yee eigenvalue is what a correct
implementation of the scheme must return on this mesh. Gating both separates
"small error" from "exactly the predicted discretization error", which is the
difference between a gate that tolerates a defect and one that identifies it.

WHY THE CLOSED PEC CONFIG IS CORRECT HERE (and what it means for the report)
--------------------------------------------------------------------------
Per the rfx validation rules: PEC for closed cavities, CPML for open
structures — never mixed. A resonant cavity is a *closed* problem, so we use
``boundary="pec"`` and NO CPML. Consequences the report must respect:

  * A lossless closed PEC cavity CONSERVES energy — the field never decays.
    The energy-based ring-down SETTLING WITNESS (a −40 dB tail check) is a
    tool for OPEN / CPML truncation and DOES NOT APPLY here. Forcing
    ``until_decay`` on a lossless closed cavity is wrong. We run a fixed
    ``num_periods`` (long enough for harminv frequency resolution) and report
    the HARMINV FIT RESIDUAL instead of a settling tail.

  * The harminv *Q* in a lossless closed cavity is infinite in physics and a
    pure window-length artefact numerically (see the warning in
    ``rfx/harminv.py``). We print Q for completeness but DO NOT gate on it or
    report it as physics — only the FREQUENCY is claims-bearing.

  * Preflight on a lossless-closed + no-CPML cavity is EXPECTED to be clean /
    advisory-free; that clean report is the correct signal, not a missing
    check. Preflight output is quoted verbatim below before any number.

MESH CHOICE (why the match is tight, honestly)
----------------------------------------------
We force ``dx`` to an EXACT DIVISOR of (a, b, d). rfx puts PEC walls at the
first/last grid planes, so the effective wall separation is ``(n-1)*dx``; with
dx = 1 mm and (a,b,d) = (50,30,40) mm the effective cavity is EXACTLY
(0.050, 0.030, 0.040) m — zero geometric-quantization error. What remains is
pure Yee numerical dispersion, which is small (~60 cells/wavelength at the
fundamental) and CONVERGES at 2nd order. The optional ``--converge`` leg halves
dx and shows the error drop ~4x, proving the residual is genuine dispersion,
not extraction luck. (Contrast ``tests/oracle/test_cavity.py``, which uses auto-dx on a
non-exact-divisor domain and honestly reports ~1.9% from wall registration +
coarse mesh + convergence to ~0.5% at 2x — same physics, different mesh
hygiene.)

FOLLOW-ON (out of scope here)
-----------------------------
A Palace / eigenmode external solver on VESSL is a documented Tier-2 comparator
for this structure class. It is NOT attempted here: the EXACT Pozar analytic is
the Tier-1 gate for this session and needs no external solver.

GATES (re-declared 2026-08-31, issue #812 crossval gate audit)
-------------------------------------------------------------
    Gate 0  wall registration: |(n-1)*dx - (a,b,d)| <= 1e-9 m on every axis
    Gate 1  TE101 vs the Pozar continuum oracle < 1%
    Gate 2  EVERY one of the seven declared modes < 2% vs Pozar, aggregated
            with max; a mode that cannot be extracted is a hard FAILURE
    Gate 3  |f_measured - f_yee| <= 0.1/T on every mode, against the EXACT
            discrete-Yee eigenvalue of this box (second in-file oracle);
            evaluated only when Gate 0 passes, because that prediction is
            exact only for walls on grid planes

What Gate 2 used to be, and why it changed (issue #812): it took ``min()``
over the six higher modes at the same 2%. Every axis has a declared target
with a zero index on it (n=0 for TE101/TE201/TE102, m=0 for TE011, l=0 for
TM110/TM210), and f_mnl depends on an extent only through m/a, so such a mode
scores identically 0.000% for ANY single-axis dimensional error -- and min()
selected exactly it. Measured: shrinking a from 50 to 25 mm (-50%) failed
Gate 1 at TE101 err 47.334% while Gate 2 still read PASS. Separately, a
NOT-FOUND mode removed itself from ``errs`` instead of failing. The 2%
number is UNCHANGED; only the aggregator and the not-found handling moved,
which is a strict tightening. Gate 0 gates a quantity this script already
computed and printed but never checked. Thresholds and their derivations:
docs/design_notes/cv14_rect_cavity_gate_predeclaration.md.

Exit contract (crossval registry, validation/crossval/manifest.json)
-------------------------------------------------------------------
    0 -> every configured gate passed (Gates 0-3 above)
    1 -> a gate failed, or a target mode could not be extracted
    (2 is not reachable: this case needs no external solver, only the
     in-file analytic oracles, so a reference can never be "unavailable".)

Usage
-----
    python validation/crossval/14_rect_cavity_pozar.py            # gate table + verdict (dx=1mm)
    python validation/crossval/14_rect_cavity_pozar.py --converge # + dx=0.5mm convergence witness
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

# --- resolve `import rfx` from THIS checkout (a bare run would otherwise pick
# --- up whatever rfx is installed or first on sys.path, which in a multi-
# --- worktree setup is a DIFFERENT copy of the solver than the one under test).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rfx.api import Simulation          # noqa: E402
from rfx.grid import C0                 # noqa: E402  speed of light (m/s)
from rfx.harminv import harminv         # noqa: E402  Matrix-Pencil estimator


# ---------------------------------------------------------------------------
# INDEPENDENT ORACLE — Pozar rectangular-cavity resonance, re-derived in-script.
# ---------------------------------------------------------------------------
def pozar_cavity_freq(a: float, b: float, d: float, m: int, n: int, l: int) -> float:
    """Exact resonant frequency of an air-filled rectangular PEC cavity (Hz).

        f_mnl = (c / 2) * sqrt( (m/a)^2 + (n/b)^2 + (l/d)^2 )

    a, b, d : cavity extents along x, y, z (metres).
    m, n, l : mode indices along x, y, z.
    Valid for both TE_mnl and TM_mnl (same frequency; differ only in which
    index combinations are physically allowed).
    """
    return (C0 / 2.0) * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (l / d) ** 2)


# ---------------------------------------------------------------------------
# INDEPENDENT ORACLE #2 — the EXACT eigenfrequency of the YEE SCHEME in this
# same PEC box, re-derived in-script. Imports no producer-side helper either.
# ---------------------------------------------------------------------------
def yee_cavity_freq(a: float, b: float, d: float, m: int, n: int, l: int,  # noqa: E741
                    hx: float, hy: float, hz: float, dt: float) -> float:
    """Exact discrete (Yee) resonant frequency of the same air-filled PEC
    cavity on a uniform staggered grid with cell sizes (hx, hy, hz) and
    timestep ``dt`` (Hz).

        sin(w*dt/2) = c*dt * sqrt( sum_i sin^2(k_i*h_i/2) / h_i^2 )
        k_x = m*pi/a,  k_y = n*pi/b,  k_z = l*pi/d
        f = w / (2*pi)

    WHY THIS IS EXACT AND NOT A FIT: rfx puts the PEC walls on the first and
    last grid planes, and the mesh here is an exact divisor of (a, b, d), so
    the effective wall separation is exactly (a, b, d) and the discrete field
    sin(m*pi*x_i/a) sampled at x_i = i*hx is an EXACT eigenvector of the
    discrete curl-curl operator with eigenvalue (2/hx)*sin(k_x*hx/2). Leapfrog
    time stepping contributes the arcsin. The result has no free parameter.

    This exactness is CONDITIONAL on exact wall registration (Gate 0). If the
    box does not land on grid planes then k_i != m*pi/a and this prediction is
    VOID, not merely loose — which is why Gate 3 is evaluated only when Gate 0
    passes.
    """
    kx, ky, kz = m * np.pi / a, n * np.pi / b, l * np.pi / d
    s = np.sqrt((np.sin(kx * hx / 2.0) / hx) ** 2
                + (np.sin(ky * hy / 2.0) / hy) ** 2
                + (np.sin(kz * hz / 2.0) / hz) ** 2)
    return float((2.0 / dt) * np.arcsin(C0 * dt * s) / (2.0 * np.pi))


# Canonical cavity: all three dimensions distinct + incommensurate so the low
# modes are well separated (no accidental degeneracy) and dx = 1 mm divides
# each dimension exactly.
A, B, D = 0.050, 0.030, 0.040  # x, y, z (m)

# Target modes and the E-component that is dominant for each (used only to hint
# which probe channel to search; identification is by nearest analytic freq).
#   TE_mnl to z: fundamental family has l>=1; TM_mnl to z: allows l=0.
#   channel key: "ex"/"ey"/"ez"
TARGET_MODES = [
    ("TE101", (1, 0, 1), "ey"),   # fundamental (dominant) mode: Ey ~ sin(pi x/a) sin(pi z/d)
    ("TM110", (1, 1, 0), "ez"),   # lowest TM: Ez ~ sin(pi x/a) sin(pi y/b)
    ("TE011", (0, 1, 1), "ex"),   # Ex-polarized TE
    ("TM111", (1, 1, 1), "ez"),   # (TE/TM)111 — appears on Ey and Ez
    ("TE201", (2, 0, 1), "ey"),
    ("TM210", (2, 1, 0), "ez"),
    ("TE102", (1, 0, 2), "ey"),
]

FREQ_MAX = 10e9  # sets the broadband source (f0 = 5 GHz, bw 0.8 -> ~1..9 GHz)

# --------------------------- GATE CONSTANTS ---------------------------------
# All four are pre-declared with their derivations in
# docs/design_notes/cv14_rect_cavity_gate_predeclaration.md, in a commit that
# PRECEDES the commit measuring them. None is fitted to any cv14 output.

# T0 (Gate 0, NEW): wall registration. dx = 1 mm divides (a,b,d) = (50,30,40) mm
# exactly and rfx puts PEC walls on the first/last grid planes, so the effective
# separation (n-1)*dx must equal (a,b,d) EXACTLY. Admissible deviation is IEEE-754
# rounding of that product: <= 2^-52 relative, i.e. <= 1.1e-17 m at 0.05 m. The
# SMALLEST geometric error the grid can EXPRESS is one cell = 1e-3 m. 1e-9 m sits
# ~8 decades above float noise and ~6 decades below the smallest expressible error.
WALL_REG_TOL_M = 1e-9

# T1 (Gate 1): unchanged, the case's own published threshold.
GATE1_TE101_PCT = 1.0

# T2 (Gate 2): the SAME published 2.0%, but aggregated with max over ALL SEVEN
# declared targets instead of min over six, and NOT-FOUND is a hard failure.
# Strict tightening: the outcomes a max-over-seven gate admits are a SUBSET of
# what the former min-over-six admitted, for every possible measurement.
GATE2_ALL_MODES_PCT = 2.0

# T3 (Gate 3, NEW): discrete-Yee residual budget, as a fraction of the Rayleigh
# (Fourier) limit 1/T of the record actually handed to harminv. 1/T is the
# resolution of a NON-parametric estimator; harminv is a parametric matrix-pencil
# estimator on a high-SNR sum of exactly-undamped exponentials (the model it
# assumes is the model the signal literally is), whose Cramer-Rao bound scales as
# ~1/(SNR*T*sqrt(N)) -- orders of magnitude below 1/T. One TENTH of one Fourier
# bin is therefore a loose bound on the estimator and a tight bound on the physics.
YEE_BUDGET_BINS = 0.1


def build_cavity(dx: float) -> Simulation:
    """Closed PEC cavity with three orthogonally-polarized broadband soft
    sources and one vector probe (records Ex, Ey, Ez, Hx, Hy, Hz).

    Sources/probe sit at generic OFF-NODE physical coordinates (absolute
    metres, not cell-relative) so every low mode is excited and observed.
    """
    sim = Simulation(freq_max=FREQ_MAX, domain=(A, B, D), boundary="pec", dx=dx)
    # Three soft point sources (impedance-free -> no cavity damping), one per
    # polarization, at distinct asymmetric interior points.
    sim.add_source((0.013, 0.011, 0.017), component="ex")
    sim.add_source((0.019, 0.023, 0.013), component="ey")
    sim.add_source((0.031, 0.013, 0.023), component="ez")
    # Vector probe at a fourth generic interior point.
    sim.add_vector_probe((0.037, 0.017, 0.029))
    return sim


# vector-probe column order is Ex,Ey,Ez,Hx,Hy,Hz
_CHAN = {"ex": 0, "ey": 1, "ez": 2}


# harminv record shaping — ONE source of truth, so the Gate-3 budget is derived
# from the record harminv actually sees rather than from the raw run length.
_MAX_HARMINV_SAMPLES = 8000


def harminv_record(res):
    """Return ``(start, n_samples, dt_h)`` describing the slice of the probe
    time series that :func:`extract_modes` hands to harminv.

    ``start`` skips the excitation region; the remainder is decimated (never
    interpolated) to at most ``_MAX_HARMINV_SAMPLES`` points. The record SPAN
    is ``n_samples * dt_h`` and its Rayleigh (Fourier) resolution limit is
    ``1 / (n_samples * dt_h)`` -- the quantity Gate 3's budget is a tenth of.
    """
    ts = np.asarray(res.time_series)
    dt = float(res.dt)
    # Skip the excitation region: default GaussianPulse f0=freq_max/2, bw=0.8,
    # cutoff=3 -> pulse completes at t0 = 3*tau; start well after (2*t0).
    tau = 1.0 / ((FREQ_MAX / 2) * 0.8 * np.pi)
    start = int(np.ceil(2.0 * 3.0 * tau / dt))
    n_avail = max(ts.shape[0] - start, 0)
    if n_avail > _MAX_HARMINV_SAMPLES:
        step = n_avail // _MAX_HARMINV_SAMPLES
        return start, min(len(range(0, n_avail, step)), _MAX_HARMINV_SAMPLES), dt * step
    return start, n_avail, dt


def harminv_freq_resolution(res) -> float:
    """Rayleigh limit 1/T (Hz) of the record :func:`harminv_record` describes."""
    _start, n, dt_h = harminv_record(res)
    return float("inf") if n < 1 else 1.0 / (n * dt_h)


def extract_modes(res, target_freq: float, channels=("ex", "ey", "ez"),
                  band=0.03):
    """Search a +/- band window around ``target_freq`` across the given E
    probe channels; return the harminv mode nearest the analytic frequency.

    The window (default +/-3%) is far wider than any plausible discretization
    error, and harminv resolves << 0.1%, so nearest-to-analytic is an
    UNAMBIGUOUS mode ID, not a bin-snap toward the expected value (same
    reasoning as tests/oracle/test_cavity.py). Gate 3 closes the residual loophole
    this window would otherwise leave open: an impostor mode of a DIFFERENT
    (m,n,l) that happened to drift into the window would have to sit within
    a tenth of a Fourier bin of the target's own discrete-Yee eigenvalue.
    """
    ts = np.asarray(res.time_series)
    dt = float(res.dt)
    start, n_keep, dt_h = harminv_record(res)
    step = max(int(round(dt_h / dt)), 1)

    f_lo, f_hi = target_freq * (1 - band), target_freq * (1 + band)
    best = None
    for ch in channels:
        sig = ts[start:, _CHAN[ch]]
        sig = sig - np.mean(sig)
        sig_h = sig[::step][:n_keep]
        if len(sig_h) < 20:
            continue
        for mode in harminv(sig_h, dt_h, f_lo, f_hi):
            cand = (abs(mode.freq - target_freq), mode.freq, mode.Q,
                    mode.amplitude, mode.error, ch)
            if best is None or cand[0] < best[0]:
                best = cand
    return best  # None or (|df|, f, Q, amp, fit_err, channel)


def run_leg(dx: float, num_periods: float):
    """Build, preflight (captured verbatim), run, and extract all target modes
    at a given resolution.

    Returns ``(rows, preflight_text, grid_info, wall_secs)`` where each row is
    ``(name, (m,n,l), f_pozar, f_yee, f_meas|None, Q, amp, fit_err, chan)`` and
    ``grid_info`` carries the effective wall separation and the harminv Rayleigh
    limit that Gate 0 and Gate 3 are evaluated against.
    """
    sim = build_cavity(dx)
    grid = sim._build_grid()
    eff = ((grid.nx - 1) * grid.dx, (grid.ny - 1) * grid.dx,
           (grid.nz - 1) * getattr(grid, "dz", grid.dx))

    # --- preflight, captured verbatim (report requirement) ---
    buf = io.StringIO()
    with redirect_stdout(buf):
        report = sim.preflight()
    pf_lines = buf.getvalue().splitlines()
    for item in report:
        pf_lines.append(str(item))
    if not pf_lines:
        pf_lines = ["(preflight returned no issues)"]

    t0 = time.time()
    res = sim.run(num_periods=num_periods, compute_s_params=False)
    wall = time.time() - t0

    hx = hy = float(grid.dx)
    hz = float(getattr(grid, "dz", grid.dx))
    dt_grid = float(grid.dt)

    rows = []
    for name, (m, n, l), hint in TARGET_MODES:
        fa = pozar_cavity_freq(A, B, D, m, n, l)
        fy = yee_cavity_freq(A, B, D, m, n, l, hx, hy, hz, dt_grid)
        # search the hinted channel first, then all channels as fallback
        found = extract_modes(res, fa, channels=(hint, "ex", "ey", "ez"))
        if found is None:
            rows.append((name, (m, n, l), fa, fy, None, None, None, None, None))
        else:
            _df, f, Q, amp, fit_err, ch = found
            rows.append((name, (m, n, l), fa, fy, f, Q, amp, fit_err, ch))
    grid_info = (grid.shape, grid.dx, hz,
                 eff, dt_grid,
                 np.asarray(res.time_series).shape[0],
                 harminv_freq_resolution(res))
    return rows, pf_lines, grid_info, wall


def print_table(rows) -> dict:
    """Print the mode table and return {name: pct_err} (Pozar continuum error).

    The ``yee GHz`` and ``meas-yee`` columns carry the SECOND oracle: the exact
    discrete-Yee eigenvalue of the same box and the residual Gate 3 gates.
    """
    print(f"{'mode':<8} {'(m,n,l)':<10} {'pozar GHz':>11} {'yee GHz':>11} "
          f"{'rfx-harminv GHz':>16} {'%err(poz)':>10} {'meas-yee MHz':>13} "
          f"{'chan':>5} {'Q(artefact)':>12} {'fit resid':>10}")
    print("-" * 118)
    errs = {}
    for name, idx, fa, fy, f, Q, amp, fit_err, ch in rows:
        if f is None:
            print(f"{name:<8} {str(idx):<10} {fa/1e9:>11.5f} {fy/1e9:>11.5f} "
                  f"{'NOT FOUND':>16} {'--':>10} {'--':>13} "
                  f"{'--':>5} {'--':>12} {'--':>10}")
            continue
        pct = abs(f - fa) / fa * 100.0
        errs[name] = pct
        print(f"{name:<8} {str(idx):<10} {fa/1e9:>11.5f} {fy/1e9:>11.5f} "
              f"{f/1e9:>16.5f} {pct:>9.4f}% {(f - fy)/1e6:>13.4f} "
              f"{ch:>5} {Q:>12.3g} {fit_err:>10.2e}")
    return errs


def harminv_record_of_leg(nsteps: int, dt: float):
    """Same record shaping as :func:`harminv_record`, from (nsteps, dt) alone,
    so the main report can print the span without holding the result object."""
    tau = 1.0 / ((FREQ_MAX / 2) * 0.8 * np.pi)
    start = int(np.ceil(2.0 * 3.0 * tau / dt))
    n_avail = max(nsteps - start, 0)
    if n_avail > _MAX_HARMINV_SAMPLES:
        step = n_avail // _MAX_HARMINV_SAMPLES
        return start, min(len(range(0, n_avail, step)), _MAX_HARMINV_SAMPLES), dt * step
    return start, n_avail, dt


def evaluate_gates(rows, eff, freq_resolution_hz):
    """Evaluate every cv14 gate. PURE: no simulation, no I/O, no globals beyond
    the pre-declared thresholds. Returns ``(ok, lines)``.

    ``rows``  as produced by :func:`run_leg`.
    ``eff``   measured effective wall separation (metres, x/y/z).
    ``freq_resolution_hz``  Rayleigh limit 1/T of the harminv record.

    Gate order is load-bearing: Gate 3's discrete-Yee prediction is EXACT only
    while the walls land on grid planes, so it is evaluated only if Gate 0 holds.
    """
    lines = []
    ok = True

    # ---- Gate 0 (NEW): wall registration --------------------------------
    # run_leg already COMPUTED this quantity and the script already PRINTED it;
    # before #812 nothing gated it, so the one zero-noise readout of the
    # geometric defect class the audit exploited was report-only.
    targets = (A, B, D)
    devs = [abs(e - t) for e, t in zip(eff, targets)]
    g0 = max(devs)
    p0 = g0 <= WALL_REG_TOL_M
    ok &= p0
    lines.append(f"  Gate 0 [wall reg <= {WALL_REG_TOL_M:.0e} m]: "
                 f"{'PASS' if p0 else 'FAIL'} (max |eff - target| = {g0:.3e} m; "
                 f"eff = ({eff[0]*1e3:.6f}, {eff[1]*1e3:.6f}, {eff[2]*1e3:.6f}) mm "
                 f"vs ({A*1e3:.1f}, {B*1e3:.1f}, {D*1e3:.1f}) mm)")

    errs = {}
    missing = []
    for name, _idx, fa, _fy, f, *_rest in rows:
        if f is None:
            missing.append(name)
        else:
            errs[name] = abs(f - fa) / fa * 100.0

    # ---- Gate 1: TE101 (fundamental) vs the continuum Pozar oracle -------
    g1 = errs.get("TE101")
    if g1 is None:
        lines.append(f"  Gate 1 [TE101 < {GATE1_TE101_PCT:g}%]: "
                     f"FAIL - TE101 not extracted")
        ok = False
    else:
        p1 = g1 < GATE1_TE101_PCT
        ok &= p1
        lines.append(f"  Gate 1 [TE101 < {GATE1_TE101_PCT:g}%]: "
                     f"{'PASS' if p1 else 'FAIL'} (TE101 err = {g1:.4f}%)")

    # ---- Gate 2: EVERY declared mode, aggregated with max -----------------
    # Pre-#812 this was min() over the six higher modes, at the same 2%. The
    # candidate set contains, for every axis, a mode with a zero index on that
    # axis (n=0 for TE101/TE201/TE102, m=0 for TE011, l=0 for TM110/TM210), and
    # f_mnl depends on an extent only through m/a, so such a mode scores
    # identically 0.000% for ANY single-axis dimensional error -- and min()
    # selected exactly that mode. NOT-FOUND also used to remove a mode from the
    # gate instead of failing it. Both are closed here; the 2% is unchanged.
    if missing:
        lines.append(f"  Gate 2 [all {len(rows)} modes < {GATE2_ALL_MODES_PCT:g}%]: "
                     f"FAIL - not extracted: {', '.join(missing)}")
        ok = False
    else:
        worst = max(errs, key=errs.get)
        p2 = errs[worst] < GATE2_ALL_MODES_PCT
        ok &= p2
        lines.append(f"  Gate 2 [all {len(rows)} modes < {GATE2_ALL_MODES_PCT:g}%]: "
                     f"{'PASS' if p2 else 'FAIL'} (worst: {worst} err = "
                     f"{errs[worst]:.4f}%; {sum(v < GATE2_ALL_MODES_PCT for v in errs.values())}"
                     f"/{len(errs)} modes < {GATE2_ALL_MODES_PCT:g}%)")

    # ---- Gate 3 (NEW): residual against the EXACT discrete-Yee eigenvalue --
    budget = YEE_BUDGET_BINS * freq_resolution_hz
    if not p0:
        lines.append("  Gate 3 [|meas - yee| <= 0.1/T]:  N/A - Gate 0 failed, so the "
                     "discrete-Yee prediction is VOID (k_i != m*pi/a off-grid)")
    elif missing:
        lines.append(f"  Gate 3 [|meas - yee| <= {budget/1e6:.3f} MHz]: "
                     f"FAIL - not extracted: {', '.join(missing)}")
        ok = False
    else:
        resid = {name: abs(f - fy)
                 for name, _idx, _fa, fy, f, *_rest in rows}
        worst3 = max(resid, key=resid.get)
        p3 = resid[worst3] <= budget
        ok &= p3
        lines.append(f"  Gate 3 [|meas - yee| <= {budget/1e6:.3f} MHz = "
                     f"{YEE_BUDGET_BINS:g}/T, T from the gated record]: "
                     f"{'PASS' if p3 else 'FAIL'} (worst: {worst3} residual = "
                     f"{resid[worst3]/1e6:.4f} MHz, "
                     f"{budget/max(resid[worst3], 1e-30):.1f}x inside)")

    # Extra honesty line: how many of ALL identified modes clear 1%
    allpct = list(errs.values())
    if allpct:
        lines.append(f"  (stronger, non-gating: {sum(v < 1.0 for v in allpct)}/"
                     f"{len(allpct)} identified modes < 1% vs Pozar; max err = "
                     f"{max(allpct):.4f}%)")
    return bool(ok), lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--converge", action="store_true",
                    help="also run a 2x-finer (dx=0.5mm) convergence witness")
    ap.add_argument("--num-periods", type=float, default=200.0,
                    help="ring-up+record length in periods at freq_max")
    args = ap.parse_args()

    print("=" * 92)
    print("CROSSVAL 14 — Rectangular PEC cavity eigenfrequencies vs EXACT Pozar oracle")
    print("=" * 92)
    print(f"Cavity (air-filled PEC): a={A*1e3:.1f} x b={B*1e3:.1f} x d={D*1e3:.1f} mm "
          f"(x, y, z)")
    print("Oracle: f_mnl = (c/2)*sqrt((m/a)^2 + (n/b)^2 + (l/d)^2)   "
          "[Pozar, re-derived in-script]")
    print()
    print("Analytic mode frequencies (sorted):")
    for name, (m, n, l), _ in sorted(
            TARGET_MODES, key=lambda t: pozar_cavity_freq(A, B, D, *t[1])):
        print(f"    {name:<8} (m,n,l)={m},{n},{l}   "
              f"{pozar_cavity_freq(A, B, D, m, n, l)/1e9:.5f} GHz")

    # ---- MAIN GATE LEG: dx = 1 mm (exact divisor of a,b,d) ----
    print()
    print("-" * 92)
    print("MAIN LEG — dx = 1.0 mm (exact divisor: effective walls land exactly on a,b,d)")
    print("-" * 92)
    rows, pf_lines, gi, wall = run_leg(1e-3, args.num_periods)
    shape, dx_, dz_, eff, dt_, nsteps, fres = gi
    print(f"grid shape = {shape}, dx = {dx_*1e3:.3f} mm, dz = {dz_*1e3:.3f} mm, "
          f"dt = {dt_:.4e} s")
    print(f"effective cavity (wall separation = (n-1)*dx) = "
          f"({eff[0]*1e3:.3f}, {eff[1]*1e3:.3f}, {eff[2]*1e3:.3f}) mm  "
          f"[target {A*1e3:.1f}, {B*1e3:.1f}, {D*1e3:.1f}]")
    print(f"timesteps = {nsteps}  (num_periods={args.num_periods:g})   "
          f"run wall = {wall:.1f} s")
    _hstart, _hn, _hdt = harminv_record_of_leg(nsteps, dt_)
    print(f"harminv record = {_hn} samples x {_hdt:.4e} s "
          f"(span T = {_hn*_hdt:.4e} s, skip {_hstart}); Rayleigh limit 1/T = "
          f"{fres/1e6:.3f} MHz -> Gate 3 budget {YEE_BUDGET_BINS:g}/T = "
          f"{YEE_BUDGET_BINS*fres/1e6:.3f} MHz")
    print()
    print("PREFLIGHT (verbatim) — lossless closed PEC + no CPML is the CORRECT")
    print("config for a resonant cavity; a clean/advisory-free report is EXPECTED:")
    for ln in pf_lines:
        print(f"    | {ln}")
    print()
    print("Ring-down settling witness: N/A — a lossless closed PEC cavity conserves")
    print("energy (no decay). We report the harminv FIT RESIDUAL (col 'fit resid',")
    print("lower=better) instead. Q is a window-length artefact (see rfx/harminv.py)")
    print("and is NOT claims-bearing — only the FREQUENCY is gated.")
    print()
    errs = print_table(rows)

    # ---- OPTIONAL CONVERGENCE WITNESS: dx = 0.5 mm ----
    if args.converge:
        print()
        print("-" * 92)
        print("CONVERGENCE WITNESS — dx = 0.5 mm (2x finer). Error should drop ~4x")
        print("(2nd-order Yee), proving the residual is real numerical dispersion,")
        print("not extraction luck.")
        print("-" * 92)
        rows2, _pf2, gi2, wall2 = run_leg(0.5e-3, args.num_periods)
        print("(witness only - NOT gated; the claim-bearing leg is dx = 1.0 mm)")
        print(f"grid shape = {gi2[0]}   run wall = {wall2:.1f} s")
        errs2 = print_table(rows2)
        print()
        print(f"{'mode':<8} {'err@1.0mm':>10} {'err@0.5mm':>10} {'ratio':>7}")
        for name in errs:
            if name in errs2 and errs2[name] > 0:
                print(f"{name:<8} {errs[name]:>9.4f}% {errs2[name]:>9.4f}% "
                      f"{errs[name]/errs2[name]:>7.2f}x")

    # ---------------------------- GATE VERDICT ----------------------------
    print()
    print("=" * 92)
    print("GATE VERDICT (comparator-first: EXACT Pozar continuum oracle for Gates 1-2,")
    print("EXACT discrete-Yee oracle for Gate 3, both re-derived in-file)")
    print("=" * 92)

    ok, gate_lines = evaluate_gates(rows, eff, fres)
    for ln in gate_lines:
        print(ln)

    print()
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — rfx harminv "
          f"{'matches' if ok else 'does NOT match'} the exact Pozar cavity "
          f"oracle within the gate.")
    if ok:
        print("Honest note: the sub-0.1% Pozar match reflects EXACT wall registration")
        print("(dx exact-divisor) + fine mesh + a well-resolved harminv fit; the")
        print("residual is 2nd-order numerical dispersion (see --converge). Gate 3")
        print("says the stronger thing: the measured frequencies land on the EXACT")
        print("discrete-Yee eigenvalues of this box, so the Pozar residual is the")
        print("PREDICTED dispersion of the scheme and not an unexplained offset.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
