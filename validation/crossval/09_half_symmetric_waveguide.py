"""Cross-validation 09: Half-symmetric rectangular-waveguide cavity (PEC + PMC).

Validates rfx's PMC boundary as a symmetry-plane image source by comparing
the TE_{101} resonance of a fully-closed PEC rectangular cavity against
a half-domain PEC+PMC cavity declared at x=a/2+dx/2 (PMC-plane convention
below), whose H_tan wall lands on the x=a/2 mirror plane.

Physics:
    Closed rectangular cavity of dimensions (a, b, d) with all-PEC walls
    supports TE_{mnp} modes (TE to z) with resonance frequencies

        f_{mnp} = (c/2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2).

    The TE_{101} mode has
        H_z = H_0 cos(pi x/a) sin(pi z/d)
        E_y ~ sin(pi x/a) cos(pi z/d)      (dominant E component)
        E_x = E_z = 0                       (n=0)

    At the x-mirror plane x = a/2:
        H_tan (= H_y, H_z) vanishes  -> PMC is the correct BC.
        E_tan (= E_y, E_z) is non-zero (E_y peaks) -> PEC would be wrong.

    Therefore the half-domain cavity (0 <= x <= a/2, PEC on x_lo, PMC on
    x_hi, PEC on the other four faces) must support the same TE_{101}
    resonance as the full-domain all-PEC cavity.

Setup:
    - a = 22.86 mm (WR-90 broad wall), b = 10.16 mm, d = 30.48 mm (1.2 in).
    - f_{101}(analytic) = 8.1964 GHz.
    - dx = 0.508 mm = 0.020 in (uniform), cpml_layers = 0 on both runs.
      Closed cavities with cpml_layers=0 sidestep the PMC+CPML composition
      architectural gap.
    - Gaussian-pulse E_y source offset from center; E_y probe off-node.
    - Harminv on ringdown (skip first 25 %) to extract the dominant mode.

Mesh / reference convention (issue #722, #724):
    REALIZE-DECLARED-BY-MESH, for the FULL cavity. dx = 0.020 in divides
    the WR-90 broad and narrow walls (0.9 in, 0.4 in) and the 1.2 in
    closure exactly (45 / 20 / 60 cells, read off the built grid), so on
    the full all-PEC run the Pozar closed form above is evaluated on the
    dimensions the solve actually has. The closure length d is
    arbitrary — no external reference pins it, unlike the WR-90 standard
    a and b — and was re-declared 30.00 -> 30.48 mm to obtain that
    commensurability. On origin/main (dx = 0.5 mm) the full cavity
    rasterized to 46 x 21 x 60 cells = 23.000 x 10.500 x 30.000 mm while
    the closed form was evaluated at 22.86 x 10.16 x 30.00 mm; that
    mismatch, not discretization, is most of what gate 1 was reporting.

PMC-plane convention: REALIZE-DECLARED, decided on issue #722's ninth
surface (2026-08-28):
    rfx enforces PMC on a `_hi` face by zeroing H_tan at array index -2,
    i.e. at the half cell 0.5*dx INSIDE the declared wall (pinned by
    tests/unit/boundaries/test_boundary_pmc_hi_faces.py — that placement is solver
    physics, measured, and is NOT changed here; see
    rfx/boundaries/pmc.py). A half-domain declared at exactly a/2
    therefore realizes its H_tan wall at a/2 - dx/2, mirroring into a
    full guide whose effective broad wall is a - dx, not a — this is
    what an EARLIER revision of this docstring measured as a fixed
    half-cell bias on gate 3 at both of the meshes below (a/2-declared,
    the convention this change REPLACES):

      mesh                 a_eff       f_half pred   f_half measured
      dx=0.635, d=30.48    22.225 mm   8.3471 GHz    8.3460 GHz (-0.013%)
      dx=0.500, d=30.00    22.500 mm   8.3276 GHz    8.3268 GHz (-0.009%)

    (a_eff = a - dx is the denominator ONLY for that a/2-declared
    convention. The predicted gate 3 against it was 1.838% / 1.405% —
    that is Pozar(a_eff)/Pozar(full realized) - 1 — against the printed
    1.835% / 1.408%; read against the printed MEASURED f_full instead of
    the closed form it is 1.847% / 1.417%. Both readings confirm the
    mechanism; neither is what this file solves any more.)

    DECISION: declare the half domain at a/2 + dx/2 (HALF_X below), not
    a/2. With a = 45 cells (ODD), a/2 = 22.5*dx is exactly an H-NODE
    plane — TE_101's H_tan (Hy, Hz) is analytically zero there. The half
    domain's declared hi face at 22.5*dx + 0.5*dx = 23*dx then has its
    H_tan zeroed (index -2, the placement above) exactly ON that H-node
    plane at 22.5*dx = a/2 — the declared mirror plane — so the half
    solve IS the image of the full cavity, not an a-dx approximation of
    it. (The alternative, quote-realized — keep a/2 and always compare
    against a_eff = a - dx — was rejected: it leaves gate 3 carrying that
    fixed half-cell bias forever, at every mesh, instead of removing it
    once.)

    The ODD-cell condition is load-bearing, not incidental: on an EVEN
    cell count a/2 sits on an E-node, and 22.5*dx does not exist on that
    lattice, so a/2 + dx/2 would NOT land the zeroed H_tan on the mirror
    plane. gcd(2286, 1016, 3048) um = 254 um, so the candidate DX values
    keeping a, b, d AND a/2+dx/2 all commensurate are 254/{1,3,5,7} um =
    2.54 / 0.8467 / 0.508 / 0.3629 mm; 0.508 mm (a = 45 cells, odd) is
    the finest of these before 0.3629 mm, whose cost is
    (0.635/0.3629)^4 = 9.4x the OLD 0.635 mm mesh (3.84x this file's
    0.508 mm) — out of scope here. A naive control confirms the failure
    mode the odd-cell condition rules out: at DX=0.635 mm, a/2+dx/2 is
    18.5 cells (not an integer — 22.86/2/0.635 = 18.0 exactly), so a
    domain declared there silently ceils to 19 cells = a + dx, a HALF
    cavity whose H-node mirror is a full cell short (measured f_half =
    8.0545 GHz, gate 3 = 1.722% — worse than either convention above).

    rfx.fidelity.fidelity_report's domain row makes this self-diagnosing
    (#729 spirit): on this file's half sim it prints `x: declared
    [0.0, 11684.0] um -> realized [0.0, 11430.0] um | face residuals
    (0.0, 254.0) um`, plus a `pmc-wall-half-cell-inside` finding naming
    this convention by name — so a future PMC-mirror script cannot ship
    the #722 ninth-surface offset silently.

    MEASURED (this pod, 2026-08-28, JAX_PLATFORMS=cpu,
    JAX_ENABLE_X64=1, each config run alone and timed):
      DX=0.508mm, HALF_X=a/2+DX/2 (THIS FILE): dx printed 0.508 mm (the
        earlier 0.635 mm mesh's header rounded 0.508 -> "0.51 mm" under
        `.2f`; fixed to `.3f` here). 'full: f = 8.1958 GHz, Q =
        5.11e+04', 'half: f = 8.1959 GHz, Q = 6.47e+04', gates
        0.007% / 0.007% / 0.001%, ALL CHECKS PASSED. Solve time
        8.9 s + 3.1 s = 12.0 s; whole script 13.2 s.
      DX=0.635mm, HALF_X=a/2 (OLD, this file's PRE-this-change history):
        'full: f = 8.1957 GHz, Q = 5.04e+04', 'half: f = 8.3460 GHz,
        Q = 8.70e+04', gates 0.009% / 1.825% / 1.835%, ALL CHECKS
        PASSED. Solve time 6.3 s + 1.6 s = 7.9 s; whole script 9.3 s.
      The new (finer) mesh costs 1.42x the OLD mesh's whole-script wall
      time (13.2 s vs 9.3 s) — the naive (0.635/0.508)^4 = 2.44x
      4th-power FDTD-cost estimate over-predicts because JIT/import
      overhead is fixed, not per-cell.
    Thresholds (10% / 10% / 5%) are UNCHANGED.

CORRECTION (2026-08-31, issue #812 Phase 1 — append-only, the block above
is left standing as the record of what was believed then):
    The MEASURED table above compares "DX=0.508mm, HALF_X=a/2+DX/2 (THIS
    FILE)" against "DX=0.635mm, HALF_X=a/2 (OLD)" and reads the resulting
    gate-3 improvement 1.835% -> 0.001% as evidence for the DECLARATION
    change. That is a confounded A/B: it moves the mesh and the declaration
    together. The declaration change contributes NOTHING to it.

    OLD VALUE / OLD CLAIM: "declare the half domain at a/2 + dx/2 (HALF_X
    below), not a/2 ... so the half solve IS the image of the full cavity",
    with gates 10% / 10% / 5% "UNCHANGED".
    WHY IT WAS WRONG: `Grid` takes n = ceil(HALF_X/dx) cells, and the
    realized H_tan mirror plane is x_m = (n - 0.5)*dx, so the half domain is
    the image half of a guide of broad wall

        a_eff = 2*x_m = (2n - 1)*dx     [an ODD multiple of dx]

    which equals a only when a/dx is ODD, and then n = (a/dx + 1)/2 is
    produced by ANY declaration in ((a/dx - 1)/2, (a/dx + 1)/2] * dx -- an
    interval containing both a/2 and a/2 + dx/2. At DX = 0.508 mm
    (a = 45 dx, odd) ceil(22.5) = ceil(23.0) = 23: both declarations build
    grid (24, 21, 61) and realize a_eff = 22.8600 mm identically (measured
    2026-08-31 at grid-build level). Where a/dx is EVEN no declaration
    works: at DX = 0.635 mm (a = 36 dx) a/2 realizes a_eff = a - dx (gate 3
    1.825%) and a/2 + dx/2 realizes a_eff = a + dx (gate 3 1.722%). So on a
    ceil-based grid the `+ dx/2` term never converts a wrong mirror plane
    into a right one; the ODD-cell mesh (0.635 -> 0.508 mm) does all of the
    work, and the odd-cell condition the block above already calls
    "load-bearing, not incidental" is in fact the WHOLE mechanism.
    HALF_X is kept at a/2 + dx/2 -- it is harmless, it puts the declared
    mesh line on a lattice node, and it states the #722 convention -- but it
    is no longer what the case relies on.

    Issue #812 also measured that gate 3's 5% window is wider than the
    signature of the error it exists to catch: a one-cell mirror-plane error
    is 2.702% (hi) / 3.001% (lo) and a half-cell error is 1.72-1.84%. The
    gate could not fail for its own stated reason. Both facts are closed by
    gating the REALIZED quantity -- see gate 0 below and
    docs/design_notes/issue812_cv09_mirror_plane_regate.md for the
    pre-declared derivations.

PASS criteria:
    0. REALIZE-DECLARED geometry, both runs (NEW, #812): every solved extent
       equals its declared length within DX/4, and the half cavity's
       realized mirror plane satisfies |a_eff - a| < DX/4 with
       a_eff = 2 * (realized H_tan wall), read off the production reporter
       rfx.fidelity.fidelity_report. DX/4 is a quarter of the smallest
       NONZERO a_eff misregistration this lattice can express (dx, reached
       when a/dx is even; 2*dx when a/dx is odd) -- derived from the lattice,
       not from any measured frequency.
    1. f_full within 10 % of analytic f_{101}.  (unchanged)
    2. f_half within 10 % of analytic f_{101}.  (unchanged)
    3. |f_full - f_half| / f_full < G3_TOL, TIGHTENED from 5% (#812) to the
       frequency image of gate 0:
           G3_TOL = |d ln f/d ln a| * (DX/4)/a = (d^2/(a^2+d^2)) * (DX/4)/a
       = 0.6400 * 127.0 um / 22.86 mm = 0.3556 % at this mesh. Gate 3 now
       refuses any full-vs-half discrepancy larger than what a QUARTER-cell
       mirror-plane misregistration would produce; a one-cell error (2.70% /
       3.00%) misses it by 7.6-8.4x and the pre-#762 half-cell error (1.838%)
       by 5.2x.
    Harminv returning no candidate is a HARD FAIL (#812): the removed
    windowed-FFT fallback quantises frequency at 1/(3072 dt) = 335.9 MHz =
    4.099 % of f_101, 11.5x gate 3's new tolerance, so it cannot honestly
    judge gate 3. The FFT spectrum survives as a printed diagnostic only.

Reference: Pozar, "Microwave Engineering", Ch. 6 (rectangular resonators).
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.harminv import harminv


C0 = 299_792_458.0

# ----------------------------------------------------------------------
# Cavity dimensions (WR-90 section with L=30.48mm (1.2 in) closure)
# ----------------------------------------------------------------------
a = 22.86e-3     # broad wall (x-axis), metres
b = 10.16e-3     # narrow wall (y-axis)
d = 30.48e-3     # cavity length (z-axis), 1.2 in — chosen so DX divides it exactly

DX = 0.508e-3    # uniform cell size, 0.020 in; a=45 b=20 d=60 cells, a odd so
                 # a/2 is an H-node plane (issue #722/#724:
                 # REALIZE-DECLARED-BY-MESH; issue #722 ninth surface:
                 # PMC-plane convention — both in the docstring)
N_STEPS = 4096
FREQ_MAX = 20e9  # covers well above f_{101}

F_101_ANALYTIC = 0.5 * C0 * np.sqrt((1.0 / a) ** 2 + (1.0 / d) ** 2)

# ----------------------------------------------------------------------
# Gate thresholds (issue #812; derivations pre-declared in
# docs/design_notes/issue812_cv09_mirror_plane_regate.md, committed before
# the measurement that judges them).
# ----------------------------------------------------------------------
#: Gate 0 — realize-declared geometry budget. A quarter of the smallest
#: NONZERO mirror-plane misregistration the Yee lattice can express
#: (a_eff = (2n-1)*dx moves in steps of 2*dx and lands off `a` by at least
#: dx whenever a/dx is even). Lattice-derived; no measured frequency in it.
GEOM_TOL = DX / 4.0
#: Gate 3 — the same budget in frequency. Pozar's
#: f_101 = (c/2) sqrt(1/a^2 + 1/d^2) has
#: d ln f / d ln a = -(1/a^2)/(1/a^2 + 1/d^2) = -d^2/(a^2 + d^2), which is
#: exactly 16/25 = 0.6400 here because a/d = 22.86/30.48 = 3/4. Kept as the
#: closed expression, not a frozen literal, so a mesh change moves the gate
#: with the physics (SPEC-00 0.2.4). Evaluates to 0.3556 % at DX = 0.508 mm.
G3_TOL = (d ** 2 / (a ** 2 + d ** 2)) * (DX / 4.0) / a


def _run_cavity(
    domain: tuple[float, float, float],
    spec: BoundarySpec,
    source_pos: tuple[float, float, float],
    probe_pos: tuple[float, float, float],
    n_steps: int = N_STEPS,
):
    """Run a single cavity sim, return (time_series, dt)."""
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=domain,
        dx=DX,
        boundary=spec,
        cpml_layers=0,
    )
    sim.add_source(source_pos, "ey")
    sim.add_probe(probe_pos, "ey")
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        sim.preflight(strict=False)
        res = sim.run(n_steps=n_steps)
    ts = np.asarray(res.time_series)[:, 0]
    dt = float(res.dt)
    return ts, dt, realized_axes(sim)


def _print_fft_diagnostic(ts: np.ndarray, dt: float) -> None:
    """Print the ringdown's five strongest FFT peaks — DIAGNOSTIC ONLY.

    Issue #812: this used to be a silent FALLBACK feeding the gates. Its
    frequency quantum is 1/(N_ringdown * dt) = 335.9 MHz = 4.099 % of
    f_101 at this mesh, 11.5x gate 3's tolerance, so a gate at G3_TOL
    cannot be honestly judged by it. Harminv returning nothing is now a
    hard failure and this only says WHY.
    """
    ringdown = ts[len(ts) // 4:]
    spec = np.abs(np.fft.rfft(ringdown * np.hanning(len(ringdown))))
    freqs = np.fft.rfftfreq(len(ringdown), dt)
    bin_hz = 1.0 / (len(ringdown) * dt)
    print(f"  FFT spectrum diagnostic (bin = {bin_hz/1e6:.1f} MHz = "
          f"{bin_hz/F_101_ANALYTIC:.3%} of f_101 — NOT gate-grade):")
    for i in np.argsort(-spec)[:5]:
        print(f"    peak: f = {freqs[i]/1e9:7.3f} GHz, |A| = {spec[i]:.3e}")


def _extract_mode_near(
    ts: np.ndarray, dt: float, f_target: float,
    rel_window: float = 0.5, min_q: float = 1.0,
):
    """Harminv on the ringdown tail; return the mode nearest f_target, or
    None. No fallback estimator (issue #812) — see _print_fft_diagnostic."""
    ringdown = ts[len(ts) // 4:]
    f_min = f_target * (1.0 - rel_window)
    f_max = f_target * (1.0 + rel_window)
    modes = harminv(ringdown, dt, f_min=f_min, f_max=f_max, min_Q=min_q)
    if not modes:
        return None
    freqs = np.asarray([m.freq for m in modes])
    return modes[int(np.argmin(np.abs(freqs - f_target)))]


def realized_axes(sim) -> dict:
    """Per-axis REALIZED geometry of ``sim``, read off the production
    reporter ``rfx.fidelity.fidelity_report`` (domain row).

    Reading the reporter rather than re-deriving the mesh here is
    deliberate: the reporter already applies the PMC half-cell rule
    (rfx/boundaries/pmc.py zeros H_tan at index -2, 0.5*dx INSIDE the
    declared mesh line), so the gate and the solve cannot disagree about
    the convention by construction. Each value is in METRES.
    """
    from rfx.fidelity import fidelity_report
    rows = fidelity_report(sim, print_report=False)
    dom = rows[0]
    assert dom["entity"].startswith("domain"), dom["entity"]
    return {ax["axis"]: dict(
        n_cells=int(ax["n_cells"]),
        mesh_extent=float(ax["mesh_extent_um"]) * 1e-6,
        realized_extent=float(ax["realized_extent_um"]) * 1e-6,
        realized_hi=float(ax["realized_um"][1]) * 1e-6,
    ) for ax in dom["axes"]}


def mirror_a_eff(axes: dict) -> float:
    """Broad wall of the FULL guide the half domain is the image half of.

    ``axes`` is a :func:`realized_axes` mapping for the half (PEC + PMC)
    run. Its x row's ``realized_hi`` IS the realized H_tan mirror plane
    x_m = (n - 0.5)*dx, so the mirrored guide is 2*x_m wide. This is the
    quantity PR #762 was about and the quantity no gate read before #812.
    """
    return 2.0 * float(axes["x"]["realized_hi"])


def geometry_rows(axes_full: dict, axes_half: dict) -> list:
    """Gate-0 rows: (label, realized_metres, declared_metres)."""
    return [
        ("full cavity a", axes_full["x"]["realized_extent"], a),
        ("full cavity b", axes_full["y"]["realized_extent"], b),
        ("full cavity d", axes_full["z"]["realized_extent"], d),
        ("half cavity a_eff (2 x realized H_tan mirror plane)",
         mirror_a_eff(axes_half), a),
        ("half cavity b", axes_half["y"]["realized_extent"], b),
        ("half cavity d", axes_half["z"]["realized_extent"], d),
    ]


def geometry_gate(axes_full: dict, axes_half: dict, tol: float = GEOM_TOL):
    """Gate 0. Returns (passed, lines) — pure, so a falsifier can drive it
    with an injected mirror-plane error without running FDTD."""
    ok = True
    lines = []
    for label, realized, declared in geometry_rows(axes_full, axes_half):
        resid = abs(realized - declared)
        verdict = "PASS" if resid < tol else "FAIL"
        if verdict == "FAIL":
            ok = False
        lines.append(
            f"{verdict}: {label} = {realized*1e3:.4f} mm vs declared "
            f"{declared*1e3:.4f} mm, |residual| = {resid*1e6:.1f} um "
            f"{'<' if verdict == 'PASS' else '>='} DX/4 = {tol*1e6:.1f} um")
    return ok, lines


def main() -> int:
    print("=" * 64)
    print("Cross-Validation 09: Half-Symmetric Waveguide Cavity (PEC + PMC)")
    print("=" * 64)
    print(f"a = {a*1e3:.2f} mm, b = {b*1e3:.2f} mm, d = {d*1e3:.2f} mm")
    print(f"dx = {DX*1e3:.3f} mm, n_steps = {N_STEPS}")
    print(f"Analytic TE_101 f = {F_101_ANALYTIC/1e9:.4f} GHz")
    print(f"Gate 0 tol = DX/4 = {GEOM_TOL*1e6:.1f} um; "
          f"gate 3 tol = |dlnf/dlna|*(DX/4)/a = {G3_TOL:.4%} "
          f"(|dlnf/dlna| = {d**2/(a**2 + d**2):.4f})")
    print()

    # ----- Full cavity: PEC on all 6 faces -----
    spec_full = BoundarySpec.uniform("pec")
    src_full = (0.25 * a, 0.5 * b, 0.5 * d)
    probe_full = (0.40 * a, 0.5 * b, 0.33 * d)

    print("Run 1: Full cavity (all-PEC)...", flush=True)
    t0 = time.time()
    ts_full, dt_full, axes_full = _run_cavity(
        domain=(a, b, d), spec=spec_full,
        source_pos=src_full, probe_pos=probe_full,
    )
    print(f"  elapsed {time.time() - t0:.1f} s  dt={dt_full*1e12:.3f} ps")

    mode_full = _extract_mode_near(ts_full, dt_full, F_101_ANALYTIC)
    if mode_full is None:
        print("FAIL: Harminv found no mode in the window around f_101 "
              "(full cavity) — hard failure, no fallback estimator (#812)")
        _print_fft_diagnostic(ts_full, dt_full)
        return 1
    f_full = float(mode_full.freq)
    q_full = float(mode_full.Q)
    print(f"  full: f = {f_full/1e9:.4f} GHz, Q = {q_full:.2e}  (via harminv)")

    # ----- Half cavity: PEC on x_lo, PMC on x_hi, PEC on remaining faces.
    #       PMC-plane convention (issue #722 ninth surface, see docstring):
    #       declared a/2 + dx/2, not a/2, so apply_pmc_faces' H_tan zero
    #       (a half-cell INSIDE the declared x_hi wall, pinned by
    #       tests/unit/boundaries/test_boundary_pmc_hi_faces.py) lands ON the a/2 mirror
    #       plane instead of a half-cell short of it. y, z unchanged. The
    #       source/probe x coords must stay inside [0, a/2 + dx/2]; since
    #       src x = 0.25 a and probe x = 0.40 a are both < 0.5 a, they
    #       carry over unchanged.
    HALF_X = 0.5 * a + 0.5 * DX
    spec_half = BoundarySpec(
        x=Boundary(lo="pec", hi="pmc"),
        y=Boundary(lo="pec", hi="pec"),
        z=Boundary(lo="pec", hi="pec"),
    )
    src_half = (0.25 * a, 0.5 * b, 0.5 * d)
    probe_half = (0.40 * a, 0.5 * b, 0.33 * d)

    print("Run 2: Half cavity (PEC + PMC, declared a/2 + dx/2)...", flush=True)
    t0 = time.time()
    ts_half, dt_half, axes_half = _run_cavity(
        domain=(HALF_X, b, d), spec=spec_half,
        source_pos=src_half, probe_pos=probe_half,
    )
    print(f"  elapsed {time.time() - t0:.1f} s  dt={dt_half*1e12:.3f} ps")

    mode_half = _extract_mode_near(ts_half, dt_half, F_101_ANALYTIC)
    if mode_half is None:
        print("FAIL: Harminv found no mode in the window around f_101 "
              "(half cavity) — hard failure, no fallback estimator (#812)")
        _print_fft_diagnostic(ts_half, dt_half)
        return 1
    f_half = float(mode_half.freq)
    q_half = float(mode_half.Q)
    print(f"  half: f = {f_half/1e9:.4f} GHz, Q = {q_half:.2e}  (via harminv)")

    # ----- Checks -----
    PASS = True
    print()

    # Gate 0 (#812): the REALIZED geometry, including the mirror plane the
    # PMC face actually lands on. Nothing here reads a declared length.
    geom_ok, geom_lines = geometry_gate(axes_full, axes_half)
    for line in geom_lines:
        print(line)
    if not geom_ok:
        PASS = False
    print()

    err_full = abs(f_full - F_101_ANALYTIC) / F_101_ANALYTIC
    if err_full < 0.10:
        print(f"PASS: full-cavity f = {f_full/1e9:.4f} GHz, "
              f"|err| = {err_full:.3%} < 10%")
    else:
        print(f"FAIL: full-cavity f = {f_full/1e9:.4f} GHz, "
              f"|err| = {err_full:.3%} >= 10%")
        PASS = False

    err_half = abs(f_half - F_101_ANALYTIC) / F_101_ANALYTIC
    if err_half < 0.10:
        print(f"PASS: half-cavity f = {f_half/1e9:.4f} GHz, "
              f"|err| = {err_half:.3%} < 10%")
    else:
        print(f"FAIL: half-cavity f = {f_half/1e9:.4f} GHz, "
              f"|err| = {err_half:.3%} >= 10%")
        PASS = False

    rel_gap = abs(f_full - f_half) / f_full
    if rel_gap < G3_TOL:
        print(f"PASS: |f_full - f_half| / f_full = {rel_gap:.4%} < "
              f"{G3_TOL:.4%} (PMC mirror reproduces full cavity)")
    else:
        print(f"FAIL: |f_full - f_half| / f_full = {rel_gap:.4%} >= "
              f"{G3_TOL:.4%} (PMC mirror does NOT match full cavity)")
        PASS = False

    print()
    if PASS:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
