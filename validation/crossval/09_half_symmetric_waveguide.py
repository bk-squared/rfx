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
    tests/test_boundary_pmc_hi_faces.py — that placement is solver
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

PASS criteria:
    1. f_full within 10 % of analytic f_{101}.
    2. f_half within 10 % of analytic f_{101}.
    3. |f_full - f_half| / f_full < 5 % (PMC mirror matches full cavity).

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
    return ts, dt


def _fft_peak_near(
    ts: np.ndarray, dt: float, f_target: float, rel_window: float = 0.5,
) -> tuple[float, float]:
    """Windowed-FFT fallback: return (f_peak, amp_peak) inside the window."""
    ringdown = ts[len(ts) // 4:]
    n = len(ringdown)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(ringdown * window))
    freqs = np.fft.rfftfreq(n, dt)
    mask = (freqs >= f_target * (1.0 - rel_window)) & \
           (freqs <= f_target * (1.0 + rel_window))
    if not np.any(mask):
        return float("nan"), 0.0
    band_freqs = freqs[mask]
    band_spec = spec[mask]
    idx = int(np.argmax(band_spec))
    return float(band_freqs[idx]), float(band_spec[idx])


def _extract_mode_near(
    ts: np.ndarray, dt: float, f_target: float,
    rel_window: float = 0.5, min_q: float = 1.0,
):
    """Harminv on the ringdown tail; return the mode nearest f_target.
    Falls back to an FFT peak-pick if Harminv returns no candidates."""
    ringdown = ts[len(ts) // 4:]
    f_min = f_target * (1.0 - rel_window)
    f_max = f_target * (1.0 + rel_window)
    modes = harminv(ringdown, dt, f_min=f_min, f_max=f_max, min_Q=min_q)
    if modes:
        freqs = np.asarray([m.freq for m in modes])
        return modes[int(np.argmin(np.abs(freqs - f_target)))], "harminv"
    f_fft, _ = _fft_peak_near(ts, dt, f_target, rel_window=rel_window)
    if np.isfinite(f_fft):
        class _FFTMode:
            freq = f_fft
            Q = float("nan")
        return _FFTMode(), "fft"
    return None, "none"


def main() -> int:
    print("=" * 64)
    print("Cross-Validation 09: Half-Symmetric Waveguide Cavity (PEC + PMC)")
    print("=" * 64)
    print(f"a = {a*1e3:.2f} mm, b = {b*1e3:.2f} mm, d = {d*1e3:.2f} mm")
    print(f"dx = {DX*1e3:.3f} mm, n_steps = {N_STEPS}")
    print(f"Analytic TE_101 f = {F_101_ANALYTIC/1e9:.4f} GHz")
    print()

    # ----- Full cavity: PEC on all 6 faces -----
    spec_full = BoundarySpec.uniform("pec")
    src_full = (0.25 * a, 0.5 * b, 0.5 * d)
    probe_full = (0.40 * a, 0.5 * b, 0.33 * d)

    print("Run 1: Full cavity (all-PEC)...", flush=True)
    t0 = time.time()
    ts_full, dt_full = _run_cavity(
        domain=(a, b, d), spec=spec_full,
        source_pos=src_full, probe_pos=probe_full,
    )
    print(f"  elapsed {time.time() - t0:.1f} s  dt={dt_full*1e12:.3f} ps")

    mode_full, src_full_tag = _extract_mode_near(ts_full, dt_full, F_101_ANALYTIC)
    if mode_full is None:
        print("FAIL: no mode found in window around f_101 (full cavity)")
        return 1
    f_full = float(mode_full.freq)
    q_full = float(mode_full.Q)
    print(f"  full: f = {f_full/1e9:.4f} GHz, Q = {q_full:.2e}  "
          f"(via {src_full_tag})")

    # ----- Half cavity: PEC on x_lo, PMC on x_hi, PEC on remaining faces.
    #       PMC-plane convention (issue #722 ninth surface, see docstring):
    #       declared a/2 + dx/2, not a/2, so apply_pmc_faces' H_tan zero
    #       (a half-cell INSIDE the declared x_hi wall, pinned by
    #       tests/test_boundary_pmc_hi_faces.py) lands ON the a/2 mirror
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
    ts_half, dt_half = _run_cavity(
        domain=(HALF_X, b, d), spec=spec_half,
        source_pos=src_half, probe_pos=probe_half,
    )
    print(f"  elapsed {time.time() - t0:.1f} s  dt={dt_half*1e12:.3f} ps")

    mode_half, src_half_tag = _extract_mode_near(ts_half, dt_half, F_101_ANALYTIC)
    if mode_half is None:
        print("FAIL: no mode found in window around f_101 (half cavity)")
        print("  FFT spectrum diagnostic:")
        ringdown = ts_half[len(ts_half) // 4:]
        spec = np.abs(np.fft.rfft(ringdown * np.hanning(len(ringdown))))
        freqs_fft = np.fft.rfftfreq(len(ringdown), dt_half)
        top = np.argsort(-spec)[:5]
        for i in top:
            print(f"    peak: f = {freqs_fft[i]/1e9:7.3f} GHz, |A| = {spec[i]:.3e}")
        return 1
    f_half = float(mode_half.freq)
    q_half = float(mode_half.Q)
    print(f"  half: f = {f_half/1e9:.4f} GHz, Q = {q_half:.2e}  "
          f"(via {src_half_tag})")

    # ----- Checks -----
    PASS = True
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
    if rel_gap < 0.05:
        print(f"PASS: |f_full - f_half| / f_full = {rel_gap:.3%} < 5% "
              f"(PMC mirror reproduces full cavity)")
    else:
        print(f"FAIL: |f_full - f_half| / f_full = {rel_gap:.3%} >= 5% "
              f"(PMC mirror does NOT match full cavity)")
        PASS = False

    print()
    if PASS:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
