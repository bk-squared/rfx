"""Cross-validation 09: Half-symmetric rectangular-waveguide cavity (PEC + PMC).

Validates rfx's PMC boundary as a symmetry-plane image source by comparing
the TE_{101} resonance of a fully-closed PEC rectangular cavity against
a half-domain PEC+PMC cavity clipped along the x-mirror plane at x=a/2.

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
    - dx = 0.635 mm = 0.025 in (uniform), cpml_layers = 0 on both runs.
      Closed cavities with cpml_layers=0 sidestep the PMC+CPML composition
      architectural gap.
    - Gaussian-pulse E_y source offset from center; E_y probe off-node.
    - Harminv on ringdown (skip first 25 %) to extract the dominant mode.

Mesh / reference convention (issue #722, #724):
    REALIZE-DECLARED-BY-MESH, for the FULL cavity. dx = 0.025 in divides
    the WR-90 broad and narrow walls (0.9 in, 0.4 in), the 1.2 in closure
    and the mirror plane a/2 exactly (36 / 16 / 48 / 18 cells, read off
    the built grid), so on the full all-PEC run the Pozar closed form
    above is evaluated on the dimensions the solve actually has. The
    closure length d is arbitrary — no external reference pins it, unlike
    the WR-90 standard a and b — and was re-declared 30.00 -> 30.48 mm to
    obtain that commensurability. On origin/main (dx = 0.5 mm) the full
    cavity rasterized to 46 x 21 x 60 cells = 23.000 x 10.500 x 30.000 mm
    while the closed form was evaluated at 22.86 x 10.16 x 30.00 mm; that
    mismatch, not discretization, is most of what gate 1 was reporting.

    THE HALF RUN IS NOT ON THAT CONVENTION, and that is a #722-class
    geometry error this change does NOT fix. rfx enforces PMC on a `_hi`
    face by zeroing H_tan at array index -2, i.e. at the half cell
    0.5*dx INSIDE the declared wall (pinned by
    tests/test_boundary_pmc_hi_faces.py). The half domain's effective
    broad half-wall is therefore a/2 - dx/2, so the guide it mirrors into
    has an effective broad wall of a - dx, not a. Predicting f_half from
    a_eff = a - dx, measured at both meshes:

      mesh                 a_eff       f_half pred   f_half measured
      dx=0.635, d=30.48    22.225 mm   8.3471 GHz    8.3460 GHz (-0.013%)
      dx=0.500, d=30.00    22.500 mm   8.3276 GHz    8.3268 GHz (-0.009%)

    The same a_eff predicts gate 3 at 1.838% against the 1.835% the
    script printed on this mesh, and 1.405% against the 1.408% it
    printed on the prior one.
    So gates 2 and 3 are dominated by that fixed half-cell offset, not
    by mesh resolution. Because the offset is a fixed HALF CELL, a
    coarser dx makes it a larger fraction of a — which is exactly why
    going 0.5 -> 0.635 mm makes gate 1 nearly exact and gates 2/3 worse.
    That is a trade, not a free win: gate 1 is the closed-form
    comparison this change targets, while gate 3 is the self-invariant
    reference this script is registered on (role=claims-bearing in
    validation/crossval/manifest.json, references entry "full PEC cavity
    versus half PEC+PMC cavity"). Gate 3 carries a geometry bias of the
    same class #722 is about, at BOTH meshes, and this change makes it
    30% larger (1.408% -> 1.835%). Removing it
    needs a PMC-plane convention (declare the half domain at a/2 + dx/2,
    or de-embed the half cell), which is out of scope here and belongs
    with the #722 follow-up.

    An earlier revision of this docstring attributed the gate 2/3
    degradation to "the half cavity's transverse resolution drops from
    23 to 18 cells". That is not the mechanism; the table above is.

    MEASURED (this pod, 2026-08-28, JAX_PLATFORMS=cpu, each script run
    alone and timed):
      dx = 0.635 mm (this file): grid nodes (37,17,49) -> cells
        (36,16,48) -> walls 22.8600 / 10.1600 / 30.4800 mm exactly;
        half-domain grid (19,17,49) -> half wall_x = 11.4300 mm, exactly
        half of 22.8600 mm. 'full: f = 8.1957 GHz, Q = 5.04e+04',
        'half: f = 8.3460 GHz, Q = 8.70e+04', gates
        0.009% / 1.825% / 1.835%, ALL CHECKS PASSED. Solve time
        6.2 s + 1.0 s = 7.2 s; whole script 10.3 s.
      dx = 0.5 mm (origin/main, d = 30.00 mm): 'full: f = 8.2112 GHz,
        Q = 3.61e+04', 'half: f = 8.3268 GHz, Q = 5.14e+04', gates
        0.397% / 1.006% / 1.408%, ALL CHECKS PASSED. Solve time
        11.0 s + 3.1 s = 14.1 s; whole script 16.3 s.
      The new mesh costs 0.51x the solve time and 0.63x the whole script
      (the gap between the two ratios is fixed import/JIT overhead). An
      earlier revision of this docstring quoted "0.508x (20.1 s vs
      39.6 s)"; re-timed alone on this pod the whole-script ratio is
      0.63x.
    Thresholds (10% / 10% / 5%) are untouched.

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

DX = 0.635e-3    # uniform cell size, 0.025 in — divides a, b, d and a/2 exactly
                 # (issue #722/#724: REALIZE-DECLARED-BY-MESH, see docstring)
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
    print(f"dx = {DX*1e3:.2f} mm, n_steps = {N_STEPS}")
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
    #       Domain x is clipped to a/2; y, z unchanged. The source/probe x
    #       coords must stay inside [0, a/2]; since src x = 0.25 a and
    #       probe x = 0.40 a are both < 0.5 a, they carry over unchanged.
    spec_half = BoundarySpec(
        x=Boundary(lo="pec", hi="pmc"),
        y=Boundary(lo="pec", hi="pec"),
        z=Boundary(lo="pec", hi="pec"),
    )
    src_half = (0.25 * a, 0.5 * b, 0.5 * d)
    probe_half = (0.40 * a, 0.5 * b, 0.33 * d)

    print("Run 2: Half cavity (PEC + PMC at x=a/2)...", flush=True)
    t0 = time.time()
    ts_half, dt_half = _run_cavity(
        domain=(0.5 * a, b, d), spec=spec_half,
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
