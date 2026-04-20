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
    - a = 22.86 mm (WR-90 broad wall), b = 10.16 mm, d = 30.0 mm.
    - f_{101}(analytic) = 8.246 GHz.
    - dx = 0.5 mm (uniform), cpml_layers = 0 on both runs.
      Closed cavities with cpml_layers=0 sidestep the PMC+CPML composition
      architectural gap documented in
      docs/research_notes/2026-04-19_v175_t10_half_symmetric_pmc.md.
    - Gaussian-pulse E_y source offset from center; E_y probe off-node.
    - Harminv on ringdown (skip first 25 %) to extract the dominant mode.

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
# Cavity dimensions (WR-90 section with L=30mm closure)
# ----------------------------------------------------------------------
a = 22.86e-3     # broad wall (x-axis), metres
b = 10.16e-3     # narrow wall (y-axis)
d = 30.0e-3      # cavity length (z-axis)

DX = 0.5e-3      # uniform cell size
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
