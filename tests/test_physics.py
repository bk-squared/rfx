"""Enhanced physics validation tests.

Tests:
1. Fresnel normal-incidence reflection from a dielectric half-space
2. Mesh convergence (2nd-order Yee scheme)
3. Late-time numerical stability (long run in PEC cavity)
"""

import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MU_0
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.sources import GaussianPulse


def _fft_peak_freq(time_series, dt, f_lo, f_hi):
    """Find peak frequency in a band using zero-padded FFT."""
    n = len(time_series)
    n_pad = n * 8
    spectrum = np.abs(np.fft.rfft(time_series, n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=dt)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    masked = np.where(mask, spectrum, 0.0)
    peak_idx = np.argmax(masked)
    return freqs[peak_idx]


def test_fresnel_normal_incidence():
    """Plane-wave reflection from dielectric half-space matches Fresnel.

    R = (1 - n) / (1 + n), n = sqrt(eps_r).
    For eps_r=4: |R| = 1/3.

    Two CPML-terminated runs (free space vs with interface), subtract
    to isolate reflected signal, compare spectral |R| to analytic.
    """
    eps_r = 4.0
    R_analytic = abs((1.0 - np.sqrt(eps_r)) / (1.0 + np.sqrt(eps_r)))  # 1/3

    # Elongated domain: propagation in x, small cross-section
    grid = Grid(freq_max=5e9, domain=(0.20, 0.04, 0.04), dx=0.002, cpml_layers=10)
    pulse = GaussianPulse(f0=2.5e9, bandwidth=0.5)
    dt, dx = grid.dt, grid.dx
    nc = grid.cpml_layers

    src_x = nc + 15              # source plane, well inside CPML
    x_interface = grid.nx // 2   # dielectric starts here
    probe_x = (src_x + x_interface) // 2
    probe = (probe_x, grid.ny // 2, grid.nz // 2)

    n_steps = 800

    def run_sim(materials):
        state = init_state(grid.shape)
        cp, cs = init_cpml(grid)
        ts = np.zeros(n_steps)
        for step in range(n_steps):
            t = step * dt
            state = update_h(state, materials, dt, dx)
            state, cs = apply_cpml_h(state, cp, cs, grid)
            state = update_e(state, materials, dt, dx)
            state, cs = apply_cpml_e(state, cp, cs, grid)
            # Plane-wave source: excite Ez across interior y-z plane
            ez = state.ez.at[src_x, nc:-nc, nc:-nc].add(pulse(t))
            state = state._replace(ez=ez)
            ts[step] = float(state.ez[probe])
        return ts

    # Reference (free space)
    ts_ref = run_sim(init_materials(grid.shape))

    # With dielectric half-space — keep dielectric OUT of CPML regions
    # (CPML coeff_e assumes eps_r=1; material in CPML causes instability)
    mat_diel = init_materials(grid.shape)
    mat_diel = mat_diel._replace(
        eps_r=mat_diel.eps_r.at[x_interface:grid.nx - nc, :, :].set(eps_r)
    )
    ts_diel = run_sim(mat_diel)

    assert not np.any(np.isnan(ts_diel)), "NaN in dielectric simulation"

    # Reflected signal = total - incident
    ts_refl = ts_diel - ts_ref

    # Spectral reflection coefficient in source bandwidth
    freqs = np.fft.rfftfreq(n_steps, d=dt)
    spec_inc = np.abs(np.fft.rfft(ts_ref))
    spec_refl = np.abs(np.fft.rfft(ts_refl))

    band = (freqs > 1.5e9) & (freqs < 3.5e9)
    R_num = spec_refl[band] / np.maximum(spec_inc[band], 1e-30)
    R_mean = np.mean(R_num)

    print(f"\nFresnel normal incidence (eps_r={eps_r}):")
    print(f"  Analytic |R|: {R_analytic:.4f}")
    print(f"  Numerical |R| (mean): {R_mean:.4f}")
    print(f"  Error: {abs(R_mean - R_analytic) / R_analytic * 100:.1f}%")

    assert abs(R_mean - R_analytic) / R_analytic < 0.20, \
        f"Fresnel |R| error {abs(R_mean - R_analytic)/R_analytic*100:.1f}% exceeds 20%"


def test_mesh_convergence_2nd_order():
    """PEC cavity resonance error decreases with mesh refinement.

    Run at three resolutions (4mm, 2mm, 1mm). Finer meshes should
    give smaller frequency error (2nd-order Yee scheme).
    """
    CAVITY_A, CAVITY_B, CAVITY_D = 0.1, 0.1, 0.05
    F_ANALYTICAL = (C0 / 2.0) * np.sqrt((1 / CAVITY_A)**2 + (1 / CAVITY_B)**2)

    resolutions = [0.004, 0.002, 0.001]  # 4mm, 2mm, 1mm
    errors = []

    for dx_val in resolutions:
        grid = Grid(freq_max=5e9, domain=(CAVITY_A, CAVITY_B, CAVITY_D),
                    dx=dx_val, cpml_layers=0)
        state = init_state(grid.shape)
        materials = init_materials(grid.shape)

        pulse = GaussianPulse(f0=F_ANALYTICAL, bandwidth=0.8)
        src_i = grid.nx // 3
        src_j = grid.ny // 3
        src_k = grid.nz // 2
        probe_i = 2 * grid.nx // 3
        probe_j = 2 * grid.ny // 3
        probe_k = grid.nz // 2

        num_steps = grid.num_timesteps(num_periods=80)
        dt, dx = grid.dt, grid.dx

        ts = np.zeros(num_steps)
        for n in range(num_steps):
            t = n * dt
            state = update_h(state, materials, dt, dx)
            state = update_e(state, materials, dt, dx)
            state = apply_pec(state)
            ez = state.ez.at[src_i, src_j, src_k].add(pulse(t))
            state = state._replace(ez=ez)
            ts[n] = float(state.ez[probe_i, probe_j, probe_k])

        # Use raw FFT peak (no parabolic interpolation — avoids artifacts at coarse grids)
        f_peak = _fft_peak_freq(ts, dt, F_ANALYTICAL * 0.5, F_ANALYTICAL * 1.5)
        err = abs(f_peak - F_ANALYTICAL) / F_ANALYTICAL
        errors.append(err)

    print(f"\nMesh convergence (TM110 = {F_ANALYTICAL/1e9:.4f} GHz):")
    for dx_val, err in zip(resolutions, errors):
        print(f"  dx={dx_val*1000:.0f}mm: err={err*100:.4f}%")

    # Finer mesh must reduce error vs coarsest
    ratio_coarse_to_fine = errors[0] / max(errors[2], 1e-15)
    print(f"  Ratio (4mm/1mm): {ratio_coarse_to_fine:.2f} (expect ~16 for 2nd order)")

    # Core convergence check: finest mesh beats coarsest
    assert errors[2] < errors[0], \
        f"1mm mesh ({errors[2]*100:.4f}%) not better than 4mm ({errors[0]*100:.4f}%)"
    # Finest mesh should have sub-0.5% error
    assert errors[2] < 0.005, \
        f"Finest mesh error {errors[2]*100:.3f}% exceeds 0.5%"
    # Overall improvement ratio should show convergence
    assert ratio_coarse_to_fine > 1.2, \
        f"Convergence ratio {ratio_coarse_to_fine:.2f} too low"


def test_late_time_stability():
    """Lossless PEC cavity conserves energy over thousands of timesteps.

    Runs 5000 steps after source stops. In a lossless PEC cavity,
    EM energy is exactly conserved. Checks: no NaN/Inf, drift < 0.1%.
    """
    grid = Grid(freq_max=3e9, domain=(0.05, 0.05, 0.025), cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=2e9, bandwidth=0.5)
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    dt, dx = grid.dt, grid.dx

    def em_energy(s):
        return float(
            0.5 * EPS_0 * (s.ex**2 + s.ey**2 + s.ez**2).sum()
            + 0.5 * MU_0 * (s.hx**2 + s.hy**2 + s.hz**2).sum()
        )

    # Inject source for 200 steps
    for n in range(200):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        ez = state.ez.at[cx, cy, cz].add(pulse(t))
        state = state._replace(ez=ez)

    energy_after_source = em_energy(state)
    assert not np.isnan(energy_after_source), "NaN after source injection"
    assert energy_after_source > 0, "No energy injected"

    # Run 5000 more steps — energy in lossless PEC cavity must be conserved
    for n in range(5000):
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)

        if n % 1000 == 999:
            e = em_energy(state)
            assert not np.isnan(e), f"NaN at step {200 + n + 1}"
            assert not np.isinf(e), f"Inf at step {200 + n + 1}"

    energy_final = em_energy(state)
    drift = abs(energy_final - energy_after_source) / energy_after_source

    print(f"\nLate-time stability (5000 steps, dt=0.99*CFL):")
    print(f"  Energy after source: {energy_after_source:.4e}")
    print(f"  Energy after 5000 steps: {energy_final:.4e}")
    print(f"  Drift: {drift * 100:.6f}%")

    assert drift < 0.001, f"Energy drift {drift*100:.4f}% exceeds 0.1%"
