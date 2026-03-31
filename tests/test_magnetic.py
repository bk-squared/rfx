"""Magnetic-material (mu_r) validation tests.

Tests:
1. Fresnel normal-incidence reflection from a magnetic slab (mu_r=4, eps_r=1)
   Expected |R| = 1/3.
2. Phase velocity in a uniform magnetic medium (mu_r=4, eps_r=1)
   Expected v = c / sqrt(mu_r * eps_r) = c/2.

Reference: Codex Spec 6D — docs/codex_specs/6D_magnetic_material_validation.md
"""

import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MU_0
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.tfsf import (
    init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e,
    apply_tfsf_e, apply_tfsf_h,
)


# =========================================================================
# Test 1: Magnetic Fresnel reflection  |R| = 1/3  for mu_r=4, eps_r=1
# =========================================================================

def test_magnetic_impedance_reflection():
    r"""Plane-wave reflection from a magnetic slab matches Fresnel.

    For a slab with eps_r=1, mu_r=4:
        η = η₀ * √(μ_r / ε_r) = 2 η₀
        R = (η₂ − η₁) / (η₂ + η₁) = (2η₀ − η₀) / (2η₀ + η₀) = 1/3

    Uses TFSF source (Ez polarization, +x direction).
    Probe in scattered-field region measures reflected wave.
    A separate vacuum run provides the incident reference.
    Spectral |R| averaged over 3–7 GHz band should match 1/3 within 15%.
    """
    mu_r = 4.0
    eps_r = 1.0
    R_analytic = abs(
        (np.sqrt(mu_r / eps_r) - 1.0) / (np.sqrt(mu_r / eps_r) + 1.0)
    )  # 1/3

    # Grid: 0.60 m in x gives plenty of room for slab + vacuum buffers
    grid = Grid(
        freq_max=10e9,
        domain=(0.60, 0.006, 0.006),
        dx=0.001,
        cpml_layers=10,
        cpml_axes="x",
    )
    dt, dx = grid.dt, grid.dx
    nc = grid.cpml_layers
    periodic = (False, True, True)

    # TFSF source configuration
    tfsf_cfg, _ = init_tfsf(
        grid.nx, dx, dt,
        cpml_layers=nc,
        tfsf_margin=5,
        f0=5e9,
        bandwidth=0.5,
        amplitude=1.0,
        polarization="ez",
        direction="+x",
    )

    # Magnetic slab: starts at nx//4, ends 10 cells before x_hi
    # Slab must NOT touch TFSF boundaries (vacuum buffer at both ends)
    x_interface = grid.nx // 4
    x_slab_end = tfsf_cfg.x_hi - 10

    # Scattered-field probe (x < x_lo) — measures only reflected wave
    probe_scat = (tfsf_cfg.x_lo - 3, grid.ny // 2, grid.nz // 2)
    # 1D reference index for incident normalization
    ref_1d_idx = tfsf_cfg.i0 + 5

    # Time budget: avoid back-face reflection contaminating front-face measurement
    v_slab = C0 / np.sqrt(mu_r * eps_r)
    slab_thick = (x_slab_end - x_interface) * dx
    t_backface = (2 * slab_thick) / v_slab  # round-trip in slab
    t_front = (x_interface - tfsf_cfg.x_lo) * dx / C0  # front-face → probe
    t_safe = t_front + t_backface
    n_steps = min(int(t_safe / dt) - 50, 2000)
    n_steps = max(n_steps, 800)

    def _run(materials):
        """Run TFSF simulation, return (scattered_trace, incident_trace)."""
        _, tfsf_st = init_tfsf(
            grid.nx, dx, dt,
            cpml_layers=nc,
            tfsf_margin=5,
            f0=5e9,
            bandwidth=0.5,
            amplitude=1.0,
            polarization="ez",
            direction="+x",
        )
        state = init_state(grid.shape)
        cp, cs = init_cpml(grid)

        ts_scat = np.zeros(n_steps)
        ts_inc = np.zeros(n_steps)

        for step in range(n_steps):
            t = step * dt

            # Correct leapfrog interleaving (Taflove Ch. 5)
            state = update_h(state, materials, dt, dx, periodic)
            state = apply_tfsf_h(state, tfsf_cfg, tfsf_st, dx, dt)
            state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
            tfsf_st = update_tfsf_1d_h(tfsf_cfg, tfsf_st, dx, dt)

            state = update_e(state, materials, dt, dx, periodic)
            state = apply_tfsf_e(state, tfsf_cfg, tfsf_st, dx, dt)
            state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
            tfsf_st = update_tfsf_1d_e(tfsf_cfg, tfsf_st, dx, dt, t)

            ts_scat[step] = float(state.ez[probe_scat])
            ts_inc[step] = float(tfsf_st.e1d[ref_1d_idx])

        return ts_scat, ts_inc, state

    # --- Slab run (magnetic material) ---
    mat_slab = init_materials(grid.shape)
    mat_slab = mat_slab._replace(
        mu_r=mat_slab.mu_r.at[x_interface:x_slab_end, :, :].set(mu_r)
    )
    ts_refl, ts_inc_slab, state_slab = _run(mat_slab)

    # --- Vacuum run (reference) ---
    mat_vac = init_materials(grid.shape)
    _, ts_inc_vac, _ = _run(mat_vac)

    # Sanity: no NaN
    assert not np.any(np.isnan(ts_refl)), "NaN in reflected trace"
    assert not np.any(np.isnan(ts_inc_slab)), "NaN in incident trace"

    # Spectral |R| in the source band
    freqs = np.fft.rfftfreq(n_steps, d=dt)
    spec_scat = np.abs(np.fft.rfft(ts_refl))
    spec_inc = np.abs(np.fft.rfft(ts_inc_vac))

    band = (freqs > 3e9) & (freqs < 7e9)
    R_num = spec_scat[band] / np.maximum(spec_inc[band], 1e-30)
    R_mean = float(np.mean(R_num))

    print(f"\nMagnetic Fresnel reflection (mu_r={mu_r}, eps_r={eps_r}):")
    print(f"  Analytic |R|:   {R_analytic:.4f}")
    print(f"  Numerical |R|:  {R_mean:.4f}")
    print(f"  Error: {abs(R_mean - R_analytic) / R_analytic * 100:.1f}%")

    assert abs(R_mean - R_analytic) / R_analytic < 0.15, (
        f"Magnetic Fresnel |R|={R_mean:.4f}, expected {R_analytic:.4f} "
        f"(error {abs(R_mean - R_analytic)/R_analytic*100:.1f}% > 15%)"
    )


# =========================================================================
# Test 2: Phase velocity in uniform magnetic medium  v = c / √(mu_r)
# =========================================================================

def test_magnetic_phase_velocity():
    r"""Pulse travels at v = c / √(μ_r · ε_r) in a uniform magnetic medium.

    Fill the entire TFSF interior with mu_r=4, eps_r=1.
    Two probes inside the total-field region measure arrival times.
    Velocity = Δx / Δt should match c/2 within 10%.
    """
    mu_r = 4.0
    eps_r = 1.0
    v_expected = C0 / np.sqrt(mu_r * eps_r)  # c/2

    grid = Grid(
        freq_max=10e9,
        domain=(0.10, 0.006, 0.006),
        dx=0.001,
        cpml_layers=10,
        cpml_axes="x",
    )
    dt, dx = grid.dt, grid.dx
    nc = grid.cpml_layers
    periodic = (False, True, True)
    y0 = grid.ny // 2
    z0 = grid.nz // 2

    tfsf_cfg, tfsf_st = init_tfsf(
        grid.nx, dx, dt,
        cpml_layers=nc,
        tfsf_margin=5,
        f0=5e9,
        bandwidth=0.5,
        amplitude=1.0,
        polarization="ez",
        direction="+x",
    )

    # Uniform magnetic medium filling most of the TFSF interior
    medium_lo = tfsf_cfg.x_lo + 5
    medium_hi = tfsf_cfg.x_hi - 5

    # Two probes well inside the magnetic region (total-field zone)
    probe_a_x = medium_lo + 10
    probe_b_x = medium_lo + 34
    assert probe_b_x < medium_hi, "Probe B must be inside the magnetic region"

    probe_a = (probe_a_x, y0, z0)
    probe_b = (probe_b_x, y0, z0)

    # Materials: uniform mu_r in the medium region
    materials = init_materials(grid.shape)
    materials = materials._replace(
        mu_r=materials.mu_r.at[medium_lo:medium_hi, :, :].set(mu_r)
    )

    # Time budget: enough for pulse to reach probe B
    # Source travels at C0 from src to medium_lo, then at v_expected
    src_to_medium = (medium_lo - tfsf_cfg.x_lo) * dx / C0
    medium_to_b = (probe_b_x - medium_lo) * dx / v_expected
    t_arrival_b = tfsf_cfg.src_t0 + src_to_medium + medium_to_b
    search_half = max(int(np.ceil(2.5 * tfsf_cfg.src_tau / dt)), 40)
    n_steps = max(
        int(np.ceil(t_arrival_b / dt)) + 2 * search_half,
        640,
    )

    state = init_state(grid.shape)
    cp, cs = init_cpml(grid)

    trace_a = np.zeros(n_steps)
    trace_b = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt

        state = update_h(state, materials, dt, dx, periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_st, dx, dt)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_h(tfsf_cfg, tfsf_st, dx, dt)

        state = update_e(state, materials, dt, dx, periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_st, dx, dt)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_e(tfsf_cfg, tfsf_st, dx, dt, t)

        trace_a[step] = float(state.ez[probe_a])
        trace_b[step] = float(state.ez[probe_b])

    # Sanity: no NaN
    assert not np.any(np.isnan(trace_a)), "NaN in trace A"
    assert not np.any(np.isnan(trace_b)), "NaN in trace B"

    # Find peak arrival at each probe
    # Estimate expected arrival indices for search windows
    src_to_medium_t = tfsf_cfg.src_t0 + src_to_medium
    expected_a = int(round((src_to_medium_t + (probe_a_x - medium_lo) * dx / v_expected) / dt))
    expected_b = int(round((src_to_medium_t + (probe_b_x - medium_lo) * dx / v_expected) / dt))

    def _peak_near(trace, expected_idx, half_w):
        lo = max(0, expected_idx - half_w)
        hi = min(len(trace), expected_idx + half_w)
        return lo + int(np.argmax(np.abs(trace[lo:hi])))

    peak_a = _peak_near(trace_a, expected_a, search_half)
    peak_b = _peak_near(trace_b, expected_b, search_half)

    assert peak_b > peak_a, (
        f"Probe B peak ({peak_b}) should arrive after probe A ({peak_a})"
    )

    delta_t = (peak_b - peak_a) * dt
    delta_x = (probe_b_x - probe_a_x) * dx
    v_measured = delta_x / delta_t

    print(f"\nMagnetic phase velocity (mu_r={mu_r}, eps_r={eps_r}):")
    print(f"  Expected v:  {v_expected:.3e} m/s  (c/{np.sqrt(mu_r * eps_r):.1f})")
    print(f"  Measured v:  {v_measured:.3e} m/s")
    print(f"  Error: {abs(v_measured - v_expected) / v_expected * 100:.1f}%")

    assert abs(v_measured - v_expected) / v_expected < 0.10, (
        f"Magnetic phase velocity {v_measured:.3e} m/s, expected {v_expected:.3e} m/s "
        f"(error {abs(v_measured - v_expected)/v_expected*100:.1f}% > 10%)"
    )
