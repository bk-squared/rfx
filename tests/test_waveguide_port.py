"""Tests for rectangular waveguide port with analytical TE mode profiles."""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MU_0
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, inject_waveguide_port,
    update_waveguide_port_probe, extract_waveguide_s11, extract_waveguide_s21,
    cutoff_frequency, modal_voltage,
)


def test_te10_cutoff_frequency():
    """TE10 cutoff for standard WR-90 waveguide (a=22.86mm, b=10.16mm)."""
    a = 22.86e-3
    b = 10.16e-3
    f_c = cutoff_frequency(a, b, 1, 0)
    # Analytical: f_c = c / (2*a) = 6.557 GHz
    f_c_exact = C0 / (2 * a)
    err = abs(f_c - f_c_exact) / f_c_exact
    print(f"\nTE10 cutoff: {f_c/1e9:.4f} GHz (exact: {f_c_exact/1e9:.4f} GHz)")
    assert err < 1e-10, f"Cutoff error {err}"


def test_te10_mode_profile_shape():
    """TE10 mode profile: Ey = -sin(pi*y/a), Ez = 0."""
    from rfx.sources.waveguide_port import _te_mode_profiles

    a, b = 0.04, 0.02
    ny, nz = 20, 10
    dx = a / ny
    y = np.linspace(0.5 * dx, a - 0.5 * dx, ny)
    z = np.linspace(0.5 * dx, b - 0.5 * dx, nz)

    ey, ez, hy, hz = _te_mode_profiles(a, b, 1, 0, y, z)

    # TE10: Ez should be zero everywhere (n=0)
    assert np.allclose(ez, 0, atol=1e-10), f"TE10 Ez not zero: max={np.max(np.abs(ez))}"

    # Ey should have sin(pi*y/a) shape along y, constant along z
    ey_mid = ey[:, nz // 2]  # slice at middle z
    expected_shape = np.sin(np.pi * y / a)
    # Normalize both for shape comparison (sign is a convention)
    ey_norm = ey_mid / np.max(np.abs(ey_mid))
    exp_norm = expected_shape / np.max(np.abs(expected_shape))
    assert np.allclose(np.abs(ey_norm), np.abs(exp_norm), atol=0.05), \
        "TE10 Ey profile shape mismatch"

    print(f"\nTE10 mode profile: Ey max={np.max(np.abs(ey)):.4f}, Ez max={np.max(np.abs(ez)):.6f}")


def test_waveguide_port_propagation():
    """TE10 mode launched above cutoff propagates and is received downstream.

    A PEC waveguide (a=40mm, b=20mm) with CPML on x-ends.
    TE10 cutoff = c/(2a) = 3.75 GHz. Source at 6 GHz (well above cutoff).
    |S21| (modal voltage at downstream probe / source DFT) should be
    significant above cutoff, proving the mode propagates.
    """
    # Waveguide dimensions
    a_wg = 0.04   # 40mm width (y)
    b_wg = 0.02   # 20mm height (z)
    length = 0.12  # 120mm length (x)

    f0 = 6e9
    f_c = C0 / (2 * a_wg)  # 3.75 GHz
    assert f0 > f_c, "Source must be above cutoff"

    dx = 0.002  # 2mm cells
    nc = 10  # CPML layers

    grid = Grid(freq_max=10e9, domain=(length, a_wg, b_wg), dx=dx, cpml_layers=nc)
    dt = grid.dt

    # Waveguide aperture in grid indices (interior, excluding CPML)
    y_lo = nc
    y_hi = grid.ny - nc
    z_lo = nc
    z_hi = grid.nz - nc

    # Source plane
    port_x = nc + 5

    port = WaveguidePort(
        x_index=port_x,
        y_slice=(y_lo, y_hi),
        z_slice=(z_lo, z_hi),
        a=a_wg, b=b_wg,
        mode=(1, 0), mode_type="TE",
    )

    freqs = jnp.linspace(4.5e9, 8e9, 25)
    # Probe 15 cells downstream of source
    port_cfg = init_waveguide_port(port, dx, freqs, f0=f0, bandwidth=0.5,
                                   amplitude=1.0, probe_offset=15)

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cp, cs = init_cpml(grid)

    n_steps = grid.num_timesteps(num_periods=40)

    for step in range(n_steps):
        t = step * dt

        state = update_h(state, materials, dt, dx)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        state = update_e(state, materials, dt, dx)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        state = apply_pec(state)  # PEC on all faces (waveguide walls + x ends)

        state = inject_waveguide_port(state, port_cfg, t, dt, dx)
        port_cfg = update_waveguide_port_probe(port_cfg, state, dt, dx)

    s21 = extract_waveguide_s21(port_cfg)
    s21_mag = np.abs(np.array(s21))

    # Above cutoff band
    f_arr = np.array(freqs)
    above_cutoff = f_arr > f_c * 1.3  # well above cutoff
    s21_above = s21_mag[above_cutoff]
    s21_db = 20 * np.log10(np.maximum(s21_above, 1e-10))
    mean_s21_db = np.mean(s21_db)

    print(f"\nWaveguide port TE10 propagation:")
    print(f"  f_cutoff = {f_c/1e9:.2f} GHz, f0 = {f0/1e9:.1f} GHz")
    print(f"  Grid: {grid.shape}")
    print(f"  Steps: {n_steps}")
    print(f"  |S21| above cutoff (mean): {np.mean(s21_above):.4f} ({mean_s21_db:.1f} dB)")
    print(f"  |S21| min/max: {np.min(s21_above):.4f} / {np.max(s21_above):.4f}")

    # TE10 above cutoff: mode should propagate and arrive at probe
    # Soft source sends half the energy each way, so |S21| ~ 0.5 (-6 dB) is expected
    # Allow > -12 dB as minimum threshold
    assert mean_s21_db > -12, \
        f"Mean |S21| = {mean_s21_db:.1f} dB, expected > -12 dB"


def test_te10_below_cutoff_evanescent():
    """Below cutoff, TE10 mode should be evanescent (|S21| → 0).

    At f < f_cutoff, the mode cannot propagate.
    Modal voltage at the downstream probe should be very small.
    """
    a_wg = 0.04
    b_wg = 0.02
    f_c = C0 / (2 * a_wg)  # 3.75 GHz

    # Source centered below cutoff
    f0 = 2.5e9
    dx = 0.002
    nc = 10
    length = 0.10

    grid = Grid(freq_max=10e9, domain=(length, a_wg, b_wg), dx=dx, cpml_layers=nc)
    dt = grid.dt

    y_lo, y_hi = nc, grid.ny - nc
    z_lo, z_hi = nc, grid.nz - nc
    port_x = nc + 5

    port = WaveguidePort(
        x_index=port_x,
        y_slice=(y_lo, y_hi), z_slice=(z_lo, z_hi),
        a=a_wg, b=b_wg, mode=(1, 0), mode_type="TE",
    )

    freqs = jnp.linspace(1e9, 3.0e9, 15)
    port_cfg = init_waveguide_port(port, dx, freqs, f0=f0, bandwidth=0.5,
                                   amplitude=1.0, probe_offset=15)

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cp, cs = init_cpml(grid)

    n_steps = grid.num_timesteps(num_periods=40)
    for step in range(n_steps):
        t = step * dt
        state = update_h(state, materials, dt, dx)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        state = update_e(state, materials, dt, dx)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        state = apply_pec(state)
        state = inject_waveguide_port(state, port_cfg, t, dt, dx)
        port_cfg = update_waveguide_port_probe(port_cfg, state, dt, dx)

    s21 = extract_waveguide_s21(port_cfg)
    s21_mag = np.abs(np.array(s21))
    s21_above = extract_waveguide_s21(port_cfg)  # reuse

    f_arr = np.array(freqs)
    below_cutoff = f_arr < f_c * 0.7
    s21_below = s21_mag[below_cutoff]
    s21_below_db = 20 * np.log10(np.maximum(np.mean(s21_below), 1e-10))

    print(f"\nWaveguide port TE10 below cutoff:")
    print(f"  f_cutoff = {f_c/1e9:.2f} GHz, f0 = {f0/1e9:.1f} GHz")
    print(f"  |S21| below cutoff (mean): {np.mean(s21_below):.4f} ({s21_below_db:.1f} dB)")

    # Below cutoff, evanescent mode → very little reaches probe
    # With 15 cells (30mm) distance and f well below cutoff, expect < -6 dB
    assert s21_below_db < -6, \
        f"Mean |S21| below cutoff = {s21_below_db:.1f} dB, expected < -6 dB"
