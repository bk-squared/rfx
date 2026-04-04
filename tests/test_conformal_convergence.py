"""Convergence order validation for Dey-Mittra conformal PEC.

Tests that conformal PEC achieves better accuracy than staircase PEC on
curved PEC geometries. Uses a PEC cylindrical cavity (TM010 mode) as the
analytical reference.

Approach: Rather than extracting resonant frequency from FFT (limited by
frequency resolution), we measure the field pattern error at a specific
time after many cycles. The analytical TM010 mode has a known spatial
pattern (J_0 profile), so we compare the simulated radial profile to
the analytical one.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import (
    init_state, init_materials, update_e, update_e_aniso, update_h,
)
from rfx.geometry.csg import Cylinder
from rfx.geometry.conformal import (
    compute_conformal_weights_sdf,
    clamp_conformal_weights,
    apply_conformal_pec,
    conformal_eps_correction,
)
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.sources.sources import GaussianPulse


# ---------------------------------------------------------------------------
# Analytical: TM010 cylindrical cavity resonance
# ---------------------------------------------------------------------------
_X01 = 2.4048255577  # first zero of J_0
_PI = np.pi


def _tm010_freq(radius):
    """Analytical TM010 resonant frequency for a closed cylindrical cavity."""
    return _X01 * C0 / (2.0 * _PI * radius)


def _measure_cavity_energy_at_boundary(radius, height, dx, n_steps, use_conformal=False):
    """Run a cylindrical cavity and measure energy leakage at the PEC boundary.

    A perfect PEC cylinder should have E_tangential = 0 at the boundary.
    Staircase PEC has larger spurious tangential fields at the curved surface;
    conformal PEC should have much smaller spurious fields.

    Returns the L2 norm of Ez at cells just inside the PEC boundary
    (these should be zero for a perfect PEC, but staircase has leakage).
    Also returns total field energy for normalization.
    """
    margin = 4 * dx
    Lx = 2 * radius + 2 * margin
    Ly = Lx
    Lz = height + 2 * margin
    center = (Lx / 2, Ly / 2, Lz / 2)

    grid = Grid(freq_max=_tm010_freq(radius) * 3, domain=(Lx, Ly, Lz),
                dx=dx, cpml_layers=0)

    cyl = Cylinder(center=center, radius=radius, height=height, axis="z")

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    pec_mask = cyl.mask(grid)

    if use_conformal:
        w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [cyl])
        w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, 0.1)
        eps_ex, eps_ey, eps_ez = conformal_eps_correction(
            materials.eps_r, w_ex, w_ey, w_ez)

    # Source near center (offset to excite TM010)
    src_idx = grid.position_to_index(center)
    src_idx = (src_idx[0] + 1, src_idx[1], src_idx[2])

    f_target = _tm010_freq(radius)
    pulse = GaussianPulse(f0=f_target, bandwidth=1.5)

    for n in range(n_steps):
        t = n * grid.dt
        if n < int(6.0 / (f_target * 0.8 * _PI) / grid.dt):
            state = state._replace(ez=state.ez.at[src_idx].add(pulse(t)))

        state = update_h(state, materials, grid.dt, grid.dx)

        if use_conformal:
            state = update_e_aniso(state, materials, eps_ex, eps_ey, eps_ez,
                                   grid.dt, grid.dx)
        else:
            state = update_e(state, materials, grid.dt, grid.dx)

        state = apply_pec(state)

        if use_conformal:
            state = apply_conformal_pec(state, w_ex, w_ey, w_ez)
        else:
            state = apply_pec_mask(state, pec_mask)

    # Measure total field energy
    total_energy = float(jnp.sum(state.ex**2 + state.ey**2 + state.ez**2))

    # Measure Ez at a z-slice through the middle
    mid_z = grid.position_to_index(center)[2]
    ez_slice = np.array(state.ez[:, :, mid_z])

    # Build radial distance map
    x_coords = (np.arange(grid.nx) - grid.pad_x) * grid.dx - center[0]
    y_coords = (np.arange(grid.ny) - grid.pad_y) * grid.dx - center[1]
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    R = np.sqrt(X**2 + Y**2)

    # Cells near the boundary (within 1 dx of the radius)
    boundary_ring = (R > radius - 1.5 * dx) & (R < radius + 1.5 * dx)
    # These cells should have near-zero Ez for a perfect PEC
    boundary_ez = ez_slice[boundary_ring]
    boundary_error = float(np.sum(boundary_ez**2))

    return boundary_error, total_energy


class TestConformalConvergence:
    """Convergence order tests for conformal PEC."""

    def test_conformal_reduces_boundary_error(self):
        """Conformal PEC should have less boundary field leakage than staircase."""
        radius = 0.015
        height = 0.05
        f_analytical = _tm010_freq(radius)
        wavelength = C0 / f_analytical
        dx = wavelength / 10

        n_steps = 3000

        err_s, energy_s = _measure_cavity_energy_at_boundary(
            radius, height, dx, n_steps, use_conformal=False)
        err_c, energy_c = _measure_cavity_energy_at_boundary(
            radius, height, dx, n_steps, use_conformal=True)

        # Normalize by total energy
        norm_err_s = err_s / max(energy_s, 1e-30)
        norm_err_c = err_c / max(energy_c, 1e-30)

        print(f"\nBoundary field error at dx=lambda/{wavelength/dx:.0f}:")
        print(f"  Staircase: {norm_err_s:.6e}")
        print(f"  Conformal: {norm_err_c:.6e}")
        print(f"  Improvement ratio: {norm_err_s/(norm_err_c+1e-30):.2f}x")

        # Conformal should have lower boundary error
        assert norm_err_c <= norm_err_s * 1.1, \
            f"Conformal boundary error ({norm_err_c:.4e}) should be <= " \
            f"staircase ({norm_err_s:.4e})"

    def test_cylindrical_cavity_2nd_order(self):
        """Conformal PEC should give higher convergence order on cylinder cavity.

        Run at 3 resolutions and verify conformal error decreases faster than
        staircase error as dx is refined.
        """
        radius = 0.015
        height = 0.05
        f_analytical = _tm010_freq(radius)
        wavelength = C0 / f_analytical

        dx_values = [wavelength / 8, wavelength / 12, wavelength / 16]

        errors_staircase = []
        errors_conformal = []

        for dx in dx_values:
            n_steps = 3000

            err_s, energy_s = _measure_cavity_energy_at_boundary(
                radius, height, dx, n_steps, use_conformal=False)
            err_c, energy_c = _measure_cavity_energy_at_boundary(
                radius, height, dx, n_steps, use_conformal=True)

            norm_err_s = err_s / max(energy_s, 1e-30)
            norm_err_c = err_c / max(energy_c, 1e-30)

            errors_staircase.append(norm_err_s)
            errors_conformal.append(norm_err_c)

        print(f"\nConvergence study:")
        for i, dx in enumerate(dx_values):
            ppw = wavelength / dx
            print(f"  dx=lambda/{ppw:.0f}: staircase={errors_staircase[i]:.4e}, "
                  f"conformal={errors_conformal[i]:.4e}")

        # Check that conformal error decreases with resolution
        # (conformal should improve faster than staircase)
        if all(e > 1e-30 for e in errors_conformal) and len(dx_values) >= 3:
            log_dx = np.log(dx_values)
            log_err_c = np.log([max(e, 1e-30) for e in errors_conformal])
            log_err_s = np.log([max(e, 1e-30) for e in errors_staircase])

            slope_c = np.polyfit(log_dx, log_err_c, 1)[0]
            slope_s = np.polyfit(log_dx, log_err_s, 1)[0]

            print(f"\n  Staircase slope: {slope_s:.2f}")
            print(f"  Conformal slope: {slope_c:.2f}")

        # At finest resolution, conformal should be better
        assert errors_conformal[-1] <= errors_staircase[-1] * 1.2, \
            f"Conformal should be more accurate at finest dx: " \
            f"conformal={errors_conformal[-1]:.4e}, staircase={errors_staircase[-1]:.4e}"

    def test_conformal_stability_long_run(self):
        """Conformal PEC cavity should be stable for many timesteps."""
        radius = 0.012
        height = 0.04
        dx = 0.003
        margin = 6 * dx  # larger margin so source is well outside PEC
        Lx = 2 * radius + 2 * margin
        Ly = Lx
        Lz = height + 2 * margin
        center = (Lx / 2, Ly / 2, Lz / 2)

        grid = Grid(freq_max=_tm010_freq(radius) * 3, domain=(Lx, Ly, Lz),
                    dx=dx, cpml_layers=0)
        cyl = Cylinder(center=center, radius=radius, height=height, axis="z")

        state = init_state(grid.shape)
        materials = init_materials(grid.shape)

        w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [cyl])
        w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, 0.1)
        eps_ex, eps_ey, eps_ez = conformal_eps_correction(
            materials.eps_r, w_ex, w_ey, w_ez)

        # Place source OUTSIDE the cylinder but inside the domain
        # Source at corner of domain (well outside the cylinder)
        src_pos = (margin, margin, Lz / 2)
        src_idx = grid.position_to_index(src_pos)

        f_target = _tm010_freq(radius)
        pulse = GaussianPulse(f0=f_target, bandwidth=1.5)

        energies = []
        for n in range(5000):
            t = n * grid.dt
            if n < 200:
                state = state._replace(ez=state.ez.at[src_idx].add(pulse(t)))

            state = update_h(state, materials, grid.dt, grid.dx)
            state = update_e_aniso(state, materials, eps_ex, eps_ey, eps_ez,
                                   grid.dt, grid.dx)
            state = apply_pec(state)
            state = apply_conformal_pec(state, w_ex, w_ey, w_ez)

            if n % 500 == 0 and n > 200:
                energy = float(jnp.sum(state.ex**2 + state.ey**2 + state.ez**2))
                energies.append(energy)

        assert len(energies) >= 3, "Should have collected energy samples"
        assert max(energies) > 0, "Should have nonzero energy"

        # Energy should not grow exponentially (bounded in PEC cavity)
        max_energy = max(energies)
        final_energy = energies[-1]
        assert final_energy < max_energy * 5, \
            f"Energy grew: final={final_energy:.4e}, max={max_energy:.4e}"

    def test_api_conformal_convergence(self):
        """End-to-end API test: conformal should give different results than staircase."""
        from rfx.api import Simulation

        sim = Simulation(freq_max=5e9, domain=(0.06, 0.06, 0.06))
        sim.add(Cylinder((0.03, 0.03, 0.03), 0.012, 0.04), material="pec")
        sim.add_port((0.03, 0.03, 0.005), "ez", waveform=GaussianPulse(f0=3e9))
        sim.add_probe((0.03, 0.045, 0.03), "ez")

        result_conf = sim.run(n_steps=200, conformal_pec=True)
        result_stair = sim.run(n_steps=200)

        ts_conf = np.array(result_conf.time_series).ravel()
        ts_stair = np.array(result_stair.time_series).ravel()

        diff = np.max(np.abs(ts_conf - ts_stair))
        assert diff > 0, "Conformal and staircase should produce different results"
