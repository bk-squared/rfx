"""CFS-CPML (Complex Frequency Shifted CPML) validation tests.

Tests:
1. Evanescent wave absorption improvement with kappa_max > 1
2. Backward compatibility: kappa_max=1.0 gives identical results to standard CPML
3. No regression for propagating waves when kappa_max > 1
"""

import numpy as np
import pytest
import jax.numpy as jnp

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MU_0
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse


def _run_simulation(grid, cpml_params, cpml_state, source_fn, n_steps,
                    probe_idx, inject_idx, inject_component="ez",
                    pec_axes=""):
    """Run an FDTD simulation and return the probe time series."""
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    dt, dx = grid.dt, grid.dx
    cpml_axes = grid.cpml_axes
    ts = np.zeros(n_steps)

    for n in range(n_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state, cpml_state = apply_cpml_h(
            state, cpml_params, cpml_state, grid, axes=cpml_axes)
        state = update_e(state, materials, dt, dx)
        state, cpml_state = apply_cpml_e(
            state, cpml_params, cpml_state, grid, axes=cpml_axes)
        if pec_axes:
            state = apply_pec(state, axes=pec_axes)
        # Inject source
        field = getattr(state, inject_component)
        field = field.at[inject_idx].add(source_fn(t))
        state = state._replace(**{inject_component: field})
        ts[n] = float(getattr(state, inject_component)[probe_idx])

    return ts, state, cpml_state


def _run_evanescent_sim(grid, cpml_params, cpml_state, n_steps, f0):
    """Run a waveguide simulation with TE10 mode excitation below cutoff.

    Excites a sinusoidal y-profile across the waveguide cross-section
    at a single x-plane, so the dominant excited mode is TE10. When f0
    is below the TE10 cutoff, this field is evanescent in x.

    Returns a dict with energy measurements in different regions.
    """
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    dt, dx = grid.dt, grid.dx
    cpml_axes = grid.cpml_axes
    n_cpml = grid.cpml_layers

    nx, ny, nz = grid.shape
    # Source plane at center of x
    x_src = nx // 2
    # TE10 mode profile: sin(pi * y / a) across the waveguide interior
    y_profile = np.sin(np.pi * np.arange(ny) / (ny - 1))
    y_profile = jnp.array(y_profile, dtype=jnp.float32)

    # CW source with smooth ramp-up
    omega = 2 * np.pi * f0
    ramp_time = 80 * dt  # smooth onset

    for n in range(n_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state, cpml_state = apply_cpml_h(
            state, cpml_params, cpml_state, grid, axes=cpml_axes)
        state = update_e(state, materials, dt, dx)
        state, cpml_state = apply_cpml_e(
            state, cpml_params, cpml_state, grid, axes=cpml_axes)
        state = apply_pec(state, axes="yz")

        # Inject TE10-like Ez source across y at x=x_src
        envelope = float(jnp.clip(t / ramp_time, 0.0, 1.0))
        src_val = envelope * np.sin(omega * t)
        for k in range(nz):
            ez = state.ez.at[x_src, :, k].add(src_val * y_profile)
            state = state._replace(ez=ez)

    # Measure field energy inside the lo-x CPML region
    pml_slice = slice(0, n_cpml)
    e2_pml = (state.ex[pml_slice] ** 2 +
              state.ey[pml_slice] ** 2 +
              state.ez[pml_slice] ** 2)
    h2_pml = (state.hx[pml_slice] ** 2 +
              state.hy[pml_slice] ** 2 +
              state.hz[pml_slice] ** 2)
    energy_pml = float(0.5 * EPS_0 * e2_pml.sum() + 0.5 * MU_0 * h2_pml.sum())

    return energy_pml


def test_cfs_cpml_evanescent_absorption():
    """CFS-CPML (kappa_max>1) should attenuate evanescent fields inside the
    PML faster than standard CPML (kappa_max=1).

    Uses a parallel-plate waveguide (PEC on y,z) with TE10 mode excitation
    below cutoff. The evanescent field penetrates the PML region. With
    CFS-CPML, the sigma profile is scaled by kappa_max (Gedney
    recommendation), providing stronger absorption while kappa stretching
    maintains impedance matching. This results in lower field energy
    inside the PML.
    """
    a = 0.06  # waveguide width (y-direction)
    f_cutoff = C0 / (2 * a)  # TE10 cutoff ≈ 2.5 GHz
    f0 = 0.5 * f_cutoff  # well below cutoff → evanescent

    freq_max = 5e9
    cpml_n = 12
    domain = (0.12, a, 0.02)
    n_steps = 800

    # --- Standard CPML ---
    grid = Grid(freq_max=freq_max, domain=domain, cpml_layers=cpml_n,
                cpml_axes="x")
    cpml_params_std, cpml_state_std = init_cpml(grid, kappa_max=1.0)
    energy_std = _run_evanescent_sim(grid, cpml_params_std, cpml_state_std,
                                      n_steps, f0)

    # --- CFS-CPML ---
    cpml_params_cfs, cpml_state_cfs = init_cpml(grid, kappa_max=7.0)
    energy_cfs = _run_evanescent_sim(grid, cpml_params_cfs, cpml_state_cfs,
                                      n_steps, f0)

    print(f"Cutoff: {f_cutoff/1e9:.2f} GHz, Source: {f0/1e9:.2f} GHz")
    print(f"Standard CPML PML energy: {energy_std:.6e}")
    print(f"CFS-CPML PML energy:      {energy_cfs:.6e}")

    if energy_std > 1e-30:
        improvement = energy_std / max(energy_cfs, 1e-30)
        print(f"CFS improvement factor: {improvement:.1f}x")
        assert improvement > 1.5, (
            f"CFS-CPML improvement {improvement:.1f}x is below 1.5x threshold"
        )
    else:
        print("Both negligible — pass")


def test_cfs_cpml_backward_compatible():
    """kappa_max=1.0 must produce bit-identical results to standard CPML."""
    grid = Grid(freq_max=3e9, domain=(0.08, 0.08, 0.08), cpml_layers=10)
    pulse = GaussianPulse(f0=2e9, bandwidth=0.5)

    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    inject = (cx, cy, cz)
    probe = (cx + 3, cy, cz)
    n_steps = 200

    # --- Run with kappa_max=1.0 ---
    cpml_params_1, cpml_state_1 = init_cpml(grid, kappa_max=1.0)
    ts_k1, _, _ = _run_simulation(
        grid, cpml_params_1, cpml_state_1,
        pulse, n_steps, probe, inject,
    )

    # --- Run with default (no kappa_max argument) ---
    cpml_params_def, cpml_state_def = init_cpml(grid)
    ts_def, _, _ = _run_simulation(
        grid, cpml_params_def, cpml_state_def,
        pulse, n_steps, probe, inject,
    )

    np.testing.assert_allclose(
        ts_k1, ts_def, atol=0, rtol=0,
        err_msg="kappa_max=1.0 should give identical results to default CPML"
    )
    print(f"Backward compatibility: PASS (max diff = {np.max(np.abs(ts_k1 - ts_def)):.2e})")


def test_cfs_cpml_no_regression_above_cutoff():
    """CFS-CPML must not degrade propagating wave absorption vs standard CPML."""
    grid = Grid(freq_max=3e9, domain=(0.12, 0.12, 0.12), cpml_layers=15)
    pulse = GaussianPulse(f0=2e9, bandwidth=0.5)

    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    inject = (cx, cy, cz)
    dt, dx = grid.dt, grid.dx
    n_steps = 700

    def _measure_energy(state):
        return float(
            0.5 * EPS_0 * (state.ex**2 + state.ey**2 + state.ez**2).sum()
            + 0.5 * MU_0 * (state.hx**2 + state.hy**2 + state.hz**2).sum()
        )

    # --- Standard CPML ---
    cpml_params_std, cpml_state_std = init_cpml(grid, kappa_max=1.0)
    state_std = init_state(grid.shape)
    materials = init_materials(grid.shape)
    for n in range(200):
        t = n * dt
        state_std = update_h(state_std, materials, dt, dx)
        state_std, cpml_state_std = apply_cpml_h(state_std, cpml_params_std, cpml_state_std, grid)
        state_std = update_e(state_std, materials, dt, dx)
        state_std, cpml_state_std = apply_cpml_e(state_std, cpml_params_std, cpml_state_std, grid)
        ez = state_std.ez.at[cx, cy, cz].add(pulse(t))
        state_std = state_std._replace(ez=ez)
    for _ in range(500):
        state_std = update_h(state_std, materials, dt, dx)
        state_std, cpml_state_std = apply_cpml_h(state_std, cpml_params_std, cpml_state_std, grid)
        state_std = update_e(state_std, materials, dt, dx)
        state_std, cpml_state_std = apply_cpml_e(state_std, cpml_params_std, cpml_state_std, grid)
    energy_std = _measure_energy(state_std)

    # --- CFS-CPML ---
    cpml_params_cfs, cpml_state_cfs = init_cpml(grid, kappa_max=5.0)
    state_cfs = init_state(grid.shape)
    for n in range(200):
        t = n * dt
        state_cfs = update_h(state_cfs, materials, dt, dx)
        state_cfs, cpml_state_cfs = apply_cpml_h(state_cfs, cpml_params_cfs, cpml_state_cfs, grid)
        state_cfs = update_e(state_cfs, materials, dt, dx)
        state_cfs, cpml_state_cfs = apply_cpml_e(state_cfs, cpml_params_cfs, cpml_state_cfs, grid)
        ez = state_cfs.ez.at[cx, cy, cz].add(pulse(t))
        state_cfs = state_cfs._replace(ez=ez)
    for _ in range(500):
        state_cfs = update_h(state_cfs, materials, dt, dx)
        state_cfs, cpml_state_cfs = apply_cpml_h(state_cfs, cpml_params_cfs, cpml_state_cfs, grid)
        state_cfs = update_e(state_cfs, materials, dt, dx)
        state_cfs, cpml_state_cfs = apply_cpml_e(state_cfs, cpml_params_cfs, cpml_state_cfs, grid)
    energy_cfs = _measure_energy(state_cfs)

    ratio_std = 10 * np.log10(energy_std / max(energy_std, 1e-30))  # baseline
    ratio_cfs = 10 * np.log10(energy_cfs / max(energy_std, 1e-30))

    print(f"Standard CPML final energy: {energy_std:.4e}")
    print(f"CFS-CPML final energy:      {energy_cfs:.4e}")
    print(f"CFS/Standard ratio:         {energy_cfs/max(energy_std,1e-30):.3f}")

    # CFS should not be more than 50% worse
    assert energy_cfs < 1.5 * energy_std, (
        f"CFS-CPML regressed: final energy {energy_cfs:.4e} > "
        f"1.5x standard {energy_std:.4e}"
    )
