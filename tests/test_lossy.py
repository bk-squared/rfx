"""Tests for lossy conductors (finite conductivity σ).

The update_e() Yee update already handles conductivity via Ca/Cb coefficients:
    Ca = (1 - σ·dt/(2ε)) / (1 + σ·dt/(2ε))
    Cb = (dt/ε) / (1 + σ·dt/(2ε))

These tests validate that the σ path produces correct energy dissipation.
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MU_0
from rfx.boundaries.pec import apply_pec
from rfx.materials import set_material
from rfx.sources.sources import GaussianPulse, add_point_source


def _total_energy(state, dx):
    """Total EM energy in the grid."""
    e_energy = 0.5 * EPS_0 * jnp.sum(state.ex**2 + state.ey**2 + state.ez**2) * dx**3
    h_energy = 0.5 * MU_0 * jnp.sum(state.hx**2 + state.hy**2 + state.hz**2) * dx**3
    return float(e_energy + h_energy)


def test_lossy_cavity_energy_decay():
    """Energy in a lossy PEC cavity decays monotonically after source turns off."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), cpml_layers=0)
    materials = init_materials(grid.shape)

    sigma = 0.1  # S/m — moderate loss
    mask = jnp.ones(grid.shape, dtype=bool)
    materials = set_material(materials, mask, sigma=sigma)

    state = init_state(grid.shape)
    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    center = (0.025, 0.025, 0.025)

    # Source shutoff step: pulse is negligible after 6·t0
    shutoff_step = int(6 * pulse.t0 / grid.dt)
    n_steps = grid.num_timesteps(num_periods=30)

    energies = []
    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)

        if step < shutoff_step:
            state = add_point_source(state, grid, center, "ez", pulse(t))

        if step % 50 == 0:
            energies.append(_total_energy(state, grid.dx))

    energies = np.array(energies)
    peak_idx = np.argmax(energies)
    decay = energies[peak_idx:]

    ratio = decay[-1] / decay[0] if decay[0] > 0 else 0
    print(f"\nLossy cavity: peak energy = {decay[0]:.4e}, final = {decay[-1]:.4e}")
    print(f"  Decay ratio = {ratio:.6f} ({10*np.log10(max(ratio, 1e-20)):.1f} dB)")

    assert ratio < 0.1, f"Lossy cavity didn't decay enough: ratio={ratio:.4f}"

    # Energy should be mostly decreasing after peak
    diffs = np.diff(decay)
    n_increasing = np.sum(diffs > decay[0] * 1e-4)
    assert n_increasing < len(diffs) * 0.1, \
        f"Energy not monotonically decreasing: {n_increasing}/{len(diffs)} steps increasing"


def test_higher_sigma_faster_decay():
    """Higher conductivity → faster energy decay from initial cavity mode.

    Initialize with a TM110-like pattern to avoid soft-source ambiguity.
    Track energy ratio (final/initial) for each σ.
    """
    from rfx.grid import C0

    shape = (20, 20, 20)
    dx = 0.003
    dt = 0.99 * dx / (C0 * np.sqrt(3))
    Lx = (shape[0] - 1) * dx
    Ly = (shape[1] - 1) * dx

    # TM110-like initial field: Ez = sin(πx/Lx) sin(πy/Ly)
    x = np.arange(shape[0]) * dx
    y = np.arange(shape[1]) * dx
    ez_init = np.sin(np.pi * x[:, None, None] / Lx) * \
              np.sin(np.pi * y[None, :, None] / Ly) * \
              np.ones((1, 1, shape[2]))

    decay_ratios = []
    for sigma_val in [0.0, 0.1, 0.5]:
        materials = init_materials(shape)
        if sigma_val > 0:
            mask = jnp.ones(shape, dtype=bool)
            materials = set_material(materials, mask, sigma=sigma_val)

        state = init_state(shape)
        state = state._replace(ez=jnp.array(ez_init, dtype=jnp.float32))
        initial_energy = _total_energy(state, dx)

        for step in range(500):
            state = update_h(state, materials, dt, dx)
            state = update_e(state, materials, dt, dx)
            state = apply_pec(state)

        final_energy = _total_energy(state, dx)
        ratio = final_energy / initial_energy if initial_energy > 0 else 0
        decay_ratios.append(ratio)

    print("\nLossy comparison (final/initial energy ratio):")
    for s, r in zip([0.0, 0.1, 0.5], decay_ratios):
        print(f"  σ={s:.2f} S/m: ratio = {r:.6f}")

    # Lossless: energy should be conserved (ratio ~ 1)
    assert decay_ratios[0] > 0.9, \
        f"Lossless energy not conserved: ratio={decay_ratios[0]:.4f}"
    # Higher σ → less energy remaining
    assert decay_ratios[1] < decay_ratios[0] * 0.5, \
        f"σ=0.1 ratio {decay_ratios[1]:.4f} not < 50% of lossless {decay_ratios[0]:.4f}"
    assert decay_ratios[2] < decay_ratios[1] * 0.5, \
        f"σ=0.5 ratio {decay_ratios[2]:.4f} not < 50% of σ=0.1 {decay_ratios[1]:.4f}"


def test_set_material_region():
    """set_material correctly applies properties to masked region only."""
    shape = (10, 10, 10)
    materials = init_materials(shape)

    # Create a mask for a sub-region
    mask = jnp.zeros(shape, dtype=bool)
    mask = mask.at[3:7, 3:7, 3:7].set(True)

    materials = set_material(materials, mask, eps_r=4.0, sigma=0.5)

    # Inside mask: eps_r=4.0, sigma=0.5
    assert float(materials.eps_r[5, 5, 5]) == 4.0
    assert float(materials.sigma[5, 5, 5]) == 0.5

    # Outside mask: eps_r=1.0, sigma=0.0
    assert float(materials.eps_r[0, 0, 0]) == 1.0
    assert float(materials.sigma[0, 0, 0]) == 0.0

    # mu_r unchanged
    assert float(materials.mu_r[5, 5, 5]) == 1.0
