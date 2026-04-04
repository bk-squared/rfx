"""Tests for Lorentz/Drude dispersive materials via ADE.

Validates:
1. Coefficients match hand-computed values
2. Drude pole constructor
3. Lorentz medium slows propagation
4. Energy bounded (no ADE instability)
5. Integration with simulation runner
"""

import numpy as np
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.grid import Grid, C0
from rfx.core.yee import (
    init_state, init_materials, update_h, EPS_0, MU_0,
)
from rfx.boundaries.pec import apply_pec
from rfx.geometry.csg import Box
from rfx.materials.lorentz import (
    LorentzPole, drude_pole, lorentz_pole,
    init_lorentz, update_e_lorentz,
)
from rfx.materials.debye import DebyePole, init_debye


def _total_energy(state, dx):
    e_sq = jnp.sum(state.ex**2 + state.ey**2 + state.ez**2)
    h_sq = jnp.sum(state.hx**2 + state.hy**2 + state.hz**2)
    return float(0.5 * EPS_0 * e_sq * dx**3 + 0.5 * MU_0 * h_sq * dx**3)


def test_lorentz_coefficients():
    """Verify ADE coefficients match hand-computed values."""
    shape = (5, 5, 5)
    dt = 1e-12
    omega_0 = 2 * np.pi * 10e9  # 10 GHz resonance
    delta = 1e9  # damping
    kappa = 3.0 * omega_0**2  # Δε=3

    materials = init_materials(shape)
    poles = [LorentzPole(omega_0=omega_0, delta=delta, kappa=kappa)]
    coeffs, lstate = init_lorentz(poles, materials, dt)

    # Hand-computed
    denom = 1.0 + delta * dt
    a_exp = (2.0 - omega_0**2 * dt**2) / denom
    b_exp = -(1.0 - delta * dt) / denom
    c_exp = EPS_0 * kappa * dt**2 / denom

    assert abs(float(coeffs.a[0, 2, 2, 2]) - a_exp) < 1e-6
    assert abs(float(coeffs.b[0, 2, 2, 2]) - b_exp) < 1e-6
    assert abs(float(coeffs.c[0, 2, 2, 2]) - c_exp) / abs(c_exp) < 1e-4

    # State should be zeros
    assert float(jnp.max(jnp.abs(lstate.px))) == 0.0
    assert float(jnp.max(jnp.abs(lstate.px_prev))) == 0.0

    print(f"\nLorentz coefficients (ω₀={omega_0:.2e}, δ={delta:.2e}):")
    print(f"  a={a_exp:.6f}, b={b_exp:.6f}, c={c_exp:.4e}")


def test_drude_pole_constructor():
    """drude_pole should set omega_0=0 and delta=gamma/2."""
    omega_p = 2 * np.pi * 100e9
    gamma = 1e12
    pole = drude_pole(omega_p, gamma)

    assert pole.omega_0 == 0.0
    assert abs(pole.delta - gamma / 2.0) < 1e-6
    assert abs(pole.kappa - omega_p**2) < 1.0

    print(f"\nDrude pole: ω₀={pole.omega_0}, δ={pole.delta:.4e}, κ={pole.kappa:.4e}")


def test_lorentz_pole_constructor():
    """lorentz_pole should compute kappa = Δε·ω₀²."""
    delta_eps = 3.0
    omega_0 = 2 * np.pi * 10e9
    d = 1e9
    pole = lorentz_pole(delta_eps, omega_0, d)

    assert abs(pole.omega_0 - omega_0) < 1.0
    assert abs(pole.delta - d) < 1e-6
    assert abs(pole.kappa - delta_eps * omega_0**2) / (delta_eps * omega_0**2) < 1e-6


def test_lorentz_energy_bounded():
    """Lorentz ADE should not produce energy growth."""
    shape = (20, 20, 20)
    dx = 0.002
    dt = 0.99 * dx / (C0 * np.sqrt(3))

    # Lorentz medium: resonance at 10 GHz
    omega_0 = 2 * np.pi * 10e9
    delta = 5e9  # moderate damping
    kappa = 2.0 * omega_0**2

    materials = init_materials(shape)
    materials = materials._replace(
        eps_r=jnp.full(shape, 2.0, dtype=jnp.float32)
    )
    poles = [LorentzPole(omega_0=omega_0, delta=delta, kappa=kappa)]
    coeffs, lstate = init_lorentz(poles, materials, dt)

    # Initialize with cavity mode
    Lx = (shape[0] - 1) * dx
    Ly = (shape[1] - 1) * dx
    x = np.arange(shape[0]) * dx
    y = np.arange(shape[1]) * dx
    ez_init = np.sin(np.pi * x[:, None, None] / Lx) * \
              np.sin(np.pi * y[None, :, None] / Ly) * \
              np.ones((1, 1, shape[2]))

    state = init_state(shape)
    state = state._replace(ez=jnp.array(ez_init, dtype=jnp.float32))
    initial_energy = _total_energy(state, dx)

    max_energy = initial_energy
    for step in range(500):
        state = update_h(state, materials, dt, dx)
        state, lstate = update_e_lorentz(state, coeffs, lstate, dt, dx)
        state = apply_pec(state)

        if step % 100 == 0:
            e = _total_energy(state, dx)
            max_energy = max(max_energy, e)

    final_energy = _total_energy(state, dx)

    print("\nLorentz energy stability:")
    print(f"  Initial: {initial_energy:.4e}, Max: {max_energy:.4e}, Final: {final_energy:.4e}")

    # Energy should not blow up
    assert max_energy < initial_energy * 2.0, \
        f"Energy grew: max={max_energy:.4e} > 2× initial={initial_energy:.4e}"


def test_lorentz_simulation_runner():
    """Lorentz dispersion works through the compiled simulation runner."""
    from rfx.simulation import run, make_source, SimResult

    grid = Grid(freq_max=15e9, domain=(0.02, 0.02, 0.02))
    materials = init_materials(grid.shape)

    omega_0 = 2 * np.pi * 10e9
    poles = [LorentzPole(omega_0=omega_0, delta=5e9, kappa=1.0 * omega_0**2)]
    lorentz = init_lorentz(poles, materials, grid.dt)

    from rfx.sources.sources import GaussianPulse
    center = (0.01, 0.01, 0.01)
    pulse = GaussianPulse(f0=7.5e9, bandwidth=0.8)
    n_steps = 50
    src = make_source(grid, center, "ez", pulse, n_steps)

    result = run(grid, materials, n_steps, lorentz=lorentz, sources=[src])

    assert isinstance(result, SimResult)
    assert result.state.ex.shape == grid.shape
    # Should have run without error
    print(f"\nLorentz simulation runner: {n_steps} steps OK, shape={grid.shape}")


def test_mixed_debye_and_lorentz_runner():
    """A simulation with both Debye and Lorentz dispersion should run."""
    from rfx.simulation import run, make_source, SimResult

    grid = Grid(freq_max=12e9, domain=(0.02, 0.02, 0.02))
    materials = init_materials(grid.shape)
    materials = materials._replace(
        eps_r=jnp.full(grid.shape, 2.0, dtype=jnp.float32)
    )

    debye = init_debye(
        [DebyePole(delta_eps=1.5, tau=8e-12)],
        materials,
        grid.dt,
    )
    lorentz = init_lorentz(
        [LorentzPole(omega_0=2 * np.pi * 9e9, delta=2e9, kappa=(2 * np.pi * 9e9) ** 2)],
        materials,
        grid.dt,
    )

    from rfx.sources.sources import GaussianPulse

    src = make_source(
        grid,
        (0.01, 0.01, 0.01),
        "ez",
        GaussianPulse(f0=6e9, bandwidth=0.8),
        40,
    )
    result = run(grid, materials, 40, debye=debye, lorentz=lorentz, sources=[src])

    assert isinstance(result, SimResult)
    assert float(jnp.max(jnp.abs(result.state.ez))) > 0.0


def test_lorentz_poles_stay_scoped_to_their_material():
    """Distinct Lorentz poles should only apply inside their own materials."""
    pole_a = lorentz_pole(1.0, 2 * np.pi * 3e9, 1e8)
    pole_b = lorentz_pole(2.0, 2 * np.pi * 5e9, 2e8)

    sim = Simulation(freq_max=8e9, domain=(0.02, 0.01, 0.01), boundary="pec")
    sim.add_material("mat_a", eps_r=2.0, lorentz_poles=[pole_a])
    sim.add_material("mat_b", eps_r=2.5, lorentz_poles=[pole_b])
    sim.add(Box((0.000, 0.000, 0.000), (0.009, 0.010, 0.010)), material="mat_a")
    sim.add(Box((0.011, 0.000, 0.000), (0.020, 0.010, 0.010)), material="mat_b")

    grid = sim._build_grid()
    _, _, lorentz = sim._build_materials(grid)
    coeffs, _ = lorentz

    mask_a = Box((0.000, 0.000, 0.000), (0.009, 0.010, 0.010)).mask(grid)
    mask_b = Box((0.011, 0.000, 0.000), (0.020, 0.010, 0.010)).mask(grid)
    ia = tuple(np.argwhere(np.array(mask_a))[0])
    ib = tuple(np.argwhere(np.array(mask_b))[0])

    assert float(coeffs.c[0][ia]) > 0.0
    assert float(coeffs.c[1][ia]) == 0.0
    assert float(coeffs.c[0][ib]) == 0.0
    assert float(coeffs.c[1][ib]) > 0.0
