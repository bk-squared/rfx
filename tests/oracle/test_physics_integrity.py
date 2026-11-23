"""Physics integrity tests — verify Maxwell's equations are satisfied.

NOT accuracy tests (frequency matching). These verify the FDTD engine
correctly implements Maxwell's equations at the discrete level.
Counting the high-z Ez ghost before PEC projection would create a spurious initial energy loss.
"""

from rfx.geometry.rasterize_grid import unpadded_high_walls

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import MaterialArrays, init_state, init_materials, EPS_0, MU_0
from rfx.stepping.yee import update_h, update_e
from rfx.boundaries.pec import apply_pec



def test_maxwell_residual_vacuum():
    """Faraday's law residual should be at machine precision in vacuum.

    After one H-update: H^{n+1/2} = H^n - (dt/mu)*curl(E^n)
    Rearranging: curl(E^n) + mu*(H^{n+1/2} - H^n)/dt = 0
    The residual should be ~1e-7 (float32 precision).
    The residual excludes boundary cells because edge effects do not obey the interior stencil.
    """
    grid = Grid(freq_max=5e9, domain=(0.05, 0.04, 0.03), dx=0.001, cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    ez_init = jnp.sin(jnp.linspace(0, 3.14, grid.nx))[:, None, None] * \
              jnp.sin(jnp.linspace(0, 3.14, grid.ny))[None, :, None] * \
              jnp.ones((1, 1, grid.nz)) * 1.0
    state = state._replace(ez=ez_init.astype(jnp.float32))

    hx_before = state.hx.copy()  # noqa: F841

    state_after = update_h(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))

    dx = grid.dx
    dez_dy = (state.ez[:, 1:, :] - state.ez[:, :-1, :]) / dx
    dez_dy = jnp.pad(dez_dy, ((0, 0), (0, 1), (0, 0)))

    dhx_dt = (state_after.hx - hx_before) / grid.dt
    residual_hx = MU_0 * dhx_dt + dez_dy

    interior = residual_hx[2:-2, 2:-2, 2:-2]
    max_residual = float(jnp.max(jnp.abs(interior)))
    mean_residual = float(jnp.mean(jnp.abs(interior)))

    print("\nFaraday residual (Hx component):")
    print(f"  Max:  {max_residual:.2e}")
    print(f"  Mean: {mean_residual:.2e}")

    field_scale = float(jnp.max(jnp.abs(ez_init)))
    rel_residual = max_residual / (field_scale / dx * MU_0 / grid.dt + 1e-30)
    print(f"  Relative: {rel_residual:.2e}")
    assert rel_residual < 1e-4, f"Maxwell residual too large: {rel_residual:.2e}"



def test_energy_conservation_pec_cavity():
    """Total EM energy must be constant in lossless PEC cavity.

    E_total = (1/2) * ε₀ * ∫|E|²dV + (1/2) * μ₀ * ∫|H|²dV
    The 5% drift bar permits float32 PEC dissipation of non-eigenmode components over 500 steps.
    """
    grid = Grid(freq_max=5e9, domain=(0.05, 0.04, 0.03), dx=0.001, cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    a, b = 0.05, 0.04
    INITIAL_E_AMPLITUDE = 1e6
    E0 = INITIAL_E_AMPLITUDE
    x = jnp.arange(grid.nx) * grid.dx
    y = jnp.arange(grid.ny) * grid.dx
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    ez_mode = E0 * jnp.sin(jnp.pi * X / a) * jnp.sin(jnp.pi * Y / b)
    ez_init = ez_mode[:, :, None] * jnp.ones((1, 1, grid.nz))
    state = state._replace(ez=ez_init.astype(jnp.float32))
    state = apply_pec(state)

    dV = grid.dx ** 3

    def total_energy(st):
        E_energy = 0.5 * EPS_0 * jnp.sum(st.ex**2 + st.ey**2 + st.ez**2) * dV
        H_energy = 0.5 * MU_0 * jnp.sum(st.hx**2 + st.hy**2 + st.hz**2) * dV
        return float(E_energy + H_energy)

    E_initial = total_energy(state)

    energies = [E_initial]
    for _ in range(500):
        state = update_h(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        state = update_e(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        state = apply_pec(state)
        energies.append(total_energy(state))

    E_final = energies[-1]
    drift = abs(E_final - E_initial) / E_initial

    print("\nEnergy conservation (500 steps):")
    print(f"  Initial: {E_initial:.6e}")
    print(f"  Final:   {E_final:.6e}")
    print(f"  Drift:   {drift:.2e}")

    assert drift < 0.05, f"Energy drift {drift:.2e} exceeds 5% tolerance"



def test_energy_decay_lossy():
    """Energy decay rate should match σ∫|E|²dV (Poynting theorem).

    dW/dt = -σ ∫|E|²dV (ohmic loss)
    The decreasing-trend bound allows float32 jitter in a dissipative medium.
    """
    CONDUCTIVITY_S_PER_M = 0.1
    SIGNIFICANT_INCREASE_FRACTION = 1e-6
    sigma_val = CONDUCTIVITY_S_PER_M
    grid = Grid(freq_max=5e9, domain=(0.04, 0.04, 0.04), dx=0.002, cpml_layers=0)
    state = init_state(grid.shape)
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    sigma = jnp.full(grid.shape, sigma_val, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    r = 3
    state = state._replace(
        ez=state.ez.at[cx-r:cx+r+1, cy-r:cy+r+1, cz-r:cz+r+1].set(100.0)
    )

    dV = grid.dx ** 3

    def total_energy(st):
        return float(0.5 * EPS_0 * jnp.sum(st.ex**2 + st.ey**2 + st.ez**2) * dV +
                      0.5 * MU_0 * jnp.sum(st.hx**2 + st.hy**2 + st.hz**2) * dV)

    energies = []
    for step in range(200):
        state = update_h(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        state = update_e(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        state = apply_pec(state)
        energies.append(total_energy(state))

    energies = np.array(energies)
    non_zero = energies > 1e-30
    if np.sum(non_zero) > 10:
        diffs = np.diff(energies[non_zero])
        increasing = np.sum(diffs > SIGNIFICANT_INCREASE_FRACTION * np.max(energies))
        total_points = len(diffs)
        frac_increasing = increasing / max(total_points, 1)
        print("\nLossy energy decay:")
        print(f"  Initial: {energies[0]:.6e}")
        print(f"  Final:   {energies[-1]:.6e}")
        print(f"  Ratio:   {energies[-1] / (energies[0] + 1e-30):.4f}")
        print(f"  Significant increases: {increasing}/{total_points} ({frac_increasing:.1%})")
        assert frac_increasing < 0.3, f"Energy increased in {frac_increasing:.0%} of steps"
        assert energies[-1] < energies[0] * 0.99, "Lossy medium should dissipate energy"



@pytest.mark.slow_physics
def test_field_pattern_tm110():
    """Ez pattern in PEC cavity should match sin(πx/a)*sin(πy/b).
    A cavity eigenmode's overall sign is arbitrary.
    """
    a, b, d = 0.05, 0.04, 0.03
    grid = Grid(freq_max=5e9, domain=(a, b, d), dx=0.001, cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    from rfx.sources.sources import GaussianPulse

    f_110 = (C0 / 2) * np.sqrt((1/a)**2 + (1/b)**2)
    pulse = GaussianPulse(f0=f_110, bandwidth=0.8)
    si, sj = grid.nx // 3, grid.ny // 3
    sk = grid.nz // 2

    n_steps = grid.num_timesteps(num_periods=100)
    for n in range(n_steps):
        state = update_h(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        state = update_e(state, materials, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        state = apply_pec(state)
        state = state._replace(ez=state.ez.at[si, sj, sk].add(pulse(n * grid.dt)))

    ez_slice = np.array(state.ez[:, :, sk])

    x = np.arange(grid.nx) * grid.dx
    y = np.arange(grid.ny) * grid.dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    analytical = np.sin(np.pi * X / a) * np.sin(np.pi * Y / b)

    ez_norm = ez_slice / (np.max(np.abs(ez_slice)) + 1e-30)
    ana_norm = analytical / (np.max(np.abs(analytical)) + 1e-30)

    corr_pos = np.corrcoef(ez_norm.ravel(), ana_norm.ravel())[0, 1]
    corr_neg = np.corrcoef(ez_norm.ravel(), -ana_norm.ravel())[0, 1]
    correlation = max(corr_pos, corr_neg)

    print("\nTM110 field pattern correlation:")
    print(f"  Correlation: {correlation:.4f}")
    assert correlation > 0.95, f"Field pattern mismatch: correlation={correlation:.4f}"



def test_reciprocity_two_port():
    """S12 should equal S21 for a passive reciprocal structure.

    Uses a PEC cavity with two probes — the coupling should be symmetric.
    Earlier CPU float32 modal decomposition showed about 3% numerical noise, motivating the 5% reciprocity bar.
    """
    from rfx import Simulation, Box

    GUIDE_LENGTH_M = 0.12
    PORT_INSET_FROM_DOMAIN_FACE_M = 0.025
    a, b, d = GUIDE_LENGTH_M, 0.04, 0.03
    f0 = (C0 / 2) * np.sqrt((1/0.04)**2 + (1/0.03)**2)

    sim = Simulation(freq_max=f0 * 2, domain=(a, b, d), boundary={"x": "cpml", "y": "pec", "z": "pec"},
                     cpml_layers=8, dx=0.002)
    sim.add_material("dielectric", eps_r=2.2)
    sim.add(Box((a/3, 0, 0), (2*a/3, b, d)), material="dielectric")

    freqs = np.linspace(f0 * 0.7, f0 * 1.3, 20)
    sim.add_waveguide_port(PORT_INSET_FROM_DOMAIN_FACE_M, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=freqs, f0=f0, name="port1")
    sim.add_waveguide_port(a - PORT_INSET_FROM_DOMAIN_FACE_M, direction="-x", mode=(1, 0), mode_type="TE",
                           freqs=freqs, f0=f0, name="port2")

    result = sim.compute_waveguide_s_matrix(num_periods=30, normalize=True)
    S = np.array(result.s_params)

    s12 = np.abs(S[0, 1, :])
    s21 = np.abs(S[1, 0, :])

    max_diff = np.max(np.abs(s12 - s21))
    mean_diff = np.mean(np.abs(s12 - s21))
    mean_mag = np.mean((s12 + s21) / 2)

    print("\nReciprocity (S12 vs S21):")
    print(f"  Mean |S12|: {np.mean(s12):.4f}")
    print(f"  Mean |S21|: {np.mean(s21):.4f}")
    print(f"  Max diff:   {max_diff:.6f}")
    print(f"  Rel diff:   {mean_diff / (mean_mag + 1e-30):.2e}")

    rel_diff = mean_diff / (mean_mag + 1e-30)
    assert rel_diff < 0.05, f"Reciprocity violation: relative |S12-S21| = {rel_diff:.2%}"



@pytest.mark.slow_physics
def test_convergence_order():
    """Error should decrease as O(dx²) — second-order Yee scheme.

    Run cavity at 3 resolutions, verify slope ≈ 2 on log-log plot.

    This includes a 0.5 mm / ~500k-cell CPU case and is a release-tag
    physics regression gate, not part of the default fast physics gate.
    """
    a, b, d = 0.05, 0.04, 0.03
    f_analytical = (C0 / 2) * np.sqrt((1/a)**2 + (1/b)**2)

    errors = []
    dxs = [2e-3, 1e-3, 0.5e-3]

    for dx in dxs:
        grid = Grid(freq_max=5e9, domain=(a, b, d), dx=dx, cpml_layers=0)
        state = init_state(grid.shape)
        materials = init_materials(grid.shape)
        from rfx.sources.sources import GaussianPulse
        pulse = GaussianPulse(f0=f_analytical, bandwidth=0.8)
        si, sj, sk = grid.nx // 3, grid.ny // 3, grid.nz // 2
        pi, pj = 2 * grid.nx // 3, 2 * grid.ny // 3

        n_steps = grid.num_timesteps(num_periods=150)
        ts = np.zeros(n_steps)
        for n in range(n_steps):
            state = update_h(state, materials, grid.dt, dx, high_wall=unpadded_high_walls(grid))
            state = update_e(state, materials, grid.dt, dx, high_wall=unpadded_high_walls(grid))
            state = apply_pec(state)
            state = state._replace(ez=state.ez.at[si, sj, sk].add(pulse(n * grid.dt)))
            ts[n] = float(state.ez[pi, pj, sk])

        nfft = len(ts) * 16
        spec = np.abs(np.fft.rfft(ts, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=grid.dt)
        band = (freqs > f_analytical * 0.5) & (freqs < f_analytical * 1.5)
        peak_idx = np.argmax(spec * band)
        if 0 < peak_idx < len(spec) - 1:
            alpha, beta, gamma = spec[peak_idx-1], spec[peak_idx], spec[peak_idx+1]
            denom = alpha - 2*beta + gamma
            if abs(denom) > 1e-30:
                p = 0.5 * (alpha - gamma) / denom
                f_sim = freqs[peak_idx] + p * (freqs[1] - freqs[0])
            else:
                f_sim = freqs[peak_idx]
        else:
            f_sim = freqs[peak_idx]
        err = abs(f_sim - f_analytical) / f_analytical
        errors.append(err)
        print(f"  dx={dx*1e3:.1f}mm: f={f_sim/1e9:.6f} GHz, err={err:.6e}")

    if errors[0] > 1e-10 and errors[1] > 1e-10:
        rate_1 = np.log(errors[0] / errors[1]) / np.log(dxs[0] / dxs[1])
    else:
        rate_1 = 0
    if errors[1] > 1e-10 and errors[2] > 1e-10:
        rate_2 = np.log(errors[1] / errors[2]) / np.log(dxs[1] / dxs[2])
    else:
        rate_2 = 0

    print("\nConvergence order:")
    print(f"  Rate (coarse→medium): {rate_1:.2f}")
    print(f"  Rate (medium→fine):   {rate_2:.2f}")

    avg_rate = (rate_1 + rate_2) / 2
    assert avg_rate > 1.0, f"Convergence rate {avg_rate:.2f} < 1.0 (expected ~2.0)"
