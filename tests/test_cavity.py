"""Stage 1 validation: PEC rectangular cavity resonance.

Analytical TM_mnp resonance frequency for a rectangular cavity:
  f_mnp = (c / 2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)

For TM_110 mode (m=1, n=1, p=0) with a=b=0.1m, d=0.05m:
  f_110 = (c / 2) * sqrt((1/0.1)^2 + (1/0.1)^2) = c * sqrt(2) / 0.2
        ≈ 2.1213 GHz

Exit criterion: resonant frequency error ≤ 0.5%.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse


def analytical_tm_freq(a: float, b: float, d: float, m: int, n: int, p: int) -> float:
    """Analytical TM_mnp resonance frequency for rectangular PEC cavity."""
    return (C0 / 2.0) * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)


@pytest.fixture
def cavity_params():
    """Standard test cavity: 0.1 x 0.1 x 0.05 m."""
    return dict(a=0.1, b=0.1, d=0.05)


def test_grid_creation():
    """Grid initializes with correct auto-resolution."""
    grid = Grid(freq_max=5e9, domain=(0.1, 0.1, 0.05))
    assert grid.dx <= C0 / 5e9 / 20.0
    assert grid.nx > 0 and grid.ny > 0 and grid.nz > 0
    assert grid.dt > 0


def test_analytical_tm110(cavity_params):
    """Verify analytical formula gives expected TM110 frequency."""
    f_110 = analytical_tm_freq(**cavity_params, m=1, n=1, p=0)
    expected = C0 * np.sqrt(2) / 0.2
    assert abs(f_110 - expected) / expected < 1e-10


def test_pec_cavity_resonance(cavity_params):
    """FDTD simulation of PEC cavity should match TM110 within 0.5%.

    This is the primary Stage 1 exit criterion.
    """
    a, b, d = cavity_params["a"], cavity_params["b"], cavity_params["d"]
    f_analytical = analytical_tm_freq(a, b, d, m=1, n=1, p=0)

    # Use coarser grid for test speed, finer for accuracy
    freq_max = 5e9
    grid = Grid(freq_max=freq_max, domain=(a, b, d), cpml_layers=0)

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    # Gaussian pulse source at off-center point to excite TM110
    pulse = GaussianPulse(f0=f_analytical, bandwidth=0.8)
    src_i = grid.nx // 3
    src_j = grid.ny // 3
    src_k = grid.nz // 2

    # Probe at another off-center point
    probe_i = 2 * grid.nx // 3
    probe_j = 2 * grid.ny // 3
    probe_k = grid.nz // 2

    num_steps = grid.num_timesteps(num_periods=40)
    dt = grid.dt
    dx = grid.dx

    # Time-stepping loop (no scan for easier debugging)
    time_series = np.zeros(num_steps)

    for n in range(num_steps):
        t = n * dt

        # H update
        state = update_h(state, materials, dt, dx)
        # E update
        state = update_e(state, materials, dt, dx)
        # PEC boundaries (cavity walls)
        state = apply_pec(state)

        # Source injection
        src_value = pulse(t)
        ez = state.ez.at[src_i, src_j, src_k].add(src_value)
        state = state._replace(ez=ez)

        # Record probe
        time_series[n] = float(state.ez[probe_i, probe_j, probe_k])

    # FFT to find resonant frequency
    spectrum = np.abs(np.fft.rfft(time_series))
    freqs = np.fft.rfftfreq(num_steps, d=dt)

    # Find peak near expected resonance (skip DC)
    search_lo = f_analytical * 0.5
    search_hi = f_analytical * 1.5
    mask = (freqs >= search_lo) & (freqs <= search_hi)
    masked_spectrum = np.where(mask, spectrum, 0.0)
    peak_idx = np.argmax(masked_spectrum)
    f_fdtd = freqs[peak_idx]

    error = abs(f_fdtd - f_analytical) / f_analytical
    print(f"Analytical: {f_analytical / 1e9:.4f} GHz")
    print(f"FDTD:       {f_fdtd / 1e9:.4f} GHz")
    print(f"Error:      {error * 100:.2f}%")

    assert error < 0.005, f"Resonance error {error*100:.2f}% exceeds 0.5% threshold"


def test_pec_zeros_boundary():
    """PEC should zero tangential E at boundaries."""
    grid = Grid(freq_max=3e9, domain=(0.05, 0.05, 0.05), cpml_layers=0)
    state = init_state(grid.shape)

    # Set non-zero fields
    state = state._replace(
        ex=jnp.ones(grid.shape),
        ey=jnp.ones(grid.shape),
        ez=jnp.ones(grid.shape),
    )

    state = apply_pec(state)

    # Check tangential components at boundaries are zero
    assert float(jnp.max(jnp.abs(state.ey[0, :, :]))) == 0.0
    assert float(jnp.max(jnp.abs(state.ey[-1, :, :]))) == 0.0
    assert float(jnp.max(jnp.abs(state.ez[0, :, :]))) == 0.0
    assert float(jnp.max(jnp.abs(state.ez[-1, :, :]))) == 0.0


def test_energy_conservation_pec():
    """In a lossless PEC cavity, total EM energy should be approximately conserved."""
    grid = Grid(freq_max=3e9, domain=(0.05, 0.05, 0.05), cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    # Inject initial energy
    pulse = GaussianPulse(f0=2e9)
    src_i, src_j, src_k = grid.nx // 2, grid.ny // 2, grid.nz // 2

    dt = grid.dt
    dx = grid.dx

    # Run source until pulse fully decays (t > 6*tau ≈ negligible)
    n_source = 100
    for n in range(n_source):
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        t = n * dt
        ez = state.ez.at[src_i, src_j, src_k].add(pulse(t))
        state = state._replace(ez=ez)

    # Let transients settle (no source)
    for _ in range(50):
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)

    # Measure energy after source fully off
    def em_energy(s):
        EPS_0 = 8.854187817e-12
        MU_0 = 1.2566370614e-6
        e_energy = 0.5 * EPS_0 * (s.ex**2 + s.ey**2 + s.ez**2).sum()
        h_energy = 0.5 * MU_0 * (s.hx**2 + s.hy**2 + s.hz**2).sum()
        return float(e_energy + h_energy)

    energy_start = em_energy(state)

    # Run 200 more steps without source
    for _ in range(200):
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)

    energy_end = em_energy(state)

    # Energy should be conserved within 1%
    drift = abs(energy_end - energy_start) / energy_start
    print(f"Energy drift: {drift * 100:.3f}%")
    assert drift < 0.01, f"Energy drift {drift*100:.2f}% exceeds 1%"
