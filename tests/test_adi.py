"""Tests for ADI-FDTD 2D TMz solver.

Validates:
  1. Unconditional stability — fields stay bounded at 5x and 10x CFL limit.
  2. Cavity resonance accuracy — eigenfrequency within 2% of analytical at 5x CFL.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import EPS_0, MU_0
from rfx import Simulation, ADIState2D, Box
from rfx.adi import thomas_solve, run_adi_2d

# Speed of light
C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


# ---------------------------------------------------------------------------
# Helper: Thomas solver unit test
# ---------------------------------------------------------------------------

class TestThomasSolve:
    """Basic correctness of the tridiagonal Thomas solver."""

    def test_simple_system(self):
        """Solve a known 4x4 tridiagonal system."""
        # System:  [2 -1  0  0] [x0]   [1]
        #          [-1 2 -1  0] [x1] = [0]
        #          [0 -1  2 -1] [x2]   [0]
        #          [0  0 -1  2] [x3]   [1]
        a = jnp.array([0.0, -1.0, -1.0, -1.0])
        b = jnp.array([2.0, 2.0, 2.0, 2.0])
        c = jnp.array([-1.0, -1.0, -1.0, 0.0])
        d = jnp.array([1.0, 0.0, 0.0, 1.0])

        x = thomas_solve(a, b, c, d)

        # Build dense matrix and solve for reference
        A = np.diag([2.0]*4) + np.diag([-1.0]*3, 1) + np.diag([-1.0]*3, -1)
        x_ref = np.linalg.solve(A, np.array([1.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(np.array(x), x_ref, atol=1e-6)

    def test_random_system(self):
        """Solve a random diagonally-dominant tridiagonal system."""
        rng = np.random.default_rng(42)
        N = 50
        a_np = rng.standard_normal(N)
        c_np = rng.standard_normal(N)
        b_np = np.abs(a_np) + np.abs(c_np) + 1.0  # diag dominant
        d_np = rng.standard_normal(N)

        x = thomas_solve(jnp.array(a_np), jnp.array(b_np),
                         jnp.array(c_np), jnp.array(d_np))

        # Dense reference
        A = np.diag(b_np) + np.diag(a_np[1:], -1) + np.diag(c_np[:-1], 1)
        x_ref = np.linalg.solve(A, d_np)
        np.testing.assert_allclose(np.array(x), x_ref, atol=1e-5)

    def test_jit_compatible(self):
        """Thomas solver works under jax.jit."""
        a = jnp.array([0.0, -1.0, -1.0])
        b = jnp.array([2.0, 2.0, 2.0])
        c = jnp.array([-1.0, -1.0, 0.0])
        d = jnp.array([1.0, 1.0, 1.0])

        x_jit = jax.jit(thomas_solve)(a, b, c, d)
        x_nojit = thomas_solve(a, b, c, d)
        np.testing.assert_allclose(np.array(x_jit), np.array(x_nojit), atol=1e-7)

    def test_differentiable(self):
        """Gradient flows through the Thomas solver."""
        def loss(d_vec):
            x = thomas_solve(
                jnp.array([0.0, -1.0, -1.0]),
                jnp.array([2.0, 2.0, 2.0]),
                jnp.array([-1.0, -1.0, 0.0]),
                d_vec,
            )
            return jnp.sum(x ** 2)

        d_vec = jnp.array([1.0, 0.0, 1.0])
        grad = jax.grad(loss)(d_vec)
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# ADI stability tests
# ---------------------------------------------------------------------------

def _cfl_dt_2d(dx: float, dy: float) -> float:
    """CFL-limited dt for standard 2D FDTD."""
    return 1.0 / (C0 * np.sqrt(1.0/dx**2 + 1.0/dy**2))


def _run_adi_stability(cfl_factor: float, n_steps: int = 300):
    """Run ADI at cfl_factor times the CFL limit; return max |Ez|."""
    Nx, Ny = 40, 40
    dx = dy = 0.01  # 1 cm cells
    dt_cfl = _cfl_dt_2d(dx, dy)
    dt = cfl_factor * dt_cfl

    ez = jnp.zeros((Nx, Ny))
    hx = jnp.zeros((Nx, Ny))
    hy = jnp.zeros((Nx, Ny))
    eps_r = jnp.ones((Nx, Ny))
    sigma = jnp.zeros((Nx, Ny))

    # Gaussian pulse source at centre
    t_arr = jnp.arange(n_steps) * dt
    tau = 20.0 * dt
    t0 = 5.0 * tau
    waveform = jnp.exp(-((t_arr - t0) / tau) ** 2)

    src_i, src_j = Nx // 2, Ny // 2
    sources = [(src_i, src_j, waveform)]
    probes = [(src_i, src_j)]

    ez_f, hx_f, hy_f, probe_data = run_adi_2d(
        ez, hx, hy, eps_r, sigma, dt, dx, dy, n_steps, sources, probes
    )
    return ez_f, probe_data


def test_adi_stability_beyond_cfl():
    """ADI fields must remain bounded at 5x and 10x CFL limit."""
    for factor in [5.0, 10.0]:
        ez_f, probe_data = _run_adi_stability(factor, n_steps=200)
        max_ez = float(jnp.max(jnp.abs(ez_f)))
        max_probe = float(jnp.max(jnp.abs(probe_data)))

        # Fields must be finite and not blow up.
        # A reasonable bound: max |Ez| < 100 (source amplitude ~ 1).
        assert np.isfinite(max_ez), f"Non-finite Ez at {factor}x CFL"
        assert max_ez < 100.0, f"|Ez| = {max_ez} blew up at {factor}x CFL"
        assert np.isfinite(max_probe), f"Non-finite probe at {factor}x CFL"


# ---------------------------------------------------------------------------
# Cavity resonance test
# ---------------------------------------------------------------------------

def test_adi_cavity_resonance():
    """PEC cavity eigenfrequency from ADI-FDTD should match analytical within 2%.

    2D TMz PEC cavity of size a x b.
    Analytical TM_mn resonance: f_mn = (C0/2) * sqrt((m/a)^2 + (n/b)^2)
    We target the TM_11 mode.
    """
    a = 0.1   # 10 cm
    b = 0.1   # 10 cm
    f_analytical = (C0 / 2.0) * np.sqrt((1.0/a)**2 + (1.0/b)**2)

    # Grid
    dx = dy = 0.002   # 2 mm  → 50 x 50 cells
    Nx = int(round(a / dx))
    Ny = int(round(b / dy))

    # CFL timestep and ADI timestep (5x CFL)
    dt_cfl = _cfl_dt_2d(dx, dy)
    cfl_factor = 5.0
    dt = cfl_factor * dt_cfl

    # Enough steps to capture several cycles of the TM11 mode
    # f_11 ~ 2.12 GHz → period ~ 0.47 ns
    # With dt = 5 * CFL ~ 5 * 4.71 ps ~ 23.6 ps, need ~20 steps/period
    # Run ~60 periods → ~1200 steps
    n_steps = 1500

    ez = jnp.zeros((Nx, Ny))
    hx = jnp.zeros((Nx, Ny))
    hy = jnp.zeros((Nx, Ny))
    eps_r = jnp.ones((Nx, Ny))
    sigma = jnp.zeros((Nx, Ny))

    # Broadband Gaussian pulse at an off-centre point (excites TM11)
    src_i = Nx // 3
    src_j = Ny // 3
    t_arr = jnp.arange(n_steps) * dt
    # Short pulse to excite wide bandwidth
    tau = 10.0 * dt
    t0 = 4.0 * tau
    waveform = jnp.exp(-((t_arr - t0) / tau) ** 2)

    sources = [(src_i, src_j, waveform)]
    # Probe at another off-centre point to capture all modes
    prb_i = 2 * Nx // 3
    prb_j = 2 * Ny // 3
    probes = [(prb_i, prb_j)]

    _, _, _, probe_data = run_adi_2d(
        ez, hx, hy, eps_r, sigma, dt, dx, dy, n_steps, sources, probes
    )

    # Extract probe time series
    signal = np.array(probe_data[:, 0])

    # Skip the initial excitation transient
    skip = int(n_steps * 0.2)
    signal_late = signal[skip:]

    # FFT to find resonant frequency
    N_fft = len(signal_late)
    freqs = np.fft.rfftfreq(N_fft, d=float(dt))
    spectrum = np.abs(np.fft.rfft(signal_late))

    # Find the dominant peak (skip DC)
    spectrum[0] = 0.0
    peak_idx = np.argmax(spectrum)
    f_peak = freqs[peak_idx]

    # Check within 2% of analytical
    rel_error = abs(f_peak - f_analytical) / f_analytical
    assert rel_error < 0.02, (
        f"Cavity resonance {f_peak/1e9:.4f} GHz vs analytical "
        f"{f_analytical/1e9:.4f} GHz — error {rel_error*100:.2f}% > 2%"
    )


# ---------------------------------------------------------------------------
# High-level Simulation integration
# ---------------------------------------------------------------------------


def test_simulation_adi_run_high_level():
    """Simulation.run should expose ADI as a real solver path."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.01),
        boundary="pec",
        mode="2d_tmz",
        solver="adi",
        adi_cfl_factor=5.0,
    )
    sim.add_source((0.01, 0.01, 0.0), "ez")
    sim.add_probe((0.01, 0.01, 0.0), "ez")
    sim.add_probe((0.01, 0.01, 0.0), "hx")

    result = sim.run(n_steps=40)

    assert isinstance(result.state, ADIState2D)
    assert result.state.ez.shape == (result.grid.nx, result.grid.ny)
    assert result.time_series.shape == (40, 2)
    assert result.dt == pytest.approx(sim._build_grid().dt * 5.0)


def test_simulation_adi_forward_contract():
    """Simulation.forward should work for ADI without exposing state."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.01),
        boundary="pec",
        mode="2d_tmz",
        solver="adi",
    )
    sim.add_source((0.01, 0.01, 0.0), "ez")
    sim.add_probe((0.01, 0.01, 0.0), "ez")

    result = sim.forward(n_steps=20)

    assert result.time_series.shape == (20, 1)
    assert result.ntff_data is None
    assert result.ntff_box is None
    assert result.grid is not None
    assert not hasattr(result, "state")


def test_simulation_adi_internal_pec_geometry_masks_ez():
    """Internal PEC geometry should be enforced through the ADI path."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.01),
        boundary="pec",
        mode="2d_tmz",
        solver="adi",
    )
    sim.add(Box((0.008, 0.008, 0.0), (0.012, 0.012, 0.01)), material="pec")
    sim.add_source((0.01, 0.01, 0.0), "ez")
    sim.add_probe((0.01, 0.01, 0.0), "ez")

    result = sim.run(n_steps=20)

    assert float(jnp.max(jnp.abs(result.time_series))) == pytest.approx(0.0)


def test_simulation_adi_rejects_unsupported_configs():
    """ADI path should fail explicitly on unsupported physics/configs."""
    with pytest.raises(ValueError, match="mode='2d_tmz'"):
        Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), boundary="pec", solver="adi")

    with pytest.raises(ValueError, match="boundary='pec'"):
        Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.01), boundary="cpml", mode="2d_tmz", solver="adi")

    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.01),
        boundary="pec",
        mode="2d_tmz",
        solver="adi",
    )
    sim.add_material("lossy", eps_r=2.2, sigma=0.1)
    sim.add(Box((0.005, 0.005, 0.0), (0.015, 0.015, 0.01)), material="lossy")
    sim.add_source((0.01, 0.01, 0.0), "ez")

    with pytest.raises(ValueError, match="lossless materials"):
        sim.run(n_steps=10)


def test_simulation_adi_rejects_port_loaded_excitation():
    """ADI integration should reject unsupported impedance-loaded ports."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.01),
        boundary="pec",
        mode="2d_tmz",
        solver="adi",
    )
    sim.add_port((0.01, 0.01, 0.0), "ez", impedance=50.0)

    with pytest.raises(ValueError, match="soft Ez sources"):
        sim.run(n_steps=10)
