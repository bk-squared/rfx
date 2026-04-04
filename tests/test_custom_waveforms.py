"""Tests for CW and custom waveform sources (Spec 6C).

Validates that:
1. CWSource reaches steady-state after ramp-up
2. CWSource DFT peak is at f0 with negligible harmonic content
3. CustomWaveform exactly reproduces a GaussianPulse
4. CWSource is differentiable through the FDTD simulation
"""

import numpy as np
import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays, init_materials
from rfx.sources.sources import GaussianPulse, CWSource, CustomWaveform
from rfx.simulation import make_source, make_probe, run


def test_cw_source_reaches_steady_state():
    """After ramp-up, CW envelope should be approximately constant."""
    grid = Grid(freq_max=10e9, domain=(0.03, 0.01, 0.01), cpml_layers=6)
    materials = init_materials(grid.shape)
    n_steps = 500

    f0 = 5e9
    ramp = 20
    cw = CWSource(f0=f0, amplitude=1.0, ramp_steps=ramp)
    src_pos = (0.005, 0.005, 0.005)
    prb_pos = (0.015, 0.005, 0.005)

    src = make_source(grid, src_pos, "ez", cw, n_steps)
    prb = make_probe(grid, prb_pos, "ez")

    result = run(grid, materials, n_steps, boundary="cpml",
                 sources=[src], probes=[prb])

    ts = np.array(result.time_series[:, 0])

    # Compute envelope via analytic signal (Hilbert transform)
    from numpy.fft import fft, ifft
    n = len(ts)
    spectrum = fft(ts)
    # Zero out negative frequencies to form analytic signal
    spectrum[n // 2 + 1:] = 0.0
    spectrum[1:n // 2] *= 2.0
    envelope = np.abs(ifft(spectrum))

    # Check the last 100 steps for steady-state
    tail = envelope[-100:]
    cv = np.std(tail) / (np.mean(tail) + 1e-30)
    print(f"Envelope CV (last 100 steps): {cv:.4f}")
    assert cv < 0.15, (
        f"CW envelope not steady: CV={cv:.4f} (need < 0.15)"
    )


def test_cw_source_dft_peak_at_f0():
    """DFT of CW probe signal should peak at f0, with harmonics 10x smaller."""
    grid = Grid(freq_max=10e9, domain=(0.03, 0.01, 0.01), cpml_layers=6)
    materials = init_materials(grid.shape)
    n_steps = 500

    f0 = 5e9
    cw = CWSource(f0=f0, amplitude=1.0, ramp_steps=20)
    src_pos = (0.005, 0.005, 0.005)
    prb_pos = (0.015, 0.005, 0.005)

    src = make_source(grid, src_pos, "ez", cw, n_steps)
    prb = make_probe(grid, prb_pos, "ez")

    result = run(grid, materials, n_steps, boundary="cpml",
                 sources=[src], probes=[prb])

    ts = np.array(result.time_series[:, 0])
    dt = grid.dt

    # Compute DFT magnitudes at f0 and 2*f0
    times = np.arange(n_steps) * dt
    dft_f0 = np.abs(np.sum(ts * np.exp(-1j * 2 * np.pi * f0 * times)))
    dft_2f0 = np.abs(np.sum(ts * np.exp(-1j * 2 * np.pi * 2 * f0 * times)))

    ratio = dft_f0 / (dft_2f0 + 1e-30)
    print(f"|DFT(f0)|={dft_f0:.4e}, |DFT(2f0)|={dft_2f0:.4e}, ratio={ratio:.1f}")
    assert ratio > 10.0, (
        f"DFT peak ratio f0/2f0 = {ratio:.1f}, expected > 10"
    )


def test_custom_waveform_matches_gaussian():
    """CustomWaveform wrapping GaussianPulse should produce identical results."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), cpml_layers=0)
    materials = init_materials(grid.shape)
    n_steps = 200

    f0 = 3e9
    pulse = GaussianPulse(f0=f0)
    custom = CustomWaveform(func=GaussianPulse(f0=f0))

    src_pos = (0.015, 0.015, 0.015)
    prb_pos = (0.02, 0.02, 0.02)

    src_pulse = make_source(grid, src_pos, "ez", pulse, n_steps)
    src_custom = make_source(grid, src_pos, "ez", custom, n_steps)
    prb = make_probe(grid, prb_pos, "ez")

    result_pulse = run(grid, materials, n_steps,
                       sources=[src_pulse], probes=[prb])
    result_custom = run(grid, materials, n_steps,
                        sources=[src_custom], probes=[prb])

    ts_pulse = np.array(result_pulse.time_series[:, 0])
    ts_custom = np.array(result_custom.time_series[:, 0])

    np.testing.assert_allclose(
        ts_custom, ts_pulse, atol=1e-6,
        err_msg="CustomWaveform(GaussianPulse) differs from GaussianPulse",
    )
    print(f"Max diff: {np.max(np.abs(ts_custom - ts_pulse)):.2e}")


def test_cw_source_differentiable():
    """CW source through checkpointed run() should produce non-zero gradient."""
    grid = Grid(freq_max=10e9, domain=(0.015, 0.015, 0.015), cpml_layers=0)
    n_steps = 30

    f0 = 5e9
    cw = CWSource(f0=f0, amplitude=1.0, ramp_steps=10)
    src = make_source(grid, (0.005, 0.0075, 0.0075), "ez", cw, n_steps)
    prb = make_probe(grid, (0.01, 0.0075, 0.0075), "ez")

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(eps_r):
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        result = run(grid, mats, n_steps, sources=[src], probes=[prb],
                     checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    grad = jax.grad(objective)(eps_r)

    grad_max = float(jnp.max(jnp.abs(grad)))
    print(f"Max |grad|: {grad_max:.4e}")
    assert grad_max > 0.0, "Gradient is all zeros — AD chain is broken"
