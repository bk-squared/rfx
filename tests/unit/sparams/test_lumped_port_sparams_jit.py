"""JIT-integrated lumped-port S-parameter path (issue #72).

Validates the AD-friendly `Simulation.forward(port_s11_freqs=...)` path
introduced for issue #72:

1. PEC cavity (lossless) → |S11| ≈ 1 in the source bandwidth.
2. JIT path agrees with the canonical Python-loop `extract_s11`
   accumulator (tests/unit/sparams/test_sparam.py reference) within physical tolerance
   on the same setup.
3. `jax.grad` flows through `minimize_s11_at_freq_wave_decomp` to
   `eps_override` (the optimisation use case from issue #72).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse


def _build_cavity_sim():
    """Closed PEC box with a single lumped port at the centre."""
    a, b, d = 0.05, 0.05, 0.025
    sim = Simulation(
        freq_max=5e9,
        domain=(a, b, d),
        dx=2.5e-3,
        boundary="pec",
    )
    sim.add_port(
        position=(a / 2, b / 2, d / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0),
    )
    return sim


def test_pec_cavity_s11_magnitude_near_one():
    """Closed PEC cavity → all power reflected → |S11| ≈ 1 in band."""
    sim = _build_cavity_sim()
    freqs = jnp.linspace(1.5e9, 4.5e9, 9, dtype=jnp.float32)

    result = sim.forward(
        num_periods=60,
        port_s11_freqs=freqs,
        skip_preflight=True,
    )

    s11 = np.array(result.s_params)
    mag = np.abs(s11)
    # Lossless PEC cavity: every frequency must be near full reflection.
    assert np.all(mag > 0.85), f"|S11| dropped below 0.85 in band: {mag}"
    assert np.all(mag < 1.10), f"|S11| > 1.10 (unphysical overshoot): {mag}"


def test_jit_path_matches_python_loop_extractor():
    """JIT-integrated path must agree with the canonical extract_s11 path."""
    from rfx.grid import Grid
    from rfx.core.yee import init_state, init_materials, update_e, update_h
    from rfx.boundaries.pec import apply_pec
    from rfx.sources.sources import LumpedPort, setup_lumped_port, apply_lumped_port
    from rfx.probes.probes import (
        init_sparam_probe, update_sparam_probe, extract_s11,
    )

    a, b, d = 0.05, 0.05, 0.025
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0)
    grid = Grid(freq_max=5e9, domain=(a, b, d), dx=2.5e-3, cpml_layers=0)
    port_pos = (a / 2, b / 2, d / 2)
    port = LumpedPort(
        position=port_pos, component="ez",
        impedance=50.0, excitation=pulse,
    )

    materials = init_materials(grid.shape)
    materials = setup_lumped_port(grid, port, materials)

    freqs = jnp.linspace(1.5e9, 4.5e9, 9, dtype=jnp.float32)
    n_steps = grid.num_timesteps(num_periods=60)
    sprobe = init_sparam_probe(grid, port, freqs, dft_total_steps=n_steps)
    state = init_state(grid.shape)
    dt, dx = grid.dt, grid.dx

    for n in range(n_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        sprobe = update_sparam_probe(sprobe, state, grid, port, dt)
        state = apply_lumped_port(state, grid, port, t, materials)

    s11_loop = np.array(extract_s11(sprobe, z0=50.0))

    # JIT path through Simulation.forward
    sim = _build_cavity_sim()
    result = sim.forward(
        num_periods=60,
        port_s11_freqs=freqs,
        skip_preflight=True,
    )
    s11_jit = np.array(result.s_params)

    # Both paths use the same V/I wave-decomposition formula on equivalent
    # measurement cells; expect close numerical agreement (tolerance set to
    # absorb ordering/precision differences between scan-body and Python
    # accumulators).
    diff = np.abs(s11_jit - s11_loop)
    assert np.max(diff) < 0.05, (
        f"JIT vs Python-loop disagree more than 0.05: "
        f"max_diff={np.max(diff):.4f}, jit={s11_jit}, loop={s11_loop}"
    )


def test_gradient_flows_through_wave_decomp_objective():
    """`jax.grad` of the new objective wrt `eps_override` returns a finite gradient."""
    from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp

    sim = _build_cavity_sim()
    freqs = jnp.linspace(2.0e9, 4.0e9, 5, dtype=jnp.float32)
    target = 3.0e9
    obj = minimize_s11_at_freq_wave_decomp(target_freq=target, port_idx=0)

    # Build a baseline eps_override of the right shape.
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss_fn(eps_arr):
        result = sim.forward(
            eps_override=eps_arr,
            num_periods=20,  # short — gradient finiteness, not accuracy
            port_s11_freqs=freqs,
            skip_preflight=True,
        )
        return obj(result)

    grad_fn = jax.grad(loss_fn)
    g = grad_fn(eps_base)
    g_np = np.array(g)
    assert np.all(np.isfinite(g_np)), "gradient contains NaN/Inf"
    assert np.linalg.norm(g_np) > 0.0, "gradient is identically zero"
