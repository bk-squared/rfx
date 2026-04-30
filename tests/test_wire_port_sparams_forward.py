"""Forward-path WirePort S-parameter wiring (issue #79 follow-up).

Verifies that ``Simulation.forward(port_s11_freqs=...)`` populates
``ForwardResult.s_params`` AND ``ForwardResult.wire_port_sparams`` when
the port is added with ``extent=...`` (multi-cell WirePort, not
single-cell lumped). Pre-fix the api dispatch only registered
``LumpedPortSParamSpec`` for lumped ports and silently dropped wire
ports, so ``result.s_params`` was None and downstream AD objectives
on wire-port arrays were unreachable from the JIT scan path.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse


def _build_wire_port_cavity():
    """Closed PEC box with a WirePort (extent spans most of the z-axis)."""
    a, b, d = 0.05, 0.05, 0.025
    sim = Simulation(
        freq_max=5e9,
        domain=(a, b, d),
        dx=2.5e-3,
        boundary="pec",
    )
    sim.add_port(
        position=(a / 2, b / 2, 0.005),
        component="ez",
        impedance=50.0,
        extent=0.015,                 # 6 cells along z → WirePort, not lumped
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0),
    )
    return sim


def test_forward_wire_port_sparams_populated():
    """forward(port_s11_freqs=...) must populate result.s_params + wire_port_sparams."""
    sim = _build_wire_port_cavity()
    freqs = jnp.linspace(1.5e9, 4.5e9, 7, dtype=jnp.float32)
    result = sim.forward(
        num_periods=40,
        port_s11_freqs=freqs,
        skip_preflight=True,
    )
    assert result.wire_port_sparams is not None, (
        "wire_port_sparams was None — api dispatch did not register the "
        "WirePortSParamSpec when port was created with extent=...")
    assert len(result.wire_port_sparams) == 1, (
        f"expected 1 wire port spec, got {len(result.wire_port_sparams)}")
    spec, accs = result.wire_port_sparams[0]
    v_dft, i_dft, v_inc_dft = accs
    assert v_dft.shape == (7,), v_dft.shape
    assert i_dft.shape == (7,), i_dft.shape
    assert result.s_params is not None, (
        "s_params still None — wire-port wave-decomp extraction did not run")
    s11 = np.asarray(result.s_params)
    assert s11.shape == (7,), s11.shape
    # Closed PEC cavity → energy reflects → |S11| should be near 1 in band
    mag = np.abs(s11)
    assert np.all(mag > 0.6), f"|S11| too low for PEC cavity: {mag}"
    assert np.all(mag < 1.20), f"|S11| > 1.20 (unphysical): {mag}"


def test_forward_wire_port_grad_flows():
    """jax.grad through forward(..., port_s11_freqs=...) must be finite for WirePort."""
    import jax

    sim = _build_wire_port_cavity()
    freqs = jnp.linspace(2.0e9, 4.0e9, 5, dtype=jnp.float32)
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss(eps):
        result = sim.forward(
            eps_override=eps,
            num_periods=15,                  # short — gradient finiteness only
            port_s11_freqs=freqs,
            skip_preflight=True,
        )
        # Mean |S11|² across the band — exercises the wave-decomp formula
        s11 = result.s_params
        return jnp.mean(jnp.abs(s11) ** 2)

    g = jax.grad(loss)(eps_base)
    g_np = np.asarray(g)
    assert np.all(np.isfinite(g_np)), "wire-port AD gradient has NaN/Inf"
    assert np.linalg.norm(g_np) > 0.0, "wire-port AD gradient is identically zero"
