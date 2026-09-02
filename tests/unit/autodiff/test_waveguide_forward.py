"""Regression test for issue #29: waveguide port + forward() broadcast error.

`_forward_from_materials` previously called the low-level ``run`` without
forwarding ``grid.cpml_axes``, so CPML state was built for all three
axes even when the grid had no CPML padding on the waveguide's
propagation axis. That produced ``(n_cpml, 1, 1) vs (nx, ny, nz)``
broadcast errors during the scan whenever a waveguide port was present.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


def _wr90_sim() -> Simulation:
    sim = Simulation(
        freq_max=12e9,
        domain=(0.05, 0.02286, 0.01016),
        dx=2e-3,
        boundary="cpml",
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        direction="+x",
        x_position=0.01,
        y_range=(0.0, 0.02286),
        z_range=(0.0, 0.01016),
        n_modes=1,
    )
    sim.add_probe(position=(0.03, 0.01143, 0.00508), component="ey")
    return sim


def test_forward_with_waveguide_port_no_broadcast_error():
    """forward() must complete without TypeError on CPML multiplication."""
    sim = _wr90_sim()
    result = sim.forward(n_steps=40)
    assert result.time_series.shape == (40, 1)
    assert bool(jnp.all(jnp.isfinite(result.time_series)))


def test_forward_with_waveguide_port_is_differentiable():
    """jax.grad through a waveguide-port forward must return a finite
    (possibly non-zero) value — the fix for #29 restores the gradient
    path that was masked by the broadcast failure.
    """
    sim = _wr90_sim()
    grid = sim._build_grid()
    eps0 = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss(alpha):
        r = sim.forward(eps_override=eps0 * alpha, n_steps=40)
        return jnp.sum(jnp.abs(r.time_series) ** 2)

    value = float(loss(jnp.float32(1.0)))
    assert np.isfinite(value)
    grad = float(jax.grad(loss)(jnp.float32(1.0)))
    assert np.isfinite(grad)
