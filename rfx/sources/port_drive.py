"""Voltage drive shared by lumped and wire ports on every admitted lane."""

import jax
import jax.numpy as jnp

from rfx.core.yee import cell_component_e_coeffs
from rfx.sources.sources import port_d_parallel


def port_drive_waveform(grid, cell, component, excitation, n_steps, materials,
                        *, n_live=1, time=None):
    """Return ``(Cb / d_parallel) * w(t) / n_live`` for one live edge.

    ``materials`` must be the realized drive materials, including overrides
    and edge-owned port/RLC stamps. Read the E update's own coefficient;
    a port waveform is a gap voltage, not a current-moment source. Keeping
    the operations in JAX preserves material and mesh derivatives.
    ``n_steps=None, time=t`` builds the same increment for the eager
    single-step port applicators instead of a precomputed table.
    """
    cb = cell_component_e_coeffs(materials, cell, component, grid.dt)[1]
    d_parallel = port_d_parallel(grid, cell, component)
    if n_steps is None:
        samples = excitation(time)
    else:
        times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
        samples = jax.vmap(excitation)(times)
    return (cb / d_parallel) * samples / n_live
