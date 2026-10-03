"""Thevenin source-voltage drive for lumped and wire ports."""

import jax
import jax.numpy as jnp

from rfx.core.yee import cell_component_e_coeffs
from rfx.sources.sources import port_dual_transverse


def port_drive_waveform(grid, cell, component, excitation, n_steps, materials,
                        *, impedance, n_live=1, time=None):
    """Return ``Cb * w(t) / (R * A_dual)`` for one live edge.

    ``w`` is the open-circuit Thevenin source voltage across the WHOLE
    port; ``impedance`` is its total series resistance R in ohms. A wire
    has n_live series edges, each stamped with resistance R/n_live and
    assigned source voltage w/n_live. Its Norton current is therefore
    w/R on every live edge: no additional 1/n_live belongs in this drive.
    A_dual is that edge's transverse dual area, including graded metrics.

    Read Cb from the realized E-update materials, including overrides and
    port/RLC stamps; do not reconstruct permittivity. JAX operations retain
    material and mesh derivatives. ``n_steps=None, time=t`` selects the
    eager single-step application instead of a precomputed waveform.
    """
    cb = cell_component_e_coeffs(materials, cell, component, grid.dt)[1]
    if getattr(grid, "dx_arr", None) is not None:
        from rfx.nonuniform import (
            e_node_dual_spacing_at, port_metric, port_metric_axes,
        )
        axis = {"ex": 0, "ey": 1, "ez": 2}[component]
        transverse = [
            port_metric(e_node_dual_spacing_at(widths, cell[a]), host)
            for a, (widths, host) in enumerate(port_metric_axes(grid))
            if a != axis
        ]
    else:
        transverse = port_dual_transverse(grid, cell, component)
    area = transverse[0] * transverse[1]
    if n_steps is None:
        samples = excitation(time)
    else:
        times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
        samples = jax.vmap(excitation)(times)
    return (cb / (impedance * area)) * samples
