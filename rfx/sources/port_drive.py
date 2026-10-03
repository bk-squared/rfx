"""Stamped-load source-voltage drive for ports."""

import jax
import jax.numpy as jnp

from rfx.core.yee import cell_component_e_coeffs


def stamped_drive(port, cell):
    """The ``(sigma_port, unit_field)`` the port's setup stamped at ``cell``.

    A port is driven through the load its setup stamped; a drive built for a
    cell with no stamp would have no source impedance to stand behind, so it
    is refused with a message naming the missing step.
    """
    try:
        return port._drive_stamps[tuple(int(c) for c in cell)]
    except KeyError:
        raise RuntimeError(
            f"port drive at cell {tuple(int(c) for c in cell)} has no stamped "
            "load: the port's setup (setup_lumped_port / setup_wire_port / "
            "setup_msl_port) must stamp its conductance before its drive is "
            "built") from None


def port_drive_waveform(grid, cell, component, excitation, n_steps, materials,
                        *, sigma_port, unit_field, time=None):
    """Return ``Cb * sigma_port * unit_field * w(t)`` for one live edge.

    ``w`` is the open-circuit Thevenin source voltage across the WHOLE
    port. ``sigma_port`` is this port's own stamped load, excluding other
    ports, RLC elements and bulk conductivity. ``unit_field`` is its
    unit-voltage field shape in inverse metres. A wire
    has n_live series edges, each stamped with resistance R/n_live and
    assigned source voltage w/n_live. Its Norton current is therefore
    w/R on every live edge: no additional 1/n_live belongs in this drive.

    Read Cb from the realized E-update materials, including overrides and
    port/RLC stamps; do not reconstruct permittivity. JAX operations retain
    material and mesh derivatives. ``n_steps=None, time=t`` selects the
    eager single-step application instead of a precomputed waveform.
    """
    cb = cell_component_e_coeffs(materials, cell, component, grid.dt)[1]
    if n_steps is None:
        samples = excitation(time)
    else:
        times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
        samples = jax.vmap(excitation)(times)
    return (cb * sigma_port * unit_field) * samples
