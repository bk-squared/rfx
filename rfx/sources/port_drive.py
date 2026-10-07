"""Stamped-load source-voltage drive for ports."""

import jax
import jax.numpy as jnp

from rfx.model.materials import e_update_coefficient_at, electric_grid_kwargs


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
    cb = e_update_coefficient_at(materials, cell, component, grid.dt, **electric_grid_kwargs(grid))
    if n_steps is None:
        samples = excitation(time)
    else:
        times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
        samples = jax.vmap(excitation)(times)
    return (cb * sigma_port * unit_field) * samples


class PortSourceQueue(list):
    """Preserve source order while deferring port builders until realization.

    Ordinary source specs can be appended/extended as usual. A deferred builder
    receives final materials at resolve time; its other arguments (including
    pre-clearing wire masks) are captured at registration time.
    """

    def defer(self, builder, *args, **kwargs):
        from functools import partial
        self.append(partial(builder, *args, **kwargs))

    def resolve(self, materials):
        from functools import partial
        sources = []
        for entry in self:
            if isinstance(entry, partial):
                built = entry(materials=materials)
                sources.extend(built if isinstance(built, list) else [built])
            else:
                sources.append(entry)
        return sources


def soft_source(grid, position, component, waveform, n_steps, *, materials,
                amplitude_kind=None, raw=False, field_dtype=None):
    """Build a soft source after the final material and load realization."""
    from rfx.simulation import make_source, make_j_source
    from rfx.api._source_semantics import guard_float16_source_increment
    builder = make_source if raw else make_j_source
    source = builder(grid, position, component, waveform, n_steps,
                     materials=materials, amplitude_kind=amplitude_kind)
    return source._replace(waveform=guard_float16_source_increment(
        source.waveform, field_dtype, amplitude_kind))
