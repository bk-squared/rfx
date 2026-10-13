"""Stored electric records from the discrete incident line."""
import jax.numpy as jnp
from jax import lax

from rfx.core.yee import FDTDState


def record_incident_line(cfg, n_steps, dt, indices, weights):
    """Advance a zero float32 line H then E and sample its weighted E field."""
    from rfx.sources.tfsf_discrete_plane_wave import (
        COMPONENTS, update_line_e, update_line_h,
    )

    zero = FDTDState(*(jnp.zeros(cfg.n_nodes, jnp.float32) for _ in COMPONENTS),
                     jnp.asarray(0, jnp.int32))

    def step(st, n):
        st = update_line_h(cfg, st, cfg.dx_1d, dt)
        st = update_line_e(cfg, st, cfg.dx_1d, dt)
        value = sum(weights[c] * getattr(st, COMPONENTS[c])[indices[c]]
                    for c in range(3))
        return st, value

    _, record = lax.scan(step, zero, jnp.arange(n_steps, dtype=jnp.int32))
    return record
