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


def _face_reads(tables):
    """Unique component/node operands, with each original table's gather order."""
    import numpy as np
    indices = tuple(np.unique(np.concatenate(
        [t.source_index for t in tables if t.source_component == c]))
        if any(t.source_component == c for t in tables) else np.array([], np.int32)
        for c in range(6))
    offsets = np.cumsum([0] + [len(row) for row in indices])
    gathers = tuple((offsets[t.source_component]
                     + np.searchsorted(indices[t.source_component], t.source_index)).astype(np.int32)
                    for t in tables)
    return indices, gathers


def record_face_values(cfg, state, dx, dt):
    """Store exactly the operands read before H and after H, before E."""
    from rfx.sources.tfsf_discrete_plane_wave import COMPONENTS, update_line_h, update_line_e

    h_indices, h_gathers = _face_reads(cfg.tables_h)
    e_indices, e_gathers = _face_reads(cfg.tables_e)

    def read(st, indices):
        return jnp.concatenate(tuple(getattr(st, name)[row]
                                     for name, row in zip(COMPONENTS, indices)))

    def step(st, _):
        h_values = read(st, h_indices)
        st = update_line_h(cfg, st, dx, dt)
        e_values = read(st, e_indices)
        st = update_line_e(cfg, st, dx, dt)
        return st, (h_values, e_values)

    from rfx.stepping.scan import recorded_scan
    _, record = recorded_scan(step, state, None, length=cfg.n_steps)
    return tuple(lax.stop_gradient(row) for row in record), (h_gathers, e_gathers)


def prepare_tfsf_for_scan(tfsf, dx, dt):
    """A marked closed box (``record_faces``) reads a face record; others carry the line."""
    from dataclasses import replace
    from rfx.sources.tfsf_discrete_plane_wave import IncidentStep

    cfg, state = tfsf
    if not getattr(cfg, "closed_box", False) or not cfg.record_faces:
        return tfsf
    if cfg.face_record is not None:
        return tfsf
    record, gathers = record_face_values(cfg, state, dx, dt)
    return replace(cfg, face_record=record, face_gathers=gathers), IncidentStep(state.step)
