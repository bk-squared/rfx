"""RLC scalar solves followed by one field scatter per component."""

import jax.numpy as jnp

from rfx.lumped import rlc_element_values


def update_rlc_elements(state, rlc_states, metas, e_prevs):
    """Read every element from the pre-RLC field, then publish all writes.

    Only solved elements (series, or parallel with an inductor) write their
    edge; folded elements write nothing. Admission already refuses two solved
    elements on one edge (#1245), so the written edges are disjoint and one
    scatter equals the old element-by-element writes.
    """
    updated_states = []
    writes = {}
    for rlc_state, meta, e_prev in zip(rlc_states, metas, e_prevs, strict=True):
        field = getattr(state, meta.component)
        cell = (meta.i, meta.j, meta.k)
        value, updated = rlc_element_values(field[cell], rlc_state, meta, e_prev)
        updated_states.append(updated)
        if meta.is_series or meta.has_inductor:
            indices, values = writes.setdefault(meta.component, ([], []))
            if cell in indices:
                raise NotImplementedError(
                    f"two solved RLC elements share the {meta.component} edge "
                    f"of cell {cell}; refuse_stacked_solved_elements (#1245) "
                    "must run before the step is built.")
            indices.append(cell)
            values.append(value.astype(field.dtype))
    fields = {}
    for component, (indices, values) in writes.items():
        index = jnp.asarray(indices, dtype=jnp.int32)
        field = getattr(state, component)
        fields[component] = field.at[tuple(index[:, axis] for axis in range(3))].set(
            jnp.stack(values))
    return state._replace(**fields), updated_states
