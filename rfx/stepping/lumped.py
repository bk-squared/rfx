"""RLC scalar solves followed by one field scatter per component."""

import jax.numpy as jnp

from rfx.lumped import rlc_element_values


def update_rlc_elements(state, rlc_states, metas, e_prevs):
    """Read every element from the pre-RLC field, then publish all writes."""
    updated_states = []
    writes = {}
    occupied = {}
    for number, (rlc_state, meta, e_prev) in enumerate(
            zip(rlc_states, metas, e_prevs, strict=True)):
        field = getattr(state, meta.component)
        cell = (meta.i, meta.j, meta.k)
        edge = (meta.component, cell)
        if edge in occupied:
            raise NotImplementedError(
                f"RLC element #{occupied[edge]} and RLC element #{number} share "
                f"the {meta.component} edge of cell {cell}; batched RLC updates "
                "require disjoint edges.")
        occupied[edge] = number
        value, updated = rlc_element_values(field[cell], rlc_state, meta, e_prev)
        updated_states.append(updated)
        if meta.is_series or meta.has_inductor:
            indices, values = writes.setdefault(meta.component, ([], []))
            indices.append(cell)
            values.append(value.astype(field.dtype))
    fields = {}
    for component, (indices, values) in writes.items():
        index = jnp.asarray(indices, dtype=jnp.int32)
        field = getattr(state, component)
        fields[component] = field.at[tuple(index[:, axis] for axis in range(3))].set(
            jnp.stack(values))
    return state._replace(**fields), updated_states
