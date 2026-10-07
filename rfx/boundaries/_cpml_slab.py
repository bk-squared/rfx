"""Slab operands and ordered write-back for the single-device CPML update."""

import jax
import jax.numpy as jnp

from rfx.core.yee import h_neighbor

ALL_FACES = tuple(f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi"))


def selected_faces(axes, faces):
    faces = frozenset(faces)
    unknown = faces.difference(ALL_FACES)
    if unknown:
        raise ValueError(f"unknown CPML faces: {sorted(unknown)}")
    return frozenset(face for face in faces if face[0] in axes)


def slab_neighbor(arr, axis, depth, lo, forward, boundary=None, *,
                  h_neighbor=h_neighbor):
    """Neighbor operand restricted to one face, including its terminal image.

    Match the whole-array forward zero extension or backward H neighbor,
    while slicing before constructing the difference. Boundary images depend
    on the explicit curl boundary; they do not select the active CPML faces.
    """
    size = arr.shape[axis]
    start, stop = (0, depth) if lo else (size - depth, size)

    def take(begin, end):
        sl = [slice(None)] * 3
        sl[axis] = slice(begin, end)
        return arr[tuple(sl)]

    if forward:
        part = take(start + 1, min(stop + 1, size))
        if stop == size:
            pads = [(0, 0)] * 3
            pads[axis] = (0, 1)
            part = jnp.pad(part, pads)
        return part

    if size == 1:
        return h_neighbor(arr, axis, boundary=boundary)

    part = take(max(0, start - 1), stop - 1)

    def terminal(index):
        # Read only the terminal plane through the shared boundary convention.
        # Interior operands above are slices of the original field.
        at = [slice(None)] * 3
        at[axis] = index
        edge = h_neighbor(arr, axis, boundary=boundary, index=tuple(at))
        return jnp.expand_dims(edge, axis)

    if start == 0:
        part = jnp.concatenate((terminal(0), part), axis=axis)
    if stop == size:
        sl = [slice(None)] * 3
        sl[axis] = slice(None, -1)
        part = jnp.concatenate((part[tuple(sl)], terminal(size - 1)), axis=axis)
    return part


def apply_ordered(field, terms):
    """Add slab corrections in production order, including overlap rounding."""
    for sl, value in terms:
        starts = tuple(window.indices(size)[0] for window, size in zip(sl, field.shape))
        # Match .at[sl].add: promote the operands for the addition, then
        # cast its result back to the destination's work dtype per update.
        updated = (field[sl] + value).astype(field.dtype)
        field = jax.lax.dynamic_update_slice(field, updated, starts)
    return field
