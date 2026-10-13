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
                  h_neighbor=h_neighbor, frame=None):
    """Neighbor operand restricted to one face, including its terminal image.

    Match the whole-array forward zero extension or backward H neighbor,
    while slicing before constructing the difference. Boundary images depend
    on the explicit curl boundary; they do not select the active CPML faces.
    """
    size = arr.shape[axis]
    begin, end = 0, size
    if frame is not None and axis == 0:
        # The opposite endpoint belongs to another rank. Retain its exchanged
        # neighbor even when this face exactly fills the owning slab (R12).
        begin, end = (frame['x'][0], size) if lo else (0, frame['x'][1])
    start, stop = (begin, begin + depth) if lo else (end - depth, end)

    def take(begin, end):
        sl = [slice(None)] * 3
        sl[axis] = slice(begin, end)
        return arr[tuple(sl)]

    if forward:
        part = take(start + 1, min(stop + 1, end))
        if stop == end:
            pads = [(0, 0)] * 3
            pads[axis] = (0, 1)
            part = jnp.pad(part, pads)
        return part

    if size == 1:
        return h_neighbor(arr, axis, boundary=boundary)

    part = take(max(begin, start - 1), stop - 1)

    def terminal(index):
        # Read only the terminal plane through the shared boundary convention.
        # Interior operands above are slices of the original field.
        at = [slice(None)] * 3
        at[axis] = index
        view = arr
        if frame is not None and axis == 0:
            # Each image retains the opposite exchanged neighbor (W0/R12).
            view = arr[begin:] if index == begin else arr[:end]
            at[axis] = 0 if index == begin else end - 1
        edge = h_neighbor(view, axis, boundary=boundary, index=tuple(at))
        return jnp.expand_dims(edge, axis)

    if start == begin:
        part = jnp.concatenate((terminal(begin), part), axis=axis)
    if stop == end:
        sl = [slice(None)] * 3
        sl[axis] = slice(None, -1)
        part = jnp.concatenate((part[tuple(sl)], terminal(end - 1)), axis=axis)
    return part


def writeback_form():
    """Choose the write-back with a Python value when the caller is traced."""
    return "padded" if jax.default_backend() == "gpu" else "slab"


def apply_ordered(field, terms, *, frame=None, owned=None):
    """Add slab corrections in production order, including overlap rounding."""
    if frame is not None:
        for sl, value in terms:
            active = True
            if sl[0].start is not None or sl[0].stop is not None:
                side = 'lo' if sl[0].start is None else 'hi'
                active = owned['x_' + side]
                lo, hi = frame['x']
                x = slice(lo, lo + sl[0].stop) if side == 'lo' else slice(hi + sl[0].start, hi)
                sl = (x, *sl[1:])
            starts = tuple(w.indices(n)[0] for w, n in zip(sl, field.shape))
            updated = jnp.where(active, (field[sl] + value).astype(field.dtype), field[sl])
            field = jax.lax.dynamic_update_slice(field, updated, starts)
        return field
    if writeback_form() == "padded":
        for sl, value in terms:
            pads = []
            mask = True
            for axis, (window, size) in enumerate(zip(sl, field.shape)):
                start, stop, _ = window.indices(size)
                pads.append((start, size - stop))
                if start != 0 or stop != size:
                    # A face slab restricts one axis; the mask is that axis's.
                    assert mask is True, "a CPML slab restricts exactly one axis"
                    shape = [1, 1, 1]
                    shape[axis] = size
                    index = jnp.arange(size).reshape(shape)
                    mask = (index >= start) & (index < stop)
            correction = jnp.pad(value, pads)
            field = jnp.where(mask, (field + correction).astype(field.dtype), field)
        return field
    for sl, value in terms:
        starts = tuple(window.indices(size)[0] for window, size in zip(sl, field.shape))
        # Match .at[sl].add: promote the operands for the addition, then
        # cast its result back to the destination's work dtype per update.
        updated = (field[sl] + value).astype(field.dtype)
        field = jax.lax.dynamic_update_slice(field, updated, starts)
    return field
