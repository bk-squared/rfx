"""Slab operands and ordered write-back for the single-device CPML update."""

import jax.numpy as jnp

ALL_FACES = tuple(f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi"))


def selected_faces(axes, faces):
    faces = frozenset(faces)
    unknown = faces.difference(ALL_FACES)
    if unknown:
        raise ValueError(f"unknown CPML faces: {sorted(unknown)}")
    return frozenset(face for face in faces if face[0] in axes)


def slab_neighbor(arr, axis, depth, lo, forward, boundary=None):
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

    part = take(max(0, start - 1), stop - 1)
    periodic = boundary.periodic if boundary is not None else (False,) * 3
    from rfx.boundaries.pmc import magnetic_image_faces
    images = (magnetic_image_faces(boundary.pmc_faces, arr.shape, periodic)
              if boundary is not None else frozenset())
    if start == 0:
        edge = (take(size - 1, size) if periodic[axis]
                else -take(0, 1) if f"{'xyz'[axis]}_lo" in images
                else jnp.zeros_like(take(0, 1)))
        part = jnp.concatenate((edge, part), axis=axis)
    if stop == size and f"{'xyz'[axis]}_hi" in images:
        edge = take(size - 1, size) + 2 * take(size - 2, size - 1)
        sl = [slice(None)] * 3
        sl[axis] = slice(None, -1)
        part = jnp.concatenate((part[tuple(sl)], edge), axis=axis)
    return part


def apply_ordered(field, terms):
    """Add slab corrections in production order, including overlap rounding."""
    for sl, value in terms:
        pads = []
        mask = True
        for axis, (window, size) in enumerate(zip(sl, field.shape)):
            start, stop, _ = window.indices(size)
            pads.append((start, size - stop))
            if start != 0 or stop != size:
                shape = [1, 1, 1]
                shape[axis] = size
                index = jnp.arange(size).reshape(shape)
                mask = (index >= start) & (index < stop)
        correction = jnp.pad(value, pads)
        field = jnp.where(mask, field + correction, field)
    return field
