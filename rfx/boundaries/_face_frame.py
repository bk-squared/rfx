"""Static physical rows and per-face ownership for wall writes.

The x-slab frame bounds a distributed shard's physical rows; a
:class:`HighPadFrame` bounds the physical rows of an array allocated with a
high pad on any axis (``rfx.stepping.vector_pad``).
"""

import jax.numpy as jnp


class HighPadFrame(dict):
    """``{axis: (0, n_phys)}`` of an array allocated with cells beyond ``n_phys``.

    The cells past ``n_phys`` are padding, not domain: the wall is at row
    ``n_phys - 1`` and an absorber face keeps its allocated depth there.
    """

    high_pad = True


def needs_face_frame(shape, faces, frame, owned):
    """Whether any axis's physical rows differ from the array, or a face is not owned."""
    if frame is not None:
        for axis, letter in enumerate("xyz"):
            if tuple(frame.get(letter, (0, shape[axis]))) != (0, shape[axis]):
                return True
    return owned is not None and any(owned.get(face, True) is not True for face in faces)


def apply_framed_faces(state, faces, frame, owned, normal_ghost_faces=()):
    """Select only the named electric face's row (and each high normal ghost row)."""
    prefix = "e"
    fields = {prefix + c: getattr(state, prefix + c) for c in "xyz"}
    for axis, letter in enumerate("xyz"):
        lo, hi = (0, state.ex.shape[axis]) if frame is None else frame.get(
            letter, (0, state.ex.shape[axis]))
        for side, row in (("lo", lo), ("hi", hi - 1)):
            face = f"{letter}_{side}"
            if face not in faces:
                continue
            owner = True if owned is None else owned.get(face, True)
            if owner is False:
                continue
            edge = [slice(None)] * 3
            edge[axis] = row
            edge = tuple(edge)
            for component in "xyz":
                if component == letter and face != "z_hi":
                    continue
                key = prefix + component
                new = 0.0
                if owner is not True:
                    old = fields[key][edge]
                    new = jnp.where(owner, jnp.zeros_like(old), old)
                fields[key] = fields[key].at[edge].set(new)
    # Only a high-padded array's ghost rows move with the frame; the x-slab
    # framed path keeps its historical writes (owned rows, no normal ghost).
    for face in (normal_ghost_faces if getattr(frame, "high_pad", False) else ()):
        letter, side = face.split("_")
        if side != "hi":
            continue
        axis = "xyz".index(letter)
        hi = state.ex.shape[axis] if frame is None else frame.get(letter, (0, state.ex.shape[axis]))[1]
        edge = [slice(None)] * 3
        edge[axis] = hi - 1
        fields[prefix + letter] = fields[prefix + letter].at[tuple(edge)].set(0.0)
    return state._replace(**fields)
