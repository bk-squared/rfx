"""Static physical rows and per-face ownership for x-slab wall writes."""

import jax.numpy as jnp


def needs_face_frame(shape, faces, frame, owned):
    """Validate an axis-name mapping and retain the original default path."""
    if frame is not None:
        for axis, letter in enumerate("xyz"):
            bounds = frame.get(letter, (0, shape[axis]))
            if letter != "x" and tuple(bounds) != (0, shape[axis]):
                raise ValueError("Only x may have a non-default face frame")
        if tuple(frame.get("x", (0, shape[0]))) != (0, shape[0]):
            return True
    return owned is not None and any(owned.get(face, True) is not True for face in faces)


def apply_framed_faces(state, faces, frame, owned, *, magnetic=False):
    """Select only the named face's row; image PMC has no low-face write."""
    prefix = "h" if magnetic else "e"
    fields = {prefix + c: getattr(state, prefix + c) for c in "xyz"}
    for axis, letter in enumerate("xyz"):
        lo, hi = (0, state.ex.shape[axis]) if frame is None else frame.get(
            letter, (0, state.ex.shape[axis]))
        for side, row in (("lo", lo), ("hi", hi - 1)):
            face = f"{letter}_{side}"
            if face not in faces or (magnetic and side == "lo"):
                continue
            owner = True if owned is None else owned.get(face, True)
            if owner is False:
                continue
            edge = [slice(None)] * 3
            edge[axis] = row
            edge = tuple(edge)
            for component in "xyz":
                if component == letter and (magnetic or face != "z_hi"):
                    continue
                key = prefix + component
                new = 0.0
                if owner is not True:
                    old = fields[key][edge]
                    new = jnp.where(owner, jnp.zeros_like(old), old)
                fields[key] = fields[key].at[edge].set(new)
    return state._replace(**fields)
