"""Slab operands and ordered write-back for the single-device CPML update."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.backend import backend_class
from rfx.core.yee import h_neighbor

ALL_FACES = tuple(f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi"))


class CPMLGridMetadata(NamedTuple):
    """Global static geometry; array operands belong to the shard inputs."""
    cpml_layers: int
    dt: float
    shape: tuple
    cpml_axes: str
    periodic_axes: str
    boundary_depths: tuple
    pad_x_lo: int
    pad_x_hi: int
    pad_y_lo: int
    pad_y_hi: int
    pad_z_lo: int
    pad_z_hi: int
    face_widths: tuple

    def boundary_cell(self, axis, side):
        return self.face_widths[2 * "xyz".index(axis) + (side == "hi")]


def cpml_grid_metadata(grid):
    """Keep global extents/depths without closing over unused mesh arrays.

    Distributed H supplies cached local component means, so h_components
    never reads grid widths. CPMLAxisParams supplies profile/face operands
    through the loop arguments. The legacy profile uses static face widths.
    """
    from rfx.boundaries.depths import grid_face_depths
    return CPMLGridMetadata(
        grid.cpml_layers, float(grid.dt), tuple(grid.shape),
        getattr(grid, "cpml_axes", "xyz"), getattr(grid, "periodic_axes", ""),
        grid_face_depths(grid),
        *(getattr(grid, f"pad_{axis}_{side}", None) for axis in "xyz" for side in ("lo", "hi")),
        tuple(grid.boundary_cell(axis, side) for axis in "xyz" for side in ("lo", "hi")))


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
    frame_lo, frame_hi = (0, size) if frame is None else frame
    framed = (frame_lo, frame_hi) != (0, size)
    start, stop = ((frame_lo, frame_lo + depth) if lo
                   else (frame_hi - depth, frame_hi))

    def take(begin, end):
        sl = [slice(None)] * 3
        sl[axis] = slice(begin, end)
        return arr[tuple(sl)]

    if forward:
        terminal_hi = stop == frame_hi and (not framed or not lo)
        part = take(start + 1, stop if terminal_hi else stop + 1)
        if terminal_hi:
            pads = [(0, 0)] * 3
            pads[axis] = (0, 1)
            part = jnp.pad(part, pads)
        return part

    if framed:
        # Only this face's physical end cuts the backward operand. In
        # particular, a one-row high frame still reads its low ghost.
        view = take(frame_lo, size) if lo else take(0, frame_hi)
        return slab_neighbor(view, axis, depth, lo, False, boundary,
                             h_neighbor=h_neighbor)

    view = arr if (frame_lo, frame_hi) == (0, size) else take(frame_lo, frame_hi)
    frame_size = frame_hi - frame_lo
    if frame_size == 1 and depth == 1:
        return h_neighbor(view, axis, boundary=boundary)

    terminal_lo = start == frame_lo and (not framed or lo)
    part = take(start if terminal_lo else start - 1, stop - 1)

    def terminal(index):
        # Read only the terminal plane through the shared boundary convention.
        # Interior operands above are slices of the original field.
        if frame_size == 1:
            return h_neighbor(view, axis, boundary=boundary)
        at = [slice(None)] * 3
        at[axis] = index
        edge = h_neighbor(view, axis, boundary=boundary, index=tuple(at))
        return jnp.expand_dims(edge, axis)

    if terminal_lo:
        part = jnp.concatenate((terminal(0), part), axis=axis)
    if stop == frame_hi and (not framed or not lo):
        sl = [slice(None)] * 3
        sl[axis] = slice(None, -1)
        part = jnp.concatenate((part[tuple(sl)], terminal(frame_size - 1)), axis=axis)
    return part


def face_windows(shape, depths, frame, periodic, grid):
    """Field/psi windows: framed faces use their realized physical depth.

    Psi retains the single-device axis depth. Its identity padding is outside
    the framed face's window and must not be read or written.
    """
    frame = tuple((frame or {}).get(axis, (0, size)) for axis, size in zip("xyz", shape))
    windows, psi_windows, face_depths = {}, {}, {}
    from rfx.boundaries.depths import grid_face_depths
    realized = ({f.name: f.realized for f in grid_face_depths(grid)}
                if any(bounds != (0, size) for bounds, size in zip(frame, shape)) else {})
    for axis, (size, depth, (lo, hi)) in enumerate(zip(shape, depths, frame)):
        full = (lo, hi) == (0, size)
        if periodic[axis] and not full:
            raise ValueError(f"periodic {'xyz'[axis]} axis requires a full CPML frame")
        for side in ("lo", "hi"):
            face = f"{'xyz'[axis]}_{side}"
            d = depth if full else realized[face]
            sl = [slice(None)] * 3
            psi_sl = [slice(None)] * 3
            if full:
                sl[axis] = slice(None, depth) if side == "lo" else slice(-depth, None)
            else:
                sl[axis] = slice(lo, lo + d) if side == "lo" else slice(hi - d, hi)
                psi_sl[axis] = slice(0, d) if side == "lo" else slice(depth - d, depth)
            windows[face] = tuple(sl)
            psi_windows[face] = tuple(psi_sl)
            face_depths[face] = d
    return frame, windows, psi_windows, face_depths


def store_psi(old, new, window, owner):
    """Select only the active psi rows, leaving identity padding untouched."""
    full = all(sl == slice(None) for sl in window)
    if owner is not None:
        new = jnp.where(owner, new, old if full else old[window])
    if full:
        return new
    starts = tuple(sl.indices(size)[0] for sl, size in zip(window, old.shape))
    return jax.lax.dynamic_update_slice(old, new, starts)


def writeback_form():
    """Choose the write-back with a Python value when the caller is traced."""
    return backend_class().cpml_writeback


def apply_ordered(field, terms):
    """Add slab corrections in production order, including overlap rounding."""
    if writeback_form() == "padded":
        for term in terms:
            sl, value = term[:2]
            owner = term[2] if len(term) == 3 else None
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
            if owner is not None:
                mask = mask & owner
            field = jnp.where(mask, (field + correction).astype(field.dtype), field)
        return field
    for term in terms:
        sl, value = term[:2]
        owner = term[2] if len(term) == 3 else None
        starts = tuple(window.indices(size)[0] for window, size in zip(sl, field.shape))
        # Match .at[sl].add: promote the operands for the addition, then
        # cast its result back to the destination's work dtype per update.
        updated = (field[sl] + value).astype(field.dtype)
        if owner is not None:
            updated = jnp.where(owner, updated, field[sl])
        field = jax.lax.dynamic_update_slice(field, updated, starts)
    return field
