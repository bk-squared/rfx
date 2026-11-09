"""Lossy subsets of realized opaque PEC faces; no geometry is released."""
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from rfx.boundaries.pec import realized_pec_edge_masks, wall_edge_masks, _shift
from rfx.core.yee import MU_0
from rfx.errors import RfxRefusal
from rfx.model.graded_grid import e_node_dual_spacings


@dataclass(frozen=True)
class WallLoss:
    sigma_bulk: object
    f0: object
    shape: object = None
    faces: tuple = ()


@dataclass(frozen=True)
class WallOperator:
    mask_ex: object
    mask_ey: object
    mask_ez: object
    sigma_sheet: object
    wall_faces: frozenset
    end_ex: object = None
    end_ey: object = None
    end_ez: object = None
    component_sigma: bool = True


def refuse_wall_loss(sim, lane, reason="the wall operator is not joined to this lane"):
    """One named refusal, always including the lane and supported route."""
    if getattr(sim, "_wall_losses", ()):
        raise RfxRefusal(f"add_wall_loss: {lane}: {reason}; use uniform or graded run()/forward() with staircase PEC walls")


def admit_wall_loss(sim, lane, *, smoothing=False, conformal=False):
    if not getattr(sim, "_wall_losses", ()):
        return
    from rfx.core.jax_utils import is_tracer
    if sim._boundary == "upml":
        refuse_wall_loss(sim, lane, "UPML replaces the E update")
    if smoothing or conformal or sim._boundary_spec.conformal_faces():
        refuse_wall_loss(sim, lane, "Kottke / conformal_pec realized solids are unsupported")
    if any(is_tracer(getattr(sim, key, None)) for key in ("_dx", "_dx_profile", "_dy_profile", "_dz_profile")):
        refuse_wall_loss(sim, lane, "traced mesh has no concrete face set")


def _empty(shape):
    return tuple(np.zeros(shape, dtype=bool) for _ in range(3))


def build_wall_operator(sim, root, film, *, lane):
    """Read production ownership, sum each face's G/full curl spacing."""
    entries = getattr(sim, "_wall_losses", ())
    if not entries:
        return film
    admit_wall_loss(sim, lane)
    grid, periodic = root.grid, root.periodic
    shape = grid.shape
    faces = frozenset(f for entry in entries for f in entry.faces)
    all_faces = frozenset(sim._boundary_spec.pec_faces())
    held = wall_edge_masks(shape, all_faces - faces)
    body = root.edges
    sigma = [jnp.zeros(shape) for _ in range(3)]
    masks = list(_empty(shape))
    widths = [e_node_dual_spacings(grid.cells(axis)) for axis in "xyz"]
    records = {key: (cells, sheet, wire) for key, cells, sheet, wire, _ in root.assembly_entries}
    for entry in entries:
        conductance = 1 / jnp.sqrt(jnp.pi * entry.f0 * MU_0 / entry.sigma_bulk)
        candidates = []
        if entry.faces:
            for face in entry.faces:
                candidates.append(("xyz".index(face[0]), wall_edge_masks(shape, (face,)), body))
        else:
            keys = [id(e) for e in sim._geometry if e.shape is entry.shape]
            own = np.zeros(shape, dtype=bool)
            for key in keys:
                cells, sheet, wire = records[key]
                if cells is None:
                    refuse_wall_loss(sim, lane, "shape must realize as a PEC solid, not a sheet or wire")
                own |= np.asarray(cells)
            for a, axis in enumerate("xyz"):
                for side in ("lo", "hi"):
                    pad = getattr(grid, f"pad_{axis}_{side}")
                    sl: list[Any] = [slice(None)] * 3
                    sl[a] = slice(0, pad) if side == "lo" else slice(shape[a] - pad, None)
                    if pad and own[tuple(sl)].any():
                        refuse_wall_loss(sim, lane, "lossy PEC solid enters a CPML pad")
            own_edges = realized_pec_edge_masks(own, periodic=periodic)
            other = list(_empty(shape))
            for key, (cells, sheet, wire) in records.items():
                if key in keys or (cells is None and sheet is None and wire is None):
                    continue
                edges = realized_pec_edge_masks(cells, sheets=() if sheet is None else (sheet,),
                                               wires=() if wire is None else (wire,), periodic=periodic)
                if edges is not None:
                    other = [p | np.asarray(q) for p, q in zip(other, edges)]
            for a in range(3):
                face_edges = list(_empty(shape))
                for c in range(3):
                    if c == a:
                        continue
                    b = 3 - a - c
                    plus = own | _shift(own, b, periodic, +1, xp=np)
                    minus = _shift(plus, a, periodic, +1, xp=np)
                    # On exactly one side of a field-facing solid boundary.
                    face_edges[c] = np.asarray(own_edges[c]) & np.asarray(body[c]) & (plus ^ minus)
                    for d in range(3):
                        if not periodic[d]:
                            sl = [slice(None)] * 3
                            sl[d] = -1
                            if d == c:
                                face_edges[c][tuple(sl)] = False
                    for face in all_faces:
                        face_edges[c] &= ~wall_edge_masks(shape, (face,))[c]
                candidates.append((a, face_edges, other))
        for a, edge_set, blocked in candidates:
            extent = [1, 1, 1]
            extent[a] = shape[a]
            density = conductance / widths[a].reshape(extent)
            for c in range(3):
                selected = np.asarray(edge_set[c]) & ~np.asarray(blocked[c]) & ~held[c]
                if not periodic[c]:
                    ghost: list[Any] = [slice(None)] * 3
                    ghost[c] = -1
                    selected[tuple(ghost)] = False
                for _, component, index in root.released_port_edges:
                    if component == "e" + "xyz"[c]:
                        selected[index] = False
                masks[c] |= selected
                sigma[c] = sigma[c] + jnp.where(selected, density, 0)
    for poles in root.assembly[1:3]:
        if poles is not None:
            for pole_mask in poles[1]:
                occupied = realized_pec_edge_masks(np.asarray(pole_mask) != 0, periodic=periodic)
                if any(np.any(m & np.asarray(p)) for m, p in zip(masks, occupied)):
                    refuse_wall_loss(sim, lane, "dispersive Debye/Lorentz/Drude overlap at a lossy face")
    ends = (None, None, None)
    if film is not None:
        ends = (film.end_ex, film.end_ey, film.end_ez)
        for c, m in enumerate((film.mask_ex, film.mask_ey, film.mask_ez)):
            masks[c] |= np.asarray(m)
            sigma[c] = sigma[c] + jnp.where(m, film.sigma_sheet, 0)
    return WallOperator(mask_ex=jnp.asarray(masks[0]), mask_ey=jnp.asarray(masks[1]),
                        mask_ez=jnp.asarray(masks[2]), sigma_sheet=jnp.stack(sigma),
                        wall_faces=faces, end_ex=ends[0], end_ey=ends[1], end_ez=ends[2])


def refuse_wall_and_f0_sheets(sim, lane):
    """Compact admission for calculators that cannot carry either operator."""
    from rfx.materials.thin_conductor import refuse_f0_sheets
    refuse_wall_loss(sim, lane)
    refuse_f0_sheets(sim._thin_conductors, lane)
