"""Surface-impedance (Leontovich) conductors for the 3-D Yee FDFD.

Physics. A good conductor much thicker than its skin depth is replaced by
the impedance boundary condition on its surface,

    E_tan = Z_s (n_out x H),     Z_s = (1 + j) R_s,   R_s = sqrt(omega mu0 / (2 sigma)),

(e^{+j omega t}; ``R_s`` is :func:`rfx.materials.thin_conductor.leontovich_rs`,
the same formula the FDTD side uses). Equivalently the conductor is a
resistive sheet carrying the surface current ``J_s = n_out x H = E_tan / Z_s``
with NO field behind it (H = 0 inside the metal).

Discrete form. Take a surface E edge (an edge with at least one air cell and
at least one metal cell among the four cells around it). Its dual face is
split into four quadrants, one per neighbouring cell. The Ampere loop of the
edge's row is restricted to the AIR quadrants:

    (1/A_air) sum_{faces f on the air part of the loop} +-H_f * len_air(f)
        = j omega eps_air E_e + Y_s E_e * len_sheet / A_air + J_e,

where ``len_air(f)`` is the length of the loop segment carried by face ``f``
that lies in air, ``A_air`` the air area of the dual face, ``eps_air`` the
area-weighted permittivity of the air quadrants, ``Y_s = 1 / Z_s`` and
``len_sheet`` the total length of conductor surface inside the loop (the
segments separating an air quadrant from a metal quadrant). In the
``curl curl E - k0^2 eps E = -j omega mu0 J`` form this is

    Ch entries of the row  ->  sign * len_air(f) / A_air,
    diagonal              +=  j omega mu0 Y_s * len_sheet / A_air
                             - k0^2 (eps_air - eps_edge),

i.e. the sheet admittance folded into the air part of the dual cell as an
equivalent conductivity ``sigma_eff = Y_s * len_sheet / A_air`` (the FDTD
side's ``sigma_sheet = G / d_dual`` with ``G = 1 / R_s``, here with the
complex ``Y_s``). H faces with metal on both sides are removed (``H = 0``
inside the conductor); edges with metal on all four sides stay PEC. On the
OUTER box walls the dual steps of :mod:`rfx.fdfd.yee3d` are already the
half cells, so there the rule reduces to ``ch_scale = 1`` and
``diag += j omega mu0 Y_s / (h/2)`` on the released wall edges: the classic
half-cell impedance wall with ``H_tan`` sampled half a cell inside (first
order in ``h`` for the field, the loss itself is second order for a mode
whose tangential H is even at the wall). The cut-cell volumes make the
discrete Poynting identity hold, so the conductor is discretely passive.

``sigma`` is a traced scalar (JAX); the geometry (cell mask, which outer
walls are lossy) is static. Scope fence: one conductivity per
:class:`Conductor`; conductors inside a PML are not supported (the cut-cell
entries are computed with the real steps); the mode of loss is the
Leontovich sheet only (no finite-thickness / skin-depth resolution).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rfx.fdfd import yee3d as y
from rfx.materials.thin_conductor import leontovich_rs

__all__ = ["Conductor", "surface_impedance", "surface_terms", "box_walls", "ConductorGeometry",
           "conductor_geometry"]


@dataclass(frozen=True)
class Conductor:
    """Static geometry of one Leontovich conductor: ``cells`` (``(nx, ny, nz)``
    bool, metal cells, or ``None``) and ``walls`` (six bools, outer box faces
    ``x_lo, x_hi, y_lo, y_hi, z_lo, z_hi`` that are lossy instead of PEC)."""
    cells: np.ndarray | None = None
    walls: tuple[bool, bool, bool, bool, bool, bool] = (False,) * 6


def box_walls(x: bool = True, y: bool = True, z: bool = False) -> Conductor:
    """Conductor made of the outer box walls normal to the chosen axes."""
    return Conductor(cells=None, walls=(x, x, y, y, z, z))


def surface_impedance(freq, sigma):
    """``Z_s = (1 + j) sqrt(omega mu0 / (2 sigma))`` (traced in both)."""
    return (1.0 + 1.0j) * leontovich_rs(freq, sigma)


@dataclass(frozen=True)
class ConductorGeometry:
    """Static classification of the edges and faces of a conductor plus the
    index bookkeeping needed to evaluate the traced cut-cell quantities."""
    free: np.ndarray          # bool (n_edges,): lossy surface edges (released / kept free)
    pec: np.ndarray           # bool (n_edges,): edges fully inside the metal
    dead_face: np.ndarray     # bool (n_faces,): faces with metal on both sides
    air_q: np.ndarray         # bool (n_edges, 2, 2): air indicator of quadrant (side_a, side_b)
    sheet_q: np.ndarray       # bool (n_edges, 2, 2): [s_a, s_b] -> segment between quadrants
    #                           (s_a, -b)/(s_a, +b) is sheet (index [s_a, 0]) and
    #                           (-a, s_b)/(+a, s_b) is sheet (index [s_b, 1]) -- see below
    cell_q: np.ndarray        # int (n_edges, 2, 2): flat cell index of each quadrant (-1 outside)
    comp: np.ndarray          # int8 (n_edges,): component of each edge
    half_idx: np.ndarray      # int (n_edges, 2, 2): [axis-slot a/b, side] -> primal step index (+1; 0 = outside)
    entry_side: np.ndarray    # int8 (n_ch,): 0 (-) / 1 (+) side of the face along the differencing axis
    entry_slot: np.ndarray    # int8 (n_ch,): 0 if the differencing axis is a (= c+1), 1 if b (= c+2)


def _quadrant_data(model: y.Yee3DModel):
    """Per edge: the flat cell index of its four neighbouring cells
    ``(-1 outside the box)``, the component, and the primal-step index
    (shifted by one, 0 = outside) of the half step on each side."""
    nx, ny, nz = model.shape
    shapes = y.edge_shapes(nx, ny, nz)
    cell_idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    cell_pad = np.pad(cell_idx, 1, mode="constant", constant_values=-1)
    cell_q, comp, half_idx = [], [], []
    for c, shp in enumerate(shapes):
        a, b = (c + 1) % 3, (c + 2) % 3
        idx = np.indices(shp)               # (3, *shp): index along x, y, z
        cq = np.zeros(shp + (2, 2), dtype=np.int64)
        hi = np.zeros(shp + (2, 2), dtype=np.int64)
        for sa in (0, 1):
            for sb in (0, 1):
                pos = [idx[0] + 1, idx[1] + 1, idx[2] + 1]      # padded cell index
                pos[a] = idx[a] + sa                             # sa=0: cell j-1 -> padded j
                pos[b] = idx[b] + sb
                cq[..., sa, sb] = cell_pad[pos[0], pos[1], pos[2]]
        for slot, ax in enumerate((a, b)):
            n_ax = model.shape[ax]
            for s in (0, 1):
                k = idx[ax] - 1 + s                              # primal step index on that side
                hi[..., slot, s] = np.where((k >= 0) & (k < n_ax), k + 1, 0)
        cell_q.append(cq.reshape(-1, 2, 2))
        comp.append(np.full(int(np.prod(shp)), c, dtype=np.int8))
        half_idx.append(hi.reshape(-1, 2, 2))
    return np.concatenate(cell_q), np.concatenate(comp), np.concatenate(half_idx)


def conductor_geometry(model: y.Yee3DModel, cond: Conductor) -> ConductorGeometry:
    """Static edge / face classification of a conductor (see module doc)."""
    nx, ny, nz = model.shape
    shapes = y.edge_shapes(nx, ny, nz)
    cell_q, comp, half_idx = _quadrant_data(model)
    lossy_cells = np.zeros(nx * ny * nz, dtype=bool)
    if cond.cells is not None:
        cells = np.asarray(cond.cells, dtype=bool)
        if cells.shape != (nx, ny, nz):
            raise ValueError(f"conductor cells must have shape {(nx, ny, nz)}")
        lossy_cells = cells.ravel()
    if len(cond.walls) != 6:
        raise ValueError("walls must be six booleans")
    walls = np.asarray(cond.walls, dtype=bool)

    inside = cell_q >= 0
    lossy_q = np.zeros(cell_q.shape, dtype=bool)
    lossy_q[inside] = lossy_cells[cell_q[inside]]
    # outside slots: lossy if every wall they lie on is lossy, PEC otherwise
    pec_q = np.zeros(cell_q.shape, dtype=bool)
    off = 0
    for c, shp in enumerate(shapes):
        a, b = (c + 1) % 3, (c + 2) % 3
        n_e = int(np.prod(shp))
        idx = np.indices(shp).reshape(3, -1)
        for sa in (0, 1):
            for sb in (0, 1):
                oa = (idx[a] == 0) if sa == 0 else (idx[a] == shp[a] - 1)
                ob = (idx[b] == 0) if sb == 0 else (idx[b] == shp[b] - 1)
                wa, wb = 2 * a + sa, 2 * b + sb
                outside = oa | ob
                any_pec = (oa & ~walls[wa]) | (ob & ~walls[wb])
                sl = slice(off, off + n_e)
                pec_q[sl, sa, sb] = outside & any_pec
                lossy_q[sl, sa, sb] = np.where(outside, ~any_pec, lossy_q[sl, sa, sb])
        off += n_e
    air_q = ~(lossy_q | pec_q)
    n_air = air_q.sum(axis=(1, 2))
    n_lossy = lossy_q.sum(axis=(1, 2))
    n_pec = pec_q.sum(axis=(1, 2))
    free = (n_air >= 1) & (n_lossy >= 1) & (n_pec == 0)
    pec = n_air == 0
    # sheet segments: [s_a, 0] separates (s_a, -b) from (s_a, +b) (runs along a,
    # length half_a[s_a]); [s_b, 1] separates (-a, s_b) from (+a, s_b) (along b).
    sheet_q = np.zeros((len(comp), 2, 2), dtype=bool)
    for s in (0, 1):
        sheet_q[:, s, 0] = (air_q[:, s, 0] & lossy_q[:, s, 1]) | (lossy_q[:, s, 0] & air_q[:, s, 1])
        sheet_q[:, s, 1] = (air_q[:, 0, s] & lossy_q[:, 1, s]) | (lossy_q[:, 0, s] & air_q[:, 1, s])

    # dead faces: both cells adjacent to the face are lossy metal
    f_shapes = y.face_shapes(nx, ny, nz)
    lossy3 = lossy_cells.reshape(nx, ny, nz)
    dead = []
    for c, shp in enumerate(f_shapes):
        pad = [(0, 0)] * 3
        pad[c] = (1, 1)
        p = np.pad(lossy3, pad, mode="constant", constant_values=False)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[c] = slice(0, p.shape[c] - 1)
        hi[c] = slice(1, p.shape[c])
        d = p[tuple(lo)] & p[tuple(hi)]
        if d.shape != shp:
            raise AssertionError(f"face bookkeeping: {d.shape} vs {shp}")
        dead.append(d.ravel())
    dead_face = np.concatenate(dead)

    # per Ch entry: differencing slot (a or b of the row's component) and side
    row_comp = comp[model.ch_rows]
    a_of_row = (row_comp + 1) % 3
    entry_slot = np.where(model.ch_axis == a_of_row, 0, 1).astype(np.int8)
    # (+1, comp b, axis a) and (-1, comp a, axis b) are the "hi" (+) faces
    entry_side = np.where((model.ch_sign > 0) == (entry_slot == 0), 1, 0).astype(np.int8)
    return ConductorGeometry(free=free, pec=pec, dead_face=dead_face, air_q=air_q,
                             sheet_q=sheet_q, cell_q=cell_q, comp=comp, half_idx=half_idx,
                             entry_side=entry_side, entry_slot=entry_slot)


def _half_steps(model: y.Yee3DModel, geo: ConductorGeometry, dx, dy, dz) -> jax.Array:
    """``(n_edges, 2, 2)`` half primal steps ``l^{s}/2`` of slot (a, b) and
    side (-, +); zero outside the box."""
    steps = [jnp.asarray(d, dtype=jnp.float64) for d in (dx, dy, dz)]
    padded = [jnp.concatenate([jnp.zeros((1,), jnp.float64), d]) for d in steps]
    out = jnp.zeros(geo.half_idx.shape, jnp.float64)
    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        sel = geo.comp == c
        for slot, ax in enumerate((a, b)):
            out = out.at[sel, slot, :].set(0.5 * padded[ax][geo.half_idx[sel, slot, :]])
    return out


def _real_dual(d) -> jax.Array:
    """Unstretched dual steps at the nodes (half cells at the two walls)."""
    d = jnp.asarray(d, dtype=jnp.float64)
    return jnp.concatenate([0.5 * d[:1], 0.5 * (d[:-1] + d[1:]), 0.5 * d[-1:]])


def surface_terms(model: y.Yee3DModel, freq, cond: Conductor, sigma, dx, dy, dz,
                  eps_r=None, geo: ConductorGeometry | None = None) -> y.BoundaryTerms:
    """:class:`rfx.fdfd.yee3d.BoundaryTerms` of the conductor at ``freq`` with
    bulk conductivity ``sigma`` (traced scalar, S/m). ``eps_r`` is the cell
    permittivity used by the solve (for the air-quadrant average on the
    surface edges); pass the static ``geo`` from :func:`conductor_geometry`
    to skip recomputing it."""
    if geo is None:
        geo = conductor_geometry(model, cond)
    omega = 2.0 * jnp.pi * jnp.asarray(freq, dtype=jnp.float64)
    k0 = omega / y.C0
    ys = 1.0 / surface_impedance(freq, sigma)
    half = _half_steps(model, geo, dx, dy, dz)              # (n_edges, 2, 2): [slot, side]
    air = jnp.asarray(geo.air_q, dtype=jnp.float64)         # [s_a, s_b]
    quad_area = half[:, 0, :, None] * half[:, 1, None, :]   # [s_a, s_b]
    a_air = jnp.sum(air * quad_area, axis=(1, 2))
    sheet = jnp.asarray(geo.sheet_q, dtype=jnp.float64)     # [s, slot]
    len_sheet = jnp.sum(sheet * half.transpose(0, 2, 1), axis=(1, 2))
    free = geo.free
    a_air_safe = jnp.where(free, a_air, 1.0)

    # sheet admittance and permittivity correction on the surface edges
    diag = jnp.where(free, 1j * omega * y.MU0 * ys * len_sheet / a_air_safe, 0.0)
    if eps_r is not None:
        eps_r = jnp.asarray(eps_r, dtype=jnp.complex128).ravel()
        eps_q = jnp.where(geo.cell_q >= 0, eps_r[jnp.clip(geo.cell_q, 0)], 0.0)
        eps_air = jnp.sum(air * quad_area * eps_q, axis=(1, 2)) / a_air_safe
        eps_edge = y._eps_on_edges(model, eps_r.reshape(model.shape))
        diag = diag + jnp.where(free, -(k0 * k0) * (eps_air - eps_edge), 0.0)

    # cut-cell dual-curl entries on the surface rows, dead faces removed
    dual = [_real_dual(d) for d in (dx, dy, dz)]
    rows = model.ch_rows
    slot = geo.entry_slot.astype(np.int64)
    side = geo.entry_side.astype(np.int64)
    other = 1 - slot
    # air length of the segment carried by the face on `side` along the
    # differencing slot: sum over the two quadrants on that side of the half
    # step along the OTHER in-plane axis
    q_air_side = jnp.where(slot[:, None] == 0, air[rows, side, :], air[rows, :, side])   # (n_ch, 2)
    len_air = jnp.sum(q_air_side * half[rows, other, :], axis=1)
    dual_d = jnp.zeros(len(rows), jnp.float64)
    for axis in range(3):
        sel = model.ch_axis == axis
        dual_d = dual_d.at[sel].set(dual[axis][model.ch_idx[sel]])
    scale = jnp.where(free[rows], len_air * dual_d / a_air_safe[rows], 1.0)
    scale = jnp.where(geo.dead_face[model.ch_cols], 0.0, scale)
    return y.BoundaryTerms(pec=geo.pec, free=free & model.wall, ch_scale=scale, diag_add=diag)
