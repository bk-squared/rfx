"""Realized electric-edge keep factors for the relaxed PEC cell input.

No run builds these arrays yet: the step bodies still form the factor from the
cell occupancy themselves, and building it a second time cost 6-10 % host
memory for nothing. A lane's owner calls the builder when it switches that
step to read the array; until then the tests pin it against each step.
"""
from functools import partial

import jax.numpy as jnp

from rfx.boundaries.pec import _shift, realized_pec_edge_masks


def _edge_factors(occupancy, sheet_edge_masks, *, dtype, periodic):
    """keep = prod(1 - o) over the four cells of each E edge, products only.

    One subtraction per cell, then multiplies in a fixed association:
    ``((1-o)(1-o_t1)) * ((1-o_t2)(1-o_t1t2))`` with t1 < t2 the two axes
    transverse to the component. Inside this expression no multiply feeds an
    add or a subtract, so the float32 array it returns is the same eager,
    under jit, inside a scan and per slab (the former ``1 - M`` with its
    chained noisy-OR was 1 ULP apart between those contexts on non-dyadic
    occupancies). That holds when the array is an output, an input, or is
    multiplied into a field. A caller that inlines this builder in its own
    compiled program and then ADDS or SUBTRACTS the factor (``1 - keep``) can
    still have the last multiply fused: build the array once, outside the
    loop, and pass it in. Outside a non-periodic wall the occupancy reads 0
    through the one neighbour rule (``_shift``), so the factor there is 1.
    Where the four-cell product is below half an ULP of 1 the old form gave
    exactly 0; this one keeps the product (at most about 3e-8).
    """
    occ = jnp.clip(occupancy.astype(dtype), 0.0, 1.0)
    keep = []
    for c in range(3):
        t1, t2 = (t for t in range(3) if t != c)
        back1 = _shift(occ, t1, periodic, +1)
        k = ((1.0 - occ) * (1.0 - back1)) * (
            (1.0 - _shift(occ, t2, periodic, +1))
            * (1.0 - _shift(back1, t2, periodic, +1)))
        if sheet_edge_masks is not None:
            k = jnp.where(sheet_edge_masks[c].astype(bool), jnp.zeros_like(k), k)
        keep.append(k)
    return tuple(keep)


def _window_factors(box, design_occupancy, occ, *, dtype, periodic):
    """The same product on a design box's window (tape stays window-sized).

    Window arithmetic of ``rfx.boundaries.pec.pec_occupancy_box_keep``: the
    write window is the box grown one cell on the plus side, the computation
    window one more cell on the minus side for the backward neighbours. A
    written value equals the whole-domain product of the merged occupancy
    bit for bit.
    """
    if any(periodic):
        # The existing refusal, verbatim, from the step's own helper.
        from rfx.boundaries.pec import pec_occupancy_box_keep
        pec_occupancy_box_keep(box, design_occupancy, shape=occ.shape,
                               dtype=dtype, pec_occupancy=occ, periodic=periodic)
    lo = tuple(int(v) for v in box[0::2])
    hi = tuple(int(v) for v in box[1::2])
    c_lo = tuple(max(v - 1, 0) for v in lo)
    c_hi = tuple(min(v + 1, n) for v, n in zip(hi, occ.shape))
    win = tuple(slice(a, b) for a, b in zip(c_lo, c_hi))
    write = tuple(slice(a, b) for a, b in zip(lo, c_hi))
    inner = tuple(slice(a - c, b - c) for a, b, c in zip(lo, c_hi, c_lo))
    box_local = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, c_lo))
    local = jnp.asarray(occ)[win].astype(dtype).at[box_local].set(
        jnp.asarray(design_occupancy).astype(dtype))
    # A window that starts inside the lattice has real neighbours in its
    # context layer; where it starts on the wall, ``_shift`` pads as the
    # whole-domain call does.
    keep = _edge_factors(local, None, dtype=dtype, periodic=(False,) * 3)
    return write, tuple(k[inner] for k in keep)


def build_edge_keep(occupancy, *, dtype=jnp.float32,
                    periodic=(False, False, False), sheet_edge_masks=None,
                    released_cells=None, shape=None, design=None):
    """The factor each E edge is multiplied by, as three arrays on the E edges.

    ``released_cells`` are integer cell indices, shape (N, 3), including N=0,
    set to 0 before the factor is formed. Design values replace their cells;
    their write window replaces the background factor (including its static
    sheet fold), as the step's window rewrite does. No occupancy and no
    design input gives None, even with static sheets.
    """
    if occupancy is None and design is None:
        return None
    if occupancy is None:
        occ = jnp.zeros(shape, dtype=dtype)
    else:
        occ = jnp.asarray(occupancy)
    if released_cells is not None:
        cells = jnp.asarray(released_cells)
        if cells.ndim != 2 or cells.shape[1] != 3 or not jnp.issubdtype(cells.dtype, jnp.integer):
            raise ValueError("released_cells must be an integer array of shape (N, 3)")
        occ = occ.at[cells[:, 0], cells[:, 1], cells[:, 2]].set(0.0)
    keep = _edge_factors(occ, sheet_edge_masks if occupancy is not None else None,
                         dtype=dtype, periodic=periodic)
    if design is not None:
        write, window_keep = _window_factors(
            design.bounds, design.occupancy, occ, dtype=dtype, periodic=periodic)
        keep = tuple(k.at[write].set(w) for k, w in zip(keep, window_keep))
    return keep


def publish_occupancy(materials, occupancy, *, dtype=jnp.float32,
                      periodic=(False, False, False), sheets=(), wires=(),
                      edge_masks=None, released_cells=None, design=None):
    """Publish the same static sheet/wire fold as the single-device steps."""
    static = None
    if occupancy is not None and (sheets or wires):
        static = realized_pec_edge_masks(None, sheets=sheets, wires=wires,
                                         periodic=periodic)
        if edge_masks is not None:
            static = tuple(s & m for s, m in zip(static, edge_masks))
    keep = build_edge_keep(occupancy, dtype=dtype, periodic=periodic,
        sheet_edge_masks=static, released_cells=released_cells,
        shape=materials.eps_r.shape, design=design)
    return materials._replace(edge_keep=keep)


def publish_slab_occupancy(materials, cut_occupancy, mesh):
    """Form each slab's factors locally, reading only its low occupancy ghost.

    Owned rows are the contract. Both published ghost rows are inert ones;
    the consumer still leaves ghosts untouched and refills E after weighting.
    There is no whole-domain weight construction or gather on this route.
    """
    if cut_occupancy is None:
        return materials._replace(edge_keep=None)
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    @partial(shard_map, mesh=mesh, in_specs=P('x'),
             out_specs=(P('x'), P('x'), P('x')), check_rep=False)
    def local(cells):
        keep = build_edge_keep(cells[:-1], dtype=jnp.float32)
        return tuple(jnp.concatenate((jnp.ones_like(k[:1]), k[1:],
                                       jnp.ones_like(k[:1])), axis=0)
                     for k in keep)
    return materials._replace(edge_keep=local(cut_occupancy))


def occupancy_from_design(shape, window, values):
    """Topology's existing full-grid float32 occupancy input, before release."""
    if values is None:
        return None
    return jnp.zeros(shape, dtype=jnp.float32).at[window].set(values)


def select_occupancy_operator(self, grid, materials, pec_occupancy_local,
                              periodic_bool, design_occupancy):
    """Keep the existing optional tensor path and all of its refusals."""
    import os
    aniso_inv_eps_run = None
    pec_occupancy_for_run = pec_occupancy_local
    if (pec_occupancy_local is not None and
            os.environ.get("RFX_PEC_OCC_KOTTKE", "0") not in ("0", "", "false", "False")):
        from rfx.current_moments import refuse_h_side_conductor
        refuse_h_side_conductor(
            self, "the Kottke occupancy lane (RFX_PEC_OCC_KOTTKE=1)")
        from rfx.geometry.smoothing import kottke_inv_eps_from_occupancy
        from rfx.core.yee import add_lumped_eps
        # The plain path's four-cell edge mean (#1213), not the per-cell
        # value: occupancy only scales it where a conductor sits (#1373).
        inv_baseline = tuple(
            (1.0 / eps_c).astype(jnp.float32)
            for eps_c in materials.components.eps)
        aniso_inv_eps_run = kottke_inv_eps_from_occupancy(
            grid,
            pec_occupancy_local,
            aniso_inv_eps_baseline=inv_baseline,
            periodic=periodic_bool,
        )
        # Occupancy acts on the volume; the capacitor stays on its
        # declared edge, added once after that correction (#1263).
        aniso_inv_eps_run = add_lumped_eps(
            aniso_inv_eps_run, materials.eps_r_lumped, inverse=True)
        pec_occupancy_for_run = None
        if design_occupancy is not None:
            raise NotImplementedError(
                "a design occupancy box (#1183) does not combine with "
                "the Kottke occupancy lane (RFX_PEC_OCC_KOTTKE=1). That "
                "lane turns the occupancy into an inverse-eps tensor "
                "inside the E update and sets pec_occupancy_for_run to "
                "None, so the design values never reach the update and "
                "the box's own 1 - M window then double-corrects the "
                "field the tensor already handled — the anti-pattern "
                "the comment above names. Measured: value 19x and "
                "gradient 2.8x off pec_occupancy_override, silently. "
                "Use pec_occupancy_override on that lane, or unset "
                "RFX_PEC_OCC_KOTTKE.")
        if os.environ.get("RFX_PEC_OCC_KOTTKE_DEBUG", "0") not in ("0", "", "false", "False"):
            import sys as _sys
            _ix, _iy, _iz = aniso_inv_eps_run
            print(f"[kottke debug] occ shape={pec_occupancy_local.shape} "
                  f"min={float(jnp.min(pec_occupancy_local)):.3e} "
                  f"max={float(jnp.max(pec_occupancy_local)):.3e}", file=_sys.stderr, flush=True)
            print(f"[kottke debug] inv_xx min={float(jnp.min(_ix)):.3e} "
                  f"max={float(jnp.max(_ix)):.3e} "
                  f"any_nan={bool(jnp.any(jnp.isnan(_ix)))}", file=_sys.stderr, flush=True)
            print(f"[kottke debug] eps_r min={float(jnp.min(materials.eps_r)):.3e} "
                  f"max={float(jnp.max(materials.eps_r)):.3e}", file=_sys.stderr, flush=True)
            print(f"[kottke debug] sigma min={float(jnp.min(materials.sigma)):.3e} "
                  f"max={float(jnp.max(materials.sigma)):.3e}", file=_sys.stderr, flush=True)
    return pec_occupancy_for_run, aniso_inv_eps_run
