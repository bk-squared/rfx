"""Realized electric-edge keep factors for the relaxed PEC cell input.

The single-device uniform and graded steps multiply by these arrays,
built once before the time loop.
"""

from rfx.errors import RfxRefusal
from functools import partial

import jax
import jax.numpy as jnp

from rfx.boundaries.pec import _shift, realized_pec_edge_masks


def clip_occupancy(x):
    """Clip to [0, 1] with slope 1 on the closed interval (JAX's clip has slope 1/2 at a bound, #1926)."""
    return jnp.where((x >= 0) & (x <= 1), x, jnp.clip(x, 0.0, 1.0))


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
    occ = clip_occupancy(occupancy.astype(dtype))
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


def _local_released(cells, view):
    """Select global release indices in the input rows and rebase x."""
    if "released_cells" in view.globals:
        return view.globals["released_cells"]
    cells = cells[(cells[:, 0] >= view.lo) & (cells[:, 0] < view.hi)]
    return cells.at[:, 0].add(-view.lo)


def build_edge_keep(occupancy, *, dtype=jnp.float32,
                    periodic=(False, False, False), sheet_edge_masks=None,
                    released_cells=None, shape=None, design=None, view=None):
    """The factor each E edge is multiplied by, as three arrays on the E edges.

    ``released_cells`` are integer cell indices, shape (N, 3), including N=0,
    set to 0 before the factor is formed. Design values replace their cells;
    their write window replaces the background factor (including its static
    sheet fold), as the step's window rewrite does. No occupancy and no
    design input gives None, even with static sheets.
    With a view, arrays and shape are local; release indices remain global.
    """
    if view is not None:
        if design is not None:
            raise NotImplementedError("build_edge_keep: design occupancy with a view is not supported")
        if periodic[0]:
            raise NotImplementedError("build_edge_keep: periodic x with a view is not supported")
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
        if view is not None:
            cells = _local_released(cells, view)
        occ = occ.at[cells[:, 0], cells[:, 1], cells[:, 2]].set(0.0)
    keep = _edge_factors(occ, sheet_edge_masks if occupancy is not None else None,
                         dtype=dtype, periodic=periodic)
    if design is not None:
        write, window_keep = _window_factors(
            design.bounds, design.occupancy, occ, dtype=dtype, periodic=periodic)
        keep = tuple(k.at[write].set(w) for k, w in zip(keep, window_keep))
    return keep


def with_edge_keep(materials, occupancy, *, dtype, periodic=(False,) * 3,
                   sheet_edge_masks=None, view=None):
    """Publish the static stage's factor using its already-resolved masks.

    ``dtype`` is the E field's. The factor is formed in that real type but
    never below float32: a half-precision run multiplies by the float32
    product rounded once, and a drive keeps its float32 weight.
    """
    if occupancy is None:
        return materials
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise NotImplementedError(
            "a partial-conductor occupancy is not supported with complex fields "
            "(the Bloch/oblique-incidence envelope): the combination has no "
            "validation. Use a real-field run, or a hard PEC geometry.")
    real = jnp.promote_types(jnp.finfo(dtype).dtype, jnp.float32)
    return materials._replace(edge_keep=build_edge_keep(
        occupancy, dtype=real, periodic=periodic,
        sheet_edge_masks=sheet_edge_masks, view=view))


def publish_occupancy(materials, occupancy, *, dtype=jnp.float32,
                      periodic=(False, False, False), sheets=(), wires=(),
                      edge_masks=None, released_cells=None, design=None, view=None):
    """Publish the same static sheet/wire fold as the single-device steps."""
    static = None
    if occupancy is not None and (sheets or wires):
        static = realized_pec_edge_masks(None, sheets=sheets, wires=wires,
                                         periodic=periodic, view=view)
        if edge_masks is not None:
            static = tuple(s & m for s, m in zip(static, edge_masks))
    keep = build_edge_keep(occupancy, dtype=dtype, periodic=periodic,
        sheet_edge_masks=static, released_cells=released_cells,
        shape=materials.eps_r.shape, design=design, view=view)
    return materials._replace(edge_keep=keep)


def publish_slab_occupancy(materials, cut_occupancy, mesh):
    """Form each slab's factors locally, reading only its low occupancy ghost.

    Owned rows are the contract. Both published ghost rows are inert ones;
    the consumer still leaves ghosts untouched and refills E after weighting.
    There is no whole-domain weight construction or gather on this route.
    """
    if cut_occupancy is None:
        return materials._replace(edge_keep=None)
    from jax.sharding import PartitionSpec as P

    @partial(jax.shard_map, mesh=mesh, in_specs=P('x'),
             out_specs=(P('x'), P('x'), P('x')), check_vma=False)
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
                "a design occupancy box does not combine with "
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
            from rfx.model.thin_conductors import conductivity_envelope  # lazy: import cycle
            print(f"[kottke debug] sigma min={float(jnp.min(conductivity_envelope(materials))):.3e} "
                  f"max={float(jnp.max(conductivity_envelope(materials))):.3e}", file=_sys.stderr, flush=True)
    return pec_occupancy_for_run, aniso_inv_eps_run


@jax.custom_jvp
def _layout_switch(value):
    """Binary layout identity; its derivative is not a physical sensitivity."""
    return value


@_layout_switch.defjvp
def _layout_switch_jvp(primals, tangents):
    raise TypeError(
        "pinned_sheet_layout is a layout switch: its derivative is not a physical "
        "sensitivity (the per-step multiply is an exponential decay); "
        "differentiate a resolved design instead")


def admit_sheet_layout(sheets, layout):
    """Validate concrete binary entries and always install the AD refusal."""
    import numpy as np
    from rfx.core.jax_utils import is_tracer
    if not isinstance(layout, tuple) or len(layout) != len(sheets):
        raise ValueError("pinned_sheet_layout must be a tuple aligned with add_pinned_sheet calls")
    addressed = []
    for index, (sheet, pair) in enumerate(zip(sheets, layout, strict=True)):
        if pair is None:
            continue
        label = f"pinned_sheet_layout sheet {index} ({sheet.name!r})"
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{label}: expected two tangential edge arrays")
        lengths = (sheet.i_range[1] - sheet.i_range[0], sheet.j_range[1] - sheet.j_range[0])
        arrays = []
        axes = tuple(a for a in range(3) if a != sheet.normal_axis)
        for t, value in enumerate(pair):
            value = jnp.asarray(value)
            shape = tuple(n + (axis != t) for axis, n in enumerate(lengths))
            if value.shape != shape:
                raise ValueError(f"{label}: edge {'xyz'[axes[t]]} shape {value.shape}, expected {shape}")
            if not is_tracer(value):
                bad = np.argwhere((np.asarray(value) != 0) & (np.asarray(value) != 1))
                if len(bad):
                    edge = tuple(int(v) for v in bad[0])
                    raise ValueError(f"{label}: edge {'xyz'[axes[t]]}{edge} must be exactly 0 or 1")
            arrays.append(_layout_switch(value))
        addressed.append((index, sheet, tuple(arrays)))
    return tuple(addressed)


def graded_sheet_masks(occupancy, sheets, wires, edge_masks, *, periodic):
    """Static sheet/wire fold, intersected with the port-released hard mask."""
    if occupancy is None or not (sheets or wires):
        return None
    static = realized_pec_edge_masks(None, sheets=sheets, wires=wires, periodic=periodic)
    if edge_masks is not None:
        static = tuple(s & m for s, m in zip(static, edge_masks))
    return static


def layout_edge_indices(grid, sheet, tangent):
    """Declared local edge coordinates mapped to padded (and periodic) nodes."""
    import numpy as np
    from rfx.materials.thin_conductor import pinned_sheet_spec
    spec = pinned_sheet_spec(grid, sheet)  # retain declaration range/plane validation
    axes = tuple(a for a in range(3) if a != sheet.normal_axis)
    ranges = (sheet.i_range, sheet.j_range)
    local = np.meshgrid(*(np.arange(lo, hi + (t != tangent)) + grid.axis_pads[a]
                          for t, (a, (lo, hi)) in enumerate(zip(axes, ranges))), indexing='ij')
    indices = [np.full(local[0].shape, int(spec.plane), dtype=int) for _ in range(3)]
    for a, values in zip(axes, local):
        indices[a] = values % grid.shape[a] if 'xyz'[a] in grid.periodic_axes else values
    return axes[tangent], tuple(indices)


def with_graded_edge_keep(materials, occupancy, layout, *, grid, sources,
                          waveguide_ports, dtype, sheet_edge_masks, periodic,
                          cells=None, sheets=(), wires=()):
    """Fold the layout switch into the cell keep.

    Full-period node 0 and node N rows address the same physical edges.
    Concrete rows must agree; traced rows multiply their keep factors,
    so either row being metal (1) makes the shared edge metal.
    """
    materials = with_edge_keep(materials, occupancy, dtype=dtype,
                              sheet_edge_masks=sheet_edge_masks, periodic=periodic)
    if not layout:
        return materials
    from rfx.model.conductors import refuse_layout_drives
    edge_sets = [(index, sheet, tangent, *layout_edge_indices(grid, sheet, tangent), values)
                 for index, sheet, pair in layout for tangent, values in enumerate(pair)]
    refuse_degenerate_layout(layout, grid)
    admit_periodic_layout(layout, grid)
    refuse_layout_seams(layout, grid, cells, sheets, wires, periodic)
    refuse_layout_drives(edge_sets, sources, waveguide_ports)
    keep = list(materials.edge_keep if materials.edge_keep is not None else
                (jnp.ones(grid.shape, dtype=dtype),) * 3)
    for _index, _sheet, _tangent, axis, indices, values in edge_sets:
        keep[axis] = keep[axis].at[indices].multiply(1 - values.astype(keep[axis].dtype))
    return materials._replace(edge_keep=tuple(keep))


def refuse_layout_seams(layout, grid, cells, sheets, wires, periodic):
    """Reject declared cross-sheet edges that no layout array can address.

    Compare the production masks: full(C, F+A, W) minus fixed(C, F, W)
    and each addressed sheet's own mask. Only static declarations are read;
    the layout values do not participate in admission.
    """
    import numpy as np
    from rfx.core.jax_utils import is_tracer
    from rfx.materials.thin_conductor import pinned_sheet_spec
    with jax.ensure_compile_time_eval():
        specs = tuple(pinned_sheet_spec(grid, sheet) for _, sheet, _ in layout)
        sheets, wires = tuple(sheets or ()), tuple(wires or ())
        # Traced cells cannot prove a seam covered: omit them, conservatively.
        cells = np.zeros(grid.shape, dtype=bool) if cells is None or is_tracer(cells) else cells
        full = realized_pec_edge_masks(cells, sheets=sheets + specs, wires=wires, periodic=periodic)
        covered = [np.asarray(mask).copy() for mask in
                   realized_pec_edge_masks(cells, sheets=sheets, wires=wires, periodic=periodic)]
        for spec in specs:
            own = realized_pec_edge_masks(None, sheets=(spec,), periodic=periodic)
            for axis in range(3):
                covered[axis] |= np.asarray(own[axis])
        for axis, mask in enumerate(full):
            seam = np.argwhere(np.asarray(mask) & ~covered[axis])
            if not len(seam):
                continue
            edge = tuple(int(v) for v in seam[0])
            neighbour = list(edge)
            neighbour[axis] = (neighbour[axis] + 1) % grid.shape[axis]
            owner = "pinned_sheet_layout"
            for (index, sheet, _), spec in zip(layout, specs):
                if axis == spec.normal_axis and grid.shape[axis] != 1:
                    continue
                footprint = np.asarray(spec.footprint)
                if footprint[edge] or footprint[tuple(neighbour)]:
                    owner = f"pinned_sheet_layout sheet {index} ({sheet.name!r})"
                    break
            raise RfxRefusal(
                f"{owner}: seam edge e{'xyz'[axis]}{edge} is outside the layout arrays; "
                "overlap the addressed sheet with its neighbour by one node, "
                "leave at least one empty node between them, "
                "or declare both as one addressed sheet")


def refuse_degenerate_layout(layout, grid):
    """The length-one normal's PEC component has no layout array (R12.1)."""
    for index, sheet, _ in layout:
        if grid.shape[sheet.normal_axis] == 1:
            _, indices = layout_edge_indices(grid, sheet, 0)
            edge = tuple(int(v.flat[0]) for v in indices)
            raise RfxRefusal(
                f"pinned_sheet_layout sheet {index} ({sheet.name!r}): "
                f"length-1 normal PEC stage owns edge e{'xyz'[sheet.normal_axis]}{edge}; "
                "the graded lane layout switch needs more than one cell along the sheet normal")


def admit_periodic_layout(layout, grid):
    """Check duplicate concrete endpoint rows without concretizing traced M."""
    import numpy as np
    from rfx.core.jax_utils import is_tracer
    for index, sheet, pair in layout:
        axes = tuple(a for a in range(3) if a != sheet.normal_axis)
        for local_axis, (axis, span) in enumerate(zip(axes, (sheet.i_range, sheet.j_range))):
            if 'xyz'[axis] not in grid.periodic_axes or span != (0, grid.shape[axis]):
                continue
            tangent = 1 - local_axis
            values = pair[tangent]
            if is_tracer(values):
                continue
            values = np.asarray(values)
            bad = np.flatnonzero(np.take(values, 0, axis=local_axis) !=
                                 np.take(values, -1, axis=local_axis))
            if len(bad):
                component, indices = layout_edge_indices(grid, sheet, tangent)
                local = (0, int(bad[0])) if local_axis == 0 else (int(bad[0]), 0)
                edge = tuple(int(v[local]) for v in indices)
                raise ValueError(
                    f"pinned_sheet_layout sheet {index} ({sheet.name!r}): periodic "
                    f"node-0/node-N rows disagree at edge e{'xyz'[component]}{edge}")


def graded_conductor_masks(cells, occupancy, sheets, wires, edge_masks, *, periodic):
    """Realize hard masks once, retaining masks already released by ports."""
    sheets, wires = tuple(sheets or ()), tuple(wires or ())
    if edge_masks is None and (cells is not None or sheets or wires):
        edge_masks = realized_pec_edge_masks(cells, sheets=sheets, wires=wires, periodic=periodic)
    return edge_masks, graded_sheet_masks(occupancy, sheets, wires, edge_masks, periodic=periodic)
