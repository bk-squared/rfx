"""Design-box admission, edge windows, and resolved E-update coefficients."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from rfx.core.yee import lumped_components


def design_box_bounds(bounds, grid):
    """``(bounds, realized cell counts)`` of a design box on this grid.

    One spelling for the permittivity box (#1179) and the occupancy box
    (#1183): the REALIZED cell counts are what a design array must match,
    and a box that is empty or leaves the grid is an error, not a clamp.
    """
    i0, i1, j0, j1, k0, k1 = (int(v) for v in bounds)
    bounds = (i0, i1, j0, j1, k0, k1)
    box_shape = (i1 - i0, j1 - j0, k1 - k0)
    if any(n <= 0 for n in box_shape):
        raise ValueError(
            f"design box {bounds} is empty (realized cell counts "
            f"{box_shape}); it must span at least one cell per axis.")
    for axis, (lo, hi, n) in enumerate(
            zip((i0, j0, k0), (i1, j1, k1), grid.shape)):
        if lo < 0 or hi > n:
            raise ValueError(
                f"design box {bounds} leaves the grid on axis "
                f"{'xyz'[axis]}: cells [{lo}, {hi}) against {n} cells.")
    return bounds, box_shape


def design_box_window(bounds, shape):
    """``(write bounds, computation window, write slice in it, box slice)``.

    The same window arithmetic :func:`rfx.boundaries.pec.pec_occupancy_box_keep`
    does for a traced occupancy, for the same reason. A cell's permittivity
    reaches the E components on the edges incident to it, and those are the
    edges at the cell and at its PLUS neighbour along each transverse axis
    (:func:`edge_averaged_materials` averages over BACKWARD neighbours). So the
    design material moves the update one cell past the box on the plus side —
    that layer must be written. One cell of context on the MINUS side is
    computed and not written: its own backward neighbour lies outside the
    window, so its value is wrong, while every cell in the write window has
    all four of its incident cells inside.
    """
    i0, i1, j0, j1, k0, k1 = (int(v) for v in bounds)
    lo = (i0, j0, k0)
    hi = (i1, j1, k1)
    c_lo = tuple(max(v - 1, 0) for v in lo)
    c_hi = tuple(min(v + 1, n) for v, n in zip(hi, shape))
    win = tuple(slice(a, b) for a, b in zip(c_lo, c_hi))
    write_bounds = (lo[0], c_hi[0], lo[1], c_hi[1], lo[2], c_hi[2])
    inner = tuple(slice(a - c, b - c) for a, b, c in zip(lo, c_hi, c_lo))
    box_local = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, c_lo))
    return write_bounds, win, inner, box_local


def held_edge_masks(held_edges, write_bounds):
    """One boolean array per E component over the write window, True on a
    held edge; ``None`` for a component with no held edge."""
    import numpy as np
    shape = tuple(write_bounds[2 * d + 1] - write_bounds[2 * d]
                  for d in range(3))
    masks = [None, None, None]
    for axis, i, j, k in held_edges:
        if masks[axis] is None:
            masks[axis] = np.zeros(shape, dtype=bool)
        masks[axis][i - write_bounds[0], j - write_bounds[2],
                    k - write_bounds[4]] = True
    return masks



class DesignBoxCoeffs(NamedTuple):
    """Resolved ``(Ca, Cb)`` for a design box, built once outside the scan.

    ``bounds`` is the WRITE window, not the declared box: since #1210 the
    design material reaches one cell past the box on the plus side of each
    transverse axis (:func:`_design_box_edge_coeffs`). ``ca`` and ``cb`` are
    3-tuples of window-shaped arrays, one per E component in x, y, z order —
    per-component because of the edge average, and because a design
    conductivity may itself be per-component (#1216).
    """
    bounds: tuple
    ca: Any
    cb: Any


def resolve_design_box(
    spec: "DesignBoxSpec",  # noqa: F821 -- declaration stays in simulation; no import cycle
    *,
    grid,
    materials,
    dt,
    use_cpml: bool,
    use_upml: bool,
    cpml_axes: str,
    use_debye: bool,
    use_lorentz: bool,
    use_kerr: bool,
    aniso_eps,
    aniso_inv_eps,
    stencil_order: int,
    bloch,
    sheet_impedance,
    cell_metas,
    periodic=(False, False, False),
) -> "DesignBoxCoeffs":
    """Check a design box against this run, then build its Ca/Cb (#1179).

    The box carries its own permittivity, so ``materials`` stays constant and
    the E update inside the box is redone from these coefficients. That is
    only the same physics while NOTHING ELSE in the step reads ``materials``
    at a design cell — and the checks below are how that is decided, in the
    one place where every resolved flag and every port/source cell is in
    hand. Everything unsupported raises; nothing is silently dropped, because
    a dropped design variable is a zero gradient that looks like a converged
    optimisation.

    Raises for: UPML, Debye/Lorentz dispersion, anisotropic / subpixel
    permittivity, Kerr, ``stencil_order=4``, the oblique-periodic Bloch path,
    a box that reaches into the CPML absorber, and a box holding a source,
    port, RLC element or surface-impedance sheet edge -- except a lumped or
    wire port's own edges listed in ``spec.held_edges``, which keep their
    ``materials`` coefficients (:func:`_design_box_edge_coeffs`).

    ``cell_metas`` entries are ``(i, j, k)`` or ``(i, j, k, component)``; a
    source or port cell is accepted only with its component, on a held edge.
    """
    unsupported = []
    if use_upml:
        unsupported.append("boundary='upml'")
    if use_debye:
        unsupported.append("debye dispersion")
    if use_lorentz:
        unsupported.append("lorentz dispersion")
    if aniso_eps is not None:
        unsupported.append("aniso_eps (subpixel/anisotropic eps)")
    if aniso_inv_eps is not None:
        unsupported.append("aniso_inv_eps (Kottke inverse-eps tensor)")
    if use_kerr:
        unsupported.append("Kerr nonlinearity")
    if stencil_order != 2:
        unsupported.append(f"stencil_order={stencil_order}")
    if bloch is not None:
        unsupported.append("oblique-periodic Bloch path")
    if unsupported:
        raise NotImplementedError(
            "the design-box permittivity rebuilds only the plain "
            "lossy Yee E update inside its box, so every path that computes "
            "E some other way is rejected rather than silently ignored. "
            "Unsupported feature(s) present: " + ", ".join(unsupported)
            + ". Use eps_override (the whole-grid traced permittivity) for "
            "these — it is exact on every path, at a grid-sized AD tape."
        )

    bounds, box_shape = design_box_bounds(spec.bounds, grid)
    i0, i1, j0, j1, k0, k1 = bounds
    # #1210: the design material reaches one cell past the box on the plus
    # side of every axis, and the redo WRITES there. Every fence below is
    # therefore held against the write window, not the declared box — a
    # source, a sheet edge or an absorber cell in that layer is the same
    # collision the declared box was already fenced against.
    _write_bounds, _, _, _ = design_box_window(bounds, tuple(grid.shape))
    w_i0, w_i1, w_j0, w_j1, w_k0, w_k1 = _write_bounds
    # #1210: the window is built with the non-periodic convention -- it clips
    # at the grid face where the grid-wide update would WRAP. That is only a
    # difference for a box that touches a periodic face, and there it is a
    # silent one, so it is refused rather than approximated. The occupancy
    # box's sibling (``pec_occupancy_box_keep``) refuses periodic axes for the
    # same reason.
    for _axis, _name in enumerate("xyz"):
        if not periodic[_axis]:
            continue
        _lo, _hi = bounds[2 * _axis], bounds[2 * _axis + 1]
        _n = grid.shape[_axis]
        if _lo == 0 or _hi == _n:
            raise NotImplementedError(
                f"a design box flush against a PERIODIC {_name} face "
                f"is not supported: cells [{_lo}, {_hi}) on a {_n}-cell "
                f"periodic axis. An E component takes the mean of its four "
                f"incident cells, and across a periodic seam those "
                f"neighbours wrap -- the box's one-cell window does not. Move "
                f"the box off the {_name} faces, or use eps_override (the "
                f"whole-grid traced permittivity), which wraps correctly.")

    # The absorber. apply_cpml_e's psi correction is written at the [:n] /
    # [-n:] face slabs, and its coefficient is dt/(eps_r*EPS_0) from
    # ``materials`` -- the one array this design stops tracing. A box inside
    # an ABSORBING slab would therefore be absorbed with the BACKGROUND
    # permittivity while it is updated with the design one.
    #
    # Per FACE, not per axis. The kernel's window width
    # (``_axis_buffer_depths``) is the axis MAXIMUM of the two pads, floored
    # at 1, and it is applied to both faces -- but a face whose profile is
    # the all-no-op one (b=1, c=0, kappa=1) adds exactly zero however wide
    # that window is: psi stays at its zero init and the kappa term is
    # (1/1 - 1). A face gets that profile exactly when its allocated pad is
    # 0, i.e. PEC / PMC / periodic (``Grid._face_pad``), and an absorbing
    # face's own pad IS its active layer count (the invariant stated in
    # ``_axis_buffer_depths``'s docstring). Reading the axis maximum instead
    # refused the ordinary patch on a ground plane: x_lo PEC + x_hi CPML at
    # cpml_layers=5 gives axis depths (5, 1, 1), so a box sitting on the PEC
    # wall was rejected for entering a 5-layer absorber that face does not
    # have, and the y/z floor of 1 rejected any box touching a PEC face.
    #
    # A per-face pad is that proxy only on an axis the GRID says it absorbs
    # on. ``init_cpml`` builds a real profile on every axis ``cpml_axes`` of
    # the RUN names, and a grid built with a narrower ``cpml_axes`` allocates
    # no pad on the others -- so "pad 0" there means "no padding cells", not
    # "no absorber" (the trap ``_axis_buffer_depths`` documents, measured at
    # a 49 % energy change). On such an axis, and on a duck-typed grid with
    # no per-face pads, fall back to the axis window: nothing is known to be
    # a no-op there.
    if use_cpml or use_upml:
        from rfx.boundaries.cpml import _axis_buffer_depths
        depths = _axis_buffer_depths(grid, grid.cpml_layers)
        grid_axes = getattr(grid, "cpml_axes", cpml_axes)
        for axis, name in enumerate("xyz"):
            if name not in cpml_axes:
                continue  # no correction is written on this axis at all
            if name in grid_axes and hasattr(grid, f"pad_{name}_lo"):
                pad_lo = int(getattr(grid, f"pad_{name}_lo"))
                pad_hi = int(getattr(grid, f"pad_{name}_hi"))
            else:
                pad_lo = pad_hi = int(depths[axis])
            lo = _write_bounds[2 * axis]
            hi = _write_bounds[2 * axis + 1]
            n = grid.shape[axis]
            if lo < pad_lo or hi > n - pad_hi:
                raise ValueError(
                    f"design box cells [{bounds[2 * axis]}, "
                    f"{bounds[2 * axis + 1]}) on axis {name} reach "
                    f"into the CPML absorber ({pad_lo} layer(s) at {name}_lo "
                    f"and {pad_hi} at {name}_hi, on a {n}-cell axis; the box's "
                    f"update window is [{lo}, {hi}), one cell wider on the "
                    f"plus side under edge averaging). The "
                    f"absorber builds its own coefficient from the background "
                    f"permittivity, which the design box no longer carries. "
                    f"Move the box into cells [{pad_lo}, {n - pad_hi - 1}).")

    def _in_box(cell) -> bool:
        i, j, k = (int(v) for v in cell[:3])
        return (w_i0 <= i < w_i1) and (w_j0 <= j < w_j1) and (w_k0 <= k < w_k1)

    # Held edges: a port's own edges, kept on the coefficients ``materials``
    # gives them (DesignBoxSpec.held_edges). Each must be an E edge in the
    # window -- one outside it holds nothing, and is a caller error.
    held = set()
    for edge in (getattr(spec, "held_edges", ()) or ()):
        edge = tuple(int(v) for v in edge)
        if len(edge) != 4 or edge[0] not in (0, 1, 2) or not _in_box(edge[1:]):
            raise ValueError(
                f"DesignBoxSpec.held_edges entry {edge} is not an (axis, i, "
                f"j, k) E edge inside the design box's update window "
                f"{_write_bounds} (axis 0, 1, 2 for Ex, Ey, Ez).")
        held.add(edge)
    held = tuple(sorted(held))
    _held_kinds = ("source", "lumped port", "wire port")
    _axis_of = {"ex": 0, "ey": 1, "ez": 2}

    def _on_held_edge(kind, cell) -> bool:
        if kind not in _held_kinds or len(cell) != 4:
            return False
        axis = _axis_of.get(str(cell[3]).lower())
        return axis is not None and (axis, *(int(v) for v in cell[:3])) in held

    # A source or port cell reads the material AT SETUP to turn a current
    # into a field increment (``_source_cell_cb``) or to fold an impedance
    # into sigma. Those reads see the background array, so a design cell
    # under one of them would drive with the wrong permittivity -- unless its
    # edge is held, and then the update there reads the background too.
    for kind, cells in cell_metas:
        hits = [tuple(int(v) for v in c[:3]) for c in cells
                if _in_box(c) and not _on_held_edge(kind, c)]
        if hits:
            how = (" A lumped or wire port may stay inside: "
                   "Simulation.forward(design_box_holds_ports=True) keeps "
                   "the port's own edges on the coefficients the drawn "
                   "materials give them (DesignBoxSpec.held_edges on this "
                   "entry)")
            if kind == "source":
                how += ("; a soft source cannot be held -- a port's drive is "
                        "accepted only on the port's held edges.")
            elif kind not in _held_kinds:
                how = (f" Only a lumped or wire port's own edges can be held "
                       f"(forward(design_box_holds_ports=True)); a {kind} "
                       f"cannot.")
            else:
                how += "."
            raise ValueError(
                f"the design box {bounds} holds {kind} cell(s) {hits}. "
                f"That cell's drive/load coefficient is built from the "
                f"background permittivity before the time loop starts, so "
                f"the design permittivity would not reach it. Move the box "
                f"off the {kind} cells, or use eps_override." + how)

    sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))

    # A per-edge design conductivity laid over a lumped load (a port's
    # termination, an RLC element's R) that is not on a held edge. The tuple
    # REPLACES the edge's conductivity at the box indices (#1216), load
    # included, so the load would silently vanish. A cell design value keeps
    # it -- the lay-in in ``_design_box_edge_coeffs`` puts the stamp back on
    # top of the cell's volume material -- so only this form is refused. A
    # passive termination that registers no accumulator leaves no cell in
    # ``cell_metas``, so the stamps are read from ``materials`` itself, and
    # only when they are concrete (a traced R or a traced mesh keeps the cell
    # checks above).
    if isinstance(spec.sigma, (tuple, list)):
        import numpy as np
        from rfx.core.jax_utils import is_tracer
        for axis, part in enumerate(lumped_components(
                getattr(materials, "sigma_lumped", None))):
            if part is None or is_tracer(part):
                continue
            nz = np.argwhere(np.asarray(jnp.asarray(part)[sl]) != 0)
            loose = [(int(a) + i0, int(b) + j0, int(c) + k0)
                     for a, b, c in nz
                     if (axis, int(a) + i0, int(b) + j0, int(c) + k0)
                     not in held]
            if loose:
                raise ValueError(
                    f"the design box {bounds} lays a per-edge design "
                    f"conductivity over a lumped load on the "
                    f"{'xyz'[axis]} edge at cell(s) {loose} (a port's "
                    f"termination or an RLC element). A per-edge "
                    f"conductivity REPLACES the edge's conductivity, load "
                    f"included, so the load would be removed. Give the "
                    f"design conductivity as one cell array instead (the "
                    f"load then stays on its own edge and the design value "
                    f"is the cell's volume material), or move the box off "
                    f"the load. A lumped or wire port inside a permittivity "
                    f"box can instead be held with "
                    f"forward(design_box_holds_ports=True); an MSL port's "
                    f"termination cannot be held.")
    w_sl = (slice(w_i0, w_i1), slice(w_j0, w_j1), slice(w_k0, w_k1))
    if sheet_impedance is not None:
        for mask in (sheet_impedance.mask_ex, sheet_impedance.mask_ey,
                     sheet_impedance.mask_ez):
            if mask is not None and bool(jnp.any(jnp.asarray(mask)[w_sl])):
                raise ValueError(
                    f"a surface_impedance_f0 sheet has loaded edges "
                    f"inside the design box {bounds}. The sheet operator "
                    f"REPLACES the E update at those edges with coefficients "
                    f"built from the background permittivity, so it would "
                    f"overwrite the design update. Move the box off the "
                    f"sheet, or use eps_override.")

    # Imported here: rfx.model.materials imports rfx.nonuniform, which imports rfx.model.
    from rfx.model.materials import _design_box_edge_coeffs
    eps_r = jnp.asarray(spec.eps_r)
    if tuple(eps_r.shape) != box_shape:
        raise ValueError(
            f"design permittivity has shape {tuple(eps_r.shape)} but the "
            f"design box {bounds} realizes {box_shape} cells.")
    sigma = materials.sigma[sl] if spec.sigma is None else spec.sigma
    if isinstance(sigma, (tuple, list)):
        # Per-component conductivity: one (Ca, Cb) pair per E component, in
        # x, y, z order. The permittivity stays shared — a design SHEET is
        # one material whose current is anisotropic, not three materials.
        if len(sigma) != 3:
            raise ValueError(
                f"a per-component design conductivity is a 3-tuple "
                f"(sigma_x, sigma_y, sigma_z); got {len(sigma)} entries.")
        for name, s in zip("xyz", sigma):
            s = jnp.asarray(s)
            if tuple(jnp.shape(s)) != box_shape:
                raise ValueError(
                    f"design conductivity sigma_{name} has shape "
                    f"{tuple(jnp.shape(s))} but the design box {bounds} "
                    f"realizes {box_shape} cells.")
        # The tuple is already per E edge (#1216) and is taken as-is; the
        # shared permittivity is still a cell quantity and is edge-averaged
        # over the window (#1210). See _design_box_edge_coeffs.
        write_bounds, ca, cb = _design_box_edge_coeffs(
            bounds, eps_r, tuple(jnp.asarray(s) for s in sigma), materials,
            dt, tuple(grid.shape), held_edges=held, grid=grid)
        return DesignBoxCoeffs(bounds=write_bounds, ca=ca, cb=cb)
    sigma = jnp.asarray(sigma)
    if tuple(jnp.shape(sigma)) != box_shape:
        raise ValueError(
            f"design conductivity has shape {tuple(jnp.shape(sigma))} but "
            f"the design box {bounds} realizes {box_shape} cells.")
    # #1210: per-component coefficients over the write window, the design
    # values laid into a copy of the background so the box-edge cells average
    # with the constant background exactly as the grid-wide update would.
    # ``sigma=None`` hands the window its own conductivity: the slice read
    # above is the cell TOTAL, a held port's load included, and laying it
    # back as a design (volume) value would count that load twice.
    write_bounds, ca, cb = _design_box_edge_coeffs(
        bounds, eps_r, None if spec.sigma is None else sigma, materials, dt,
        tuple(grid.shape), held_edges=held, grid=grid)
    return DesignBoxCoeffs(bounds=write_bounds, ca=ca, cb=cb)

