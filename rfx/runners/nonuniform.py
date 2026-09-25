"""Non-uniform grid run path extracted from Simulation."""

from __future__ import annotations

import warnings
from dataclasses import replace

import jax
import numpy as np
import jax.numpy as jnp

from rfx.grid import C0
from rfx.core.jax_utils import is_tracer
from rfx.core.yee import MaterialArrays
from rfx.materials.debye import init_debye
from rfx.materials.lorentz import init_lorentz
from rfx.materials.thin_conductor import check_sheet_occupancy, sheet_bounds
from rfx.sources.waveguide_port import _node_span_to_cell_span
from rfx.sources.sources import stamp_lumped_sigma as _stamp_lumped_sigma
from rfx.nonuniform import (
    NonUniformGrid,
    e_node_dual_spacing_at,
    e_node_dual_spacings,
    interior_cells,
    make_nonuniform_grid,
    port_metric,
    port_metric_axes,
    run_nonuniform,
    run_nonuniform_until_decay,
    make_current_source,
)


INTERFACE_EPS_RULES = ("sampled", "dual_average")


def _validate_interface_eps_nu(sim, *, subpixel_smoothing=False, eps_override=None):
    """Refuse combinations that cannot consume the component eps rule."""
    if getattr(sim, "_interface_eps", "sampled") == "sampled":
        return
    from rfx.core.jax_utils import is_tracer

    if subpixel_smoothing or eps_override is not None:
        raise ValueError("interface_eps='dual_average' cannot combine with subpixel_smoothing or eps_override")
    if sim._thin_conductors or sim._lumped_rlc:
        raise ValueError("interface_eps='dual_average' cannot combine with thin conductors or lumped RLC")
    if any(is_tracer(getattr(sim, name, None))
           for name in ("_dx_profile", "_dy_profile", "_dz_profile")):
        raise ValueError("interface_eps='dual_average' requires concrete profiles, not traced profiles")


def assemble_interface_eps_nu(sim, grid, materials):
    """Cell-centre eps averaged over the four cells sharing each E edge.

    Tangential weights are dual-face areas, with PEC cells excluded. Along
    its own axis the edge stays in its own cell (no node-aligned interface
    cuts that edge). End nodes are one-sided. All-PEC edges retain sampled
    eps as a finite fallback; the existing PEC mask enforces their fields.
    Scalar materials, conductivity and source normalization are untouched.
    """
    from rfx.geometry.rasterize_grid import GridCoords, coords_from_nonuniform_grid, rasterize_geometry

    _validate_interface_eps_nu(sim)
    nodes = coords_from_nonuniform_grid(grid)
    widths = (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)
    if any(d is None for d in widths):
        raise ValueError("interface_eps='dual_average' requires the exact float64 grid spine")
    centres = [np.asarray(x) + np.asarray(d) / 2 for x, d in zip(nodes[:3], widths)]
    # The bounding node has no outgoing real cell: copy the last centre.
    for x in centres:
        x[-1] = x[-2]
    from rfx.geometry.smoothing import continued_conductor_shape
    geometry = [replace(entry, shape=continued_conductor_shape(
                    sim, grid, entry.shape, entry=entry, unextendable=[]))
                if sim._resolve_material(entry.material_name).sigma >= sim._PEC_SIGMA_THRESHOLD
                else entry for entry in sim._geometry]
    cell, debye, lorentz, pec, *_ = rasterize_geometry(
        geometry, sim._resolve_material, GridCoords(*centres, grid.shape),
        pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD)
    if debye is not None or lorentz is not None:
        raise ValueError("interface_eps='dual_average' cannot combine with Debye/Lorentz materials")
    eps = np.asarray(cell.eps_r, dtype=np.float64)
    live = np.ones(grid.shape, dtype=np.float64) if pec is None else (~np.asarray(pec)).astype(np.float64)
    components = []
    for c in range(3):
        num, den = eps * live, live
        for a in range(3):
            if a == c:
                continue
            shape = [1, 1, 1]
            shape[a] = grid.shape[a]
            d = np.asarray(widths[a]).reshape(shape)
            num, den = num * d, den * d
            lower = np.maximum(np.arange(grid.shape[a]) - 1, 0)
            num = num + np.take(num, lower, axis=a)
            den = den + np.take(den, lower, axis=a)
        out = np.asarray(materials.eps_r, dtype=np.float64).copy()
        np.divide(num, den, out=out, where=den > 0)
        components.append(jnp.asarray(out, dtype=materials.eps_r.dtype))
    return tuple(components)


def build_nonuniform_grid(
    freq_max: float,
    domain: tuple,
    dx: float | None,
    cpml_layers: int,
    dz_profile: np.ndarray,
    *,
    dx_profile: np.ndarray | None = None,
    dy_profile: np.ndarray | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    cpml_axes: str = "xyz",
    dt: float | None = None,
    dt_min_cell: float | None = None,
    dt_caller: str | None = None,
) -> NonUniformGrid:
    """Build a NonUniformGrid from per-axis profiles.

    ``dx_profile`` / ``dy_profile`` are optional; when omitted the
    corresponding axis is uniform with spacing ``dx`` across
    ``domain``. ``pec_faces`` / ``pmc_faces`` force per-face pad=0 on
    the listed faces (PMC+CPML composition fix on the NU path, 2026-04).
    ``dt`` / ``dt_min_cell`` pin a concrete time step across a mesh
    deformation family — see :func:`rfx.nonuniform.make_nonuniform_grid`.
    """
    if dx is None:
        dx = C0 / freq_max / 20.0
    if dz_profile is None:
        # Synthesise the uniform dz profile LOCALLY (pure) when only
        # dx/dy are non-uniform. This used to be done at every
        # forward()/run()/runner call site via
        # ``sim._dz_profile = np.full(...)``, permanently mutating sim
        # state as a side effect of execution (roadmap W1.3).
        nz_phys = max(1, int(round(domain[2] / dx)))
        dz_profile = np.full(nz_phys, float(dx))
    domain_xy = (domain[0], domain[1])
    # An outer jit must not turn a STATIC mesh into traced coordinates:
    # conductor classification needs the same concrete lattice as preflight.
    # Actual mesh design variables retain their differentiable build path.
    from contextlib import nullcontext
    traced = any(is_tracer(value) for value in (
        dx, dx_profile, dy_profile, dz_profile))
    with nullcontext() if traced else jax.ensure_compile_time_eval():
        return make_nonuniform_grid(
            domain_xy, dz_profile, dx, cpml_layers,
            dx_profile=dx_profile, dy_profile=dy_profile,
            pec_faces=pec_faces, pmc_faces=pmc_faces, cpml_axes=cpml_axes,
            dt=dt, dt_min_cell=dt_min_cell, dt_caller=dt_caller,
        )


def assemble_materials_nu(
    sim,
    grid: NonUniformGrid,
    sheet_specs: list | None = None,
    pec_sheets: list | None = None,
    pec_wires: list | None = None,
) -> tuple[MaterialArrays, object, object, jnp.ndarray | None]:
    """Build material arrays and dispersion specs for non-uniform grid.

    Delegates to the shared rasterize_geometry() with non-uniform coordinates.
    Supports all shape types, Debye/Lorentz poles, chi3, and thin conductors.
    Lattice ownership contract (#931): PEC volumes are centre-sampled into
    ``pec_mask``; PEC thin conductors and zero-thickness PEC Boxes are
    :class:`~rfx.boundaries.pec.SheetSpec` sheets (``pec_sheets``
    out-parameter), sub-cell ``PolylineWire`` filaments are ``WireSpec``
    (``pec_wires``); neither is in ``pec_mask``. Lossy sheets fold into
    sigma with the LOCAL E-node DUAL spacing normal to the sheet (#373,
    corrected #669-review) — both the DC fold (sigma_bulk*t/d_norm) and the
    opt-in Leontovich surface-impedance mode (sigma_eff = 1/(Rs0*d_norm)
    when surface_impedance_f0 is set, issue #669). Only non-Box LOSSY
    sheets on the legacy DC path warn-and-skip.

    Returns
    -------
    materials, debye_spec, lorentz_spec, pec_mask
    """
    from rfx.geometry.rasterize_grid import (
        rasterize_geometry, coords_from_nonuniform_grid, extend_cpml_pad_materials,
        cell_sizes_from_nonuniform_grid, centres_from_nonuniform_grid,
        sheet_footprint_traced, sheet_spec_from_shape,
    )

    coords = coords_from_nonuniform_grid(grid)
    cell_sizes = cell_sizes_from_nonuniform_grid(grid)
    centres = centres_from_nonuniform_grid(grid, coords)
    _pec_sheets = pec_sheets if pec_sheets is not None else []
    _pec_wires = pec_wires if pec_wires is not None else []
    from rfx.geometry.smoothing import continued_conductor_shape, warn_unextendable_shapes
    conductor_findings = []
    geometry = [replace(entry, shape=continued_conductor_shape(
                    sim, grid, entry.shape, entry=entry, unextendable=conductor_findings))
                if sim._resolve_material(entry.material_name).sigma >= sim._PEC_SIGMA_THRESHOLD
                else entry for entry in sim._geometry]

    result = rasterize_geometry(
        geometry,
        sim._resolve_material,
        coords,
        pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD,
        pole_geometry_entries=sim._geometry,
        centres=centres,
        cell_sizes=cell_sizes,
        sheets=_pec_sheets,
        wires=_pec_wires,
        # The NU stepper installs no periodic BC and NU grids are 3-D, so the
        # non-periodic #689 convention is the one this lane's step function
        # uses (see the realization comment further down); the guard must ask
        # with the SAME flags or it judges a seam the solve never has.
        periodic=(False, False, False),
    )
    materials, debye_spec, lorentz_spec, pec_mask, _pec_shapes, _kerr_chi3 = result

    # Extend material properties into CPML padding so that guided modes in
    # dielectric waveguides see an impedance-matched absorber (equivalent to
    # UPML). Each CPML face copies the interior-edge slice outward, as if
    # the geometry continued beyond the domain. Mirrors the uniform path's
    # identical step (rfx/api/_compile.py) exactly via the shared
    # extend_cpml_pad_materials helper, adapted to the NU per-face pad
    # bookkeeping (``grid.pad_{axis}_{lo,hi}``, already zero on
    # PEC/PMC/periodic faces). Before this fix the NU assembler had NO such
    # step, so an edge-touching structure saw a different absorber medium
    # per path (#582: measured 736 pad cells eps 4.0-vs-1.0 at the slab's
    # k=9 layer) and the uniform-mesh reduction anchor diverged wherever
    # that mismatch interacted with subpixel smoothing. #627 moved this
    # (and the uniform mirror) onto one shared implementation.
    #
    # Dispersion-pole masks are deliberately NOT extended here (#627b tried
    # and reverted): see extend_cpml_pad_materials's docstring — extending
    # a high-Q Lorentz pole into the pad turns a stable edge-touching
    # simulation into a divergent one, with no NaN/exception to catch it.
    # And a pole-carrying column's STATICS are not promoted by the hi-face
    # fallback either (#808, same shared rule, same gate as the uniform
    # lane): the promoted eps_inf-without-pole material matches no
    # declared model and moved a committed Debye recovery past its gate.
    if sim._boundary in ("cpml", "upml") and sim._cpml_layers > 0:
        plx, phx = grid.pad_x_lo, grid.pad_x_hi
        ply, phy = grid.pad_y_lo, grid.pad_y_hi
        plz, phz = grid.pad_z_lo, grid.pad_z_hi
        _pole_mask_any = None
        for _spec in (debye_spec, lorentz_spec):
            if _spec is not None:
                for _pmask in _spec[1]:
                    _pole_mask_any = (_pmask if _pole_mask_any is None
                                      else (_pole_mask_any | _pmask))
        eps_r_ext, sigma_ext, mu_r_ext = extend_cpml_pad_materials(
            materials.eps_r, materials.sigma, materials.mu_r,
            plx, phx, ply, phy, plz, phz,
            dispersion_pole_mask=_pole_mask_any,
        )
        materials = MaterialArrays(eps_r=eps_r_ext, sigma=sigma_ext, mu_r=mu_r_ext)

    # Thin conductors. A PEC thin conductor is a SHEET (#931 §1.3): one
    # node plane (nearest node to its mid-plane, tie -> lower), a closed
    # footprint, no cell. It is emitted as a SheetSpec, never OR'd into
    # pec_mask; the run lanes realize it through
    # rfx.boundaries.pec.realized_pec_edge_masks. A shape thicker than one
    # local cell along its normal is refused ("not a sheet; use add()").
    if sim._thin_conductors:
        conductors = [replace(tc, shape=continued_conductor_shape(
                        sim, grid, tc.shape, entry=tc, unextendable=conductor_findings))
                      for tc in sim._thin_conductors]
        pec_tcs = [tc for tc in conductors
                   if getattr(tc, "is_pec", False)]
        lossy_tcs = [tc for tc in conductors
                     if not getattr(tc, "is_pec", False)]
        for tc in pec_tcs:
            _pec_sheets.append(sheet_spec_from_shape(
                tc.shape, coords, cell_sizes, name="thin_conductor",
                lane="non-uniform", refuse_thick=True))
        # #373: lossy (non-PEC) thin conductors fold into sigma using the LOCAL
        # spacing NORMAL to the sheet, not a uniform grid.dx. The sheet has
        # bulk conductivity sigma_bulk and physical thickness t but is realized
        # on ONE E node along its normal; preserving the sheet resistance
        # R_s = 1/(sigma_bulk*t) needs sigma_eff = sigma_bulk*t/d_norm with
        # d_norm the length that node's sigma actually acts over.
        #
        # That length is the DUAL spacing (d[k-1]+d[k])/2, NOT the primal cell
        # d[k] (#669 review). The NU E update divides the curl at node k by
        # inv_d_e[k] = 2/(d[k-1]+d[k]) (rfx/nonuniform.py), so multiplying the
        # discrete Ampere law at that node by (d[k-1]+d[k])/2 turns the loss
        # term into a surface current sigma_eff*dual*E — the realized sheet
        # conductance is sigma_eff*dual. Dividing by the primal d[k] instead
        # realizes R_s*d[k]/dual, correct only where the two adjacent cells are
        # equal (all uniform meshes, and NU nodes away from a grading step) and
        # wrong by the local cell ratio ON a transition: measured attenuation
        # ratio 1.2021 (0.25/0.50 mm step) and 0.6214 (1.00/0.25 mm step)
        # against the matched-mesh case, where a mesh-independent sheet must
        # give 1.000. This corrects the legacy #373 DC fold as well as the
        # #669 Leontovich mode — NU DC thin conductors at grading transitions
        # now realize their specified sheet resistance instead of a
        # cell-ratio-scaled one.
        #
        # AD-safe: sigma_eff is a smooth function of the sigma_bulk*t DoF times
        # a grid quantity built with jnp (so a traced dz_profile still flows),
        # and jnp.where keeps the sigma field (an AD-live material)
        # differentiable.
        for tc in lossy_tcs:
            _f0 = getattr(tc, "surface_impedance_f0", None)
            if _f0 is not None:
                # #674: a surface-impedance sheet may be ANY
                # ``mask_on_coords`` shape — the fold below is per occupied
                # cell and shape-agnostic, and the rasterization on this lane
                # has been coords-based since #369. What the shape still owes
                # is an axis-aligned bounding box, because the sheet NORMAL
                # (hence which axis' dual spacing normalizes the fold) is read
                # from it. ``sheet_bounds`` reads Box's corner_lo/corner_hi
                # first, so a Box takes bit-identically the arithmetic it took
                # before this generalization.
                lo, hi = sheet_bounds(tc.shape)
                if lo is None or hi is None:
                    # Defensive mirror of the add-time check (issue #669):
                    # a surface-impedance sheet must fail LOUD, never
                    # warn-and-skip (the #369 silently-vaporized-metal
                    # class). Reachable only for a ThinConductor built
                    # outside add_thin_conductor().
                    raise ValueError(
                        "surface-impedance (surface_impedance_f0) thin "
                        "conductor requires a shape with an axis-aligned "
                        "bounding box (Box corner_lo/corner_hi, or "
                        "Shape.bounding_box()) to locate its normal; "
                        "refusing to skip it on the non-uniform path.")
            else:
                # Legacy DC fold: left as #373 shipped it for Box sheets. A
                # non-Box DC sheet used to warn and be skipped, which solved
                # the board without that conductor; since 2.0 it is refused.
                # Folding it instead stays the separate decision #674 left.
                lo = getattr(tc.shape, "corner_lo", None)
                hi = getattr(tc.shape, "corner_hi", None)
                if lo is None or hi is None:
                    raise NotImplementedError(
                        "a lossy thin conductor (add_thin_conductor without "
                        "surface_impedance_f0) with a non-Box shape "
                        f"({type(tc.shape).__name__}) is not implemented on "
                        "the non-uniform lane; skipping it would leave the "
                        "conductor out of the solve. Instead: draw it as a "
                        "Box, or run on a uniform mesh.")
            extents = [float(hi[i]) - float(lo[i]) for i in range(3)]
            n_axis = min(range(3), key=lambda i: extents[i])  # sheet normal axis
            d_norm = e_node_dual_spacings(
                (grid.dx_arr, grid.dy_arr, grid.dz)[n_axis])
            bshape = [1, 1, 1]
            bshape[n_axis] = int(d_norm.shape[0])
            _plane = None
            if _f0 is not None:
                # #931 G4 by construction: the f0 sheet takes the SAME
                # footprint and plane a PEC sheet on this shape gets
                # (sheet_spec_from_shape: nearest node to the mid-plane,
                # closed Box footprint). A traced mesh (mesh-as-design-
                # variable) has no static plane; there the footprint keeps
                # the shape's own traced node sampler, as before #931.
                if any(is_tracer(c) for c in (coords.x, coords.y, coords.z)):
                    m = sheet_footprint_traced(tc.shape, coords, n_axis)
                else:
                    _spec = sheet_spec_from_shape(
                        tc.shape, coords, cell_sizes, normal_axis=n_axis,
                        name="thin_conductor", lane="non-uniform",
                        refuse_thick=True)
                    m = _spec.footprint
                    _plane = _spec.plane
                # #674 guard: the realization normalizes ONE E node along the
                # sheet normal, so the rasterized sheet must occupy exactly
                # one layer there — and must not have vaporized.
                check_sheet_occupancy(m, n_axis, lane="non-uniform")
                # Leontovich band-centre surface-impedance mode (#669/#677):
                # since #677 the sheet does NOT fold into materials.sigma
                # (that realized it as a full-cell slab and moved resonances
                # by geometry, issue #677). It is emitted as a
                # SheetImpedanceSpec with sigma_sheet = (1/Rs0)/d_norm per
                # node — d_norm the LOCAL E-node dual spacing along the
                # sheet normal (#671), so sigma_sheet * Rs0 * d_norm == 1 on
                # every layer of a graded mesh — and realized NODE-THIN by
                # the per-step operator on the sheet edge set. Thickness
                # deliberately does not enter (Leontovich loss is
                # thickness-independent). eps_r stays at background (a sheet
                # is a surface, not a dielectric fill).
                #
                # Invariant (NU two-run S reference): the sheet is NOT
                # resident in materials.sigma, so the reference run's
                # sigma_override does NOT strip it — the reference must
                # strip the sheet ctx EXPLICITLY
                # (run_nonuniform_path(strip_sheet_impedance=True); the
                # rfx/api/_sparams.py NU vacuum-reference call site does).
                from rfx.materials.thin_conductor import (
                    SheetImpedanceSpec, leontovich_rs)
                rs0 = leontovich_rs(_f0, tc.sigma_bulk)
                g_sheet = 1.0 / rs0
                if sheet_specs is not None:
                    sigma_sheet = jnp.where(
                        m, (g_sheet / d_norm).reshape(bshape) *
                        jnp.ones_like(materials.sigma), 0.0)
                    sheet_specs.append(SheetImpedanceSpec(
                        mask=m, normal_axis=n_axis, g_sheet=g_sheet,
                        sigma_sheet=sigma_sheet, plane=_plane))
                continue
            m = tc.shape.mask_on_coords(coords.x, coords.y, coords.z)
            sigma_eff = tc.sigma_bulk * (tc.thickness / d_norm.reshape(bshape))
            materials = materials._replace(
                eps_r=jnp.where(m, tc.eps_r, materials.eps_r),
                sigma=jnp.where(m, sigma_eff, materials.sigma),
            )
    # Node-pinned PEC sheets (add_pinned_sheet): built from node indices, so
    # they need no node POSITION and are the one sheet declaration a traced
    # mesh can carry. Same helper as the uniform lane, so the two cannot
    # disagree on what a pinned range realizes.
    if getattr(sim, "_pinned_sheets", None):
        from rfx.materials.thin_conductor import pinned_sheet_spec
        for _ps in sim._pinned_sheets:
            _pec_sheets.append(pinned_sheet_spec(grid, _ps))

    from rfx.materials.thin_conductor import (
        warn_sheet_planes_inside_dielectric,
    )
    warn_sheet_planes_inside_dielectric(_pec_sheets, materials.eps_r)
    from rfx.api._compile import _refuse_uncollected_pec
    _refuse_uncollected_pec(_pec_sheets if pec_sheets is None else (),
                            _pec_wires if pec_wires is None else (),
                            lane="non-uniform")
    warn_unextendable_shapes(conductor_findings)
    return materials, debye_spec, lorentz_spec, pec_mask


def pos_to_nu_index(grid: NonUniformGrid, pos) -> tuple[int, int, int]:
    """Convert physical (x, y, z) to non-uniform grid indices.

    Delegates to ``rfx.nonuniform.position_to_index`` so the cumulative
    lookup works on non-uniform xy as well.
    """
    from rfx.nonuniform import position_to_index
    return position_to_index(grid, pos)


def _nu_flux_tangential_bounds(d_arr, pad_lo: int, pad_hi: int,
                               center, size) -> tuple[int, int]:
    """Resolve a finite graded-axis CELL window inside the physical interior.

    Uses nearest cumulative edges, with the lower edge at a tie. Every
    requested endpoint outside the interior emits a clamp warning, including
    sub-cell overflow. Ordinary interior snapping remains a geometry result.
    """
    from rfx.probes.flux_region import resolve_flux_axis

    interior = interior_cells(np.asarray(d_arr, dtype=float), pad_lo, pad_hi)
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    result = resolve_flux_axis(edges, pad_lo, center, size)
    if result["clamped_low"] or result["clamped_high"]:
        warnings.warn(
            f"flux monitor center={center!r} size={size!r}: CLAMPED to interior "
            f"[0, {float(edges[-1]):.12g}] m; requested bounds "
            f"{result['requested_bounds_m']} m, realized bounds "
            f"{result['realized_bounds_m']} m. The monitor integrates the realized extent.",
            UserWarning, stacklevel=2,
        )
    return result["cell_slice"]


def _build_waveguide_port_config_nu(sim, entry, grid: NonUniformGrid,
                                     freqs: jnp.ndarray, n_steps: int):
    """NU-aware waveguide port config builder.

    Mirrors the uniform-path ``Simulation._build_waveguide_port_config``
    but resolves indices and per-axis aperture spans against the
    NonUniformGrid (cumulative-sum cell edges).
    """
    from rfx.sources.waveguide_port import (
        WaveguidePort,
        init_waveguide_port,
        init_multimode_waveguide_port,
    )

    normal_axis = entry.direction[1]
    axis_idx = {"x": 0, "y": 1, "z": 2}[normal_axis]
    pos_vec = [0.0, 0.0, 0.0]
    pos_vec[axis_idx] = entry.x_position
    x_index = pos_to_nu_index(grid, tuple(pos_vec))[axis_idx]

    # Per-face NU allocation (2026-04): pass (pad_lo, pad_hi) for each axis.
    pads_lo_hi = {
        "x": (grid.pad_x_lo, grid.pad_x_hi),
        "y": (grid.pad_y_lo, grid.pad_y_hi),
        "z": (grid.pad_z_lo, grid.pad_z_hi),
    }

    def _range_to_slice_nu(value_range, d_arr_jnp, n_axis, pad_lo, pad_hi):
        d_np = np.asarray(d_arr_jnp)
        # Cell-edge positions in physical coords (interior only, edge=0 at first
        # interior face). Length = n_interior + 1.
        interior = interior_cells(d_np, pad_lo, pad_hi)
        edges = np.insert(np.cumsum(interior), 0, 0.0)
        if value_range is None:
            return (pad_lo, n_axis - pad_hi), float(edges[-1])
        lo, hi = value_range
        lo_local = int(np.argmin(np.abs(edges - float(lo))))
        hi_local = int(np.argmin(np.abs(edges - float(hi))))
        if hi_local <= lo_local:
            raise ValueError(
                f"range {value_range!r} does not resolve to a valid aperture on the NU grid"
            )
        lo_idx = lo_local + pad_lo
        hi_idx = hi_local + pad_lo + 1
        actual_span = float(edges[hi_local] - edges[lo_local])
        if actual_span <= 0.0:
            raise ValueError(
                f"range {value_range!r} resolves to invalid aperture span {actual_span}"
            )
        return (lo_idx, hi_idx), actual_span

    if normal_axis == "x":
        u_slice, a_span = _range_to_slice_nu(entry.y_range, grid.dy_arr, grid.ny, *pads_lo_hi["y"])
        v_slice, b_span = _range_to_slice_nu(entry.z_range, grid.dz, grid.nz, *pads_lo_hi["z"])
    elif normal_axis == "y":
        u_slice, a_span = _range_to_slice_nu(entry.x_range, grid.dx_arr, grid.nx, *pads_lo_hi["x"])
        v_slice, b_span = _range_to_slice_nu(entry.z_range, grid.dz, grid.nz, *pads_lo_hi["z"])
    else:
        u_slice, a_span = _range_to_slice_nu(entry.x_range, grid.dx_arr, grid.nx, *pads_lo_hi["x"])
        v_slice, b_span = _range_to_slice_nu(entry.y_range, grid.dy_arr, grid.ny, *pads_lo_hi["y"])

    # NODE span -> CELL span, same conversion and same reason as the uniform
    # builder (issue #868): ``_range_to_slice_nu`` reports the aperture as
    # ``edges[hi] - edges[lo]``, i.e. the cells lo..hi-1, while
    # ``WaveguidePort.u_slice`` is the field-array slice the transverse mode
    # operator is sized from.
    u_slice, v_slice = _node_span_to_cell_span(u_slice), _node_span_to_cell_span(v_slice)

    # Snapped source-plane physical coordinate (for waveguide_plane_positions).
    # Use the cumulative cell-edge position corresponding to the snapped cell
    # index along the port-normal axis.
    if normal_axis == "x":
        d_axis_np = np.asarray(grid.dx_arr)
    elif normal_axis == "y":
        d_axis_np = np.asarray(grid.dy_arr)
    else:
        d_axis_np = np.asarray(grid.dz)
    _pad_lo_axis, _pad_hi_axis = pads_lo_hi[normal_axis]
    _interior = interior_cells(d_axis_np, _pad_lo_axis, _pad_hi_axis)
    _edges_axis = np.insert(np.cumsum(_interior), 0, 0.0)
    _local_axis = max(0, min(x_index - _pad_lo_axis, len(_edges_axis) - 1))
    snapped_source_plane = float(_edges_axis[_local_axis])

    port = WaveguidePort(
        x_index=x_index,
        y_slice=None,
        z_slice=None,
        a=a_span,
        b=b_span,
        mode=entry.mode,
        mode_type=entry.mode_type,
        direction=entry.direction,
        x_position=snapped_source_plane,
        normal_axis=normal_axis,
        u_slice=u_slice,
        v_slice=v_slice,
    )
    if entry.n_modes > 1:
        return init_multimode_waveguide_port(
            port, grid, freqs,
            n_modes=entry.n_modes,
            f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
            bandwidth=entry.bandwidth,
            amplitude=entry.amplitude,
            probe_offset=entry.probe_offset,
            ref_offset=entry.ref_offset,
            dft_total_steps=n_steps,
            dt=float(grid.dt),
        )
    return init_waveguide_port(
        port, grid, freqs,
        f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
        bandwidth=entry.bandwidth,
        amplitude=entry.amplitude,
        probe_offset=entry.probe_offset,
        ref_offset=entry.ref_offset,
        dft_total_steps=n_steps,
        dt=float(grid.dt),
    )


def _setup_msl_ports_nu(sim, grid, materials, materials_drive, sources,
                        n_steps, pec_edge_masks, *, geometry_edge_masks=None,
                        sheet_specs=(), drawn_eps_r=None):
    """Set up MSL ports on the non-uniform mesh (Ez static-Laplace feed only).

    Mirrors the uniform MSL block (``rfx/runners/uniform.py``: ``_msl_ports``)
    but with two NU-specific differences:
      * the eigenmode J+M launch is FENCED — ``run_nonuniform`` carries no
        magnetic-source channel, so the Schelkunoff H-source would have
        nowhere to go; and
      * the feed's own termination conductance is stamped into
        ``materials_drive`` as well as ``materials`` before the feed is
        built from it (#1256: the drive's Cb has to be the E update's, port
        load included), and ``materials_drive`` carries any whole-grid
        eps/sigma override, traced or not (#1267).

    The substrate ``eps_r`` of the launch fixture -- the static-Laplace mode
    shape, when the port does not state ``eps_r_sub`` -- is read at the
    feed's centre cell from ``drawn_eps_r``, the permittivity as drawn,
    whenever an override is in force, and from ``materials_drive`` (which is
    then the drawn one) otherwise. The fixture is a static shape: under an
    override the uniform ``forward()`` reads it from the registered
    materials too (#483), so a finite difference through the override and
    ``jax.grad`` differentiate one function, and a traced override never
    reaches the host-side Laplace solve. Only the Cb each feed cell is
    driven through follows the override.

    The Ez point-sources are appended to ``sources`` (they ride the generic
    NU point-source scan injection). The per-probe DFT planes are registered
    separately by ``compute_msl_s_matrix`` via ``add_dft_plane_probe`` and
    flow through the existing NU ``dft_plane_probes`` accumulation. Returns
    the (possibly σ-updated) ``materials`` and the port-cleared realized PEC
    edge masks (#931 §1.9).
    """
    from rfx.sources.msl_port import (
        _msl_yz_cells,
        msl_normal_component as _msl_normal_component,
        compute_msl_mode_profile,
        make_msl_port_sources,
        msl_cell,
        msl_cross_section_span,
        msl_port_from_entry,
        setup_msl_port,
    )
    original_edges = pec_edge_masks if geometry_edge_masks is None else geometry_edge_masks
    for pe in sim._msl_ports:
        # Issue #661: one shared projection of position -> port frame.
        mp = msl_port_from_entry(pe)
        port_mode = getattr(pe, "mode", "laplace")
        if port_mode == "eigenmode":
            raise NotImplementedError(
                "NU MSL S-params support mode='laplace'/'uniform' (Ez feed) "
                "only: the eigenmode J+M launch needs the magnetic-source "
                "channel, which run_nonuniform does not carry. Use "
                "mode='laplace' (the add_msl_port default) on the "
                "non-uniform lane."
            )
        # laplace / uniform: build the static-Laplace Ez mode profile from the
        # substrate eps_r read concretely under the trace centre.
        from rfx.sources.msl_port import validate_msl_port_geometry
        validate_msl_port_geometry(
            grid, mp, pec_edge_masks=original_edges, sheet_specs=sheet_specs,
            pec_faces=sim._boundary_spec.pec_faces(), name=pe.name)
        span = msl_cross_section_span(grid, mp)
        k_mid = (span["n_lo"] + span["n_hi"]) // 2
        eps_cell = msl_cell(
            pe.direction, span["i_feed"], span["w_centre"], k_mid
        )
        if pe.eps_r_sub is not None:
            eps_r_sub = float(pe.eps_r_sub)
        else:
            _fixture_eps = (materials_drive.eps_r if drawn_eps_r is None
                            else drawn_eps_r)
            eps_r_sub = float(np.asarray(_fixture_eps[eps_cell]))
        mode_profile = compute_msl_mode_profile(grid, mp, eps_r_sub)

        materials = setup_msl_port(grid, mp, materials, mode_profile=mode_profile)
        materials_drive = setup_msl_port(      # #1256
            grid, mp, materials_drive, mode_profile=mode_profile)
        if pe.excite and pe.waveform is not None:
            sources.extend(make_msl_port_sources(
                grid, mp, materials_drive, n_steps, mode_profile=mode_profile,
            ))
        if pec_edge_masks is not None:
            from rfx.boundaries.pec import clear_edges
            # Only the SUBSTRATE-NORMAL component: the edge the modal
            # source drives.  The three-component form opened a
            # width-long slot in the ground plane at the feed (#931
            # §1.9, corrected).
            pec_edge_masks = clear_edges(
                pec_edge_masks, list(_msl_yz_cells(grid, mp)),
                component=_msl_normal_component(mp))
    return materials, pec_edge_masks


def run_nonuniform_path(sim, *, n_steps, compute_s_params=None, s_param_freqs=None,
                        eps_override=None, sigma_override=None,
                        pec_mask_override=None, pec_occupancy_override=None,
                        checkpoint=False,
                        emit_time_series=True, checkpoint_every=None,
                        n_warmup=0,
                        report_every=None, report_label="",
                        subpixel_smoothing: bool = False,
                        attach_waveguide_flux: bool = False,
                        strip_interior_pec: bool = False,
                        strip_sheet_impedance: bool = False,
                        until_decay: float | None = None,
                        decay_check_interval: int = 50,
                        decay_min_steps: int = 100,
                        decay_max_steps: int = 50_000,
                        decay_energy_consecutive: int = 2,
                        radiated_flux_box: tuple | None = None,
                        flux_env_checks: int = 4,
                        design_box=None):
    """Run simulation on non-uniform grid with graded dz.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance (read-only access to its fields).
    n_steps : int
        Number of timesteps.
    until_decay : float or None
        When not None, run via :func:`run_nonuniform_until_decay`
        (issue #383 — NU port of the #169 interior-energy stop): every
        step-count-sized build (source tables, waveguide port record
        buffers, DFT planes, flux monitors) is sized with
        ``decay_max_steps`` instead of ``n_steps``, and the scan is
        replaced by the chunked host loop that stops when the
        dV-weighted interior energy has decayed to this fraction of
        peak. The routing layer only sends this on absorbing
        (cpml/upml) boundaries. ``checkpoint`` (the jax.checkpoint grad
        tape flag) is accepted-and-ignored on the decay path — it is a
        forward-only host loop (mirrors the uniform run_until_decay);
        ``checkpoint_every`` / ``n_warmup`` raise NotImplementedError
        below. Flux monitors must use the rectangular DFT window (the
        NU default) — a streaming windowed DFT needs the true total
        step count in advance, which a decay-terminated run does not
        have, so non-rect windows raise ValueError instead of silently
        mis-windowing.
    decay_check_interval, decay_min_steps, decay_max_steps,
    decay_energy_consecutive :
        Threaded to :func:`run_nonuniform_until_decay` (same semantics
        as ``Simulation.run``'s ``decay_*`` kwargs).
    compute_s_params : bool or None
    s_param_freqs : array or None
    eps_override, sigma_override : jnp.ndarray or None
        When provided, replace the assembled material arrays before
        source/port setup. Used by the differentiable ``forward()``
        path to inject optimisation variables.
    pec_mask_override : jnp.ndarray or None
        Extra hard-PEC mask ORed into the geometry-derived pec_mask.
    strip_interior_pec : bool
        When True, drop the interior-geometry PEC returned by
        ``assemble_materials_nu`` — both the ``pec_mask`` VOLUME cells
        (the rasterized iris / wall / post) and the declared SHEETS and
        WIRES (#931 §1.9: a sheet iris owns no cell, so stripping only
        ``pec_mask`` would leave it in the "empty guide" reference and the
        two-run S11 would come out 0) — forcing a clean
        vacuum-plus-boundary-walls reference run. The
        boundary-wall PEC (the y/z guide walls from the BoundarySpec) is
        NOT carried in ``pec_mask`` — it is enforced separately via
        ``pec_faces`` (grid pad=0 + ``apply_pec`` / CPML face split), so
        stripping ``pec_mask`` keeps the guide walls while removing the
        scatterer. This is the NU analogue of the uniform two-run
        S-matrix reference (``_sparams.py``: the reference run uses
        ``dielectric_shapes=[]`` + boundary-only PEC, and a comment there
        warns that applying the interior ``pec_mask`` to the reference
        makes it bit-identical to the device → ``(device-reference)=0`` →
        ``S11=0`` for any reflector). Used ONLY by the NU two-run S-matrix
        vacuum reference; the device run leaves this False.
    design_box : DesignBoxSpec or None
        Issue #1183. One static box whose E update is redone from its own
        (usually traced) permittivity, so reverse-mode AD keeps box-shaped
        arrays per timestep instead of grid-shaped ones. The graded-mesh
        counterpart of the uniform lane's ``rfx.simulation.run(design_box=)``
        (#1179); ``rfx.simulation._resolve_design_box``, called from
        ``_build_nu_scan``, carries the fences that need the resolved step
        context, and the ones below are this lane's own.

    Returns
    -------
    Result
    """
    # every single-device non-uniform solve (run AND forward) enters here
    sim._require_mode_the_nonuniform_lane_solves()
    # compute_waveguide_s_matrix's graded branch reaches this lane without
    # _dispatch_plan, so the lane refuses the refinement it drops (#1240).
    sim._require_no_refinement_on_the_nonuniform_lane()
    from rfx.api import Result

    _validate_interface_eps_nu(sim, subpixel_smoothing=subpixel_smoothing,
                               eps_override=eps_override)

    # ---- #1183 design box: the fences only this lane can see ----
    # Everything the resolved step context decides (the absorber, the
    # source/wire-port/RLC cells, dispersion, subpixel eps, a sheet inside
    # the box) is rfx.simulation._resolve_design_box's, from _build_nu_scan.
    # These three are declarations that never reach it: a modal port writes
    # its own E over a whole plane from a mode profile, and an MSL port
    # REWRITES ``materials`` across its cross-section (_setup_msl_ports_nu)
    # -- both from the background permittivity the box no longer carries.
    if design_box is not None:
        if sim._waveguide_ports:
            raise NotImplementedError(
                "a design box (#1183) does not combine with a waveguide "
                "port on the non-uniform lane: the port injects and "
                "extracts a modal field over its whole plane, built from "
                "the background permittivity before the time loop. Use "
                "eps_override.")
        if getattr(sim, "_msl_ports", None):
            raise NotImplementedError(
                "a design box (#1183) does not combine with an MSL port on "
                "the non-uniform lane: the port rewrites ``materials`` "
                "across its cross-section at setup, which the design "
                "permittivity would not reach. Use eps_override.")
        if until_decay is not None:
            raise NotImplementedError(
                "a design box (#1183) does not combine with until_decay on "
                "the non-uniform lane: the decay stop is a forward-only "
                "host loop, so there is no gradient tape for the box to "
                "keep small. Use a fixed n_steps run.")

    # Flux monitors: the NU scan body accumulates Poynting-flux DFTs (parity
    # with the uniform path). Full-plane AND finite-region (``size=``)
    # monitors are supported; the finite-region tangential CELL window is
    # resolved against the graded cumulative cell edges in the monitor-build
    # block below (``_nu_flux_tangential_bounds``, audit c/B1, issue #764).

    # ---- until_decay (issue #383) fences + sizing ----
    if until_decay is not None:
        if checkpoint_every is not None:
            raise NotImplementedError(
                "checkpoint_every is not supported with until_decay on the "
                "non-uniform path: the decay stop drives a chunked host "
                "loop, not a single lax.scan, so segmented remat does not "
                "apply (mirrors run_until_decay's checkpoint_segments "
                "raise on the uniform lane). Use a fixed n_steps run for "
                "checkpoint_every."
            )
        if n_warmup:
            raise NotImplementedError(
                "n_warmup is not supported with until_decay on the "
                "non-uniform path: the decay stop is a forward-only host "
                "loop, so the warmup AD-tape split does not apply."
            )
        for _fm in getattr(sim, "_flux_monitors", None) or []:
            _win = getattr(_fm, "dft_window", "rect")
            if _win != "rect":
                raise ValueError(
                    f"flux monitor {_fm.name!r} uses dft_window={_win!r}, "
                    "which is not supported with until_decay on the "
                    "non-uniform path: a streaming windowed DFT needs the "
                    "true total step count in advance, and a "
                    "decay-terminated run does not know it (sizing by "
                    "decay_max_steps would silently mis-weight every "
                    "sample). Use dft_window='rect' (the default) or a "
                    "fixed n_steps run."
                )

    # Step-count-sized builds use ``sizing_n``: on the decay path every
    # source table / waveguide-port record buffer / DFT accumulator is
    # allocated for the worst case (decay_max_steps) so nothing truncates
    # before the stop fires; stopping early is safe (the waveguide
    # post-scan rect DFT masks by n_steps_recorded, and the streaming
    # DFT-plane / flux / wire-port accumulators simply stop accumulating).
    sizing_n = int(decay_max_steps) if until_decay is not None else n_steps

    grid = build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers, sim._dz_profile,
        dx_profile=sim._dx_profile, dy_profile=sim._dy_profile,
        pec_faces=sim._boundary_spec.pec_faces()
            if sim._boundary_spec is not None else None,
        pmc_faces=sim._boundary_spec.pmc_faces()
            if sim._boundary_spec is not None else None,
        cpml_axes="".join(
            ax for ax in "xyz"
            if ax not in (sim._periodic_axes or "")
        ),
        dt=getattr(sim, "_dt_pin", None),
        dt_min_cell=getattr(sim, "_dt_min_cell", None),
        dt_caller="Simulation",
    )
    _sheet_specs: list = []
    _pec_sheets: list = []
    _pec_wires: list = []
    materials, debye_spec, lorentz_spec, pec_mask = assemble_materials_nu(
        sim, grid, sheet_specs=_sheet_specs, pec_sheets=_pec_sheets,
        pec_wires=_pec_wires)
    if getattr(sim, "_interface_eps", "sampled") == "dual_average" and (
            debye_spec is not None or lorentz_spec is not None):
        raise ValueError("interface_eps='dual_average' cannot combine with Debye/Lorentz materials")
    if strip_sheet_impedance:
        # #677 EXPLICIT reference strip: the surface-impedance sheet no
        # longer rides materials.sigma, so the two-run vacuum reference's
        # sigma_override cannot strip it — dropping the ctx here is the
        # reference-run analogue of strip_interior_pec below. Pinned by the
        # G8 negative control (tests/unit/materials/test_sheet_impedance.py).
        _sheet_specs = []

    # Two-run S-matrix vacuum reference: drop the interior-geometry PEC
    # (the rasterized iris / wall / post) so the reference is a clean
    # empty guide. The boundary-wall PEC (y/z guide walls) is NOT in
    # ``pec_mask`` — it is enforced via ``pec_faces`` (grid pad=0 +
    # ``apply_pec`` / CPML face split) — so the guide walls survive. This
    # mirrors the uniform reference run (``_sparams.py``: dielectric/PEC
    # interior shapes dropped, boundary PEC kept). Without this, the
    # vacuum override replaces only eps_r/sigma and the reference keeps
    # the same interior PEC mask as the device → both DFTs are
    # bit-identical → ``(device-reference)=0`` → ``S11=0`` for any
    # PEC reflector on the NU path.
    if strip_interior_pec:
        pec_mask = None
        # #931: a sheet iris owns no cell, so stripping only ``pec_mask``
        # would leave it in the vacuum reference and give S11 = 0.
        _pec_sheets = []
        _pec_wires = []

    # ``eps_override`` / ``sigma_override`` replace the assembled material
    # arrays for the scan, and every source and port DRIVE is built from
    # ``materials_drive``, which starts from the SAME arrays -- overridden,
    # and traced when the override is (#1267). A drive pushes its current
    # through the Cb = dt/(eps + sigma*dt/2) of the edge it feeds, so it has
    # to read the permittivity the E update reads there. Built from the
    # arrays as drawn, as this lane did until #1267, a device whose
    # permittivity under the port came from the override was driven
    # Cb_drawn/Cb_override times too hard: a layered substrate (eps_r
    # 3.38/3.38/10.2 under a 50 ohm wire) supplied by override read S11
    # 0.027 away from the same board declared as materials, +0.065 dB at
    # 6 GHz, and any permittivity derivative taken through the override
    # carried that ratio's derivative as a false amplitude term. The
    # uniform ``forward()`` builds its drives from the traced overridden
    # materials (``rfx/api/_execute.py``); a traced override now makes this
    # lane's source table traced in the same way (``make_current_source``
    # and ``make_msl_port_sources`` stay in jnp for a tracer). A two-run
    # reference (the waveguide S-matrix's vacuum override) likewise drives
    # any current source through its own arrays; its modal port drives never
    # read them. The one thing still read from the arrays as drawn is the
    # MSL launch fixture's substrate eps_r (``_setup_msl_ports_nu``,
    # ``drawn_eps_r``): a static mode shape, as on the uniform lane (#483).
    #
    # #1256: ``materials_drive`` is not a frozen snapshot. Each port below
    # stamps its termination conductance into BOTH copies. A port drives a
    # current into an edge whose update coefficient Cb = dt/(eps + sigma*dt/2)
    # includes that port's own conductance; a drive built without it
    # injected (1 + sigma_port*dt/(2*eps)) times the declared current --
    # 2.9 on a 50 ohm port across three 0.5 mm cells of eps_r 3.38 at the
    # Courant step, so the absolute field depended on dt and on the dual
    # cell sizes at the feed. S-parameters, ratios of two responses to one
    # drive, hid it only where every cell of the port had the same factor;
    # a layered substrate, graded cells or a lumped R/C under the port
    # made the factors unequal and moved S11 too. The uniform lane builds
    # its port sources after the stamp and never had this. On a concrete mesh the stamp is a host float and the copy stays
    # concrete; on a traced mesh (#1207) sigma_port is traced, and so is the
    # drive, exactly as the E update is.
    materials_drive = materials
    _drawn_eps_r = None     # only an override separates drawn from stepped
    if eps_override is not None or sigma_override is not None:
        _drawn_eps_r = materials.eps_r
        # A whole-grid override REPLACES the array, so any lumped stamp that
        # was folded into it is gone; its #1210 record goes with it, or the
        # E update would add back a load the override does not carry.
        materials = materials._replace(
            eps_r=eps_override if eps_override is not None else materials.eps_r,
            sigma=sigma_override if sigma_override is not None else materials.sigma,
            eps_r_lumped=(None if eps_override is not None
                          else materials.eps_r_lumped),
            sigma_lumped=(None if sigma_override is not None
                          else materials.sigma_lumped),
        )
        materials_drive = materials     # #1267: the drive sees the override
    elif design_box is not None:
        # #1183, the same rule at the same place: the design permittivity
        # sets the precision of the material arithmetic, exactly as an
        # ``eps_override`` array does one branch up. Without it the two
        # halves of one grid step at two precisions -- the float32 rounding
        # of ``eps_r * EPS_0`` alone moves the field by ~1e-7 relative.
        _design_dtype = jnp.promote_types(
            materials.eps_r.dtype, jnp.result_type(design_box.eps_r))
        if _design_dtype != materials.eps_r.dtype:
            materials = materials._replace(
                eps_r=materials.eps_r.astype(_design_dtype))

    if pec_mask_override is not None:
        pec_mask = pec_mask_override if pec_mask is None else (pec_mask | pec_mask_override)

    # #931 §1.7: realize (Mx, My, Mz) ONCE here.  The NU stepper installs no
    # periodic BC and NU grids are 3-D, so the non-periodic #689 convention
    # is the one the step function uses.  Port clearing, wire-port liveness
    # and the sheet ctx all read THIS object from here on.
    from rfx.boundaries.pec import (
        clear_edges as _clear_edges,
        edge_is_pec as _edge_is_pec,
        realized_pec_edge_masks as _rpem,
    )
    _pec_sheets = tuple(_pec_sheets)
    _pec_wires = tuple(_pec_wires)
    pec_edge_masks = None
    if pec_mask is not None or _pec_sheets or _pec_wires:
        pec_edge_masks = _rpem(pec_mask, sheets=_pec_sheets,
                               wires=_pec_wires)
    _msl_geometry_edges = pec_edge_masks  # before ANY port clearing

    # ── Subpixel smoothing on non-uniform mesh ─────────────────────────
    # Builds Kottke tensor-averaged ε per E-component using per-axis
    # cell-size arrays. Only the non-dispersive scan branch consumes
    # ``aniso_eps`` (matches the uniform-path semantics — see
    # rfx/simulation.py::_update_e_with_optional_dispersion).
    aniso_eps = None
    if subpixel_smoothing:
        from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform
        if debye_spec is not None or lorentz_spec is not None:
            sim._refuse_unsupported_run_kwargs(
                "non-uniform mesh with Debye/Lorentz materials",
                {"subpixel_smoothing": subpixel_smoothing},
                instead="remove the Debye/Lorentz poles",
                reason_overrides={"subpixel_smoothing":
                    "the dispersive E update does not read the smoothed "
                    "per-component permittivity tensor, so interfaces "
                    "would get scalar eps"},
            )
        else:
            # #1043 stage B: the pairs carry the CPML/UPML pad continuation,
            # the NU mirror of the two uniform sites. Same shared builder, so
            # the three cannot drift the way the array-side replication did
            # before #627.
            from rfx.geometry.smoothing import (
                smoothed_shape_pairs, warn_unextendable_shapes,
            )
            shape_eps_pairs, _unextendable = smoothed_shape_pairs(sim, grid)
            warn_unextendable_shapes([u for u in _unextendable if not u.conductor])
            if shape_eps_pairs:
                aniso_eps = compute_smoothed_eps_nonuniform(
                    grid, shape_eps_pairs, background_eps=1.0,
                )

    if getattr(sim, "_interface_eps", "sampled") == "dual_average":
        aniso_eps = assemble_interface_eps_nu(sim, grid, materials)

    # Fold RLC R/C into materials before other port/source setup
    # (mirrors the uniform path). #1256: into the drive's copy too -- an
    # R or C across a driven edge is part of that edge's Cb, and a drive
    # built without it is off by the element's own (eps + sigma*dt/2) ratio.
    if sim._lumped_rlc:
        from rfx.lumped import setup_rlc_materials
        for spec in sim._lumped_rlc:
            materials = setup_rlc_materials(grid, spec, materials)
            materials_drive = setup_rlc_materials(grid, spec, materials_drive)

    sources = []
    probes = []
    wire_port_specs = []

    # Domain extents (for auto-detecting port direction).
    dom_lx = float(sim._domain[0])
    dom_ly = float(sim._domain[1])

    def _auto_direction(position) -> str:
        """Pick the outward-normal direction (+x/-x/+y/-y) of the port
        by finding the smallest *relative* distance to a boundary
        face. Relative (not absolute) distance avoids the trap where
        a short domain extent makes a completely-centered port appear
        "close" to its narrow-axis boundary.
        """
        x, y, _z = position
        rel = {
            "-x": float(x) / max(dom_lx, 1e-30),
            "+x": float(dom_lx - x) / max(dom_lx, 1e-30),
            "-y": float(y) / max(dom_ly, 1e-30),
            "+y": float(dom_ly - y) / max(dom_ly, 1e-30),
        }
        return min(rel, key=rel.get)

    for pe in sim._ports:
        idx = pos_to_nu_index(grid, pe.position)
        if pe.impedance == 0.0:
            # Current source with dV normalization; amplitude_kind (issue
            # #571) threaded — None/'current' are bit-identical no-ops here.
            src = make_current_source(
                grid, idx, pe.component, pe.waveform, sizing_n,
                materials_drive, amplitude_kind=pe.amplitude_kind)
            sources.append(src)
        elif pe.extent is not None:
            # Wire port on non-uniform grid
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            axis = axis_map[pe.component]
            end_pos = list(pe.position)
            end_pos[axis] += pe.extent
            idx_end = pos_to_nu_index(grid, tuple(end_pos))
            lo_k = min(idx[axis], idx_end[axis])
            hi_k = max(idx[axis], idx_end[axis])

            # HALF-OPEN in edges, from the shared spelling: the driven
            # edges are the ones whose own location lies inside the
            # declared extent. This lane had its own endpoint-INCLUSIVE
            # copy, so the same declaration drove one more edge here than
            # on the uniform lane once that lane was corrected.
            from rfx.sources.sources import wire_port_edge_span
            _first_k, _last_k = wire_port_edge_span(
                grid, axis, lo_k, hi_k,
                float(pe.position[axis]), float(end_pos[axis]))
            wire_cells = list(range(_first_k, _last_k + 1))

            # Live-cell split (issue #318): a cell whose extent lies inside
            # PEC (assembled-geometry mask, read BEFORE this port's own
            # clearing below) is dead — it carries no port sigma and no
            # source, and drops out of the 1/n scaling. With no dead cells
            # this is bit-identical to the historical all-cells formula.
            _cells_ijk = []
            for k in wire_cells:
                cell = list(idx)
                cell[axis] = k
                _cells_ijk.append(tuple(cell))
            if pec_edge_masks is not None:
                live_flags = [
                    not _edge_is_pec(pec_edge_masks, pe.component,
                                     c[0], c[1], c[2])
                    for c in _cells_ijk]
            else:
                live_flags = [True] * len(_cells_ijk)
            n_live = sum(live_flags)
            if n_live == 0:
                raise ValueError(
                    f"WirePort at {pe.position} ({pe.component}, extent "
                    f"{pe.extent}): all {len(_cells_ijk)} extent cells land "
                    "inside PEC geometry, so the port has no live cell to "
                    "terminate or drive (issue #318). Shorten the extent or "
                    "move the port so at least one cell center sits outside "
                    "PEC."
                )

            # float64 so these are the SAME numbers the extractor reads at
            # rfx/nonuniform.py (`_dx_arr_np`/`_dy_arr_np`, both float64).
            # sigma and I must be sized on one set of metrics. Bit-identical
            # on a uniform profile. A TRACED axis stays a tracer instead —
            # ``port_metric_axes`` is the one place that decides which.
            (_dx_np, _x_host), (_dy_np, _y_host), (_dz_m, _z_host) = \
                port_metric_axes(grid)
            for (ci, cj, ck), live in zip(_cells_ijk, live_flags):
                dxi = port_metric(_dx_np[ci], _x_host)
                dyj = port_metric(_dy_np[cj], _y_host)
                dual_xi = port_metric(
                    e_node_dual_spacing_at(_dx_np, ci), _x_host)
                dual_yj = port_metric(
                    e_node_dual_spacing_at(_dy_np, cj), _y_host)
                dual_zk = port_metric(
                    e_node_dual_spacing_at(_dz_m, ck), _z_host)
                # 3D wire port: σ = n_live * d_parallel / (Z0 * d_perp1 * d_perp2)
                # Each LIVE cell in the wire carries 1/n_live of total
                # impedance Z0 (issue #318 — dead cells excluded).
                #
                # #688: d_perp1/d_perp2 are the E-node DUAL spacings on the
                # two axes TRANSVERSE to the port component — the same
                # metrics `wire_port_current` weights its Ampere legs by
                # (#672). They are the sides of the dual face the conduction
                # current pierces, so V/I closes only if the termination and
                # the measurement use one metric family. d_parallel stays
                # PRIMAL: it is the length of the E edge V is taken over.
                # The old primal spelling for d_perp is right only where the
                # two transverse axes are locally uniform (measured 1.7778x
                # too small a conductance on a 2:1 step on both).
                if live:
                    if axis == 2:
                        d_cell = port_metric(_dz_m[ck], _z_host)
                        dp1, dp2 = dual_xi, dual_yj
                    elif axis == 1:
                        d_cell = dyj
                        dp1, dp2 = dual_xi, dual_zk
                    else:
                        d_cell = dxi
                        dp1, dp2 = dual_yj, dual_zk
                    sigma_port = n_live * d_cell / (pe.impedance * dp1 * dp2)
                    # #1210: a port's load is a device across ONE edge, so
                    # it is recorded as an edge-owned stamp and kept out of
                    # the edge average. Stamped bare it was quartered, and a
                    # 50 ohm termination presented 200 ohm.
                    materials = _stamp_lumped_sigma(
                        materials, (ci, cj, ck), sigma_port, pe.component)
                    # #1256: and into the copy the drive is built from.
                    materials_drive = _stamp_lumped_sigma(
                        materials_drive, (ci, cj, ck), sigma_port, pe.component)
                    # No PEC clearing here (#931 §1.9, corrected): a cell
                    # is LIVE exactly when the port component's own edge is
                    # not PEC, so releasing that component is a no-op, and
                    # releasing the two tangential edges would open the
                    # conductor the port foot stands on.

            # Create per-cell sources — only when the port is excited.
            # Passive (excite=False) ports contribute just the σ
            # resistive termination above, acting as a matched load.
            #
            # Issue #764: the V/I reference cell is the midpoint of the
            # LIVE run, not of the raw extent. A dead extent cell (inside
            # PEC, #318) carries essentially no port current
            # (|I_dead|/|I_mid| = 0.003-0.03 measured on the #313 thru),
            # so an all-extent midpoint landing on a dead cell read a
            # quenched Ampere loop as the port current. With no dead
            # cells the live midpoint is bit-identical to the historical
            # all-extent midpoint.
            _live_cells = tuple(
                c for c, live in zip(_cells_ijk, live_flags) if live)
            mid_cell = list(_live_cells[len(_live_cells) // 2])

            if pe.excite:
                for cell_ijk, live in zip(_cells_ijk, live_flags):
                    # Dead extent cells get no source (issue #318).
                    if not live:
                        continue
                    src = make_current_source(
                        grid, cell_ijk, pe.component,
                        pe.waveform, sizing_n, materials_drive)
                    # Scale by 1/n_live for distributed excitation. A traced
                    # source table stays traced: on a mesh design variable
                    # the injected current moment is normalized by the port
                    # cell's own control volume (#672), so the table carries
                    # a real term of the derivative.
                    _wf = src[4]
                    scaled_wf = (_wf if is_tracer(_wf) else np.array(_wf)) \
                        / n_live
                    sources.append(
                        (src[0], src[1], src[2], src[3], scaled_wf))

            # Wire port S-param spec — include excite/direction so the
            # runner scan body can record V/I and the post-processing
            # can orient the wave decomposition correctly.
            port_direction = pe.direction or _auto_direction(pe.position)
            wire_port_specs.append({
                'mid_i': mid_cell[0], 'mid_j': mid_cell[1],
                'mid_k': mid_cell[2],
                'component': pe.component,
                'impedance': pe.impedance,
                'excite': bool(pe.excite),
                'direction': port_direction,
                # Issue #764: static live-cell run for the whole-port gap
                # voltage V_port = sum_live(-E_c * d_par,c) and its count.
                'live_cells': _live_cells,
                'n_live': int(n_live),
            })
        else:
            # Single-cell lumped port
            i, j, k = idx
            (_dx_np, _x_host), (_dy_np, _y_host), (_dz_m, _z_host) = \
                port_metric_axes(grid)
            dxi = port_metric(_dx_np[i], _x_host)
            dyj = port_metric(_dy_np[j], _y_host)
            dual_xi = port_metric(e_node_dual_spacing_at(_dx_np, i), _x_host)
            dual_yj = port_metric(e_node_dual_spacing_at(_dy_np, j), _y_host)
            dual_zk = port_metric(e_node_dual_spacing_at(_dz_m, k), _z_host)
            # 3D lumped port: σ = d_parallel / (Z0 * d_perp1 * d_perp2)
            # This ensures correct power dissipation P = V²/Z0 in
            # anisotropic cells where dz ≠ dx.  The old formula
            # σ = 1/(Z0*d_parallel) is only valid for cubic cells.
            #
            # #688: d_perp1/d_perp2 are the E-node DUAL spacings on the two
            # axes transverse to the port component (the sides of the dual
            # face the conduction current pierces), matching the metrics
            # the wire-port Ampere loop uses (#672). d_parallel stays PRIMAL
            # — it is the E edge V is taken over. Sharpening the sentence
            # above: the primal spelling for d_perp is valid whenever the
            # two TRANSVERSE axes are locally uniform, which is strictly
            # weaker than cubic and is what actually makes the uniform lane
            # right.
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            port_axis = axis_map[pe.component]
            if port_axis == 2:
                d_parallel = port_metric(_dz_m[k], _z_host)
                d_perp1, d_perp2 = dual_xi, dual_yj
            elif port_axis == 1:
                d_parallel = dyj
                d_perp1, d_perp2 = dual_xi, dual_zk
            else:
                d_parallel = dxi
                d_perp1, d_perp2 = dual_yj, dual_zk
            sigma_port = d_parallel / (pe.impedance * d_perp1 * d_perp2)
            materials = _stamp_lumped_sigma(      # #1210, #1236
                materials, (i, j, k), sigma_port, pe.component)
            materials_drive = _stamp_lumped_sigma(    # #1256, #1236
                materials_drive, (i, j, k), sigma_port, pe.component)
            if pec_edge_masks is not None:
                # The lumped port drives ONE edge: its own component at
                # its own cell (#931 §1.9, corrected).
                pec_edge_masks = _clear_edges(
                    pec_edge_masks, [(i, j, k)], component=pe.component)
            if pe.excite:
                src = make_current_source(
                    grid, idx, pe.component, pe.waveform, sizing_n, materials_drive)
                sources.append(src)

    for pe in sim._probes:
        idx = pos_to_nu_index(grid, pe.position)
        probes.append((*idx, pe.component))

    # DFT plane probes — mirror the uniform-path setup in
    # runners/uniform.py, but use pos_to_nu_index so the plane
    # coordinate resolves on a possibly-graded mesh.
    dft_plane_probes = []
    if sim._dft_planes:
        from rfx.probes.probes import init_dft_plane_probe
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        for pe in sim._dft_planes:
            axis_idx = axis_to_index[pe.axis]
            plane_pos = [0.0, 0.0, 0.0]
            plane_pos[axis_idx] = pe.coordinate
            grid_index = pos_to_nu_index(grid, tuple(plane_pos))[axis_idx]
            freqs_arr = (
                pe.freqs
                if pe.freqs is not None
                else jnp.linspace(sim._freq_max / 10, sim._freq_max, pe.n_freqs)
            )
            dft_plane_probes.append(
                init_dft_plane_probe(
                    axis=axis_idx,
                    index=grid_index,
                    component=pe.component,
                    freqs=freqs_arr,
                    grid_shape=(grid.nx, grid.ny, grid.nz),
                    dft_total_steps=sizing_n,
                    region=getattr(sim, "_dft_plane_regions", {}).get(pe.name),
                )
            )

    # Flux monitors. Full-plane monitors integrate the entire plane;
    # finite-region (``size=``) monitors restrict the DFT accumulation to the
    # CELL window whose cumulative face area matches the requested physical
    # size, resolved against the graded cumulative cell edges via
    # ``_nu_flux_tangential_bounds`` (audit c/B1, issue #764). Tangential cell
    # sizes are passed as per-cell arrays so a graded tangential axis is
    # integrated correctly by FluxMonitor.dA.
    flux_monitor_objs = []
    if getattr(sim, "_flux_monitors", None):
        from rfx.probes.probes import init_flux_monitor
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        # Per-axis tangential cell-size arrays for dA.
        _d_arr = {
            0: (np.asarray(grid.dy_arr), np.asarray(grid.dz)),
            1: (np.asarray(grid.dx_arr), np.asarray(grid.dz)),
            2: (np.asarray(grid.dx_arr), np.asarray(grid.dy_arr)),
        }
        from rfx.probes.flux_region import resolve_flux_region
        for pe in sim._flux_monitors:
            axis_idx = axis_to_index[pe.axis]
            plane_pos = [0.0, 0.0, 0.0]
            plane_pos[axis_idx] = pe.coordinate
            grid_index = pos_to_nu_index(grid, tuple(plane_pos))[axis_idx]
            freqs_arr = (
                pe.freqs
                if pe.freqs is not None
                else jnp.linspace(sim._freq_max / 10, sim._freq_max, pe.n_freqs)
            )
            d1_arr, d2_arr = _d_arr[axis_idx]
            # Finite-region window (B1): physical size/center -> CELL slice on
            # each tangential axis against the graded cumulative edges. Full
            # plane keeps init_flux_monitor's (0, -1) full-extent defaults.
            lo1, hi1, lo2, hi2 = 0, -1, 0, -1
            region = resolve_flux_region(grid, pe, sim._domain)
            if region is not None:
                (lo1, hi1), (lo2, hi2) = region["cell_slices"]
                grid_index = region["normal_index"]
            flux_monitor_objs.append(
                init_flux_monitor(
                    axis=axis_idx,
                    index=grid_index,
                    freqs=freqs_arr,
                    grid_shape=(grid.nx, grid.ny, grid.nz),
                    d1=d1_arr,
                    d2=d2_arr,
                    dft_total_steps=sizing_n,
                    dft_window=getattr(pe, "dft_window", "rect"),
                    dft_window_alpha=getattr(pe, "dft_window_alpha", 0.25),
                    lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2,
                )
            )

    sp_freqs = None
    if wire_port_specs and (compute_s_params is None or compute_s_params):
        sp_freqs = s_param_freqs
        if sp_freqs is None:
            sp_freqs = np.linspace(
                sim._freq_max / 10, sim._freq_max, 50)

    # Lumped RLC: build per-element metadata + zero-init ADE states
    rlc_metas: tuple = ()
    rlc_states_init: tuple = ()
    if sim._lumped_rlc:
        from rfx.lumped import (build_rlc_meta, init_rlc_state,
                                refuse_stacked_solved_elements)
        rlc_metas = tuple(
            build_rlc_meta(grid, spec, materials) for spec in sim._lumped_rlc
        )
        # #1245: at most one element with its own solve per realized edge.
        refuse_stacked_solved_elements(sim._lumped_rlc, rlc_metas)
        rlc_states_init = tuple(init_rlc_state() for _ in sim._lumped_rlc)

    # Waveguide ports: build per-port config via NU-aware
    # init_waveguide_port (duck-types on grid for per-axis widths).
    waveguide_port_cfgs = []
    if sim._waveguide_ports:
        wg_freqs = None
        for pe in sim._waveguide_ports:
            if pe.freqs is not None:
                wg_freqs = jnp.asarray(pe.freqs, dtype=jnp.float32)
                break
        if wg_freqs is None:
            wg_freqs = jnp.linspace(
                sim._freq_max * 0.5, sim._freq_max, 20, dtype=jnp.float32
            )
        for pe in sim._waveguide_ports:
            waveguide_port_cfgs.append(
                _build_waveguide_port_config_nu(
                    sim, pe, grid, wg_freqs, sizing_n,
                )
            )

    # MSL ports — full cross-section distributed Ez feed (NU lane). Sources
    # ride the generic point-source `sources` list; per-probe DFT planes are
    # registered by compute_msl_s_matrix via add_dft_plane_probe.
    if getattr(sim, "_msl_ports", None):
        materials, pec_edge_masks = _setup_msl_ports_nu(
            sim, grid, materials, materials_drive, sources, sizing_n,
            pec_edge_masks,
            geometry_edge_masks=_msl_geometry_edges, sheet_specs=_sheet_specs,
            drawn_eps_r=_drawn_eps_r,
        )

    # Debye/Lorentz coefficients, AFTER the last stamp into ``materials``
    # (the wire, lumped and MSL port loads above), as the uniform lane builds
    # them. With a dispersive material anywhere in the model the E update runs
    # on these coefficients alone and never reads ``materials.sigma`` (#1257):
    # built before the port stamps they left every port unterminated, and a
    # 50 ohm and a 5000 ohm port gave the same waveform.
    debye = None
    if debye_spec is not None:
        debye_poles, debye_masks = debye_spec
        debye = init_debye(debye_poles, materials, grid.dt, mask=debye_masks)

    lorentz = None
    if lorentz_spec is not None:
        lorentz_poles, lorentz_masks = lorentz_spec
        lorentz = init_lorentz(lorentz_poles, materials, grid.dt, mask=lorentz_masks)

    # Optional per-waveguide-port Poynting flux monitors at each port's
    # probe plane (issue #88 flux-extractor path). Built from the same
    # cfg geometry the uniform ``extract_waveguide_s_matrix_flux`` uses,
    # but with per-cell tangential cell-size arrays so a graded tangential
    # axis integrates correctly via FluxMonitor.dA.
    wg_flux_monitors = []
    if attach_waveguide_flux and waveguide_port_cfgs:
        from rfx.probes.probes import init_flux_monitor
        _axis_idx = {"x": 0, "y": 1, "z": 2}
        _tang_arrs = {
            0: (np.asarray(grid.dy_arr), np.asarray(grid.dz)),
            1: (np.asarray(grid.dx_arr), np.asarray(grid.dz)),
            2: (np.asarray(grid.dx_arr), np.asarray(grid.dy_arr)),
        }
        for cfg in waveguide_port_cfgs:
            ax = _axis_idx[cfg.normal_axis]
            d1a, d2a = _tang_arrs[ax]
            wg_flux_monitors.append(
                init_flux_monitor(
                    axis=ax,
                    index=cfg.probe_x,
                    freqs=cfg.freqs,
                    grid_shape=(grid.nx, grid.ny, grid.nz),
                    d1=d1a,
                    d2=d2a,
                    dft_total_steps=sizing_n,
                    lo1=cfg.u_lo, hi1=cfg.u_hi,
                    lo2=cfg.v_lo, hi2=cfg.v_hi,
                )
            )

    # TFSF plane-wave source. Scope: axis-aligned +x / -x incidence with
    # angle_deg=0 so the 1D auxiliary grid runs along the uniform x axis
    # with scalar cell size grid.dx. Oblique angles (2D auxiliary grid)
    # and +z / -z incidence (which would require a z-nonuniform 1D aux)
    # are rejected here with actionable messages.
    tfsf_pair = None
    if sim._tfsf is not None:
        entry = sim._tfsf
        if abs(entry.angle_deg) > 0.01:
            raise ValueError(
                "TFSF oblique incidence (angle_deg != 0) is not yet "
                "supported on nonuniform z mesh — the 2D auxiliary grid "
                "is uniform-only. Use angle_deg=0 or run on the uniform lane."
            )
        if entry.direction in ("+z", "-z"):
            raise ValueError(
                "TFSF z-directed incidence is not yet supported on "
                "nonuniform z mesh (1D auxiliary grid would need to be "
                "z-nonuniform). Use direction='+x' or '-x', or run on the "
                "uniform lane."
            )
        if entry.direction not in ("+x", "-x"):
            raise ValueError(
                "TFSF on nonuniform mesh supports only direction='+x' or "
                f"'-x'; got {entry.direction!r}."
            )
        from rfx.sources.tfsf import init_tfsf
        tfsf_pair = init_tfsf(
            grid.nx,
            grid.dx,
            grid.dt,
            cpml_layers=grid.cpml_layers,
            tfsf_margin=entry.margin,
            f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
            bandwidth=entry.bandwidth,
            amplitude=entry.amplitude,
            polarization=entry.polarization,
            direction=entry.direction,
            angle_deg=entry.angle_deg,
            ny=grid.ny,
            nz=grid.nz,
            waveform=getattr(entry, 'waveform', 'differentiated_gaussian'),
        )

    # NTFF box: build once (indices are Python-static) + zero-init
    # DFT accumulators that will be threaded through the scan carry.
    # NonUniformGrid is a NamedTuple without ``position_to_index``, so
    # we build the box directly via ``pos_to_nu_index`` (which uses
    # the cumulative dx_arr/dy_arr/dz lookup) and skip make_ntff_box.
    ntff_box = None
    ntff_data_init = None
    if sim._ntff is not None:
        from rfx.farfield import (
            NTFFBox, init_ntff_data, with_face_centre_collocation,
        )
        corner_lo, corner_hi, ntff_freqs = sim._ntff
        lo_idx = pos_to_nu_index(grid, corner_lo)
        hi_idx = pos_to_nu_index(grid, corner_hi)
        # Per-face CPML depths must come from THIS grid's pads. A
        # non-uniform grid carries no `face_layers`, so both the previous
        # direct construction and NTFFBox.from_grid fall back to the scalar
        # `cpml_layers` on every face — wrong whenever the pads are
        # asymmetric, which they are for any non-absorbing face. Measured:
        # a z_lo PEC face gives pad_z_lo = 0 while the scalar is 6, so
        # every NTFF face coordinate was displaced by six cells in z, and
        # the pattern came back with no warning (#743).
        ntff_box = NTFFBox(
            i_lo=lo_idx[0], i_hi=hi_idx[0],
            j_lo=lo_idx[1], j_hi=hi_idx[1],
            k_lo=lo_idx[2], k_hi=hi_idx[2],
            freqs=jnp.asarray(ntff_freqs, dtype=jnp.float32),
            cpml_lo_x=int(grid.pad_x_lo),
            cpml_lo_y=int(grid.pad_y_lo),
            cpml_lo_z=int(grid.pad_z_lo),
        )
        # Accumulate at the centre of each face cell (second-order surface
        # integral). The half-cell interpolation weights for the tangential
        # H come from this grid's own cell widths, so a graded axis gets the
        # right pair instead of a flat 1/2.
        ntff_box = with_face_centre_collocation(ntff_box, grid)
        ntff_data_init = init_ntff_data(ntff_box)

    # #677: assemble the surface-impedance sheet ctx from the specs the
    # assembler emitted, against the FINAL realized PEC edges of this run
    # (PEC wins on overlapping edges). Crossing-normal refusal lives in the
    # builder.
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx
    sheet_ctx = build_sheet_impedance_ctx(
        _sheet_specs, pec_edge_masks=pec_edge_masks)
    if sheet_ctx is not None:
        # v1 fences (loud, never silent): the sheet operator replaces the
        # standard E update at its edges, which is only correct against the
        # plain isotropic update_e_nu path.
        if debye_spec is not None or lorentz_spec is not None:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets combined "
                "with dispersive (Debye/Lorentz) materials in one run are "
                "not supported (#677 v1): the sheet operator would "
                "silently override the ADE dispersion update at its edges. "
                "Remove the dispersive material or the f0 sheet.")
        if aniso_eps is not None:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets combined "
                "with subpixel_smoothing / anisotropic permittivity are "
                "not supported (#677 v1): the sheet operator assumes the "
                "isotropic E update at its edges. Disable "
                "subpixel_smoothing or drop the f0 sheet.")

    _shared_run_kwargs = dict(
        design_box=design_box,
        sheet_impedance=sheet_ctx,
        aniso_eps=aniso_eps,
        pec_mask=pec_mask,
        pec_edge_masks=pec_edge_masks,
        pec_sheets=_pec_sheets,
        pec_wires=_pec_wires,
        pec_occupancy=pec_occupancy_override,
        sources=sources,
        probes=probes,
        wire_ports=wire_port_specs if wire_port_specs else None,
        s_param_freqs=sp_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=getattr(sim, '_pec_faces', None),
        pmc_faces=sim._boundary_spec.pmc_faces() if getattr(sim, '_boundary_spec', None) is not None else None,
        dft_planes=dft_plane_probes if dft_plane_probes else None,
        flux_monitors=(
            (flux_monitor_objs + wg_flux_monitors)
            if (flux_monitor_objs or wg_flux_monitors) else None
        ),
        rlc_metas=rlc_metas,
        rlc_states=rlc_states_init,
        ntff_box=ntff_box,
        ntff_data=ntff_data_init,
        waveguide_ports=waveguide_port_cfgs if waveguide_port_cfgs else None,
        tfsf=tfsf_pair,
        emit_time_series=emit_time_series,
    )
    if until_decay is not None:
        # #383: chunked host loop with the interior-energy stop. The
        # fences above already rejected checkpoint_every / n_warmup /
        # non-rect flux windows; ``checkpoint`` is accepted-and-ignored
        # (forward-only host loop — mirrors uniform run_until_decay's
        # treatment of the grad-tape flag).
        r = run_nonuniform_until_decay(
            grid, materials,
            decay_by=until_decay,
            check_interval=decay_check_interval,
            min_steps=decay_min_steps,
            max_steps=decay_max_steps,
            decay_energy_consecutive=decay_energy_consecutive,
            radiated_flux_box=radiated_flux_box,
            flux_env_checks=flux_env_checks,
            report_every=report_every,
            report_label=report_label,
            **_shared_run_kwargs,
        )
    elif (report_every is not None and not checkpoint
          and checkpoint_every is None and not n_warmup):
        # #667 on the NU lane: drive the SAME chunked host loop the decay
        # stop uses, with the stop disabled (decay_by=0.0 is that loop's
        # documented forced-N escape) and min_steps past the end so no
        # energy check ever runs. Chunk re-entry threads the full carry
        # (bit-identity locked by tests/unit/nonuniform/test_nu_progress_chunking.py).
        from rfx.progress import validate_report_every
        _re = validate_report_every(report_every, n_steps=n_steps)
        r = run_nonuniform_until_decay(
            grid, materials,
            decay_by=0.0,
            check_interval=_re,
            min_steps=n_steps + 1,
            max_steps=n_steps,
            decay_energy_consecutive=1,
            report_every=_re,
            report_label=report_label,
            **_shared_run_kwargs,
        )
    else:
        if report_every is not None:
            import warnings
            warnings.warn(
                f"report_every={report_every} is ignored on this non-uniform "
                "run: chunked progress (#667) composes with neither "
                "checkpoint/segmented-remat nor n_warmup on the NU lane "
                "(the chunked loop is forward-only).",
                UserWarning, stacklevel=2)
        r = run_nonuniform(
            grid, materials, n_steps,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            n_warmup=n_warmup,
            **_shared_run_kwargs,
        )

    s_params = r.get("s_params")
    freqs_out = r.get("s_param_freqs")

    # Waveguide-port output: dict[name -> cfg], optional sparams dict.
    waveguide_ports_result = None
    waveguide_sparams_result = None
    if sim._waveguide_ports and "waveguide_ports" in r:
        from rfx.sources.waveguide_port import (
            extract_waveguide_sparams,
            waveguide_plane_positions,
        )
        from rfx.api import WaveguideSParamResult
        final_cfgs = r["waveguide_ports"]
        waveguide_ports_result = {
            entry.name: cfg
            for entry, cfg in zip(sim._waveguide_ports, final_cfgs)
        }
        waveguide_sparams_result = {}
        for entry, cfg in zip(sim._waveguide_ports, final_cfgs):
            plane_positions = waveguide_plane_positions(cfg)
            source_plane = plane_positions["source"]
            measured_reference_plane = plane_positions["reference"]
            measured_probe_plane = plane_positions["probe"]
            if entry.calibration_preset == "source_to_probe":
                reference_plane = source_plane
                probe_plane = measured_probe_plane
                calibration_preset = "source_to_probe"
            elif entry.reference_plane is not None or entry.probe_plane is not None:
                reference_plane = (
                    entry.reference_plane
                    if entry.reference_plane is not None
                    else measured_reference_plane
                )
                probe_plane = (
                    entry.probe_plane
                    if entry.probe_plane is not None
                    else measured_probe_plane
                )
                calibration_preset = "explicit"
            else:
                reference_plane = measured_reference_plane
                probe_plane = measured_probe_plane
                calibration_preset = "measured"
            s11, s21 = extract_waveguide_sparams(
                cfg,
                ref_shift=reference_plane - measured_reference_plane,
                probe_shift=probe_plane - measured_probe_plane,
            )
            # jnp-native (NU flux-AD): this per-port diagnostic is a SIDE
            # output the normalize='flux' S-matrix path does not consume, but
            # run_nonuniform_path always assembles it. np.array(tracer) here
            # detaches / crashes the eps_override-traced device run (the same
            # concretization class as issue #70 / #148). Keeping these as
            # jnp arrays is forward bit-identical for concrete consumers
            # (np.asarray accepts jnp transparently — test_nonuniform_api /
            # test_api read .s11 via np.abs / np.isfinite).
            waveguide_sparams_result[entry.name] = WaveguideSParamResult(
                freqs=cfg.freqs,
                s11=s11,
                s21=s21,
                calibration_preset=calibration_preset,
                source_plane=float(source_plane),
                measured_reference_plane=measured_reference_plane,
                measured_probe_plane=measured_probe_plane,
                reference_plane=reference_plane,
                probe_plane=probe_plane,
            )

    # Repack DFT planes into {name: DFTPlaneProbe} dict to match
    # the uniform path's Result schema.
    dft_planes_dict = None
    if sim._dft_planes and "dft_planes" in r:
        dft_planes_dict = {
            entry.name: probe
            for entry, probe in zip(sim._dft_planes, r["dft_planes"])
        }

    # Repack flux monitors into {name: FluxMonitor} dict (uniform schema).
    # The accumulated list is sim._flux_monitors (front) + per-waveguide-port
    # flux monitors (back, only when attach_waveguide_flux=True).
    flux_monitors_dict = None
    if getattr(sim, "_flux_monitors", None) and "flux_monitors" in r:
        flux_monitors_dict = {
            entry.name: mon
            for entry, mon in zip(sim._flux_monitors, r["flux_monitors"])
        }

    # Extract per-waveguide-port Poynting flux spectra (issue #88 flux path).
    waveguide_port_flux_result = None
    if attach_waveguide_flux and wg_flux_monitors and "flux_monitors" in r:
        from rfx.probes.probes import flux_spectrum
        n_sim_flux = len(flux_monitor_objs)
        wg_final = r["flux_monitors"][n_sim_flux:]
        # jnp-native (NU flux-AD, mirrors the uniform PR #172 fix): keep
        # flux_spectrum on the AD tape — np.array() here would detach the
        # eps_override-traced gradient (the issue #148 concretization bug).
        # The sole consumer (compute_waveguide_s_matrix_nu, normalize='flux')
        # is jnp-native; np.asarray(...) downstream still accepts these for
        # any concrete-path numpy reader, so forward values are unchanged.
        waveguide_port_flux_result = tuple(
            flux_spectrum(m) for m in wg_final
        )

    # The per-port raw DFT accumulators (v, i, v_inc, v_port), the same
    # diagnostic channel the uniform lane fills. ``run_nonuniform`` already
    # builds them (#764); dropping them here left a caller that needs the
    # port's INCIDENT wave -- absorbed power as a fraction of what the port
    # delivered, say -- with only the S-parameter RATIO, which is
    # load-independent by construction and carries no level.
    #
    # Paired with the port metadata so the field has the SAME shape as the
    # uniform lane's (rfx/simulation.py, ``final_wire_sparams``): a tuple of
    # ``(meta, accs)``, so ``for spec, accs in result.wire_port_sparams``
    # runs unchanged on either lane. The first entry differs in TYPE between
    # the lanes -- see ``Result.wire_port_sparams`` in rfx/api/_spec.py.
    wire_port_sparams_result = None
    _wire_raw = r.get("wire_sparams_raw")
    if _wire_raw is not None:
        wire_port_sparams_result = tuple(
            zip(r.get("wire_sparams_meta", ()), _wire_raw)
        )

    return Result(
        state=r["state"],
        time_series=r["time_series"],
        s_params=s_params,
        freqs=freqs_out,
        ntff_data=r.get("ntff_data"),
        ntff_box=ntff_box,
        dft_planes=dft_planes_dict,
        wire_port_sparams=wire_port_sparams_result,
        flux_monitors=flux_monitors_dict,
        waveguide_ports=waveguide_ports_result,
        waveguide_sparams=waveguide_sparams_result,
        waveguide_port_flux=waveguide_port_flux_result,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
