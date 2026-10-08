"""Cell and per-component material assembly for uniform and profiled grids.

The legacy entry points adapt only the return tuple. Grid-type branches keep
existing lane-specific sampling, sheet folds and refusals; no new physics rule.
Conductors own their realized products in rfx.model.conductors.
`rasterize_geometry` (rfx/geometry/rasterize_grid.py) still holds the subgridded fine lane's and the NU dual-average path's copy of the geometry loop; a later S1 step retires it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import NamedTuple

import jax

import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer
from rfx.core.yee import (
    MaterialArrays, add_lumped_eps, cell_owned_component_materials,
    component_h_materials, edge_averaged_materials, edge_mean_components,
    lumped_components, lumped_total, permittivity_without_lumped,
)
from rfx.geometry.csg import Box, _grid_coords
from rfx.geometry._pole_keying import _accumulate_pole_mask, _spec_from_pole_masks
from rfx.geometry.rasterize_grid import (
    GridCoords, _material_cell_mask, assert_declared_span_is_filled,
    cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
    centres_from_nonuniform_grid, centres_from_uniform_grid,
    classify_pec_entry, coords_from_nonuniform_grid, extend_cpml_pad_materials,
    refuse_vaporized_sheets, sheet_end_rows, sheet_footprint_traced, sheet_spec_from_shape,
)
from rfx.geometry.smoothing import continued_conductor_shape
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.materials.thin_conductor import (
    apply_thin_conductor, check_sheet_occupancy, sheet_bounds,
)
from rfx.nonuniform import NonUniformGrid, e_node_dual_spacings


def assemble_cells(
    sim, grid, *, include_thin_conductors=True, include_cpml_pad_extension=True,
    sheet_specs=None, pec_sheets=None, pec_wires=None, pad_fill_findings=None,
    check_declared_span=True,
    geometry_masks=None, assembly_entries=None,
):
    """Build material arrays plus per-pole dispersion masks.

    Lattice ownership contract (#931): PEC geometry entries are
    classified as VOLUME (centre-sampled cells, into ``pec_mask``),
    SHEET (a zero-thickness Box, into ``pec_sheets``) or WIRE (a
    sub-cell ``PolylineWire``, into ``pec_wires``); PEC thin
    conductors are sheets.  Sheets and wires own no cell and are NOT
    in ``pec_mask``; realize them with
    ``rfx.boundaries.pec.realized_pec_edge_masks``.

    Parameters
    ----------
    sheet_specs : list or None
        Out-parameter for surface-impedance sheets. These carry a separate
        node-thin operator rather than folding into the volume sigma (#677).
    geometry_masks, assembly_entries : list or None
        Out-parameters for the geometry masks and conductor classification
        records consumed by the realized-conductor object. Entries retain
        the original declaration's identity through conductor continuation.
    pec_sheets, pec_wires : list or None
        Out-parameters (same pattern as ``sheet_specs``) so the
        positional return tuple stays unchanged.  Sheets and wires are
        always classified, but they own no cell, so a caller that
        passes no collector gets a ``pec_mask`` with the sheet MISSING
        — not a mask that contains it.  Omitting the collectors on a
        model that HAS a sheet or a wire is therefore a ``ValueError``
        naming the caller (:func:`rfx.api._compile._refuse_uncollected_pec`), not a
        mask the caller cannot tell from a conductor-free model.  A
        caller that steps fields realizes what it collected; a caller
        that only reads cells still passes ``pec_sheets=[],
        pec_wires=[]`` and drops the result, which makes "I read cells
        only" an explicit decision at the call site.
    pad_fill_findings : list or None
        Uniform-grid out-parameter. When a list is given, a declared-but-unfilled
        span at a padded hi face (#1070) is APPENDED to it instead of
        raising ``PadFillShortfall``. ``fidelity_report`` passes one:
        an audit whose job is to show where the realized model differs
        from the declared one has to show this difference too, and a
        report that refuses to run instead of naming the defect is the
        opposite of useful (review of PR #1136, A). A solve passes
        nothing and gets the raise, because there the pad would be
        filled with vacuum and the answer would be wrong.
    include_thin_conductors : bool, default True
        When False, stop one step short of the finished arrays and
        return the state as it is *before* the ``_thin_conductors``
        stage below — i.e. geometry rasterized and the CPML pad
        extended, but no thin conductor applied yet. Everything else
        (pole masks, conformal-face PEC injection, the ``has_pec``
        decision) is unchanged, so the only difference is that a
        lossy conductor's cells still carry their background material
        and PEC thin sheets are absent from ``pec_sheets`` (they own no cells).

        Only ``rfx.vmap_sweep`` passes False, and it needs this
        because the ORDER here is load-bearing: the pad is extended
        BEFORE thin conductors are applied. Their geometry now continues
        through reached absorbing faces before application. The batched sweep path has to re-extend
        the pad for each swept value, and extending the *finished*
        arrays would copy the conductor's array values instead of realizing
        its continued geometry (issue #642). Handing that path the
        pre-conductor arrays and letting it re-apply the same shared
        ``apply_thin_conductor`` afterwards reproduces this order
        instead of approximating it.

        The default is True and no other caller passes it, so
        ``run()`` and every existing path are unaffected by
        construction.
    include_cpml_pad_extension : bool, default True
        When False, skip the CPML pad extension entirely and return the
        arrays exactly as the geometry rasterized them — vacuum in the
        padding, and vacuum at any boundary node the rasterizer dropped.

        Same caller, same reason, one issue later. #637/#643 let the
        batched sweep re-extend already-extended arrays, on the argument
        that "every pad cell is overwritten by one of the three passes,
        so the result depends only on the INTERIOR values — which are
        the batch-correct ones". #655 made the shared rule repair the
        dropped hi-face boundary NODE as well as the pad, which is an
        interior cell that ``Shape.mask`` does not cover — so that
        premise stopped holding and the batched path inherited the BASE
        material there instead of its own swept value. Handing it the
        un-extended arrays and letting the shared rule take the whole
        decision per swept element restores the premise rather than
        patching around it (issue #642's lesson: the batched path was
        given the wrong INPUT, not running the wrong algorithm).

    Returns
    -------
    materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, boundary_pec_shapes, kerr_chi3
        pec_mask is a boolean array (True at PEC cells) or None.
        pec_shapes is a list of Shape objects that are PEC.
        boundary_pec_shapes is a list of PEC shapes from boundary conditions.
        kerr_chi3 is a float32 array of chi3 values or None.

    Materials/poles retain declaration order and float32 initialization.
    Overrides and design boxes remain downstream until M4.
    """
    # Start with vacuum
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    chi3_arr = jnp.zeros(grid.shape, dtype=jnp.float32)
    pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)
    pec_shapes = []
    has_kerr = False
    # Track whether any PEC cells were added as a Python-side (static)
    # predicate. This replaces a later ``bool(jnp.any(pec_mask))``, which
    # is a host-side boolean conversion on a device array — fine eagerly,
    # but it raises TracerBoolConversionError when the whole forward()
    # is wrapped in an outer ``jax.jit`` (the geometry-derived pec_mask
    # becomes a tracer). PEC cells enter pec_mask only from a PEC geometry
    # entry, keyed on static config; PEC thin conductors own sheets, not cells.
    # A Python flag set when volume cells are added is equivalent (it can only
    # differ when a PEC shape's mask is empty — e.g. entirely outside the
    # grid — where returning the all-False mask is a downstream no-op).
    has_pec_cells = False

    # Collect per-pole masks so distinct materials do not inherit
    # each other's dispersion poles. Keyed per
    # ``rfx.geometry._pole_keying._pole_key`` (#274): pole value when
    # hashable (equal poles dedupe/merge as before), ``id(pole)``
    # only for unhashable traced poles. Values are (pole, mask).
    debye_masks_by_pole: dict[DebyePole | int, tuple[DebyePole, jnp.ndarray]] = {}
    lorentz_masks_by_pole: dict[LorentzPole | int, tuple[LorentzPole, jnp.ndarray]] = {}

    # Grid type, not constant widths: constant-profile NU retains its sampler
    # and carrier restrictions. Uniform custom shapes may only implement mask(grid).
    nonuniform = isinstance(grid, NonUniformGrid)
    lane = "non-uniform" if nonuniform else "uniform"
    sampling_grid = None if nonuniform else grid
    if nonuniform:
        _coords = coords_from_nonuniform_grid(grid)
        _cell_sizes = cell_sizes_from_nonuniform_grid(grid)
        _centres = centres_from_nonuniform_grid(grid, _coords)
    else:
        _cx, _cy, _cz = _grid_coords(grid)
        _coords = GridCoords(x=_cx, y=_cy, z=_cz, shape=grid.shape)
        _centres = centres_from_uniform_grid(grid)
        _cell_sizes = cell_sizes_from_uniform_grid(grid)
    _pec_sheets = pec_sheets if pec_sheets is not None else []
    _pec_wires = pec_wires if pec_wires is not None else []

    # #1070: a structure declared out to a padded face must rasterize to
    # within the one node the half-open Box rule costs it. Checked only
    # when a pad will actually be filled from that edge, and only on a
    # concrete mask -- under an outer jit the mask is a tracer and the
    # question cannot be asked on the host.
    # The declared-span audit is a uniform-only contract (#1070).
    # The shared solve builder runs the same check on every grid itself
    # (rfx.model.pad_fill) and passes check_declared_span=False.
    _check_pad_fill = (check_declared_span and not nonuniform and include_cpml_pad_extension
                       and sim._boundary in ("cpml", "upml")
                       and sim._cpml_layers > 0)
    from rfx.geometry.smoothing import continued_conductor_shape, warn_unextendable_shapes
    conductor_findings = []

    # NU historically continues all PEC declarations before sampling any entry.
    geometry = [replace(entry, shape=continued_conductor_shape(
                    sim, grid, entry.shape, entry=entry, unextendable=conductor_findings))
                if sim._resolve_material(entry.material_name).sigma >= sim._PEC_SIGMA_THRESHOLD
                else entry for entry in sim._geometry] if nonuniform else sim._geometry
    for declared_entry, entry in zip(sim._geometry, geometry, strict=True):
        mat = sim._resolve_material(entry.material_name)
        mask = _material_cell_mask(entry.shape, _coords, _centres, grid=sampling_grid)

        if mat.sigma >= sim._PEC_SIGMA_THRESHOLD:
            solved_shape = entry.shape if nonuniform else continued_conductor_shape(
                sim, grid, entry.shape, entry=entry, unextendable=conductor_findings)
            # True PEC (#931): volume cells into pec_mask (centre
            # sampled, §1.1); a zero-thickness Box is a sheet; a
            # sub-cell PolylineWire is a filament. eps/sigma stay at
            # vacuum values either way.
            cells, sheet, wire = classify_pec_entry(
                solved_shape, _coords, _centres, _cell_sizes,
                name=entry.material_name, grid=sampling_grid)
            if assembly_entries is not None:
                assembly_entries.append((id(declared_entry), cells, sheet, wire, solved_shape))
            if cells is not None:
                pec_mask = pec_mask | cells
                has_pec_cells = True
                mask = cells
            elif sheet is not None:
                _pec_sheets.append(sheet)
                # Uniform chi3 uses the sheet footprint; NU retains its sampler.
                if not nonuniform:
                    mask = sheet.footprint
            else:
                _pec_wires.append(wire)
            pec_shapes.append(solved_shape)
        else:
            # #1070, and only here (review of PR #1136, C): the pad
            # extension replicates eps/sigma/mu, never ``pec_mask``, so
            # the vacuum-in-the-pad failure this checks for cannot happen
            # to a PEC entry. Asking about one would report a condition
            # that does not exist, in a message about dielectric pads.
            # It has to sit AFTER classify_pec_entry, because that is
            # what decides which an entry is.
            # Read the DECLARED domain, not ``sim._domain``: that
            # attribute is a mesh descriptor and reading it RESOLVES the
            # mesh. differentiable_material_fit builds its grid once and
            # then assembles a brand-new Simulation carrying traced
            # materials on every step; that object has never resolved
            # its mesh, so the read would run the auto-mesh planner on a
            # tracer, which refuses. Assembly is handed a built ``grid``
            # and must not plan a mesh. ``domain`` is a required
            # constructor argument, so the declaration is always there.
            # GPU run 369367262302 on a6d6fce1, regression of PR #1136.
            if _check_pad_fill and not is_tracer(mask):
                assert_declared_span_is_filled(
                    entry.material_name, entry.shape, mask, grid,
                    sim._unresolved_domain,
                    record=pad_fill_findings)
            eps_r = jnp.where(mask, mat.eps_r, eps_r)
            sigma = jnp.where(mask, mat.sigma, sigma)
            mu_r = jnp.where(mask, mat.mu_r, mu_r)

        if geometry_masks is not None and mat.sigma < sim._PEC_SIGMA_THRESHOLD:
            geometry_masks.append((id(declared_entry), mask))

        if mat.chi3 != 0.0:
            chi3_arr = jnp.where(mask, mat.chi3, chi3_arr)
            has_kerr = True

        pole_mask = mask
        if mat.debye_poles or mat.lorentz_poles:
            # Poles retain their declared occupancy, before continuation.
            pole_mask = _material_cell_mask(declared_entry.shape, _coords, _centres, grid=sampling_grid)
            if mat.sigma >= sim._PEC_SIGMA_THRESHOLD and cells is not None:
                from rfx.geometry.rasterize_grid import pec_volume_cell_mask
                pole_mask = pec_volume_cell_mask(declared_entry.shape, _centres, _cell_sizes, grid=sampling_grid)

        if mat.debye_poles:
            for pole in mat.debye_poles:
                _accumulate_pole_mask(debye_masks_by_pole, pole, pole_mask)

        if mat.lorentz_poles:
            for pole in mat.lorentz_poles:
                _accumulate_pole_mask(lorentz_masks_by_pole, pole, pole_mask)

    # The NU stepper installs no periodic BC and NU grids are 3-D, so the
    # non-periodic #689 convention is the one this lane's step function
    # uses; the guard must ask
    # with the SAME flags or it judges a seam the solve never has.
    # NU's geometry-sheet guard precedes pads and thin sheets. Its stepper
    # installs no periodic boundary; preserve both the timing and those flags.
    if nonuniform:
        pec_mask = _optional_pec_mask(pec_mask, has_pec_cells)
        refuse_vaporized_sheets(_pec_sheets, lane=lane, periodic=(False, False, False))

    # Extend material properties into CPML padding so that guided
    # modes in dielectric waveguides see an impedance-matched absorber
    # (equivalent to UPML).  Each CPML face copies the interior-edge
    # slice outward, as if the geometry continued beyond the domain.
    # extend_cpml_pad_materials holds the #627 fix: the former
    # hand-duplicated copies (#582) both carried a hi-face vacuum
    # column for a domain-touching box; the fix lives once.
    #
    # Dispersion-pole masks are deliberately NOT extended here (#627b
    # tried and reverted): extending a high-Q (Q~60) Lorentz pole into
    # the pad turns a stable edge-touching simulation into a divergent
    # one (last/mid energy ratio 649 vs 0.12-0.16 decaying on every
    # other tested variant, including the static extension below with
    # the same pole left un-extended in the interior) — with no NaN
    # and no exception, so nothing downstream catches it. See
    # extend_cpml_pad_materials's docstring and the follow-up issue
    # (filed separately, tracking the stability factorial).
    #
    # And the hi-face fallback never promotes a pole-carrying column's
    # STATICS either (#808): promoting them puts the material's eps_inf
    # without its poles into the pad and the repaired boundary node — a
    # material no declared model has — which moved a committed Debye
    # recovery from its pinned 11% error to 32%, past its 20% gate.
    # The combined pole mask below gates exactly that fallback; static
    # materials keep the full #627a/#655 behaviour.
    # Before #582 the NU assembler had no pad extension, so an edge-touching
    # structure saw a different absorber medium per path: measured 736 pad
    # cells eps 4.0-vs-1.0 at the slab's k=9 layer. The uniform-mesh reduction
    # anchor diverged wherever that mismatch interacted with subpixel smoothing.
    # Extend statics only, before thin folds. Never extend poles or Kerr (#627b);
    # pole occupancy gates hi-face static promotion (#808).
    if (include_cpml_pad_extension
            and sim._boundary in ("cpml", "upml")
            and sim._cpml_layers > 0):
        # Per-face allocation (2026-04): (pad_{axis}_lo / _hi). Reflector /
        # periodic faces have pad=0 on that side and the corresponding
        # replicate step is skipped so the interior cells are not
        # overwritten. The replicate depth matches the actual
        # allocation on that face (``pad_*_lo`` or ``pad_*_hi``).
        plx, phx = grid.pad_x_lo, grid.pad_x_hi
        ply, phy = grid.pad_y_lo, grid.pad_y_hi
        plz, phz = grid.pad_z_lo, grid.pad_z_hi
        _pole_mask_any = None
        for _, _pmask in (list(debye_masks_by_pole.values())
                          + list(lorentz_masks_by_pole.values())):
            _pole_mask_any = (_pmask if _pole_mask_any is None
                              else (_pole_mask_any | _pmask))
        eps_r, sigma, mu_r = extend_cpml_pad_materials(
            eps_r, sigma, mu_r, plx, phx, ply, phy, plz, phz,
            dispersion_pole_mask=_pole_mask_any,
        )

    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    # Apply thin conductors (#931: PEC thin sheets go to ``pec_sheets``,
    # never to pec_mask; f0 sheets to ``sheet_specs``; DC folds to sigma).
    # This runs AFTER material-array extension, with continued geometry.
    # rfx.vmap_sweep depends
    # on being able to observe the state just before this stage — see
    # ``include_thin_conductors`` in this function's docstring (#642).
    if include_thin_conductors:
        materials, pec_mask = _fold_thin_conductors(
            sim, grid, materials, pec_mask, pec_shapes, _pec_sheets,
            sheet_specs, geometry_masks, assembly_entries, conductor_findings)

    # Node-pinned PEC sheets (add_pinned_sheet): built from node indices,
    # so the same call gives the same footprint here and on the NU lane.
    # Gated with the metric thin conductors so vmap_sweep's
    # ``include_thin_conductors=False`` observation point still sees the
    # state before EVERY sheet.
    # Node-pinned PEC sheets (add_pinned_sheet): built from node indices, so
    # they need no node POSITION and are the one sheet declaration a traced
    # mesh can carry. Same helper as the uniform lane, so the two cannot
    # disagree on what a pinned range realizes.
    if include_thin_conductors:
        from rfx.materials.thin_conductor import pinned_sheet_spec
        for _ps in getattr(sim, "_pinned_sheets", ()) or ():
            _pec_sheets.append(pinned_sheet_spec(grid, _ps))

    # Stage 1 conformal PEC face-shift (issue: WR-90 mesh-conv xfail).
    # When an axis is declared ``Boundary(conformal=True)`` we promote
    # its boundary-face PEC into a half-space ``Box`` injected into
    # ``pec_shapes`` so the existing Dey-Mittra path
    # (``run_uniform(conformal_pec=True, pec_shapes=…)``) sees a real
    # PEC volume at the physical wall coordinate. Default off keeps
    # the current binary ``apply_pec_faces`` semantics bit-identical.
    # Boundary-face half-space boxes (conformal=True faces only).
    # Tracked separately from geometry pec_shapes so the normalize=True
    # reference run sees only boundary walls — not interior PEC obstacles.
    boundary_pec_shapes: list = []
    # Only the uniform Dey-Mittra consumer carries boundary half-space shapes.
    conformal_faces = () if nonuniform else sim._boundary_spec.conformal_faces()
    if conformal_faces:
        # Declared, not resolved: see the pad-fill check above.
        _conformal_domain = sim._unresolved_domain
        big = max(_conformal_domain) * 100.0
        for face in conformal_faces:
            axis_name, side = face.split("_")
            axis_idx = "xyz".index(axis_name)
            # Auto-derive wall coordinate from waveguide ports whose
            # propagation direction is *transverse* to this axis.
            # Take the most restrictive aperture: max(lo) for the
            # lo-face wall, min(hi) for the hi-face wall — that is
            # the largest waveguide-interior region all ports agree
            # to leave free of PEC.
            wall_lo = 0.0
            wall_hi = float(_conformal_domain[axis_idx])
            for entry in sim._waveguide_ports:
                if entry.direction[1] == axis_name:
                    # Port-normal axis — no transverse wall on this
                    # face from this port.
                    continue
                rng = (entry.x_range, entry.y_range,
                       entry.z_range)[axis_idx]
                if rng is None:
                    # Port covers full domain along this axis —
                    # contributes no fractional cell.
                    continue
                wall_lo = max(wall_lo, float(rng[0]))
                wall_hi = min(wall_hi, float(rng[1]))

            corner_lo = [-big, -big, -big]
            corner_hi = [big, big, big]
            if side == "lo":
                # Skip when the wall coincides with the grid origin
                # (y=0 PEC face): the binary apply_pec_faces handles
                # this exactly and a Dey-Mittra Box at corner_hi=0
                # would impose a spurious 0.5 weight on the cell at
                # j=0. Only inject when an actual interior region
                # past the lo face needs to be PEC-fied.
                if wall_lo <= 0.0:
                    continue
                corner_hi[axis_idx] = wall_lo
            else:  # hi
                # Always inject on the hi side. The grid often
                # extends past the declared domain due to dx-snap or
                # CPML padding on other axes, so a fractional cell
                # exists at the wall even when ``wall_hi`` equals
                # the user-declared domain extent. When no
                # fractional cell is present (grid edge ≤
                # wall_hi), the SDF naturally produces weight=1
                # everywhere and the Box is a harmless no-op.
                corner_lo[axis_idx] = wall_hi
            _bpec_box = Box(tuple(corner_lo), tuple(corner_hi))
            pec_shapes.append(_bpec_box)
            boundary_pec_shapes.append(_bpec_box)

    debye_spec = _spec_from_pole_masks(debye_masks_by_pole)
    lorentz_spec = _spec_from_pole_masks(lorentz_masks_by_pole)

    if not nonuniform:
        pec_mask = _optional_pec_mask(pec_mask, has_pec_cells)
    kerr_chi3 = chi3_arr if has_kerr else None
    from rfx.materials.thin_conductor import (
        warn_sheet_planes_inside_dielectric,
    )
    # #931: a sheet plane buried strictly inside a dielectric body is
    # the geometry the declaration did NOT describe. Warn (preflight
    # names it too); nothing is re-sampled.
    warn_sheet_planes_inside_dielectric(_pec_sheets, materials.eps_r)
    from rfx.api._compile import _refuse_uncollected_pec
    _refuse_uncollected_pec(_pec_sheets if pec_sheets is None else (),
                            _pec_wires if pec_wires is None else (),
                            lane=lane)
    if not nonuniform:
        refuse_vaporized_sheets(_pec_sheets, lane=lane, periodic=sim._periodic_flags())
    warn_unextendable_shapes(conductor_findings)
    return materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, boundary_pec_shapes, kerr_chi3


def _optional_pec_mask(mask, has_pec_cells):
    """Keep eager empty masks absent and the outer-jit decision static."""
    # Eager path keeps the exact ``jnp.any`` test (a PEC shape whose mask is
    # empty -- e.g. entirely outside the grid -- still returns None, so the
    # eager result is bit-identical). Only under an outer ``jax.jit`` trace,
    # where pec_mask is a tracer and cannot be host-converted to bool, do we
    # fall back to the static Python predicate (which can over-approximate
    # only in that empty-mask corner). This makes forward()/optimize()
    # wrappable in an outer jax.jit without changing any eager behaviour.
    has_pec = has_pec_cells if is_tracer(mask) else bool(jnp.any(mask))
    return mask if has_pec else None


def _fold_thin_conductors(sim, grid, materials, pec_mask, pec_shapes, pec_sheets,
                          sheet_specs, geometry_masks, assembly_entries, findings):
    """Continue declarations and apply the one model fold in declared order."""
    for declared in sim._thin_conductors:
        key = id(declared)
        tc = replace(declared, shape=continued_conductor_shape(
            sim, grid, declared.shape, entry=declared, unextendable=findings))
        materials, pec_mask = apply_thin_conductor(
            grid, tc, materials, pec_mask=pec_mask, sheet_specs=sheet_specs,
            sheets=pec_sheets, geometry_masks=geometry_masks, geometry_key=key)
        if assembly_entries is not None:
            assembly_entries.append((key, None, pec_sheets[-1] if tc.is_pec else None,
                                     None, tc.shape))
        if tc.is_pec:
            pec_shapes.append(tc.shape)
    return materials, pec_mask


class ComponentCells(NamedTuple):
    """Finished cell arrays and the occupancy specs returned by assemble_cells."""

    materials: MaterialArrays
    debye_spec: object = None
    lorentz_spec: object = None


class ComponentDispersion(NamedTuple):
    """ADE pole coefficients; weights exist only for dt-less direct callers."""

    poles: tuple
    coefficients: object
    weights: object
    dt: object


class KernelComponents(NamedTuple):
    """Only component operands read by a time step (no assembly arrays)."""

    eps_update: tuple
    sigma_update: tuple
    mu_update: tuple


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ComponentMaterials:
    """Realized volume operands, separate stamps, and finished kernel arrays.

    ``eps``/``sigma``/``mu`` hold volume values. ``*_update`` include only
    the corresponding component's lumped/wire stamp, already added once.
    UPML retains its historical cell-owned E convention explicitly. Tensor
    corrections and design windows remain separate until their assembly stage
    migrates; this step does not change their constitutive rule.
    """

    eps: tuple
    sigma: tuple
    mu: tuple
    eps_lumped: tuple
    sigma_lumped: tuple
    mu_wire: tuple
    eps_update: tuple
    sigma_update: tuple
    mu_update: tuple
    upml_eps: tuple
    upml_sigma: tuple
    debye: ComponentDispersion | None
    lorentz: ComponentDispersion | None
    cells_view: MaterialArrays
    periodic: tuple = field(metadata={"static": True})
    grid_key: tuple = field(metadata={"static": True})
    source_upml: bool = field(default=False, metadata={"static": True})


from types import SimpleNamespace  # noqa: E402


class EdgePoles(NamedTuple):
    """Pole specs for a lane that keeps no whole-domain realization (the
    distributed runners): carried in the ``components`` slot of the materials
    handed to the drive builders only. An edge's ADE terms are built from the
    cells around that edge, by the functions the kernels' coefficients use."""

    debye_spec: object
    lorentz_spec: object
    grid: object = None

    def at(self, cell, dt):
        lo = tuple(max(int(i) - 1, 0) for i in cell)
        window = tuple(slice(a, int(i) + 1) for a, i in zip(lo, cell))
        local = (slice(None),) + tuple(int(i) - a for a, i in zip(lo, cell))
        shape = tuple(int(i) + 1 - a for a, i in zip(lo, cell))
        out = []
        for spec, kind in ((self.debye_spec, "Debye"), (self.lorentz_spec, "Lorentz")):
            if spec is None:
                out.append(None)
                continue
            poles, masks = spec
            masks = masks if isinstance(masks, (tuple, list)) else [masks] * len(poles)
            cut = (poles, [m if m is None else m[window] for m in masks])
            widths = electric_cell_sizes(self.grid)
            widths = None if widths is None else tuple(
                None if d is None else d[w] for d, w in zip(widths, window))
            terms = _realize_poles(cut, (False,) * 3, kind, SimpleNamespace(dt=dt), shape,
                                   cell_sizes=widths).coefficients
            out.append(tuple(tuple(v[local] for v in t) if isinstance(t, tuple) else t[local]
                             for t in terms))
        return out


def _ade_cb(eps, sigma, dt, axis, debye_terms, lorentz_terms):
    """The dispersive E update's curl coefficient on one edge."""
    from rfx.materials.debye import debye_e_coeffs
    from rfx.materials.lorentz import lorentz_e_coeffs, mixed_e_component_coeffs
    operands = ((eps,) * 3, (sigma,) * 3)
    dc = lc = None
    if debye_terms is not None:
        dc = debye_e_coeffs(operands, dt, *debye_terms)
    if lorentz_terms is not None:
        lc = lorentz_e_coeffs(operands, dt, *lorentz_terms)
    if dc is not None and lc is not None:
        return mixed_e_component_coeffs(dc, lc, axis, dt)[1]
    return (dc if dc is not None else lc).cb[axis]


def e_update_material_at(materials, cell, component, periodic=(False, False, False), *, grid=None, cell_sizes=None):
    """Relative epsilon and conductivity of the realized E update at one edge.

    Stamps are already included in the stored operands. Legacy callers without
    a realization retain the single-edge averaging rule until they migrate.
    """
    c = getattr(materials, "components", None)
    if c is None or isinstance(c, EdgePoles):
        from rfx.core.yee import cell_component_e_materials
        return cell_component_e_materials(materials, cell, component, periodic,
                                         cell_sizes=(cell_sizes if cell_sizes is not None else
                                             electric_cell_sizes(c.grid if isinstance(c, EdgePoles) else grid)))
    axis = {"ex": 0, "ey": 1, "ez": 2}[str(component).lower()]
    cell = tuple(cell)
    return c.eps_update[axis][cell], c.sigma_update[axis][cell]


def e_update_coefficient_at(materials, cell, component, dt,
                            periodic=(False, False, False), *, host=False, grid=None):
    """Read the run's electric-current coefficient on one realized edge.

    Pole coefficients are sampled before constructing the scalar ADE operator;
    this avoids rebuilding grid-sized coefficient arrays for every source.
    ``host`` retains the graded current source's historical scalar arithmetic
    on plain edges. All dispersive coefficients use the kernel's algebra.
    """
    from rfx.core.yee import e_update_coeffs
    eps, sigma = e_update_material_at(materials, cell, component, periodic, grid=grid)
    c = getattr(materials, "components", None)
    axis = {"ex": 0, "ey": 1, "ez": 2}[str(component).lower()]
    if isinstance(c, EdgePoles):
        terms = c.at(cell, dt)
        if terms[0] is not None or terms[1] is not None:
            return _ade_cb(eps, sigma, dt, axis, *terms)
        c = None  # Plain graded drives retain their historical host arithmetic.
    if c is not None and c.source_upml:
        # Admission excludes pads. There sigma_perp=0; UPML still uses its
        # cell-owned E operands, even at an interior material interface.
        from rfx.boundaries.upml import _upml_e_coeffs_si, _upml_e_coeffs_eps_r
        from rfx.core.yee import si_value_eps_r_grad
        return si_value_eps_r_grad(
            _upml_e_coeffs_si, _upml_e_coeffs_eps_r, jnp.float32(0),
            c.upml_eps[axis][tuple(cell)],
            c.upml_sigma[axis][tuple(cell)].astype(jnp.float32), jnp.float32(dt))[1]
    if c is not None and (c.debye is not None or c.lorentz is not None):
        pole_cell = (slice(None),) + tuple(cell)
        dt_terms = lt_terms = None
        if c.debye is not None:
            alpha, beta = c.debye.coefficients
            dt_terms = (alpha[pole_cell], tuple(b[pole_cell] for b in beta))
        if c.lorentz is not None:
            a, b, strength = c.lorentz.coefficients
            lt_terms = (a[pole_cell], b[pole_cell], tuple(v[pole_cell] for v in strength))
        return _ade_cb(eps, sigma, dt, axis, dt_terms, lt_terms)
    if host:
        from rfx.nonuniform import current_source_cb
        return current_source_cb(eps, sigma, dt,
                                 traced=is_tracer(eps) or is_tracer(sigma))
    return e_update_coeffs(eps, sigma, dt)[1]


def electric_cell_sizes(grid):
    """One-dimensional primal metrics; None axes retain the equal-cell path."""
    if grid is None:
        return None
    if hasattr(grid, "cells"):
        return tuple(None if grid.is_constant(a) else grid.cells(a) for a in range(3))
    # Legacy direct realization accepts metric records as well as grid classes.
    # A shape/dt-only record declares equal cells. Width records use the same
    # terminal-node duplicate convention as component_h_materials.
    if not hasattr(grid, "dx_arr"):
        return None
    import numpy as np
    out = []
    for d, n in zip((grid.dx_arr, grid.dy_arr, grid.dz), grid.shape):
        xp = jnp if is_tracer(d) else np
        d = xp.asarray(d)
        if d.shape[0] == n - 1:
            d = xp.concatenate((d, d[-1:]))
        out.append(None if not is_tracer(d) and np.all(d == d[0]) else d)
    return tuple(out)


def electric_grid_kwargs(grid):
    """Equal-cell readers keep their original call signature and arithmetic."""
    widths = electric_cell_sizes(grid)
    return {} if widths is None or all(d is None for d in widths) else dict(grid=grid)


def pole_component_weights(mask, periodic=(False, False, False), *, cell_sizes=None):
    """Dual-volume pole occupancy, using the electric material rule."""
    if mask is None:
        return None
    return edge_mean_components(jnp.asarray(mask, dtype=bool).astype(jnp.float32), periodic,
                                cell_sizes=cell_sizes)


def _realize_poles(spec, periodic, kind, grid, shape, *, cell_sizes=None):
    if spec is None:
        return None
    poles, masks = spec
    if isinstance(masks, (tuple, list)):
        if len(masks) != len(poles):
            raise ValueError(f"Expected {len(poles)} {kind} masks, got {len(masks)}")
    else:
        masks = [masks] * len(poles)
    weights = tuple(pole_component_weights(m, periodic, cell_sizes=cell_sizes) for m in masks)
    if grid is None:
        return ComponentDispersion(tuple(poles), None, weights, None)
    if kind == "Debye":
        from rfx.materials.debye import debye_pole_coeffs as build
    else:
        from rfx.materials.lorentz import lorentz_pole_coeffs as build
    return ComponentDispersion(tuple(poles), build(poles, grid.dt, shape, weights), None, grid.dt)


def realize_components(cells, grid, *, periodic):
    """Realize final cells once, using the existing E/pole and H rules.

    ``cells`` is a MaterialArrays or ComponentCells with dispersion specs.
    ``grid=None`` is the uniform-metric compatibility path for direct ADE
    callers. E materials and pole fractions use the same primal dual-volume
    weights; constant transverse axes retain the original arithmetic.
    """
    if isinstance(cells, ComponentCells):
        materials, debye_spec, lorentz_spec = cells
    else:
        materials, debye_spec, lorentz_spec = cells, None, None
    # A cells view never contains another realization.
    materials = materials._replace(components=None)
    eps_lumped = lumped_components(materials.eps_r_lumped)
    sigma_lumped = lumped_components(materials.sigma_lumped)
    sigma_volume = materials.sigma
    sigma_stamp = lumped_total(sigma_lumped)
    if sigma_stamp is not None:
        sigma_volume = sigma_volume - sigma_stamp
    eps, sigma = edge_averaged_materials(
        permittivity_without_lumped(materials), sigma_volume, periodic,
        cell_sizes=electric_cell_sizes(grid))
    widths = ((grid.dx_arr, grid.dy_arr, grid.dz)
              if hasattr(grid, "dx_arr") else None)
    mu = component_h_materials(materials._replace(mu_r_wire=None),
                               periodic, cell_sizes=widths)
    mu_wire = lumped_components(materials.mu_r_wire)
    eps_update = add_lumped_eps(eps, materials.eps_r_lumped)
    sigma_update = tuple(s if p is None else s + p for s, p in zip(sigma, sigma_lumped))
    mu_update = tuple(m if p is None else m + p for m, p in zip(mu, mu_wire))
    upml_eps, upml_sigma = cell_owned_component_materials(materials)
    return ComponentMaterials(
        eps, sigma, mu, eps_lumped, sigma_lumped, mu_wire,
        eps_update, sigma_update, mu_update, upml_eps, upml_sigma,
        _realize_poles(debye_spec, periodic, "Debye", grid, materials.eps_r.shape,
                       cell_sizes=electric_cell_sizes(grid)),
        _realize_poles(lorentz_spec, periodic, "Lorentz", grid, materials.eps_r.shape,
                       cell_sizes=electric_cell_sizes(grid)),
        materials, tuple(periodic), _grid_key(grid))


def with_components(materials, grid, *, periodic, debye_spec=None, lorentz_spec=None):
    """Attach one realization after the last cell/stamp write, or reuse it.

    Callers must not modify the cells after this boundary. Raw low-level
    runner inputs enter here too; no global or identity cache is involved.
    """
    if materials.components is not None:
        validate_components(materials, periodic=periodic, grid=grid)
        return materials
    components = realize_components(
        ComponentCells(materials, debye_spec, lorentz_spec), grid, periodic=periodic)
    return materials._replace(components=components)


def _grid_key(grid):
    if grid is None:
        return ()
    return (tuple(grid.shape), tuple(id(getattr(grid, name, None))
                                    for name in ("dx_arr", "dy_arr", "dz")))


def validate_components(materials, *, periodic, grid=None):
    """Reject stale assembly views before entering any compiled time loop."""
    components = materials.components
    if components.periodic != tuple(periodic):
        raise ValueError("realized material periodic flags differ from the kernel")
    # The lumped/wire records are inputs of the realization too (review of
    # M2a, round 3): replacing one alone must not reuse the old components.
    for name in ("eps_r", "sigma", "mu_r", "sigma_lumped", "eps_r_lumped", "mu_r_wire"):
        if getattr(components.cells_view, name, None) is not getattr(materials, name, None):
            raise ValueError(f"stale realized material: {name} cell array changed")
    if grid is not None and components.grid_key and components.grid_key != _grid_key(grid):
        raise ValueError("realized material grid differs from the kernel")


def kernel_materials(materials, *, keep_eps=False, electric=True, magnetic=True, epsilon=True):
    """Drop assembly-only leaves before a scan/jitted step sees materials.

    Diagnostic captures explicitly observe cell arrays too. Kerr reads cell
    epsilon; all other step operands come from the realized components.
    """
    from rfx import _realized
    if _realized.ACTIVE is not None:
        return materials
    c = materials.components
    return MaterialArrays(
        materials.eps_r if keep_eps else jnp.zeros((), materials.eps_r.dtype), None, None,
        mu_r_wire=True if materials.mu_r_wire is not None else None,
        components=KernelComponents(c.eps_update if electric and epsilon else (),
                                    c.sigma_update if electric else (),
                                    c.mu_update if magnetic else ()))


def kernel_context(ctx):
    """Select only the operands of the enabled uniform update branches."""
    return replace(ctx, materials=kernel_materials(
        ctx.materials, keep_eps=ctx.use_kerr,
        electric=not (ctx.use_debye or ctx.use_lorentz or ctx.use_upml or ctx.use_fast_he),
        magnetic=not (ctx.use_upml or ctx.use_fast_he),
        epsilon=ctx.aniso_eps is None and ctx.aniso_inv_eps is None))


def h_components(materials, grid, *, periodic):
    """H-only compatibility entry for raw CPML callers; no E/pole allocation."""
    components = getattr(materials, "components", None)  # legacy views lack it
    if components is not None:
        return components.mu_update
    widths = (grid.dx_arr, grid.dy_arr, grid.dz) if hasattr(grid, "dx_arr") else None
    return component_h_materials(materials, periodic, cell_sizes=widths)


def validate_dispersion(realized, poles, dt):
    """Cached ADE coefficients belong to the realization's poles and timestep."""
    if realized.coefficients is None:
        return
    def same(a, b):
        return a is b or (not is_tracer(a) and not is_tracer(b)
                          and jnp.ndim(a) == jnp.ndim(b) == 0 and bool(a == b))
    if not same(realized.dt, dt):
        raise ValueError("realized ADE coefficients have a different timestep")
    if not all(same(a, b) for p, q in zip(realized.poles, poles)
               for a, b in zip(p, q)):
        raise ValueError("realized ADE coefficients have different pole parameters")


def _design_box_edge_coeffs(bounds, eps_r_box, sigma_box, materials, dt, shape,
                            held_edges=(), *, grid=None):
    """Per-component ``(Ca, Cb)`` over the design box's write window (#1210).

    The box works in EDGE (per-E-component) values, and what it is handed
    decides how each value is made:

    * a CELL quantity -- the design permittivity always, and a single design
      conductivity array -- is written into a COPY of the background over the
      computation window and turned into edge values by the same four-cell
      average as the rest of the grid (``component_e_materials``). The
      window's minus-side context supplies the neighbour cells outside the box
      on its faces, and its plus-side layer is written because a cell's value
      reaches the edges on its plus faces. The box lane then equals what
      ``update_e`` would do with the design values in ``materials``.
    * a 3-tuple ``(sigma_x, sigma_y, sigma_z)`` is ALREADY per edge (#1216,
      the sheet lane: one conductivity per Yee edge, component ``c`` at the
      same box index). It is taken as-is and NOT averaged again: it replaces
      the edge conductivity at the box indices, and every other edge in the
      window -- the plus-side layer included -- keeps the background edge
      value. A box-shaped edge array cannot address the plus-face edge layer
      a cell array reaches, which is why a single array ``s`` and the tuple of
      its own averages agree on the box's edges and not on that layer.

    ``sigma_box=None`` keeps the window's own conductivity.

    ``held_edges`` (``(axis, i, j, k)`` tuples, all inside the write window)
    are a port's edges held out of the design (:class:`DesignBoxSpec`). Each
    one gets the coefficient ``materials`` itself gives it — the same
    arithmetic over the same window with no design value in it, which is the
    grid-wide ``update_e`` coefficient bit for bit, the port's own load
    included — so the drive built from ``materials`` before the time loop
    meets the update it was built for, and the design values' derivative
    through that edge is exactly zero.

    At a box cell carrying a lumped load -- a held port's, or a passive
    termination's (an MSL port with ``excite=False``) -- a CELL design value
    is that cell's volume material: the load is added back on top of it, so
    the average that removes every stamp before it spreads the cell over its
    edges finds the design value there and not the design value less the
    load (which was a negative conductance, and NaN, before). That is what a
    whole-grid ``eps_override`` / ``sigma_override`` does, since the ports
    stamp their loads on top of the override. A per-edge conductivity has no
    such reading and is refused over an unheld load
    (:func:`_resolve_design_box`).
    """
    from rfx.simulation import _design_box_window, _held_edge_masks
    from rfx.core.yee import component_e_materials, e_update_coeffs, map_lumped
    held = tuple(held_edges or ())
    write_bounds, win, inner, box_local = _design_box_window(bounds, shape)

    def _cell_value(box_value, record):
        # A lumped load lives on its own edge; keep it off the cell volume.
        total = lumped_total(record)
        if total is None:
            return box_value
        return box_value + jnp.asarray(total)[win][box_local]

    eps_box = jnp.asarray(eps_r_box)
    eps = jnp.asarray(materials.eps_r)[win]
    # Promote the BACKGROUND to the design dtype, never the other way: a
    # traced float64 design permittivity cast down to the float32 background
    # would silently lose the precision the x64 AD lanes run for (#646).
    eps = eps.astype(jnp.promote_types(eps.dtype, eps_box.dtype))
    eps = eps.at[box_local].set(_cell_value(
        eps_box, getattr(materials, "eps_r_lumped", None)))

    per_edge = isinstance(sigma_box, (tuple, list))
    sig = jnp.asarray(materials.sigma)[win]
    if not per_edge and sigma_box is not None:
        sig_box = jnp.asarray(sigma_box)
        sig = sig.astype(jnp.promote_types(sig.dtype, sig_box.dtype))
        sig = sig.at[box_local].set(_cell_value(
            sig_box, getattr(materials, "sigma_lumped", None)))

    # A lumped stamp in the window's context layer is edge-owned, not a cell
    # volume, so it is removed before the average and added back at its cell,
    # on its own component — the same rule ``component_e_materials`` applies
    # grid-wide (#1210, #1236). The design box itself is fenced off port and
    # source cells, except a port's own edges on explicit opt-in (held).
    win_mats = MaterialArrays(
        eps_r=eps, sigma=sig, mu_r=None,
        eps_r_lumped=map_lumped(
            getattr(materials, "eps_r_lumped", None),
            lambda a: jnp.asarray(a)[win].astype(eps.dtype)),
        sigma_lumped=map_lumped(
            getattr(materials, "sigma_lumped", None),
            lambda a: jnp.asarray(a)[win].astype(sig.dtype)))
    widths = electric_cell_sizes(grid)
    widths = None if widths is None else tuple(
        None if d is None else d[w] for d, w in zip(widths, win))
    eps_c, sig_c = component_e_materials(win_mats, (False, False, False), cell_sizes=widths)

    if per_edge:
        def _put(edge_bg, edge_box):
            edge_box = jnp.asarray(edge_box)
            edge_bg = edge_bg.astype(
                jnp.promote_types(edge_bg.dtype, edge_box.dtype))
            return edge_bg.at[box_local].set(edge_box)
        sig_c = tuple(_put(bg, v) for bg, v in zip(sig_c, sigma_box))

    pairs = [e_update_coeffs(e, s_, dt) for e, s_ in zip(eps_c, sig_c)]
    ca = tuple(p[0][inner] for p in pairs)
    cb = tuple(p[1][inner] for p in pairs)
    if held:
        bg_mats = MaterialArrays(
            eps_r=jnp.asarray(materials.eps_r)[win],
            sigma=jnp.asarray(materials.sigma)[win], mu_r=None,
            eps_r_lumped=map_lumped(
                getattr(materials, "eps_r_lumped", None),
                lambda a: jnp.asarray(a)[win]),
            sigma_lumped=map_lumped(
                getattr(materials, "sigma_lumped", None),
                lambda a: jnp.asarray(a)[win]))
        bg = [e_update_coeffs(e, s_, dt) for e, s_ in
              zip(*component_e_materials(bg_mats, (False, False, False), cell_sizes=widths))]
        masks = _held_edge_masks(held, write_bounds)
        ca = tuple(a if m is None else jnp.where(m, b[0][inner], a)
                   for a, b, m in zip(ca, bg, masks))
        cb = tuple(c if m is None else jnp.where(m, b[1][inner], c)
                   for c, b, m in zip(cb, bg, masks))
    return write_bounds, ca, cb


def slab_pole_fractions(mask, nx_per, nx, rank=None, *, cell_sizes=None):
    """Pole-mask means with slab_e_component_materials' halo/face convention."""
    from rfx.runners._distributed_common import _slab_x_lo_view, _slab_model_rows
    if mask is None:
        return None
    if rank is None:
        raise ValueError("slab rank must be supplied as data")
    cell = jnp.asarray(mask, dtype=bool).astype(jnp.float32)
    edge = edge_mean_components(_slab_x_lo_view(cell, rank), cell_sizes=cell_sizes)
    real = _slab_model_rows(cell.shape[0], nx_per, nx, rank)
    return tuple(jnp.where(real, e, cell) for e in edge)


def geometric_interface_components(eps, live, fallback, widths):
    """Geometric cell-centre epsilon with PEC cells excluded by its opt-in rule.

    Both its numerator and live volume take the same dual-volume edge rule.
    NumPy retains this concrete geometric stage's float64 arithmetic; the
    caller casts its final tensor to the existing material dtype.
    """
    import numpy as np
    numerator = edge_mean_components(eps * live, cell_sizes=widths, array_module=np)
    denominator = edge_mean_components(live, cell_sizes=widths, array_module=np)
    out = []
    for num, den in zip(numerator, denominator):
        value = fallback.copy()
        np.divide(num, den, out=value, where=den > 0)
        out.append(value)
    return tuple(out)
