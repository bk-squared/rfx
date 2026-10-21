"""Setup and dispersive update for the graded time step."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax.numpy as jnp

from rfx import _realized
from rfx.boundaries.axes import resolve_cpml_axes
from rfx.core.drives import drive_layout
from rfx.core.jax_utils import is_tracer
from rfx.core.yee import FDTDState, MaterialArrays, init_state
from rfx.measurement import (
    graded_acc_indices, graded_accumulators, graded_s_ports, graded_specs,
)
from rfx.model import graded_conductor_masks, with_graded_edge_keep
from rfx.model.graded_grid import NonUniformGrid
from rfx.model.source_coefficients import drive_table


def _waveguide_port_axis_metrics(grid, cfg):
    """``(dx_h, dx_e)`` for one waveguide port: the metric each half of its
    one-sided TFSF boundary divides by.

    A waveguide port injects through two corrections half a cell apart, and
    on a graded mesh they do not sit on the same metric. Before this, both
    took the grid's BOUNDARY cell, so a port inside a fine band injected at
    the boundary cell's scale: on the WR-90 fixture of
    ``tests/unit/sparams/test_waveguide_nu_sparam.py``, 1.5 mm boundary and
    0.75 mm band, both corrections ran at 0.5000 of the local metric (gap
    G13, program repro R2). Where the port sits in a boundary-sized cell the
    two numbers ARE the boundary cell, which is why every existing NU port
    test is unaffected.

    Which metric each half is entitled to is the interface's table
    (``rfx/_grid_metric.py``), applied at the plane each correction acts on,
    not at the port's nominal index:

    * the H correction lands on the upstream half-cell, ``x_index - 1`` for
      a ``+`` port and ``x_index`` for a ``-`` one, and differences two E
      nodes across that one cell, so it takes the PRIMAL width there;
    * the E correction lands on the source plane, ``x_index`` for ``+`` and
      ``x_index + 1`` for ``-``, and sits ON a node, so it takes that node's
      DUAL spacing.

    The axis is the port's own, read from ``direction``; ``x_index`` is the
    plane index along it and the name is historical. A ``+y`` port reads the
    y cells (``test_nonuniform_api`` has one). Before this, every port took
    the x boundary scalar whatever it propagated along, which on a dz-only
    graded mesh happened to be the right number because dx and dy are equal
    there.

    On a uniform axis the two collapse to the same number and to
    ``grid.dx``, so nothing on the uniform lane moves.

    Host reads, per #1190: a concrete axis is resolved to Python floats here,
    once, outside the scan. A traced axis keeps its tracer, the way the CPML
    z pair already does, so the mesh-as-design-variable path survives.

    On a grid built by hand -- the ``NonUniformGrid`` constructor called
    directly and then handed to ``run_nonuniform``, both public -- there is
    no float64 cell spine, so ``cells`` and ``duals`` each widen the float32
    store and each warn. Measured: two ``ExactNodeSpineMissingWarning`` per
    port per call, and the metric comes back 8.692e-09 relative above the
    spine's value (0.000750000006519258 against 0.00075 on the WR-90 band).
    That is the pre-#802 float32 quantization reaching the injection
    coefficient, far below anything a user reads, and it is the existing
    degraded path rather than something this helper introduces. Build the
    grid with ``make_nonuniform_grid`` and neither happens.
    """
    axis = cfg.direction[-1].lower()
    forward = cfg.direction.startswith("+")
    h_plane = cfg.x_index - 1 if forward else cfg.x_index
    e_plane = cfg.x_index if forward else cfg.x_index + 1

    n = int((grid.nx, grid.ny, grid.nz)[("x", "y", "z").index(axis)])
    h_plane = max(0, min(int(h_plane), n - 1))
    e_plane = max(0, min(int(e_plane), n - 1))

    cells = grid.cells(axis)
    duals = grid.duals(axis)
    if is_tracer(cells):
        return cells[h_plane], duals[e_plane]
    return float(cells[h_plane]), float(duals[e_plane])


def _curl_h_nu(state, inv_dx, inv_dy, inv_dz, *, boundary=None, frame=None, owned=None):
    """ADE adapter to the shared graded E-update curl (no dtype change)."""
    from rfx.core.yee import curl_h_nu
    return curl_h_nu(state.hx, state.hy, state.hz, inv_dx, inv_dy, inv_dz,
                     boundary=boundary, frame=frame, owned=owned)


def _update_e_nu_dispersive(
    state: FDTDState,
    materials: MaterialArrays,
    dt: float,
    inv_dx: jnp.ndarray,
    inv_dy: jnp.ndarray,
    inv_dz: jnp.ndarray,
    *,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    e_old: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    boundary=None, frame=None, owned=None,
) -> tuple[FDTDState, object | None, object | None]:
    """E-field update with ADE dispersion on non-uniform grid.

    Uses per-axis inverse spacing arrays for curl(H), then applies the
    same ADE coefficient math as the uniform path. The ADE coefficients
    (ca, cb, cc, alpha, beta, etc.) are pre-baked spatial arrays that do
    not depend on dx, so they work unchanged on non-uniform grids.

    Mirrors the structure of ``_update_e_with_optional_dispersion`` in
    ``rfx/simulation.py`` but replaces the uniform ``curl / dx`` with
    non-uniform ``curl * inv_d[axis]``.

    Parameters
    ----------
    e_old : (ex_old, ey_old, ez_old) tuple of arrays, optional
        Explicit pre-step E snapshots to use as ``*_old`` in the ADE
        polarisation update.  When ``None`` (single-device default), the
        helper reads ``state.ex/ey/ez`` directly — these are the fields
        from the start of the step in the single-device runner because
        no E ghost exchange occurs before this call.

        On the distributed NU path (V3 Phase 2D), the caller MUST pass
        explicit snapshots taken BEFORE any ghost exchange so that the
        polarisation update uses pre-exchange E values.  See the ADE
        Ordering Contract documented in ``run_nonuniform_distributed``.
    """
    from rfx.materials.debye import DebyeState
    from rfx.materials.lorentz import LorentzState

    curl_x, curl_y, curl_z = _curl_h_nu(state, inv_dx, inv_dy, inv_dz, boundary=boundary, frame=frame, owned=owned)
    if e_old is not None:
        ex_old, ey_old, ez_old = e_old
    else:
        ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    # Narrow every output back to the dtype of the carry it came from
    # (issue #656) — the NU lane runs the same ADE recurrences off the same
    # ``init_debye``/``init_lorentz`` coefficients, which are built from
    # ``grid.dt`` (an np.float64 scalar, strongly typed in JAX). The full
    # rationale lives on ``rfx.materials.lorentz.update_e_lorentz``.
    _fdtype = state.ex.dtype

    # Every coefficient is per E component (#1260): component c takes index c.
    from rfx.materials.debye import debye_e_component, per_component
    from rfx.materials.lorentz import (
        lorentz_e_component, lorentz_p_component, mixed_e_component_coeffs,
    )
    e_old3 = (ex_old, ey_old, ez_old)
    curls = (curl_x, curl_y, curl_z)

    # --- Debye only ---
    if debye is not None and lorentz is None:
        debye_coeffs, debye_state = debye
        _pdtype = jnp.promote_types(debye_state.px.dtype, _fdtype)
        p_d = (debye_state.px, debye_state.py, debye_state.pz)
        out = [debye_e_component(debye_coeffs, c, e_old3[c], curls[c], p_d[c],
                                 _fdtype, _pdtype) for c in range(3)]
        (ex_new, px_new), (ey_new, py_new), (ez_new, pz_new) = out

        new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                                  step=state.step + 1)
        new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)
        return new_fdtd, new_debye, None

    # --- Lorentz only ---
    if lorentz is not None and debye is None:
        lorentz_coeffs, lor_state = lorentz
        _pdtype = jnp.promote_types(lor_state.px.dtype, _fdtype)
        p_l = (lor_state.px, lor_state.py, lor_state.pz)
        p_l_prev = (lor_state.px_prev, lor_state.py_prev, lor_state.pz_prev)

        px_new, py_new, pz_new = (
            lorentz_p_component(lorentz_coeffs, c, e_old3[c], p_l[c],
                                p_l_prev[c], _pdtype) for c in range(3))

        dpx = jnp.sum(px_new - lor_state.px, axis=0)
        dpy = jnp.sum(py_new - lor_state.py, axis=0)
        dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

        ex_new = lorentz_e_component(lorentz_coeffs, 0, ex_old, curl_x, dpx,
                                     _fdtype)
        ey_new = lorentz_e_component(lorentz_coeffs, 1, ey_old, curl_y, dpy,
                                     _fdtype)
        ez_new = lorentz_e_component(lorentz_coeffs, 2, ez_old, curl_z, dpz,
                                     _fdtype)

        new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                                  step=state.step + 1)
        new_lor = LorentzState(
            px=px_new, py=py_new, pz=pz_new,
            px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
        )
        return new_fdtd, None, new_lor

    # --- Mixed Debye + Lorentz ---
    debye_coeffs, debye_state = debye
    lorentz_coeffs, lor_state = lorentz
    _dpdtype = jnp.promote_types(debye_state.px.dtype, _fdtype)
    _lpdtype = jnp.promote_types(lor_state.px.dtype, _fdtype)

    # Explicit Lorentz polarization update first
    p_l = (lor_state.px, lor_state.py, lor_state.pz)
    p_l_prev = (lor_state.px_prev, lor_state.py_prev, lor_state.pz_prev)
    p_d = (debye_state.px, debye_state.py, debye_state.pz)
    p_l_new = tuple(
        lorentz_p_component(lorentz_coeffs, c, e_old3[c], p_l[c], p_l_prev[c],
                            _lpdtype) for c in range(3))
    px_l_new, py_l_new, pz_l_new = p_l_new

    e_new, p_d_new = [], []
    for c in range(3):
        dp_l = jnp.sum(p_l_new[c] - p_l[c], axis=0)
        ca, cb, cc_debye, cc_lorentz = mixed_e_component_coeffs(
            debye_coeffs, lorentz_coeffs, c, dt)
        e_c = (ca * e_old3[c] + cb * curls[c]
               + jnp.sum(cc_debye * p_d[c], axis=0)
               - cc_lorentz * dp_l).astype(_fdtype)
        e_new.append(e_c)
        beta_c = per_component(debye_coeffs.beta, "beta")[c]
        p_d_new.append((debye_coeffs.alpha * p_d[c]
                        + beta_c * (e_c[None] + e_old3[c][None])
                        ).astype(_dpdtype))
    ex_new, ey_new, ez_new = e_new

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                              step=state.step + 1)
    new_debye = DebyeState(px=p_d_new[0], py=p_d_new[1], pz=p_d_new[2])
    new_lor = LorentzState(
        px=px_l_new, py=py_l_new, pz=pz_l_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )
    return new_fdtd, new_debye, new_lor


class _NUScanSetup(NamedTuple):
    """Host-side bundle built by :func:`_build_nu_scan` (#383 code motion).

    Carries the scan step function, its initial carry, the stacked source
    table, and the post-scan assembly metadata shared by
    :func:`run_nonuniform` (single ``lax.scan``) and
    :func:`run_nonuniform_until_decay` (chunked host loop). This tuple
    never crosses a JAX transform boundary — ``step_fn`` is a Python
    closure and the ``use_*`` flags are Python bools.

    ``step_fn_inv(carry, xs, invariants)`` is the same step taking the
    per-cell arrays it reads (materials, PEC edge masks and occupancy, the
    CPML permittivity, dispersion, anisotropic, sheet and design-box
    coefficients) as ``invariants``; ``step_fn`` is it with ``invariants``
    bound. A jitted loop passes them as an argument (see
    :func:`rfx.core.jax_utils.split_loop_invariants`) instead of compiling
    them into the program as grid-sized constants.
    """
    step_fn: object
    step_fn_inv: object
    invariants: dict
    carry_init: dict
    src_waveforms: jnp.ndarray
    dt: object
    sources: list
    probes: list
    wire_ports: list
    dft_planes: list
    flux_monitors: list
    waveguide_meta: tuple
    wp_meta: list
    sp_freqs: object
    use_wire_ports: bool
    use_dft_planes: bool
    use_flux_monitors: bool
    use_lumped_rlc: bool
    use_ntff: bool
    use_waveguide_ports: bool
    use_current_moments: bool = False
    current_moments: object = None


def _build_nu_scan(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    cpml_axes: str | None = None,
    pec_mask=None,
    pec_sheets=(),
    pec_wires=(),
    pec_edge_masks=None,
    pec_occupancy=None,
    pinned_sheet_layout=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    dft_planes: list | None = None,
    rlc_metas: tuple = (),
    rlc_states: tuple = (),
    ntff_box=None,
    ntff_data=None,
    current_moments=None,
    waveguide_ports: list | None = None,
    tfsf: tuple | None = None,
    _feature_tfsf: bool = False,
    flux_monitors: list | None = None,
    emit_time_series: bool = True,
    aniso_eps: tuple | None = None,
    sheet_impedance=None,
    design_box=None,
) -> _NUScanSetup:
    """Build the NU scan carry + step function (pure code motion, #383).

    Extracted verbatim from :func:`run_nonuniform` so the until-decay
    entry point can drive the SAME ``step_fn`` / carry through a chunked
    host loop. ``n_steps`` is used only to size the zero-source
    placeholder table; the source waveform arrays themselves define the
    table length when sources are present.
    """
    cpml_axes = resolve_cpml_axes(grid, cpml_axes)
    from rfx.model.materials import kernel_materials, with_components
    from rfx.geometry.node_tensor import SmoothedEps
    from rfx.model.tensor_operands import TensorOperands
    tensor = TensorOperands.of(aniso_eps) if isinstance(aniso_eps, SmoothedEps) else None
    materials = with_components(materials, grid, periodic=tuple(a in grid.periodic_axes for a in "xyz"),
                                tensor=tensor)
    if tensor is not None:
        aniso_eps = tensor.eps
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(
        materials, lane="non-uniform Yee with dispersion/tensor or design-box updates",
        unsupported=(debye is not None or lorentz is not None
                     or aniso_eps is not None or design_box is not None))
    sources = sources or []
    if _realized.ACTIVE is not None:
        _realized.sources(grid, materials, sources, "graded.sources")
    probes = probes or []
    wire_ports = wire_ports or []
    dft_planes = dft_planes or []
    waveguide_ports = waveguide_ports or []
    flux_monitors = flux_monitors or []
    dt = grid.dt
    from rfx.model.pole_stability import judge_prebuilt_poles
    from rfx.materials.lorentz import pole_sets_judged
    if lorentz is not None and not pole_sets_judged(lorentz[0]):
        judge_prebuilt_poles(grid, lorentz, debye, periodic=tuple(a in grid.periodic_axes for a in "xyz"))
    use_wire_ports = len(wire_ports) > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_dft_planes = len(dft_planes) > 0
    use_flux_monitors = len(flux_monitors) > 0
    use_lumped_rlc = len(rlc_metas) > 0
    use_ntff = ntff_box is not None and ntff_data is not None
    use_current_moments = current_moments is not None
    use_waveguide_ports = len(waveguide_ports) > 0
    use_tfsf = tfsf is not None

    # The far-field integral is over the SCATTERED field, which it only is
    # when the Huygens box encloses the whole injected region. Same check the
    # uniform lane runs (rfx/simulation.py), same helper.
    from rfx.boundaries.tfsf import admit_setup

    from rfx.boundaries.setup import nonuniform_boundaries
    from rfx.stepping.absorber import apply_cpml_h, apply_cpml_e_step as apply_cpml_e, graded_cpml_operands
    (use_cpml, cpml_params, cpml_state_init, cpml_grid, cpml_axes_eff,
     curl_boundary, _pec_faces_frozen, _pmc_faces_frozen, use_pmc_faces) = nonuniform_boundaries(
         grid, cpml_axes, pec_faces, pmc_faces)

    pec_edge_masks, pec_static_edge_masks = graded_conductor_masks(
        pec_mask, pec_occupancy, pec_sheets, pec_wires, pec_edge_masks,
        periodic=tuple(a in grid.periodic_axes for a in "xyz"))
    from rfx.model.node_pec import restrict_node_pec
    materials = restrict_node_pec(materials, pec_edge_masks, _pec_faces_frozen)
    use_pec_edges = pec_edge_masks is not None
    use_pec_occupancy = pec_occupancy is not None or bool(pinned_sheet_layout)

    admit_setup(grid=grid, tfsf=tfsf, materials=materials, waveguide_ports=waveguide_ports,
                ntff=ntff_box if use_ntff else None, periodic=tuple(a in grid.periodic_axes for a in "xyz"),
                pec_faces=pec_faces, pmc_faces=pmc_faces, feature_owned=_feature_tfsf, nonuniform=True,
                updates=dict(debye=debye, lorentz=lorentz, aniso_eps=aniso_eps, pec_mask=pec_mask,
                             pec_edge_masks=pec_edge_masks, pec_occupancy=pec_occupancy,
                             sheet_impedance=sheet_impedance, design_box=design_box),
                localized=(("sources", sources), ("waveguide_ports", waveguide_ports),
                           ("wire_ports", wire_ports), ("rlc_metas", rlc_metas)))

    # #677 surface-impedance sheet: exponential-stepping A/B built once
    # from the FINAL scan materials; applied per step at tangential edges
    # in step_fn (after apply_pec_mask, before sources/DFT sampling).
    use_sheet_impedance = sheet_impedance is not None
    if use_sheet_impedance:
        from rfx.materials.thin_conductor import sheet_update_coeffs
        sheet_coeffs = sheet_update_coeffs(
            sheet_impedance.sigma_sheet, materials, dt)

    state = init_state((grid.nx, grid.ny, grid.nz))
    src_meta = [(s[0], s[1], s[2], s[3]) for s in sources]
    materials = with_graded_edge_keep(materials, pec_occupancy, pinned_sheet_layout, grid=grid, sources=sources, waveguide_ports=waveguide_ports, dtype=state.ex.dtype, sheet_edge_masks=pec_static_edge_masks, periodic=tuple(a in grid.periodic_axes for a in "xyz"), cells=pec_mask, sheets=pec_sheets, wires=pec_wires)
    src_waveforms = drive_table([jnp.array(s[4]) for s in sources], src_meta, n_steps,
        occupancy=pec_occupancy, sheet_edge_masks=pec_static_edge_masks, keep=materials.edge_keep, periodic=tuple(a in grid.periodic_axes for a in "xyz"))
    prb_meta = [(p[0], p[1], p[2], p[3]) for p in probes]


    inv_dx_h = grid.inv_dx_h
    inv_dy_h = grid.inv_dy_h
    inv_dz_h = grid.inv_dz_h
    inv_dx = grid.inv_dx
    inv_dy = grid.inv_dy
    inv_dz = grid.inv_dz

    # ---- #1183 design box on the graded mesh: fence, then build Ca/Cb ----
    # The SAME resolver the uniform lane runs (rfx.simulation), because the
    # question it answers is the same one: does anything else in this step
    # read ``materials`` at a design cell. What differs on this lane is only
    # the curl, and ``update_e_box`` takes it as ``inv_d``. The flags this
    # lane cannot produce are passed as absent: no UPML runner, no Kerr, no
    # fourth-order stencil, no Bloch phase, no Stage-2 inverse-eps tensor.
    design_box_coeffs = None
    if design_box is not None:
        from rfx.model import resolve_design_box
        design_box_coeffs = resolve_design_box(
            design_box,
            grid=grid,
            materials=materials,
            dt=dt,
            use_cpml=use_cpml,
            use_upml=False,
            cpml_axes=cpml_axes_eff,
            use_debye=use_debye,
            use_lorentz=use_lorentz,
            use_kerr=False,
            aniso_eps=aniso_eps,
            aniso_inv_eps=None,
            stencil_order=2,
            bloch=None, periodic=tuple(a in grid.periodic_axes for a in "xyz"),
            sheet_impedance=sheet_impedance,
            cell_metas=(
                ("source", [(s[0], s[1], s[2], s[3]) for s in sources]),
                ("wire port", [
                    (int(wp["mid_i"]), int(wp["mid_j"]), int(wp["mid_k"]),
                     wp["component"])
                    for wp in wire_ports
                ]),
                ("lumped RLC element", [(m.i, m.j, m.k) for m in rlc_metas]),
            ),
        )
    use_design_box = design_box_coeffs is not None

    carry_init = {"fdtd": state}
    if use_cpml:
        carry_init["cpml"] = cpml_state_init

    # Debye/Lorentz ADE state
    if use_debye:
        debye_coeffs, debye_state = debye
        carry_init["debye"] = debye_state

    if use_lorentz:
        lorentz_coeffs, lorentz_state = lorentz
        carry_init["lorentz"] = lorentz_state

    sp_freqs, wp_meta, use_wire_ports = graded_accumulators(wire_ports, grid, s_param_freqs, carry_init)

    # DFT plane probe carry + static metadata
    if use_dft_planes:
        carry_init["dft_planes"] = tuple(probe.accumulator for probe in dft_planes)
        from rfx.measurement.accumulators import plane_metadata
        dft_meta = plane_metadata(dft_planes)
    else:
        dft_meta = ()

    # Flux monitor carry + static metadata (mirrors the uniform scan body
    # in rfx/simulation.py). The NU dt is scalar, so the DFT kernels need
    # no per-axis time weighting; the axis-aware area element dA already
    # lives on each FluxMonitor (handles graded tangential cells).
    if use_flux_monitors:
        from rfx.measurement.accumulators import flux_metadata
        flux_meta = flux_metadata(flux_monitors, window=True)
        carry_init["flux_monitors"] = tuple(
            (fm.e1_dft, fm.e2_dft, fm.h1_dft, fm.h2_dft)
            for fm in flux_monitors
        )
    else:
        flux_meta = ()

    # Lumped RLC ADE state (one per element) — metas are Python-static
    if use_lumped_rlc:
        carry_init["rlc_states"] = tuple(rlc_states)

    # NTFF accumulators — seeded from caller, updated per step via
    # accumulate_ntff. Box indices and freqs are Python-static.
    if use_ntff:
        carry_init["ntff"] = ntff_data
    if use_current_moments:
        from rfx.current_moments import (
            accumulate_current_moments as _accumulate_cm,
            init_current_moment_data as _init_cm,
            slab_e_snapshot as _slab_e_snapshot,
        )
        # Same dtype policy as the NTFF carry (#646).
        carry_init["current_moments"] = _init_cm(
            current_moments, field_dtype=carry_init["fdtd"].ex.dtype)

    # Waveguide-port time-series carry (mirrors uniform path).
    # Phase 2 cleanup (2026-04-25) removed in-scan DFT accumulators;
    # spectra are computed POST-SCAN by a rect full-record DFT on the
    # recorded modal V/I time series.
    if use_waveguide_ports:
        carry_init["waveguide_port_accs"] = tuple(
            (
                cfg.v_probe_t,
                cfg.v_ref_t,
                cfg.i_probe_t,
                cfg.i_ref_t,
                cfg.v_inc_t,
                cfg.n_steps_recorded,
            )
            for cfg in waveguide_ports
        )
        waveguide_meta = tuple(waveguide_ports)
        waveguide_port_metrics = tuple(
            _waveguide_port_axis_metrics(grid, cfg) for cfg in waveguide_ports
        )
    else:
        waveguide_meta = ()
        waveguide_port_metrics = ()

    # TFSF 1D auxiliary state carry. Injection axis is x (uniform on
    # NU paths we support — dz-only nonuniformity), so the 1D aux runs
    # with grid.dx spacing and the E/H corrections use scalar
    # coeff = dt / (EPS_0 * dx) etc. Oblique / +z,-z cases are
    # rejected upstream (see rfx/runners/nonuniform.py and rfx/api.py).
    if use_tfsf:
        from rfx.sources.tfsf import is_tfsf_2d as _is_tfsf_2d
        tfsf_cfg, tfsf_state = tfsf
        if _is_tfsf_2d(tfsf_cfg):
            raise ValueError(
                "TFSF oblique incidence (2D auxiliary grid) is not yet "
                "supported on nonuniform z mesh. Use angle_deg=0 along x."
            )
        if tfsf_cfg.direction not in ("+x", "-x"):
            raise ValueError(
                "TFSF on nonuniform mesh supports only direction='+x' or "
                f"'-x' (injection along uniform x axis); got {tfsf_cfg.direction!r}."
            )
        carry_init["tfsf"] = tfsf_state

    _cpml_inv_eps_r, _cpml_loss = graded_cpml_operands(
        materials, aniso_eps, use_cpml=use_cpml,
        use_debye=use_debye, use_lorentz=use_lorentz)

    # The per-cell arrays and x spacing vectors reach the step through
    # ``invariants`` so a jitted loop can pass arrays as an argument.
    # Spacing vectors are not grid-sized and stay compile-time constants
    # on one device; CPML profiles stay in the closure (_NUScanSetup).
    from rfx.stepping.graded import GradedStepContext, make_graded_step, graded_probe_table
    invariants = {
        "inv_dx": inv_dx,
        "inv_dx_h": inv_dx_h,
        "dx_arr": grid.dx_arr,
        "drives": drive_layout(src_meta, src_waveforms.dtype),
        "probe_table": graded_probe_table(prb_meta, grid.shape, dtype=state.ex.dtype),
        "materials": kernel_materials(materials, electric=not (use_debye or use_lorentz), epsilon=aniso_eps is None),
        "pec_edge_masks": pec_edge_masks,
        "pec_occupancy": pec_occupancy,
        "pec_static_edge_masks": pec_static_edge_masks,
        "cpml_inv_eps_r": _cpml_inv_eps_r,
        "e_loss": _cpml_loss,
        "debye_coeffs": debye_coeffs if use_debye else None,
        "lorentz_coeffs": lorentz_coeffs if use_lorentz else None,
        "aniso_eps": aniso_eps,
        "sheet_impedance": sheet_impedance,
        "sheet_coeffs": sheet_coeffs if use_sheet_impedance else None,
        "design_box_coeffs": design_box_coeffs,
    }

    ctx = GradedStepContext(
        use_lumped_rlc=use_lumped_rlc,
        rlc_metas=rlc_metas,
        use_current_moments=use_current_moments,
        _slab_e_snapshot=_slab_e_snapshot if use_current_moments else None,
        current_moments=current_moments,
        dt=dt,
        inv_dy_h=inv_dy_h,
        inv_dz_h=inv_dz_h,
        grid=grid,
        use_tfsf=use_tfsf,
        tfsf_cfg=tfsf_cfg if use_tfsf else None,
        use_waveguide_ports=use_waveguide_ports,
        waveguide_meta=waveguide_meta,
        waveguide_port_metrics=waveguide_port_metrics,
        use_cpml=use_cpml,
        apply_cpml_h=apply_cpml_h,
        cpml_params=cpml_params,
        cpml_grid=cpml_grid,
        cpml_axes_eff=cpml_axes_eff,
        use_pmc_faces=use_pmc_faces,
        _pmc_faces_frozen=_pmc_faces_frozen,
        use_sheet_impedance=use_sheet_impedance,
        use_design_box=use_design_box,
        use_debye=use_debye,
        use_lorentz=use_lorentz,
        inv_dy=inv_dy,
        inv_dz=inv_dz,
        curl_boundary=curl_boundary,
        apply_cpml_e=apply_cpml_e,
        _pec_faces_frozen=_pec_faces_frozen,
        use_pec_edges=use_pec_edges,
        use_pec_occupancy=use_pec_occupancy,
        use_wire_ports=use_wire_ports,
        wp_meta=wp_meta,
        port_record_specs=graded_specs(wire_ports),
        port_acc_indices=graded_acc_indices(wire_ports),
        sp_freqs=sp_freqs,
        use_dft_planes=use_dft_planes,
        dft_meta=dft_meta,
        use_flux_monitors=use_flux_monitors,
        flux_meta=flux_meta,
        use_ntff=use_ntff,
        ntff_box=ntff_box,
        _accumulate_cm=_accumulate_cm if use_current_moments else None,
        emit_time_series=emit_time_series,
    )
    step = make_graded_step(ctx)

    return _NUScanSetup(
        step_fn=partial(step, invariants=invariants),
        step_fn_inv=step,
        invariants=invariants,
        carry_init=carry_init,
        src_waveforms=src_waveforms,
        dt=dt,
        sources=sources,
        probes=probes,
        wire_ports=graded_s_ports(wire_ports),
        dft_planes=dft_planes,
        flux_monitors=flux_monitors,
        waveguide_meta=waveguide_meta,
        wp_meta=wp_meta,
        sp_freqs=sp_freqs,
        use_wire_ports=use_wire_ports,
        use_dft_planes=use_dft_planes,
        use_flux_monitors=use_flux_monitors,
        use_lumped_rlc=use_lumped_rlc,
        use_ntff=use_ntff,
        use_waveguide_ports=use_waveguide_ports,
        use_current_moments=use_current_moments,
        current_moments=current_moments,
    )
