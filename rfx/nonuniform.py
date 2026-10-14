"""Non-uniform Yee grid FDTD runner.

Supports spatially-varying dx, dy, dz profiles. The most common use
is dz-graded meshes for thin-substrate structures where z-resolution
must be fine near the substrate but coarse in the air region; dx/dy
graded meshes additionally allow fine cells near metal edges
(fringing-field physics in patch antennas, microstrip filters, etc.)
without paying the cost of a uniform fine mesh everywhere.

Back-compat: `make_nonuniform_grid(domain_xy, dz_profile, dx, ...)`
with a scalar `dx` still produces a uniform-xy grid; only `dz` is
graded. Pass `dx_profile=` / `dy_profile=` to enable per-cell x/y
spacing.

Uses update_h_nu / update_e_nu from core/yee.py with pre-computed
per-axis inverse spacing arrays. Fully JIT-compiled via jax.lax.scan.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from typing import NamedTuple

from rfx import _realized
from rfx.boundaries.axes import padded_axes, resolve_cpml_axes
from rfx.boundaries.depths import resolve_face_depths

import jax
import jax.numpy as jnp

from rfx.core.drives import drive_layout
from rfx.model.source_coefficients import drive_table
from rfx.model import with_edge_keep
import numpy as np

from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    EPS_0, MU_0,
)
from rfx.boundaries.pec import (
    realized_pec_edge_masks,
)
from rfx.core.jax_utils import is_tracer, recorded_scan
from rfx.stepping import segmented_scan
from rfx._grid_metric import (
    DECLARED_SPAN_TOL_M,
    axis_name as _axis_name,
    dual_spacings_from_cells as _dual_spacings_from_cells,
    nearest_node_index,
    normalize_axis as _normalize_axis,
)

from rfx.model.graded_grid import (  # noqa: F401 -- moved; re-exported
    C0,
    NonUniformGrid,
    _pad_profile,
    _append_bounding_node,
    interior_cells,
    assert_cells_span_declared_profile,
    node_positions_from_profile,
    _profile_to_inv_arrays,
    e_node_dual_spacings,
    e_node_dual_spacing_at,
    _pinned_dt,
    _assert_declared_floor_at_runtime,
    make_nonuniform_grid,
    _assert_traced_index_matches_realized,
    z_position_to_index,
    _axis_position_to_index,
    _nearest_on_line,
    position_to_index,
)
from rfx.model.graded_profile import (  # noqa: F401 -- moved; re-exported
    _BandSeg,
    _Anchor,
    _RATIO_FLOOR,
    _MAX_PROFILE_CELLS,
    _STEP_TOL,
    _EDGE_TOL,
    _ramp_steps,
    _ramp_spec,
    _ramp_sum,
    _ramp_cells,
    _band_realize_free,
    _band_realize_protected,
    _build_band_profile,
    make_band_profile,
    make_z_profile,
)


def port_metric_axes(grid):
    """The three cell-size arrays a port sizes itself on, each with a flag
    saying whether it has a host value.

    Returns ``((dx, x_host), (dy, y_host), (dz, z_host))``.

    A CONCRETE x/y axis comes back as the float64 host copy — the same
    numbers :func:`_build_wp_meta` reads — so the termination conductance,
    the Ampere-loop weights and the gap-voltage line integral are sized on
    one set of numbers rather than three roundings of it. z is never
    converted: ``grid.dz`` is already the float32 array both sites index,
    and a float64 round trip would move the last bits of every shipped
    non-uniform port.

    A TRACED axis comes back untouched, so the cell sizes at the port's own
    cells keep their gradient. That is the whole point: when the mesh is the
    design variable, the port's cells change size with the deformation, and
    a port sized on a host copy of the nominal mesh drops that term out of
    the derivative while the physics keeps it.
    """
    dx_arr = grid.dx_arr
    dy_arr = grid.dy_arr
    x_traced, y_traced = is_tracer(dx_arr), is_tracer(dy_arr)
    return (
        (dx_arr if x_traced else np.asarray(dx_arr, dtype=np.float64),
         not x_traced),
        (dy_arr if y_traced else np.asarray(dy_arr, dtype=np.float64),
         not y_traced),
        (grid.dz, not is_tracer(grid.dz)),
    )


def port_metric(value, host):
    """``float(value)`` on an axis with a host value, the tracer itself
    otherwise. Keeps the concrete spelling byte-for-byte what it was."""
    return float(value) if host else value


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


def make_current_source(grid: NonUniformGrid, position_ijk, component,
                        waveform_fn, n_steps, materials, amplitude_kind=None):
    """Create a properly normalized current source for non-uniform grid.

    The waveform specifies CURRENT (Amperes). The E-field addition is:
    E += (dt/ε) × I_source / dV
    where dV is the E node's control volume: the PRIMAL per-cell width on
    the component's own axis and the DUAL spacing ``(d[k-1]+d[k])/2`` on the
    two transverse axes (issue #672). On a uniform mesh the two coincide and
    this is the familiar ``dx × dy × dz``.

    This gives resolution-independent injected POWER regardless of cell size.
    Same approach as Meep's internal source normalization.

    Native amplitude convention (issue #571): ``'current'`` —
    ``amplitude_kind`` of None or ``'current'`` is bit-identical to the
    historical output (no extra multiply). ``amplitude_kind='field'``
    returns the waveform directly, without reading a material coefficient.
    The shared builder preserves the current drive's differentiable Cb/dV.
    """
    i, j, k = position_ijk

    # Read the edge's own eps and sigma, averaged over its four cells
    # (#1210), with the same differentiable drive formula on every trace.
    from rfx.model.materials import e_update_coefficient_at
    if str(component).lower() not in ("ex", "ey", "ez"):
        raise ValueError(
            f"unknown component {component!r}: a current source injects an "
            f"electric field component, one of 'ex', 'ey', 'ez'.")
    from rfx.model.materials import electric_materials_traced
    materials_traced = electric_materials_traced(materials)
    from rfx.model.source_coefficients import source_table
    dV, grid_traced = current_source_volume(grid, (i, j, k), component)
    any_traced = materials_traced or grid_traced
    waveform = source_table(
        current_source_samples(grid, waveform_fn, n_steps), amplitude_kind,
        native="cb_over_dv", volume=dV,
        coefficient=lambda: e_update_coefficient_at(
            materials, (i, j, k), component, grid.dt, host=True, grid=grid))

    waveform_out = waveform if any_traced or is_tracer(waveform) else np.array(waveform)
    return (i, j, k, component, waveform_out)


def current_source_cb(eps_r_c, sigma_c, dt):
    """``Cb = (dt/eps) / (1 + sigma*dt/(2*eps))`` of the edge a current
    source feeds, from that edge's ``(eps_r, sigma)``.

    ``eps_r_c`` / ``sigma_c`` are the edge's own values -- the mean over the
    four cells incident to it, :func:`rfx.core.yee.cell_component_e_materials`
    (#1210). Concrete and traced inputs use the same jnp arithmetic,
    keeping a traced permittivity on the tape (GEO-C3).
    The one spelling of the drive coefficient: :func:`make_current_source`
    and the distributed graded forward, which builds it inside its jitted
    program from the slabs its E update receives (#1279), both call it.
    """
    # In eps_r units the reverse pass avoids squaring eps ~ 1e-11,
    # which overflowed float32 cotangents into NaN gradients (#1279).
    eps_r = jnp.asarray(eps_r_c)
    sigma = jnp.asarray(sigma_c)
    return (dt / EPS_0) / (eps_r + sigma * (dt / (2.0 * EPS_0)))


def current_source_samples(grid, waveform_fn, n_steps):
    """The current ``I(t)`` in amperes at the ``n_steps`` step times
    ``t = n*dt`` (float32), before any ``Cb/dV`` normalisation."""
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    return jax.vmap(waveform_fn)(times)


def current_source_volume(grid, position_ijk, component):
    """``(dV, grid_traced)``: the control volume of the E node a current
    source at ``position_ijk`` injects into, and whether the profile is
    traced (then ``dV`` is a jnp scalar, else a Python float)."""
    i, j, k = position_ijk

    # Control volume of the E node this source injects into (issue #672).
    # An E_a component is an EDGE along its own axis a and sits ON a node on
    # the two transverse axes, so its control volume is MIXED:
    #   axis == a  -> primal per-cell width d[idx]
    #   axis != a  -> dual spacing (d[idx-1]+d[idx])/2, which is exactly the
    #                 metric the NU E update divides curl H by (CORE-C2,
    #                 ``_profile_to_inv_arrays``); the Ex update, for
    #                 instance, uses inv_dy and inv_dz and never inv_dx.
    # Using the primal width on all three axes (pre-#672) is right only where
    # both neighbour pairs are equal. On a 2:1 grading step it mis-normalizes
    # the injected current moment by the local cell ratio on each graded
    # TRANSVERSE axis (measured: dV too large by 1.333 with one graded
    # transverse axis, 1.778 with two).
    # Stay in jnp when the profile is traced so a mesh-as-design variable
    # propagates the gradient into the waveform normalisation.
    _axis_of = {"ex": 0, "ey": 1, "ez": 2}
    if component not in _axis_of:
        raise ValueError(
            f"make_current_source: unknown component {component!r} "
            f"(expected one of {sorted(_axis_of)}). The control volume is "
            f"component-dependent on a non-uniform mesh, so there is no safe "
            f"fallback (issue #672)."
        )
    p_axis = _axis_of[component]
    grid_traced = (
        is_tracer(grid.dx_arr) or is_tracer(grid.dy_arr) or is_tracer(grid.dz)
    )
    _profiles = (grid.dx_arr, grid.dy_arr, grid.dz)
    _idx = (i, j, k)
    _widths = []
    for _a in range(3):
        if _a == p_axis:
            _w = (jnp.asarray(_profiles[_a])[_idx[_a]] if grid_traced
                  else float(np.asarray(_profiles[_a])[_idx[_a]]))
        elif grid_traced:
            _w = e_node_dual_spacing_at(jnp.asarray(_profiles[_a]), _idx[_a])
        else:
            _w = float(e_node_dual_spacing_at(
                np.asarray(_profiles[_a], dtype=np.float64), _idx[_a]))
        _widths.append(_w)
    dx_local, dy_local, dz_local = _widths
    return dx_local * dy_local * dz_local, grid_traced


def _bwd_neighbor(h, idx, axis, boundary=None):
    """``h`` one cell back along ``axis``, using the SAME out-of-domain
    convention the NU E-update uses.

    ``rfx.core.yee._shift_bwd`` (rfx/core/yee.py) pads explicitly with zero,
    so ``_curl_h_nu`` reads 0 for H outside the domain. Before #689 this
    helper spelled the backward read as a raw ``h[i - 1]``, which at
    ``i == 0`` is Python's negative index — H at the OPPOSITE face of the
    domain. That made a 4-term local Ampere loop depend on a field cell it
    does not enclose, and it disagreed with the very curl the port current
    is supposed to be the loop integral of.

    A length-1 axis carries no variation along itself, so the difference
    along it is 0 (return the cell itself, matching the uniform lane's
    2-D behaviour). NU grids are 3-D today, so that branch is defensive.
    """
    from rfx.core.yee import h_neighbor
    return h_neighbor(h, axis, index=idx, boundary=boundary)


def _build_wp_meta(wire_ports, grid):
    """Static per-port metadata for the wire-port DFT scan body.

    Pre-computes the port cell's metrics OUTSIDE the scan (repo engineering
    principle 4). Two metric families, and they are not interchangeable
    (issue #672):
      PRIMAL  d[idx]                 -> V = -E_a * d_parallel, the length of
                                        the E_a edge itself.
      DUAL    (d[idx-1]+d[idx])/2    -> the Ampere-loop legs, each weighted
                                        by the dual spacing on that H
                                        component's own axis.
    A CONCRETE x/y axis stays float() as before; a TRACED one keeps its
    tracer, so the V and I the scan records are weighted by the port cells'
    own sizes on the mesh being solved rather than by a host copy of the
    nominal one. The z metrics were already indexed without np.asarray for
    the same reason. Which axis is which comes from
    :func:`port_metric_axes`, the same decision the assembly makes for the
    termination conductance — the two must agree or V/I does not close.
    `excite` and `direction` are carried for the post-processing;
    `direction` no longer enters the wave split (issue #673).

    Slot order (guarded by
    tests/unit/nonuniform/test_nonuniform_source_port_dual_spacing.py):
      0..2  mid_i, mid_j, mid_k     (midpoint of the LIVE run, issue #764)
      3     component
      4     impedance (whole-port Z0)
      5, 6  primal dx[mid_i], dy[mid_j]
      7     excite
      8     direction
      9,10  dual x/y spacings at mid
      11    primal dz[mid_k]        (traced-safe)
      12    dual z spacing at mid
      13    live_cells              (static tuple of (i, j, k), issue #764)
      14    d_par per live cell     (primal width on the component's own
                                     axis at each live cell, same order as
                                     slot 13; z entries traced-safe)
    Slots 13/14 feed the whole-port gap voltage
    V_port = sum_live(-E_c * d_par,c) — the #672 PRIMAL family, per cell.
    """
    (_dx_arr_np, _x_host), (_dy_arr_np, _y_host), (_dz_m, _z_host) = \
        port_metric_axes(grid)

    def _d_par(cell, comp):
        ci, cj, ck = cell
        if comp == "ex":
            return port_metric(_dx_arr_np[ci], _x_host)
        if comp == "ey":
            return port_metric(_dy_arr_np[cj], _y_host)
        # ez: index grid.dz WITHOUT np.asarray so a traced dz_profile
        # keeps its gradient through the extractor.
        return _dz_m[ck]

    meta = []
    for wp in wire_ports:
        live_cells = tuple(
            tuple(int(x) for x in c)
            for c in wp.get('live_cells',
                            ((wp['mid_i'], wp['mid_j'], wp['mid_k']),))
        )
        comp = wp['component']
        meta.append((
            wp['mid_i'], wp['mid_j'], wp['mid_k'],
            comp, wp['impedance'],
            port_metric(_dx_arr_np[wp['mid_i']], _x_host),
            port_metric(_dy_arr_np[wp['mid_j']], _y_host),
            bool(wp.get('excite', True)),
            str(wp.get('direction', '-x')),
            port_metric(e_node_dual_spacing_at(_dx_arr_np, wp['mid_i']),
                        _x_host),
            port_metric(e_node_dual_spacing_at(_dy_arr_np, wp['mid_j']),
                        _y_host),
            _dz_m[wp['mid_k']],
            e_node_dual_spacing_at(_dz_m, wp['mid_k']),
            live_cells,
            tuple(_d_par(c, comp) for c in live_cells),
        ))
    return meta


def wire_port_current(hx, hy, hz, comp, mi, mj, mk,
                      dual_x, dual_y, dual_z, boundary=None):
    """Enclosed current from the discrete Ampere loop at a wire-port cell.

    For component ``a`` with ``(a, b, c)`` cyclic,
    ``curl_a = dH_c/db - dH_b/dc`` and the pierced DUAL face has area
    ``dual_b * dual_c``, so

        I_a = (H_c[b] - H_c[b-1]) * dual_c - (H_b[c] - H_b[c-1]) * dual_b

    i.e. each H-difference leg is weighted by the DUAL spacing along THAT H
    COMPONENT'S OWN axis (the leg's own length as a contour segment) — never
    by the axis it is differenced along, and never by a primal cell width.

    Before issue #672 every branch got both of those wrong at once: one leg
    carried the primal width of the right axis, the other carried a width
    from an unrelated axis. Both errors are invisible on a uniform cubic
    mesh, which is why the uniform lane (``rfx/simulation.py``) can spell the
    whole thing with a single ``dx`` and still be correct there.

    Signs match the pre-#672 code exactly; only the metrics changed.

    Out-of-domain H is ZERO (#689). Each backward read goes through
    :func:`_bwd_neighbor`, which pads the way ``_curl_h_nu`` /
    ``rfx.core.yee._shift_bwd`` pad, so the port current is the loop
    integral of the same curl the E-update integrates. The previous raw
    ``[mi - 1]`` spelling was a NEGATIVE index at ``mi == 0``, i.e. H at
    the opposite domain face, which a 4-term LOCAL loop must never touch.

    Scope, stated because #689's commit body overstated it: this is a
    STENCIL correctness fix, not a repair of a measured wrong number. The
    cell it used to read, the last index of the axis, holds 0.0 at runtime
    anyway — ``inv_d_h[N-1] = 0`` by construction
    (``_profile_to_inv_arrays``) and the last row is grid pad. Measured on
    a 21-cell NU run with an ez wire port flush against x_lo (``mi == 0``),
    sampling every H component's last plane on every axis at every one of
    300 steps: max|H| there was 0.000000e+00 throughout, under
    ``boundary="pec"`` and under ``boundary="cpml"``, against global
    max|H| of 1.9e+07 and 1.8e+04. So the two spellings agree on any state
    the stepper can produce, and the +1000 perturbation the #689 oracles
    use is not reachable by the solver. Keep the fix — a local loop should
    not read a cell it does not enclose, and the two spellings agreeing
    only by a coincidence of the pad arrays is not a property worth
    depending on — but do not describe it as a live defect.
    """
    idx = (mi, mj, mk)
    if comp == "ez":
        return ((hy[idx] - _bwd_neighbor(hy, idx, 0, boundary)) * dual_y
                - (hx[idx] - _bwd_neighbor(hx, idx, 1, boundary)) * dual_x)
    if comp == "ex":
        return ((hz[idx] - _bwd_neighbor(hz, idx, 1, boundary)) * dual_z
                - (hy[idx] - _bwd_neighbor(hy, idx, 2, boundary)) * dual_y)
    if comp == "ey":
        return ((hx[idx] - _bwd_neighbor(hx, idx, 2, boundary)) * dual_x
                - (hz[idx] - _bwd_neighbor(hz, idx, 0, boundary)) * dual_z)
    raise ValueError(f"wire_port_current: unknown component {comp!r}")


def _curl_h_nu(state, inv_dx, inv_dy, inv_dz, *, boundary=None):
    """ADE adapter to the shared graded E-update curl (no dtype change)."""
    from rfx.core.yee import curl_h_nu
    return curl_h_nu(state.hx, state.hy, state.hz, inv_dx, inv_dy, inv_dz,
                     boundary=boundary)


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
    boundary=None,
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

    curl_x, curl_y, curl_z = _curl_h_nu(state, inv_dx, inv_dy, inv_dz, boundary=boundary)
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
    materials = with_components(materials, grid, periodic=(False, False, False))
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

    # #931 §1.7: realize the PEC edges ONCE here.  The NU stepper installs
    # no periodic BC at all and NU grids are 3-D, so the non-periodic #689
    # convention is the right one; a lane that pre-realized (and
    # port-cleared) its masks passes them in instead.
    pec_sheets = tuple(pec_sheets or ())
    pec_wires = tuple(pec_wires or ())
    if pec_edge_masks is None and (
            pec_mask is not None or pec_sheets or pec_wires):
        pec_edge_masks = realized_pec_edge_masks(
            pec_mask, sheets=pec_sheets, wires=pec_wires)
    use_pec_edges = pec_edge_masks is not None
    use_pec_occupancy = pec_occupancy is not None
    # Same as the uniform lane (§1.6/§1.9): the static sheet/wire masks are
    # intersected with the masks handed in, so a port that cleared its own
    # edge before the run is not put back inside the conductor by a
    # re-realization from the declarations.
    pec_static_edge_masks = None
    if use_pec_occupancy and (pec_sheets or pec_wires):
        pec_static_edge_masks = realized_pec_edge_masks(
            None, sheets=pec_sheets, wires=pec_wires)
        if pec_edge_masks is not None:
            pec_static_edge_masks = tuple(
                s & m for s, m in zip(pec_static_edge_masks, pec_edge_masks))

    admit_setup(grid=grid, tfsf=tfsf, materials=materials, waveguide_ports=waveguide_ports,
                ntff=ntff_box if use_ntff else None, periodic=(False, False, False),
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
    materials = with_edge_keep(materials, pec_occupancy, dtype=state.ex.dtype, sheet_edge_masks=pec_static_edge_masks)
    src_waveforms = drive_table([jnp.array(s[4]) for s in sources], src_meta, n_steps,
        occupancy=pec_occupancy, sheet_edge_masks=pec_static_edge_masks, keep=materials.edge_keep)
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
        from rfx.simulation import _resolve_design_box
        design_box_coeffs = _resolve_design_box(
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
            bloch=None,
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

    # Wire port S-param DFT accumulators
    # (defaults bound unconditionally so _NUScanSetup can carry them;
    # both stay unused unless the branch below runs — no behavior change)
    sp_freqs = None
    wp_meta: list = []
    if use_wire_ports and s_param_freqs is not None:
        sp_freqs = jnp.asarray(s_param_freqs, dtype=jnp.float32)
        nf = len(sp_freqs)
        carry_init["wire_sparams"] = tuple(
            (jnp.zeros(nf, dtype=jnp.complex64),  # v_dft
             jnp.zeros(nf, dtype=jnp.complex64),  # i_dft
             jnp.zeros(nf, dtype=jnp.complex64),  # v_inc_dft
             jnp.zeros(nf, dtype=jnp.complex64))  # v_port_dft (issue #764)
            for _ in wire_ports
        )
        wp_meta = _build_wp_meta(wire_ports, grid)
    else:
        use_wire_ports = False

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
        wire_ports=wire_ports,
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


def run_nonuniform(
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
    checkpoint: bool = False,
    emit_time_series: bool = True,
    checkpoint_every: int | None = None,
    n_warmup: int = 0,
    aniso_eps: tuple | None = None,
    sheet_impedance=None,
    design_box=None,
    record_dft: bool = False,
) -> dict:
    """Run non-uniform FDTD via jax.lax.scan.

    Parameters
    ----------
    cpml_axes : str or None: axes with absorber pads when omitted; subsets accepted
    sources : list of (i, j, k, component, waveform_array)
    probes : list of (i, j, k, component)
    wire_ports : list of dict with keys:
        mid_i, mid_j, mid_k, component, impedance, waveform_array
    s_param_freqs : (n_freqs,) array for S-param DFT
    debye : (DebyeCoeffs, DebyeState) or None
    lorentz : (LorentzCoeffs, LorentzState) or None
    dft_planes : list of DFTPlaneProbe or None
        Frequency-domain plane accumulators. The accumulation is
        identical to the uniform path (acc += field * exp(-j2pi f t) * dt);
        dt is scalar on both paths so no per-axis weighting is required.
    """
    setup = _build_nu_scan(
        grid, materials, n_steps,
        cpml_axes=cpml_axes,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        s_param_freqs=s_param_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=pec_faces,
        pmc_faces=pmc_faces,
        dft_planes=dft_planes,
        rlc_metas=rlc_metas,
        rlc_states=rlc_states,
        ntff_box=ntff_box,
        ntff_data=ntff_data,
        current_moments=current_moments,
        waveguide_ports=waveguide_ports,
        tfsf=tfsf, _feature_tfsf=_feature_tfsf,
        flux_monitors=flux_monitors,
        emit_time_series=emit_time_series,
        aniso_eps=aniso_eps,
        sheet_impedance=sheet_impedance,
        design_box=design_box,
    )
    step_fn = setup.step_fn
    carry_init = setup.carry_init
    if record_dft and dft_planes:
        carry_init["dft_time_records"] = tuple(
            jnp.zeros((n_steps,) + p.accumulator.shape[1:], dtype=carry_init["fdtd"].ex.dtype)
            for p in dft_planes)
    if "wire_sparams" in carry_init:
        carry_init["sparam_time_records"] = tuple(
            jnp.zeros((n_steps, 3), dtype=carry_init["fdtd"].ex.dtype) for _ in carry_init["wire_sparams"])
    src_waveforms = setup.src_waveforms

    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)

    # ---- n_warmup split --------------------------------------------------
    # When n_warmup > 0, run the first n_warmup steps with the carry
    # stop_gradient'd so AD builds no tape for that transient lead-in
    # (issue #40). Only the trailing n_optimize = n_steps - n_warmup
    # steps participate in reverse-mode autodiff.
    #
    # The truncation error DEPENDS ON DISTANCE FROM THE SOURCE TO THE
    # DESIGN REGION -- it is not a blanket property of n_warmup itself
    # (corrects an earlier version of this comment that said the
    # design-relevant support "generally" extends into the warmup window;
    # see the #626 addendum below for why that was an overgeneralization
    # from a single, worst-case fixture). Mechanism: severing the carry
    # severs every gradient path from a design variable's influence during
    # steps < n_warmup back into the loss -- but only steps during which
    # the WAVEFRONT HAS ALREADY REACHED the design region carry any such
    # influence to sever. Before the wavefront arrives, the field there
    # (and hence the loss's sensitivity to that cell) is ~0, so severing
    # those steps' carry discards ~nothing and the gradient is
    # (near-)exact -- n_warmup is genuinely free compute/memory relief in
    # that regime, not merely an approximation. Define
    #     K_safe ~= floor(min_distance(source, design_region) / (C0 * dt))
    # where min_distance is the MINIMUM over EVERY active source AND over
    # each source's own spatial extent (not just its nominal position) --
    # a TFSF plane-wave source, for example, illuminates from an entire
    # box face, not a point, so "distance from the source" means distance
    # from the NEAREST point of that box to the nearest point of the
    # design region. (grid steps for the wavefront to reach the closest
    # design cell, C0 = vacuum lightspeed -- a conservative floor valid
    # for any slower, non-vacuum propagation). For n_warmup <= K_safe
    # truncation error is negligible; beyond it, error grows and can
    # reach the full gradient magnitude by the time n_warmup reaches the
    # loss window.
    #
    # Forward output is exactly n_warmup-invariant (bit-identical) in every
    # placement measured. Two measured regimes (issue #626 part 2 /
    # addendum, both vs an independent central-FD oracle):
    #   - NEAR-SOURCE placement (design cell ~3 cells from the source, so
    #     K_safe ~ 0 -- the wavefront is already present at every step):
    #     error grows smoothly and monotonically from a ~1.5% noise floor
    #     at small n_warmup up to 58% at the loss-window boundary itself,
    #     and EXACTLY zero (gradient fully vanished) once n_warmup extends
    #     far enough into the loss window (fixture:
    #     tests/unit/autodiff/test_n_warmup.py::test_warmup_truncation_error_grows_with_k,
    #     n_steps=100, loss window [80,100)).
    #   - FAR-FROM-SOURCE placement (design cell 62 cells from the source,
    #     K_safe=108 measured from the grid's own dt): AD matches the K=0
    #     FD oracle to <0.01% rel_err through K=88 (deep sub-wavefront,
    #     K_safe-20), then grows SMOOTHLY -- not a sharp cliff -- as K
    #     approaches K_safe: 0.015% at K=98, 0.098% at K=102, 0.259% at
    #     K=104, 0.601% at K=106, 1.186% AT K=108 (=K_safe itself). Every
    #     one of those K<=K_safe values sits comfortably inside this
    #     repo's own established ~1.5% AD-vs-FD noise floor. Past K_safe
    #     the curve keeps growing (1.56% at K=109, ~2.5% around K=112),
    #     though non-monotonically farther out (numerical-dispersion
    #     ripple, not noise) before the broader trend continues upward
    #     toward the loss window (75% by K=200). K_safe is therefore a
    #     BOUND on where truncation stays within the established noise
    #     floor, not a literal discontinuity -- an earlier version of
    #     this comment read the original coarse (multiples-of-20) sweep,
    #     which skipped every point in [101,119], as "exactly the
    #     wavefront-arrival prediction" and overstated how sharp the
    #     transition is (fixture:
    #     scripts/diagnostics/i626_n_warmup_wavefront_locality.py,
    #     n_steps=220, loss window [180,220), finely sampled around
    #     K_safe).
    # The near-source curve is the WORST case, not the general case --
    # use K_safe to decide whether a given source/design placement is in
    # the exact-or-noise-floor or the truncated regime; do not read
    # "n_warmup truncates the gradient" as true unconditionally.
    if n_warmup > 0:
        if n_warmup >= n_steps:
            raise ValueError(
                f"n_warmup ({n_warmup}) must be < n_steps ({n_steps})"
            )
        warmup_steps = jnp.arange(n_warmup, dtype=jnp.int32)
        warmup_xs = (warmup_steps, src_waveforms[:n_warmup])
        warmup_final, warmup_ys = recorded_scan(step_fn, carry_init, warmup_xs)
        carry_init = jax.tree_util.tree_map(
            jax.lax.stop_gradient, warmup_final
        )
        warmup_ys = jax.lax.stop_gradient(warmup_ys)
        xs = (
            jnp.arange(n_warmup, n_steps, dtype=jnp.int32),
            src_waveforms[n_warmup:],
        )
        n_steps_opt = n_steps - n_warmup
    else:
        warmup_ys = None
        n_steps_opt = n_steps

    use_segmented = (
        checkpoint_every is not None
        and 0 < int(checkpoint_every) < n_steps_opt
    )
    if use_segmented:
        final, opt_ys = segmented_scan(
            step_fn, carry_init, xs, int(checkpoint_every), scan=recorded_scan
        )
    else:
        body = jax.checkpoint(step_fn) if checkpoint else step_fn
        final, opt_ys = recorded_scan(body, carry_init, xs)
    # Merge warmup + optimize outputs back into one time_series so the
    # downstream result shape stays (n_steps, n_probes).
    if warmup_ys is not None:
        time_series = jnp.concatenate([warmup_ys, opt_ys], axis=0)
    else:
        time_series = opt_ys

    return _assemble_nu_result(setup, final, time_series)


def _assemble_nu_result(setup: _NUScanSetup, final: dict, time_series, n_record=None) -> dict:
    """Assemble the NU result dict (pure code motion from run_nonuniform,
    #383). Shared by :func:`run_nonuniform` and
    :func:`run_nonuniform_until_decay` so the two paths return the exact
    same schema (state, time_series, dt, conditional dft_planes /
    flux_monitors / ntff_data / waveguide_ports / s_params...)."""
    dt = setup.dt
    dft_planes = setup.dft_planes
    flux_monitors = setup.flux_monitors
    wire_ports = setup.wire_ports
    waveguide_meta = setup.waveguide_meta
    wp_meta = setup.wp_meta
    sp_freqs = setup.sp_freqs
    use_wire_ports = setup.use_wire_ports
    use_dft_planes = setup.use_dft_planes
    use_flux_monitors = setup.use_flux_monitors
    use_lumped_rlc = setup.use_lumped_rlc
    use_ntff = setup.use_ntff
    use_waveguide_ports = setup.use_waveguide_ports

    result = {
        "dft_time_records": tuple(r[:n_record] for r in final.get("dft_time_records", ())),
        "state": final["fdtd"],
        "time_series": time_series,
        "dt": dt,
    }

    # Surface final RLC ADE states (per element).
    if use_lumped_rlc:
        result["rlc_states"] = tuple(final["rlc_states"])

    # Repack DFT plane probes with their final accumulators.
    if use_dft_planes:
        result["dft_planes"] = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, final["dft_planes"])
        )

    # Repack flux monitors with their final E/H DFT accumulators so the
    # caller can call flux_spectrum() on them (same schema as uniform).
    if use_flux_monitors:
        result["flux_monitors"] = tuple(
            mon._replace(e1_dft=e1, e2_dft=e2, h1_dft=h1, h2_dft=h2)
            for mon, (e1, e2, h1, h2) in zip(
                flux_monitors, final["flux_monitors"]
            )
        )

    # Surface final NTFF DFT accumulators
    if use_ntff:
        result["ntff_data"] = final["ntff"]

    # Surface the in-loop block current moments and the slab they belong to.
    # getattr: run(ringdown=) assembles S through here with a stand-in setup
    # (rfx.ringdown._assemble) that carries no monitor.
    if getattr(setup, "use_current_moments", False):
        result["current_moment_data"] = final["current_moments"]
        result["current_moment_monitor"] = setup.current_moments

    # Surface final waveguide-port configs (with recorded modal V/I
    # time series; spectra are extracted post-scan via rect-DFT).
    if use_waveguide_ports:
        result["waveguide_ports"] = tuple(
            cfg_meta._replace(
                v_probe_t=accs[0],
                v_ref_t=accs[1],
                i_probe_t=accs[2],
                i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(
                waveguide_meta, final["waveguide_port_accs"]
            )
        )

    # ---- Extract full S-matrix column from wire port DFTs ----
    #
    # Each port has a V/I DFT pair. The wave decomposition below is
    # DIRECTION-FREE, and that is a physics statement, not a shortcut
    # (issue #673). The port's outward normal is not a degree of freedom
    # for a lumped gap V/I pair:
    #
    #   V = -E_comp * d_parallel      (the E edge's own length)
    #   I = (curl H)_comp * A_perp    (right-hand loop about the SAME axis)
    #
    # Neither expression contains the port's position or an outward
    # normal, so their RELATIVE sign is fixed by the sampling code and no
    # port placement can flip it. The E update at that edge enforces
    # eps*dE/dt + sigma*E = (curl H)_comp, so in phasor form, with
    # C = eps*A/d and the runner's sigma = n_live*d/(Z0*dp1*dp2),
    #
    #       V / I = -1 / (n_live/Z0 + j*w*C + Y_ext)
    #
    # for any passive Y_ext attached across the gap (Re Y_ext >= 0), at a
    # port that is NOT itself driven. Hence Z_in = -V/I sits in the right
    # half plane and
    #
    #       S = (Z_in - Z0)/(Z_in + Z0) = (V + Z0*I)/(V - Z0*I),
    #
    # which is the same ratio as a = (-V + Z0*I)/2, b = (-V - Z0*I)/2.
    # The reciprocal form maps that passive disk to its EXTERIOR, i.e. it
    # returns |S| >= 1 at EVERY bin on ANY passive structure — which is
    # what the pre-#673 direction switch did on "-x"/"-y". That is what
    # fixes the CONVENTION; the convention is then applied unconditionally,
    # exactly as the uniform lane does.
    # Scope note, measured: the |S| <= 1 consequence is a statement about an
    # UNDRIVEN port, and it is a property of the port CELL rather than of the
    # structure across the gap (vacuum, PEC plates shorting the gap and an
    # eps_r=10 slab filling it all read S11 = -0.600000 at n_live = 4).
    # That is exactly why the all-passive fallback below is INTENTIONALLY
    # frozen on this per-cell convention: no physical falsifier can gate a
    # load-independent reading, so it is a diagnostic/self-consistency
    # channel, not S_jj (S_jj requires driving port j) — and the
    # excite=False locks (lane-parity closed forms, sigma ORACLE 1/2) keep
    # their n_live sensitivity as witnesses. Do not "unify" it with the
    # driven diagonal below.
    #
    # DRIVEN diagonal (issue #764). At a genuinely excited port the per-cell
    # convention above stops tracking the load entirely — PROVENANCE, all
    # measured before the #764 fix: one geometry with only `excite` flipped
    # read S11(0.2 GHz) = -0.600000 passive against +0.999670-0.022145j
    # driven though the quasi-static input impedance is identical; the
    # 2-port MSL stub in tests/unit/sparams/test_twoport_wire_port.py reached
    # max|S11| = 4.648 (the reciprocal class that (-V - Z0*I)/(-V + Z0*I)
    # applied to a driven port yields); a matched 50 ohm load read
    # S11 = +0.35426 and a PEC short +0.26780. Root cause was the #313/#318
    # frame mismatch: V and I sampled at ONE cell measure the per-cell
    # Z0/n_live while the reflection formula references the whole-port Z0.
    # The fix is frame-consistent terminal quantities:
    #
    #   V_port = sum_live(-E_c * d_par,c)   (whole-gap line integral,
    #                                        PRIMAL metric per cell, #672)
    #   I      = Ampere loop at the LIVE-run midpoint cell (DUAL metrics)
    #   S_kk   = (V_port - Z0*I) / (V_port + Z0*I)
    #
    # where Z0 is the whole-port port.impedance — the impedance #318
    # physically realizes as the series termination (n_live cells of
    # sigma-folded Z0/n_live in series). Sense: from the pinned per-cell law
    # with equal per-cell injection I0, V_port + Z0*I = Z0*I0 is a
    # drive-only constant (the incident wave), and closing the loop through
    # the external DUT gives V_port = +Z_L*I and I = Z0*I0/(Z0+Z_L) — the
    # measured #683 circuit law. The opposite branch sign (V_port = -Z_L*I)
    # would put I = Z0*I0/(Z0-Z_L), divergent at a matched load — refuted.
    # Only the SUM is KVL/Faraday-constrained on the staggered grid; a
    # single cell is not (a PEC short forces sum V_c = 0 while V_mid stays
    # finite — why the short used to read +0.268: a structural error, not a
    # scale factor).
    # Quasi-static check on the PASSIVE path: with Y_ext -> 0 and
    # w*C*Z0/n_live << 1 the per-cell convention gives
    # S11 -> (1 - n_live)/(1 + n_live), measured to 5 decimals on both
    # lanes (tests/unit/nonuniform/test_nu_wire_port_lane_parity.py).
    #
    # `direction` is still carried on the port spec — the reference-plane
    # path (add_port(reference_plane_cells=...)) needs it for the outboard
    # sign — but it must NOT enter this wave split.
    #
    # Given these, with ONE excited port `k` the full k-th column of
    # the S-matrix is
    #       S[j, k] = b_j / a_k          (j = every port)
    # which reduces to the familiar S11 = b_k/a_k for j==k (reflection)
    # and to S21 = b_2/a_1 for j != k (transmission). Other columns of
    # the S matrix stay zero — callers need to run additional sims
    # with different excited ports to fill them (reciprocity lets us
    # infer S12 from S21 for passive networks).
    #
    # Which (a, b) pair fills a GENUINELY excited column is the issue
    # #770 whole-port pair (see the in-loop derivation below); the
    # per-cell `_ab` split above survives only in the all-passive
    # diagnostic fallback.
    if use_wire_ports and "wire_sparams" in final:
        import numpy as _np
        n_wp = len(wire_ports)
        nf = len(sp_freqs)
        # Tracer-safe accumulation (issue #70): build S with jnp + .at[].set
        # so jax.grad on an objective that pulls through the wire_sparams
        # extractor does not hit TracerArrayConversionError. Concrete-path
        # callers still get a jnp.ndarray that numpy consumers accept via
        # np.asarray(...) (jnp arrays implement __array__).
        S = jnp.zeros((n_wp, n_wp, nf), dtype=jnp.complex64)

        # Pick the excited ports as the filled columns. If no port is
        # excited (all passive), fall back to the legacy diagnostic
        # extraction (no meaningful S-matrix in that case).
        # ``genuinely_excited`` (issues #764 + #770) keys the whole-port
        # wave pair to meta[7] ONLY — the all-passive fallback stays on
        # the frozen per-cell convention (see the scope note above).
        genuinely_excited = [
            idx for idx, meta in enumerate(wp_meta) if meta[7]]
        excited_idx = genuinely_excited
        if not excited_idx:
            excited_idx = list(range(n_wp))   # legacy: treat all as self-excited

        def _ab(v_dft, i_dft, z0):
            """Return (a_incoming, b_outgoing) at one port.

            Direction-free, written in the uniform lane's sign so it reads
            line-for-line against rfx/runners/uniform.py (the overall -1
            cancels in the ratio b/a). See the derivation above (#673).
            """
            zi = z0 * i_dft
            return (-v_dft + zi) / 2.0, (-v_dft - zi) / 2.0

        ab_per_port = []
        for (v_dft, i_dft, _, _vp), meta in zip(final["wire_sparams"],
                                                wp_meta):
            z0 = meta[4]
            a, b = _ab(v_dft, i_dft, z0)
            ab_per_port.append((a, b))

        # All-passive fallback ONLY: the frozen per-cell diagnostic
        # convention (see the scope note above — no physical falsifier
        # can gate a load-independent reading; do not "unify").
        if not genuinely_excited:
            for k in excited_idx:
                a_k = ab_per_port[k][0]
                safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k,
                                     jnp.ones_like(a_k))
                for j in range(n_wp):
                    b_j = ab_per_port[j][1]
                    S = S.at[j, k, :].set(b_j / safe_a_k)

        # Issue #764 + #770: every GENUINELY excited column is the
        # whole-port wave pair, frame-consistent across diagonal and
        # off-diagonal (adjudicated against external physics 2026-08-29,
        # docs/design_notes/issue770_offdiag_adjudication_predeclaration.md
        # + results note):
        #
        #   a_k = (V_port,k + Z0*I_k) / (2*sqrt(Z0))   # measured drive-only
        #                                              # constant Z0*I0 (#683)
        #   b_j = (V_port,j - Z0*I_j) / (2*sqrt(Z0))   # outgoing at port j
        #   S[j, k] = b_j / a_k    (diagonal reduces to the #764
        #                           (V_port - Z0*I)/(V_port + Z0*I))
        #
        # The old mixed split (per-cell midpoint v against the whole-port
        # Z0, the open defect #764 section 6 named) measured neither
        # frame on the canonical thru (|S21| 0.65-0.97, drifting); the
        # whole-port pair measured |S21| = 0.934-0.995 vs the external
        # flux referee 0.971-0.997, power closure deficit 0.9-4.4% vs
        # the 0.2-4.0% flux gap, reciprocity 2.67e-4, and lane parity
        # 3.97e-6 against the uniform decomposer (#770 harness).
        # Global receive sign: +1, DC-witness pinned (S21(DC) -> +1,
        # dev -0.058/-0.115 rad at 0.5/1 GHz; flipped channel at ~pi).
        for k in genuinely_excited:
            v_port_k = final["wire_sparams"][k][3]
            i_k = final["wire_sparams"][k][1]
            z0_k = wp_meta[k][4]
            a_k = v_port_k + z0_k * i_k
            safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k, jnp.ones_like(a_k))
            sqrt_z0_k = jnp.sqrt(z0_k)
            for j in range(n_wp):
                v_port_j = final["wire_sparams"][j][3]
                i_j = final["wire_sparams"][j][1]
                z0_j = wp_meta[j][4]
                b_j = (v_port_j - z0_j * i_j) * (
                    sqrt_z0_k / jnp.sqrt(z0_j))
                S = S.at[j, k, :].set(b_j / safe_a_k)

        result["s_params"] = S
        result["s_param_freqs"] = _np.array(sp_freqs)
        # Raw per-port DFT accumulators (v_dft, i_dft, v_inc_dft,
        # v_port_dft), surfaced for diagnostics/validation harnesses
        # (issue #764; mirrors the uniform lane's raw-acc access via
        # forward()'s wire_port_sparams).
        result["sparam_time_records"] = tuple(
            record[:n_record] for record in final.get("sparam_time_records", ()))
        result["wire_sparams_raw"] = final["wire_sparams"]
        # The static per-port metadata those accumulators were recorded
        # with, in the same order (:func:`_build_wp_meta` slots). The
        # accumulators alone say nothing about which port they came from
        # or at what impedance; ``run_nonuniform_path`` zips the two into
        # the ``(meta, accs)`` pairs the uniform lane's
        # ``Result.wire_port_sparams`` already carries. The existing
        # ``wire_sparams_raw`` shape is unchanged — validation harnesses
        # index it per port.
        result["wire_sparams_meta"] = tuple(wp_meta)

    return result


def run_nonuniform_until_decay(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    *,
    cpml_axes: str | None = None,
    decay_by: float = 1e-3,
    check_interval: int = 50,
    min_steps: int = 100,
    max_steps: int = 50_000,
    decay_energy_consecutive: int = 2,
    radiated_flux_box: tuple | None = None,
    flux_env_checks: int = 4,
    report_every: int | None = None,
    report_label: str = "",
    pec_mask=None,
    pec_sheets=(),
    pec_wires=(),
    pec_edge_masks=None,
    pec_occupancy=None,
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
    stop_fn=None,
    record_dft: bool = False,
) -> dict:
    """Run non-uniform FDTD until the interior-domain energy decays (#383).

    NU port of the issue #169 total-domain-energy stop criterion: a
    Python loop drives constant-length jitted chunks (``chunk =
    check_interval`` steps) of the SAME ``step_fn`` that
    :func:`run_nonuniform` scans, threading the SAME carry (NTFF Kahan
    compensation arrays included) across chunks. Per-chunk ``xs``
    continue the GLOBAL step indices, exactly like the ``n_warmup``
    split in :func:`run_nonuniform`, so source waveforms / DFT phases
    are identical to a fixed-step run of the same length.

    Stop criterion (absorbing boundaries — the ONLY supported class on
    this entry point; the routing layer in ``rfx/api/_execute.py`` gates
    on ``boundary in ('cpml', 'upml')``):

    * At each chunk boundary (after ``k * check_interval`` steps), once
      ``steps_done >= min_steps``, compute the host-side dV-weighted
      interior energy ``U = sum((Ex^2 + ... + Hz^2) * dV)`` over the
      non-CPML interior slice, with ``dV = dx[i] * dy[j] * dz[k]`` the
      per-cell primal volume (all six components weighted by the primal
      cell dV — the same staggering simplification the uniform path
      makes; on a uniform mesh this reduces to the uniform criterion up
      to the constant dV factor).
    * ``peak_U`` is updated at checks only, BEFORE the compare.
    * The stop fires after ``decay_energy_consecutive`` CONSECUTIVE
      sub-threshold checks (``U < decay_by * peak_U``); any
      above-threshold check resets the counter. ``>= 2`` is mandatory
      for the same reason as the uniform path: the interior energy is
      not null-free (transient inter-packet minima recover).
    * ``decay_by = 0.0`` preserves the forced-N escape (``U < 0`` is
      never true for ``U >= 0``), so the loop runs exactly
      ``max_steps`` steps.
    * The loop is hard-bounded by ``max_steps``; a final partial chunk
      (shorter than ``check_interval``) is allowed so the bound is
      exact.

    Check cadence note: the uniform ``run_until_decay`` checks after
    steps ``k * interval + 1`` (its 0-based ``step % interval == 0``
    fires at step index ``k * interval``); the chunked NU loop checks
    after ``k * interval`` steps. The <= 1-step cadence difference is
    deliberate (one XLA program per chunk length) — cross-lane stop-step
    equality is NOT a contract (host/device float order already differs
    between the lanes).

    On a closed domain (no absorbing pads) the interior energy does not
    decay and this function runs to ``max_steps``; there is NO
    point-field fallback on the NU lane (no legacy behavior to
    preserve). The routing layer warn-and-drops ``until_decay`` for
    non-absorbing NU boundaries instead of calling this.

    Forward-only: this is a host loop (not ``lax.scan``); it is not
    reachable from ``forward()``/``optimize()`` and carries no AD tape
    contract. ``checkpoint`` / ``checkpoint_every`` / ``n_warmup`` are
    deliberately absent from the signature (the caller raises before
    dispatching here).

    ``stop_fn`` (issue #1254, ``run(..., until_identified=True)``), when
    given, is called at every chunk boundary after the energy check as
    ``stop_fn(steps_done, peek)``; ``peek()`` returns
    :func:`_assemble_nu_result` of the record so far, and a true return ends
    the loop there. ``None`` (default) leaves the loop unchanged.

    Returns
    -------
    dict
        Same schema as :func:`run_nonuniform` (assembled by the shared
        :func:`_assemble_nu_result`), plus a ``"decay_checks"``
        diagnostic: a list of ``(step, U, peak_U)`` host-float tuples,
        one per eligible energy check. The trace exists only on THIS
        runner-dict return — ``run_nonuniform_path`` does not copy it
        onto the public ``Result`` — so consumers that need to verify
        the fire condition actually held (workspace rule R5) must call
        this module function directly.
    """
    if check_interval < 1:
        raise ValueError(
            f"check_interval must be >= 1, got {check_interval}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    setup = _build_nu_scan(
        grid, materials, max_steps,
        cpml_axes=cpml_axes,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        s_param_freqs=s_param_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=pec_faces,
        pmc_faces=pmc_faces,
        dft_planes=dft_planes,
        rlc_metas=rlc_metas,
        rlc_states=rlc_states,
        ntff_box=ntff_box,
        ntff_data=ntff_data,
        current_moments=current_moments,
        waveguide_ports=waveguide_ports,
        tfsf=tfsf, _feature_tfsf=_feature_tfsf,
        flux_monitors=flux_monitors,
        emit_time_series=emit_time_series,
        aniso_eps=aniso_eps,
        sheet_impedance=sheet_impedance,
        design_box=design_box,
    )
    carry = setup.carry_init
    if record_dft and dft_planes:
        carry["dft_time_records"] = tuple(
            jnp.zeros((max_steps,) + p.accumulator.shape[1:], dtype=carry["fdtd"].ex.dtype)
            for p in dft_planes)
    if "wire_sparams" in carry:
        carry["sparam_time_records"] = tuple(
            jnp.zeros((max_steps, 3), dtype=carry["fdtd"].ex.dtype) for _ in carry["wire_sparams"])

    # Source table: pad/truncate to max_steps so every chunk slice is
    # full-length (mirrors the uniform run_until_decay pad/truncate).
    src_waveforms = jnp.asarray(setup.src_waveforms)
    n_have = int(src_waveforms.shape[0])
    if n_have < max_steps:
        src_waveforms = jnp.pad(
            src_waveforms, ((0, max_steps - n_have), (0, 0)))
    elif n_have > max_steps:
        src_waveforms = src_waveforms[:max_steps]

    # One compiled program per chunk length: full chunks share one XLA
    # executable; the final partial chunk (if any) compiles once more. The
    # per-cell arrays are an argument, not constants compiled into it.
    from rfx.core.jax_utils import split_loop_invariants
    inv_args, inv_rebuild = split_loop_invariants(
        setup.invariants, (grid.nx, grid.ny, grid.nz))
    step_fn_inv = setup.step_fn_inv

    @jax.jit
    def _run_chunk(carry_in, xs, inv_args):
        step = partial(step_fn_inv, invariants=inv_rebuild(inv_args))
        return jax.lax.scan(step, carry_in, xs)

    # Non-CPML interior slice bounds + per-cell primal volume dV
    # (Python ints / concrete arrays — the reduction below is host-side;
    # NonUniformGrid is always 3D).
    _ix0, _ix1 = grid.pad_x_lo, grid.nx - grid.pad_x_hi
    _iy0, _iy1 = grid.pad_y_lo, grid.ny - grid.pad_y_hi
    _iz0, _iz1 = grid.pad_z_lo, grid.nz - grid.pad_z_hi
    _dV = (
        jnp.asarray(grid.dx_arr)[_ix0:_ix1, None, None]
        * jnp.asarray(grid.dy_arr)[None, _iy0:_iy1, None]
        * jnp.asarray(grid.dz)[None, None, _iz0:_iz1]
    )

    def _interior_energy(state) -> float:
        """dV-weighted interior field energy (host float)."""
        sx, sy, sz = (slice(_ix0, _ix1), slice(_iy0, _iy1), slice(_iz0, _iz1))
        u = (state.ex[sx, sy, sz] ** 2 + state.ey[sx, sy, sz] ** 2
             + state.ez[sx, sy, sz] ** 2 + state.hx[sx, sy, sz] ** 2
             + state.hy[sx, sy, sz] ** 2 + state.hz[sx, sy, sz] ** 2)
        return float(jnp.sum(u * _dV))

    # #388 opt-in RADIATED-FLUX stop (NU lane; mirrors the uniform run_until_decay path):
    # stop on the outgoing Poynting flux through a Huygens box instead of the interior energy,
    # so the non-radiating static soft-source charge (which FLOORS the energy criterion) and
    # near-Nyquist buzz do not stall the stop. Opt-in; default keeps the energy criterion.
    use_flux_stop = radiated_flux_box is not None
    if use_flux_stop:
        _flo = position_to_index(grid, radiated_flux_box[0])
        _fhi = position_to_index(grid, radiated_flux_box[1])
        _bl = (min(_flo[0], _fhi[0]), max(_flo[0], _fhi[0]),
               min(_flo[1], _fhi[1]), max(_flo[1], _fhi[1]),
               min(_flo[2], _fhi[2]), max(_flo[2], _fhi[2]))

    def _radiated_power(state) -> float:
        """Net outgoing Poynting flux ∮ (E×H)·n̂ over the box (co-located approx; a stop
        criterion needs only the DECAY, so the half-cell stagger and per-cell dA are neglected
        — they are fixed weightings that do not change the decay rate)."""
        ex, ey, ez = state.ex, state.ey, state.ez
        hx, hy, hz = state.hx, state.hy, state.hz
        il, ih, jl, jh, kl, kh = _bl
        jj, kk, ii = slice(jl, jh), slice(kl, kh), slice(il, ih)
        p = jnp.sum(ey[ih, jj, kk] * hz[ih, jj, kk] - ez[ih, jj, kk] * hy[ih, jj, kk])
        p -= jnp.sum(ey[il, jj, kk] * hz[il, jj, kk] - ez[il, jj, kk] * hy[il, jj, kk])
        p += jnp.sum(ez[ii, jh, kk] * hx[ii, jh, kk] - ex[ii, jh, kk] * hz[ii, jh, kk])
        p -= jnp.sum(ez[ii, jl, kk] * hx[ii, jl, kk] - ex[ii, jl, kk] * hz[ii, jl, kk])
        p += jnp.sum(ex[ii, jj, kh] * hy[ii, jj, kh] - ey[ii, jj, kh] * hx[ii, jj, kh])
        p -= jnp.sum(ex[ii, jj, kl] * hy[ii, jj, kl] - ey[ii, jj, kl] * hx[ii, jj, kl])
        return float(p)

    peak_U = 0.0           # running peak, updated at checks only
    energy_below = 0       # consecutive sub-threshold energy checks
    peak_flux = 0.0        # #388 flux-stop: running peak of the |P| envelope (from first check)
    flux_below = 0         # #388 flux-stop: consecutive sub-threshold checks
    flux_hist: list = []   # recent |P| samples for the max-envelope
    decayed_fired = False  # #388: did the energy criterion fire (vs cap-hit)?
    decay_checks: list = []
    ys_chunks = []
    steps_done = 0
    reporter = None
    if report_every:
        from rfx.progress import ProgressReporter
        # decay_by == 0.0 is the documented forced-N escape: max_steps is then
        # the exact run length, not a cap, and the line should not say "(cap)".
        reporter = ProgressReporter(max_steps, label=report_label,
                                    total_is_cap=(decay_by > 0.0
                                                  or stop_fn is not None))

    while steps_done < max_steps:
        this_chunk = min(int(check_interval), max_steps - steps_done)
        if stop_fn is not None and max_steps - steps_done - this_chunk == 1:
            this_chunk += 1   # #1254: no one-step last chunk (XLA inlines it)
        xs = (
            jnp.arange(steps_done, steps_done + this_chunk, dtype=jnp.int32),
            src_waveforms[steps_done:steps_done + this_chunk],
        )
        carry, ys = _run_chunk(carry, xs, inv_args)
        ys_chunks.append(ys)
        steps_done += this_chunk
        if reporter is not None and (
                steps_done - reporter.last_reported >= int(report_every)
                or steps_done >= max_steps):
            jax.block_until_ready(carry["fdtd"])   # honest wall-clock rate
            reporter.report(steps_done)

        # Stop check at the chunk boundary (check-step-only — the whole-domain
        # reduction / surface integral is the expensive part).
        if use_flux_stop:
            # RADIATED-FLUX stop: track the flux peak from the FIRST check (radiation peaks
            # during the source drive), gate only the STOP on min_steps.
            flux_hist.append(abs(_radiated_power(carry["fdtd"])))
            env = max(flux_hist[-flux_env_checks:])
            if env > peak_flux:
                peak_flux = env
            if steps_done >= min_steps and peak_flux > 0.0 and env < decay_by * peak_flux:
                flux_below += 1
                if flux_below >= decay_energy_consecutive:
                    decayed_fired = True
                    break
            else:
                flux_below = 0
        elif decay_by > 0.0 or steps_done >= min_steps:
            # Track the PEAK from the first chunk boundary, like the flux
            # branch above; only the STOP (and the recorded check trace) waits
            # for min_steps. Gating the peak on min_steps too reads the
            # RESIDUAL as the reference on a domain that empties before
            # min_steps, and no residual falls another ``decay_by`` below
            # itself, so such a run reaches max_steps without firing (#1078;
            # the uniform lane carried the same defect). ``decay_by == 0.0``
            # is the forced-N escape and keeps the pre-min_steps chunks
            # uncomputed, so its recorded trace is unchanged.
            U = _interior_energy(carry["fdtd"])
            if U > peak_U:
                peak_U = U
            if steps_done >= min_steps:
                decay_checks.append((steps_done, U, peak_U))
                # decay_by=0.0 forced-N escape: U >= 0 so U < 0 never fires.
                if U < decay_by * peak_U:
                    energy_below += 1
                    if energy_below >= decay_energy_consecutive:
                        decayed_fired = True
                        break
                else:
                    energy_below = 0

        # #1254: the caller's stop check at the chunk boundary.
        if stop_fn is not None:
            from rfx.progress import concat_chunks
            if stop_fn(steps_done, lambda: _assemble_nu_result(
                    setup, carry, concat_chunks(ys_chunks), n_record=steps_done)):
                if reporter is not None and reporter.last_reported != steps_done:
                    reporter.report(steps_done)
                break

    # #388: measured static-remnant advisory on cap-hit (until_decay is absorbing-only on
    # the NU lane too, so a cap-hit without firing means the energy criterion could not
    # self-terminate — warn if the remnant is electrostatic). Function-local import avoids
    # a module-level simulation<->nonuniform cycle.
    # decay_by == 0.0 is the forced-N escape (fixed-step progress route):
    # running to max_steps is the DESIGN there, not a failed stop — the
    # static-remnant advisory would be a false alarm.
    if not decayed_fired and not use_flux_stop and decay_by > 0.0:
        from rfx.simulation import _warn_static_remnant_cap_hit
        _warn_static_remnant_cap_hit(carry["fdtd"], materials, grid)

    # Per-chunk ys concatenate to exactly steps_done rows; the
    # emit_time_series=False convention ((steps, 0)) is preserved
    # because each chunk emits (this_chunk, 0).
    time_series = jnp.concatenate(ys_chunks, axis=0)

    result = _assemble_nu_result(setup, carry, time_series, n_record=steps_done)
    result["decay_checks"] = decay_checks
    return result
