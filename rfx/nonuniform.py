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

from rfx.boundaries.axes import resolve_cpml_axes
from rfx.measurement import assemble_graded_result as _assemble_nu_result

import jax.numpy as jnp

from rfx.model import current_source_samples
import numpy as np

from rfx.core.yee import EPS_0
from rfx.core.jax_utils import is_tracer

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
            materials, (i, j, k), component, grid.dt, periodic=tuple(a in grid.periodic_axes for a in "xyz"), host=True, grid=grid))

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
    _profiles = tuple(store if grid_traced or exact is None else exact for store, exact in zip((grid.dx_arr, grid.dy_arr, grid.dz), (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)))
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


from rfx.stepping import run_nonuniform as run_nonuniform  # noqa: E402
from rfx.stepping import run_nonuniform_until_decay as run_nonuniform_until_decay  # noqa: E402
