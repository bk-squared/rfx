"""Static per-port metadata the graded V/I DFT scan reads (primal and dual port-cell metrics)."""
from rfx.model.graded_grid import e_node_dual_spacing_at
from rfx.model.graded_ports import port_metric, port_metric_axes


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
