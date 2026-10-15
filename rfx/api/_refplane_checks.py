"""Conductor footprint for the API reference-plane trace scan."""
import jax.numpy as jnp


def _refplane_conductor_mask(pec_mask, sheet_ctx, pec_sheets=()):
    """Full conductor footprint for the reference-plane trace scan (#695).

    ``pec_mask | (sheet footprints)`` — #931 §1.9: the footprint is the
    cells of the PEC VOLUMES union the footprints of the SHEETS.  Neither
    an f0 sheet (#677) nor a PEC sheet (#931) contributes to ``pec_mask``
    or to ``materials.sigma``, so the bare ``pec_mask`` this used to pass
    made a sheet-traced transmission line look like an empty
    cross-section.  ``sheet_ctx.sigma_sheet`` is ``> 0`` exactly on the f0
    sheet cells; PEC sheets arrive as ``SheetSpec``s.

    Returns ``pec_mask`` unchanged when there is nothing to add, or when
    an input is traced — this caller already requires concrete values and
    a traced OR would only relocate the failure.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.materials.thin_conductor import conductor_footprint

    masks = []
    if pec_mask is not None and is_tracer(pec_mask):
        return pec_mask
    if sheet_ctx is not None:
        sig = getattr(sheet_ctx, "sigma_sheet", None)
        if sig is not None:
            if is_tracer(sig):
                return pec_mask
            masks.append(jnp.asarray(sig) > 0.0)
    for sp in (pec_sheets or ()):
        masks.append(jnp.asarray(sp.footprint))
    if not masks:
        return pec_mask
    return conductor_footprint(pec_mask=pec_mask, sheet_masks=masks)



def _refplane_reject_msl_ladder_in_plane_zone(
    sim, grid, port_entry, *, line_axis, outboard_sign, i_port, i_far,
):
    """Crossing guard, microstrip leg: no microstrip de-embedding probe may
    sit inside a lumped/wire port's reference-plane zone.

    A microstrip port is not a point: its N-probe de-embedding ladder
    reaches back toward the DUT by ``n_probe_offset + n * n_probe_spacing``
    cells, so the port position can sit far outside the plane zone while an
    inner rung sits INSIDE it (on the probe-fed microstrip fixture the feed
    snaps to index 77, far beyond a 2N plane at 53, while the default
    ``n_probes=5`` ladder's last rung is index 51 — invisible to the
    port-position check above).

    The ladder walked here is the one the S-matrix drivers actually probe:
    ``_resolve_msl_auto_offsets`` re-derives auto probe offsets (#469) and
    auto spacings (#681) from the registered geometry, and the resolved
    ladder can be LONGER than the registration ladder (a widened spacing)
    or SHORTER (a short feed floors the spacing at two cells). Walking
    ``sim._msl_ports`` as registered would miss a crossing the run makes,
    or report one it never makes. The resolver is idempotent (it restarts
    from the stored lower edge every call), so the result is the same
    whether the caller installed the registered or the resolved entries.
    Its clearance warnings are suppressed here: they belong to the driver
    that probes the ladder, which has already emitted them once.

    Message class stays "reach past another port" (frozen by
    tests/locks/test_refplane_port_waves.py::
    test_refplane_crossing_guard_rejects_planes_past_other_port), with the
    offending rung named. Line-axis indices only (conservative): a
    TRANSVERSELY separated microstrip port on a parallel trace also trips.
    """
    import warnings

    from rfx.api._sparams import _resolve_msl_auto_offsets
    from rfx.sources.msl_port import (
        MSLPort,
        msl_axis_roles,
        msl_probe_x_coords_n,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entries = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), grid)
    axis_index = {"x": 0, "y": 1, "z": 2}
    for msl_entry in entries:
        x_feed, y_centre, z_lo = (float(c) for c in msl_entry.position)
        port_model = MSLPort(
            feed_x=x_feed,
            y_lo=y_centre - float(msl_entry.width) / 2.0,
            y_hi=y_centre + float(msl_entry.width) / 2.0,
            z_lo=z_lo,
            z_hi=z_lo + float(msl_entry.height),
            direction=msl_entry.direction,
            impedance=float(msl_entry.impedance),
            excitation=msl_entry.waveform,
        )
        prop_axis_name = msl_axis_roles(msl_entry.direction)[0]
        prop_axis = axis_index[prop_axis_name]
        probe_coords = msl_probe_x_coords_n(
            grid, port_model,
            n_probes=int(msl_entry.n_probes),
            n_offset_cells=int(msl_entry.n_probe_offset),
            n_spacing_cells=int(msl_entry.n_probe_spacing),
        )
        for rung, rung_coord in enumerate(probe_coords):
            rung_point = list(msl_entry.position)
            rung_point[prop_axis] = float(rung_coord)
            rung_index = int(
                grid.position_to_index(tuple(rung_point))[line_axis])
            in_zone = (
                (i_port < rung_index <= i_far) if outboard_sign > 0
                else (i_far <= rung_index < i_port))
            if in_zone:
                raise ValueError(
                    "reference_plane_cells: the reference planes of the "
                    f"port at {port_entry.position} (indices up to "
                    f"{i_far} on axis {line_axis}) reach past another "
                    f"port at {msl_entry.position} — its MSL de-embedding "
                    f"probe {rung} at {prop_axis_name} = {rung_coord:.6g} m "
                    f"(index {rung_index}) lies inside the plane zone. "
                    "Reduce N so both planes stay on the uniform line "
                    "between the ports, or reduce n_probes / "
                    "n_probe_offset / n_probe_spacing so the ladder stays "
                    "outboard of the 2N plane (an AUTO offset/spacing is "
                    "the resolved value the run probes, not the "
                    "registration default). Note: this check compares "
                    "line-axis indices only (conservative), so a "
                    "TRANSVERSELY separated MSL port on a different "
                    "parallel trace also trips it.")
