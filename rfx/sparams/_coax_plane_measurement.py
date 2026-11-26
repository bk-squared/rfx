"""Coax plane measurement setup shared by the two-port and transition calculators."""

from rfx.measurement.calculator_plan import calculator_plan
from rfx.measurement.ports import coax_owner
from rfx.measurement.flux_weights import monitor_periodic_flags


def coax_plane_measurement(sim, grid, n_steps, *, plane_indices, center_xy,
                           pin_radius, outer_radius, frequencies, pec_cell_mask,
                           probes, planes, flux_monitors):
    """Return the coax owner and calculator plan for the supplied plane measurements."""
    owner = coax_owner(
        grid, port_id=0, plane_indices=plane_indices, center_xy=center_xy,
        pin_radius=pin_radius, outer_radius=outer_radius, frequencies=frequencies,
        pec_cell_mask=pec_cell_mask)
    plan = calculator_plan(
        grid, n_steps, probes=probes, planes=planes,
        flux_monitors=flux_monitors,
        flux_periodic=monitor_periodic_flags(sim, grid), extra_owners=(owner,))
    return owner, plan
