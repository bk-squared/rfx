import numpy as np

from .plan import (
    KERNEL_KNOWN_DIFFERENCES, MeasurementPlan, Node, Owner, TimeBase,
    _frequencies, _plane_nodes, field_channel, flux_owner,
)


def calculator_plan(grid, n_steps, *, probes, planes, flux_monitors=(),
                    flux_periodic=None, extra_owners=()):
    if flux_monitors and flux_periodic is None:
        raise TypeError("calculator_plan() needs flux_periodic when flux_monitors are given.")
    owners = [Owner(f"witness:{i}", "probe", (field_channel(
        probe.component, (Node(probe.component, probe.i, probe.j, probe.k, 1.0),),
        e_time_offset=0., h_time_offset=0.),)) for i, probe in enumerate(probes)]
    owners.extend(Owner(f"dft_plane:{i}", "dft_plane", (field_channel(
        plane.component,
        _plane_nodes(grid, plane.axis, plane.index, plane.component, plane.region),
        box=(plane.axis, plane.index, plane.region),
        e_time_offset=0., h_time_offset=0.),),
        _frequencies(None, freqs=plane.freqs)) for i, plane in enumerate(planes))
    owners.extend(flux_owner(
        grid, i, cfg, _frequencies(None, freqs=cfg.freqs),
        availability="sampled", periodic=flux_periodic)
        for i, cfg in enumerate(flux_monitors))
    owners.extend(extra_owners)
    return MeasurementPlan("run_uniform", TimeBase(grid.dt, n_steps), tuple(owners),
                           known_differences=KERNEL_KNOWN_DIFFERENCES)


def coax_plane_voltages(result, owner):
    record = next((record for record in getattr(result, "sparam_time_records", None) or ()
                   if record.id == owner.id), None)
    if record is None:
        return None
    return np.stack([np.asarray(record[f"V:{i}"], dtype=np.complex128)
                     for i in range(len(owner.channels))], axis=1)
