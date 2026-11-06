"""Attach a host-built measurement snapshot to an execution's prepared solve."""
from rfx.measurement.plan import build_measurement_plan


def attach_measurement(sim, conductors, *, n_steps, path, grid=None, **inputs):
    from rfx.model.conductors import PreparedSolve
    prepared = conductors._prepared_solve
    if prepared is None:
        # Internal material-based entries also own an execution-local snapshot.
        prepared = PreparedSolve(sim)
        prepared._taken = True
        object.__setattr__(conductors, '_prepared_solve', prepared)
    if prepared.measurement is not None:
        return
    prepared.measurement = build_measurement_plan(
        sim, conductors.grid if grid is None else grid, n_steps=n_steps, path=path,
        conductors=conductors, pec_edges=conductors.pec_edges,
        periodic=conductors.periodic, **inputs)
