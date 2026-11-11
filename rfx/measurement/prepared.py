"""Retain realized inputs, then attach after the runner's structure admission."""
from dataclasses import replace

from rfx.measurement.plan import build_measurement_plan


def attach_measurement(sim, conductors, *, n_steps, path, grid=None, **inputs):
    from rfx.model.conductors import PreparedSolve
    from rfx.measurement.identification import read_ringdown_probes
    identification = read_ringdown_probes(getattr(sim, "_probes", None), "attach_measurement")
    prepared = conductors._prepared_solve
    if prepared is None:
        # Internal material-based entries also own an execution-local snapshot.
        prepared = PreparedSolve(sim)
        prepared._taken = True
        object.__setattr__(conductors, '_prepared_solve', prepared)
    if prepared.measurement is not None or getattr(prepared, '_measurement_inputs', None) is not None:
        return
    # Stage mutates the prepared owner; retain its pre-release geometry view.
    prepared._measurement_inputs = dict(
        sim=sim, grid=conductors.grid if grid is None else grid, n_steps=n_steps, path=path,
        conductors=replace(conductors), pec_edges=conductors.pec_edges,
        periodic=conductors.periodic, identification=identification, **inputs)
    if path == 'run_subgridded':
        finish_measurement(conductors)


def finish_measurement(conductors):
    """Build once after path admission, releasing the retained dense inputs."""
    prepared = conductors._prepared_solve
    inputs = getattr(prepared, '_measurement_inputs', None)
    if inputs is None:
        return
    prepared.measurement = build_measurement_plan(**inputs)
    prepared._measurement_inputs = None
