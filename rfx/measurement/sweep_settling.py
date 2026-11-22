"""Adapt batched records to the same result witness run() uses."""
import numpy as np


def sweep_member_witnesses(sim, grid, series, names, planes, accumulators, *, param_name, param_values):
    from rfx.api._spec import Result
    from rfx.measurement.settling import attach_run_settling_witness

    import copy
    from dataclasses import replace

    dbs, witnesses = [], []
    from rfx.vmap_sweep import _parse_param_name
    mat_name, field = _parse_param_name(param_name)
    for index, samples in enumerate(series):
        member = copy.deepcopy(sim)
        for name in ([mat_name] if mat_name else list(member._materials)):
            member._materials[name] = replace(member._resolve_material(name), **{field: float(param_values[index])})
        realized = {name: plane._replace(accumulator=accumulators[name][index])
                    for name, plane in zip(names, planes)}
        result = Result(state=None, time_series=samples, s_params=None, freqs=None,
                        dft_planes=realized or None, grid=grid, dt=grid.dt)
        result = attach_run_settling_witness(member, result, n_steps=len(samples))
        dbs.append(result.settling_db)
        witnesses.append(result.settling_witness)
    return np.asarray(dbs, dtype=float), tuple(witnesses)
