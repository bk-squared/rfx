"""Adapt batched records to the same result witness run() uses."""
import numpy as np


def sweep_member_witnesses(sim, grid, series, names, planes, accumulators):
    from rfx.api._spec import Result
    from rfx.measurement.settling import attach_run_settling_witness

    dbs, witnesses = [], []
    for index, samples in enumerate(series):
        realized = {name: plane._replace(accumulator=accumulators[name][index])
                    for name, plane in zip(names, planes)}
        result = Result(state=None, time_series=samples, s_params=None, freqs=None,
                        dft_planes=realized or None, grid=grid, dt=grid.dt)
        result = attach_run_settling_witness(sim, result, n_steps=len(samples))
        dbs.append(result.settling_db)
        witnesses.append(result.settling_witness)
    return np.asarray(dbs, dtype=float), tuple(witnesses)
