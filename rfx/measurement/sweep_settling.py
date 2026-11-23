"""Adapt batched records and their realized bounds to the shared witness."""
import numpy as np


def sweep_member_witnesses(sim, grid, series, names, planes, accumulators, *, echo_bounds):
    from rfx.api._spec import Result
    from rfx.measurement.settling import attach_run_settling_witness

    dbs, witnesses = [], []
    for index, samples in enumerate(series):
        realized = {name: plane._replace(accumulator=accumulators[name][index])
                    for name, plane in zip(names, planes)}
        result = Result(state=None, time_series=samples, s_params=None, freqs=None,
                        dft_planes=realized or None, grid=grid, dt=grid.dt,
                        settling_echo_bound=echo_bounds[index])
        result = attach_run_settling_witness(sim, result, n_steps=len(samples))
        witness = result.settling_witness
        dbs.append(witness['db'])
        witnesses.append(witness)
    return np.asarray(dbs, dtype=float), tuple(witnesses)
