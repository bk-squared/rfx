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
        unavailable = None
        for name in ([mat_name] if mat_name else
                     dict.fromkeys(entry.material_name for entry in sim._geometry)):
            try:
                member._materials[name] = replace(
                    member._resolve_material(name), **{field: float(param_values[index])})
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                unavailable = dict(tau_max=None, D=None, n_max=None, g=None,
                                   reason=f'cannot judge: sweep material {name!r} cannot be resolved: {exc}')
        realized = {name: plane._replace(accumulator=accumulators[name][index])
                    for name, plane in zip(names, planes)}
        result = Result(state=None, time_series=samples, s_params=None, freqs=None,
                        dft_planes=realized or None, grid=grid, dt=grid.dt,
                        settling_echo_bound=unavailable)
        result = attach_run_settling_witness(member, result, n_steps=len(samples))
        witness = result.settling_witness
        if unavailable is not None:
            reason = witness.get('reason') or ''
            if unavailable['reason'] not in reason:
                reason = '; '.join(filter(None, (reason, unavailable['reason'])))
            witness = dict(witness, reason=reason, echo_bound_refused=True,
                           echo_bound_reason=unavailable['reason'])
            if witness.get('tail_status', witness['status']) == 'pass':
                witness.update(status='undetermined', db=float('nan'))
        dbs.append(witness['db'])
        witnesses.append(witness)
    return np.asarray(dbs, dtype=float), tuple(witnesses)
