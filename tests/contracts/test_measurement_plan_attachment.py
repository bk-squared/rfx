"""Prepared solves own one plan before entering any time-step loop."""
from dataclasses import replace
from unittest.mock import patch

import jax
import pytest

from rfx.measurement.plan import MeasurementPlan
from rfx.model.conductors import PreparedSolve
from tests.contracts.path_equivalence.builders import build, FREQS


PATHS = ('run_uniform', 'fwd_uniform', 'run_nonuniform', 'fwd_nonuniform',
         'run_distributed', 'fwd_distributed', 'run_distributed_nu',
         'fwd_distributed_nu', 'run_subgridded')


class AtStep(Exception):
    pass


def capture(path, *, sim=None, decay=False):
    graded = 'nonuniform' in path or path.endswith('_nu')
    subgrid = path == 'run_subgridded'
    if sim is None:
        if subgrid:
            from tests.contracts.test_subgrid_requires_experimental import _sim
            sim = _sim('research')
        else:
            sim = build(('_probes', 'probe'),
                        'run_nonuniform' if graded else 'run_uniform', graded=graded)
    prepared, reached = [], []
    original = PreparedSolve.__init__

    def remember(self, *args, **kwargs):
        original(self, *args, **kwargs)
        prepared.append(self)

    def stop(product, stepper, *args, **kwargs):
        reached.append(product)
        raise AtStep

    from rfx.measurement.prepared import build_measurement_plan
    with patch.object(PreparedSolve, '__init__', remember), patch(
            'rfx.stepping.prepared.advance', stop), patch(
            'rfx.measurement.prepared.build_measurement_plan', wraps=build_measurement_plan) as builder:
        options = dict(n_steps=12, skip_preflight=True)
        if 'distributed' in path:
            # Root conftest forces two virtual CPU devices before importing JAX.
            devices = jax.devices('cpu')[:2]
            if len(devices) < 2:
                pytest.skip('needs two virtual CPU devices from root conftest XLA_FLAGS')
            options['devices'] = devices
        if path.startswith('fwd'):
            options.update(checkpoint=False, port_s11_freqs=FREQS if any(p.impedance > 0 for p in sim._ports) else None)
            if 'distributed' in path:
                options['distributed'] = True
            call = sim.forward
        else:
            options['compute_s_params'] = False
            if any(p.impedance > 0 for p in sim._ports):
                options['s_param_freqs'] = FREQS
            if decay:
                options.update(until_decay=1e-4, decay_max_steps=40,
                               decay_min_steps=10, decay_check_interval=5)
            call = sim.run
        with pytest.raises(AtStep):
            call(**options)
        assert len(reached) == 1
        assert len(prepared) == 1
        solve = prepared[0]
        assert reached[0]._prepared_solve is solve
        plans = [v for v in vars(solve).values() if isinstance(v, MeasurementPlan)]
        assert len(plans) == 1
        assert plans[0] is solve.measurement
        assert builder.call_count == 1
        assert solve.measurement.path == path
        assert solve.measurement.time_base.record_length == (40 if decay else 12)
        return solve.measurement


@pytest.mark.parametrize('path', PATHS)
def test_b4_attached_once(path):
    plan = capture(path)
    if path == 'run_subgridded':
        assert plan.owners
        assert all(not owner.channels and 'not on the plan' in owner.availability
                   for owner in plan.owners)


@pytest.mark.parametrize('run,forward', [
    ('run_uniform', 'fwd_uniform'), ('run_nonuniform', 'fwd_nonuniform'),
    ('run_distributed', 'fwd_distributed'),
    ('run_distributed_nu', 'fwd_distributed_nu'),
])
def test_b3_same_request(run, forward):
    graded = 'nonuniform' in run or run.endswith('_nu')
    sim = build(('_dft_planes', 'dft_plane'),
                'run_nonuniform' if graded else 'run_uniform', graded=graded)
    sim.add_flux_monitor(axis='x', coordinate=4/1024, freqs=FREQS)
    plans = [capture(path, sim=sim) for path in (run, forward)]
    normalized = [replace(plan, path='', owners=tuple(
        replace(owner, availability='') for owner in plan.owners)) for plan in plans]
    assert normalized[0] == normalized[1]


@pytest.mark.parametrize('path', ['run_uniform', 'run_nonuniform'])
def test_record_capacity_until_decay(path):
    capture(path, decay=True)
