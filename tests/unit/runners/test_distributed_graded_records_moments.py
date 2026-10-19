"""Graded volume moments use the shared pre-E snapshot and owned weights."""
import numpy as np
import pytest
from rfx.current_moments import current_moment_far_field
from tests.unit.runners._graded_records_scene import radiating_model, pair, measure, record, mutation_result, with_bars


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_moments(n_devices, entry):
    sim, grid, seam = radiating_model('cpml', n_devices, grading='z')
    sim.add_current_moment_monitor((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 7e-3),
        block_size=2e-3, freqs=[3e9, 7e9])
    one, many = pair(sim, n_devices, entry)
    theta, phi = np.linspace(0, np.pi, 9), np.linspace(0, 2 * np.pi, 11)
    ff = [current_moment_far_field(r, theta, phi) for r in (one, many)]
    # The first array is the Kahan-accumulated per-block physical result,
    # also consumed by current_moment_far_field; the second is internal carry.
    values = {'moments': measure(many.current_moment_data[0], one.current_moment_data[0])}
    for name in ('E_theta', 'E_phi'):
        values[name] = measure(getattr(ff[1], name), getattr(ff[0], name))
    # CPML wrapped step: compile-form rounding, operands byte-equal (rfx-archive DESIGN.md 4g amendment 1).
    with_bars(values, 'relative_peak')
    record('moments', dict(boundary='cpml', devices=n_devices, entry=entry, shape=grid.shape), values)
    assert values['moments']['peak'] > 0
    assert all(v['passed'] for v in values.values()), ('moment bar', values)


@pytest.mark.parametrize('mutation', ['previous_e', 'double_weights'])
def test_moment_mutations(mutation, monkeypatch):
    from rfx.runners._distributed_records import MomentRecords
    if mutation == 'previous_e':
        original = MomentRecords.update
        def wrong(self, buffer, state, dt, step, *, ranks, previous):
            return original(self, buffer, state, dt, step, ranks=ranks,
                            previous=self.prepare(state, ranks=ranks))
        monkeypatch.setattr(MomentRecords, 'update', wrong)
    else:
        original = MomentRecords.__init__
        def wrong(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.parts = [m._replace(w_ex=2*m.w_ex, w_ey=2*m.w_ey, w_ez=2*m.w_ez) for m in self.parts]
        monkeypatch.setattr(MomentRecords, '__init__', wrong)
    with pytest.raises(AssertionError, match='moment bar') as failure:
        test_moments(2, 'run')
    mutation_result('moments', mutation, failure.value)
