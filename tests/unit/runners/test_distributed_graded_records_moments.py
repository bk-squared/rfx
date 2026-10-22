"""Graded volume moments use the shared pre-E snapshot and owned weights."""
import numpy as np
import pytest
from rfx.current_moments import current_moment_far_field
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import radiating_model, pair, measure, record, mutation_result, with_bars

# x graded next to the 1 mm boundary cells (the CPML reads the boundary dx),
# uniform 1 mm from 3.3 to 13.3 mm. The monitor refuses graded in-plane cells
# under its slab, so this is the graded-x mesh it admits. The slab's first x
# row reads half of the graded 1.1 mm cell before it: the one graded number in
# the slab's x spacings. On 3 devices the last seam leaves one slab row.
X_OUTSIDE_SLAB = np.r_[1e-3, 1.2e-3, 1.1e-3, np.full(10, 1e-3), .7e-3, 1e-3]
X_SLAB = ((3.3e-3, 1e-3, 1e-3), (12.3e-3, 7e-3, 7e-3))


def compare(sim, n_devices, entry):
    one, many = pair(sim, n_devices, entry)
    theta, phi = np.linspace(0, np.pi, 9), np.linspace(0, 2 * np.pi, 11)
    ff = [current_moment_far_field(r, theta, phi) for r in (one, many)]
    # The first array is the Kahan-accumulated per-block physical result,
    # also consumed by current_moment_far_field; the second is internal carry.
    values = {'moments': measure(many.current_moment_data[0], one.current_moment_data[0])}
    for name in ('E_theta', 'E_phi'):
        values[name] = measure(getattr(ff[1], name), getattr(ff[0], name))
    # CPML wrapped step: compile-form rounding, operands byte-equal (rfx-archive DESIGN.md 4g amendment 1).
    return with_bars(values, 'relative_peak'), many


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_moments(n_devices, entry):
    sim, grid, seam = radiating_model('cpml', n_devices, grading='z')
    sim.add_current_moment_monitor((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 7e-3),
        block_size=2e-3, freqs=[3e9, 7e9])
    values, _ = compare(sim, n_devices, entry)
    record('moments', dict(boundary='cpml', devices=n_devices, entry=entry, shape=grid.shape), values)
    assert values['moments']['peak'] > 0
    assert all(v['passed'] for v in values.values()), ('moment bar', values)


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_moments_x_graded_outside_slab(n_devices, entry):
    """Owned x rows carry their own x spacing where the x cells are graded."""
    sim, grid, seam = radiating_model('cpml', n_devices, profile=X_OUTSIDE_SLAB)
    sim.add_current_moment_monitor(*X_SLAB, block_size=2e-3, freqs=[3e9, 7e9])
    values, many = compare(sim, n_devices, entry)
    m = many.current_moment_monitor
    # Realized, not declared: the slab's first row reads the graded dual
    # spacing (1.1 mm + 1 mm) / 2, the others 1 mm, and every seam cuts the slab.
    inv = np.asarray(m.inv_dx_e, dtype=np.float64)
    np.testing.assert_allclose(inv, np.r_[2 / 2.1e-3, np.full(inv.size - 1, 1 / 1e-3)], rtol=1e-6)
    assert all(m.i_lo < k * seam < m.i_hi for k in range(1, n_devices)), (m.i_lo, m.i_hi, seam)
    record('moments', dict(boundary='cpml', devices=n_devices, entry=entry, shape=grid.shape,
                           mesh='x graded outside the slab'), values)
    assert values['moments']['peak'] > 0
    assert all(v['passed'] for v in values.values()), ('moment bar', values)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_graded_cells_under_the_slab_refused_on_devices(entry):
    """Graded x cells under the slab: refused on several devices as on one."""
    sim, _, _ = radiating_model('cpml', 2)
    sim.add_current_moment_monitor((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 7e-3),
        block_size=2e-3, freqs=[3e9, 7e9])
    with pytest.raises(NotImplementedError, match='uniform in-plane mesh'):
        getattr(sim, entry)(n_steps=24, skip_preflight=True, devices=host_devices(2),
                            **({'distributed': True} if entry == 'forward' else {}))


def _wrong_parts(monkeypatch, mutation):
    from rfx.runners._distributed_records import MomentRecords
    original = MomentRecords.__init__
    def wrong(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if mutation == 'double_weights':
            self.parts = [m._replace(w_ex=2*m.w_ex, w_ey=2*m.w_ey, w_ez=2*m.w_ez) for m in self.parts]
        else:
            # Every device reads the x spacings from the slab's first row
            # instead of from its own rows.
            self.parts = [m._replace(inv_dx_e=self.monitor.inv_dx_e[:m.i_hi - m.i_lo]) for m in self.parts]
    monkeypatch.setattr(MomentRecords, '__init__', wrong)


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
        _wrong_parts(monkeypatch, mutation)
    with pytest.raises(AssertionError, match='moment bar') as failure:
        test_moments(2, 'run')
    mutation_result('moments', mutation, failure.value)


@pytest.mark.parametrize('mutation', ['x_spacing_from_slab_start', 'double_weights'])
def test_x_graded_moment_mutations(mutation, monkeypatch):
    _wrong_parts(monkeypatch, mutation)
    with pytest.raises(AssertionError, match='moment bar') as failure:
        test_moments_x_graded_outside_slab(2, 'run')
    mutation_result('moments', mutation, failure.value)
