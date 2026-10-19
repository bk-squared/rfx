"""Debye in CPML: DFT cross-trace bar and law C for every snapshot step."""
import jax
import numpy as np
import pytest
from rfx import SnapshotSpec
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import radiating_model, measure, record, mutation_result
from tests.unit.runners.test_distributed_graded_records_planes import add_planes
from tests.unit.runners.test_distributed_graded_records_flux import add_flux
from tests.unit.runners.test_distributed_graded_records_ntff import add_box


@pytest.mark.parametrize('n_devices', [2, 3])
def test_debye_cpml(n_devices, monkeypatch):
    sim, grid, seam = radiating_model('cpml', n_devices, pole=True)
    add_planes(sim, grid, seam)
    add_flux(sim, grid, seam)
    add_box(sim)
    steps = 16
    components = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
    frames = []
    from rfx.stepping import graded
    from rfx.stepping.sequence import HookPoint
    original = graded.make_graded_step
    def observe(context, **kwargs):
        def capture(frame):
            jax.debug.callback(lambda *fields: frames.append(tuple(np.array(f) for f in fields)),
                *(getattr(frame.st, c) for c in components), ordered=True)
        return original(context, hooks={HookPoint.STEP_END: (capture,)}, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(graded, 'make_graded_step', observe)
        one = sim.run(n_steps=steps, skip_preflight=True)
        jax.block_until_ready(one.state)
        jax.effects_barrier()
    many = sim.run(n_steps=steps, skip_preflight=True, devices=host_devices(n_devices),
                   snapshot=SnapshotSpec(interval=1, components=components))
    values = {}
    for key, p in one.dft_planes.items():
        values['plane/' + key] = measure(many.dft_planes[key].accumulator, p.accumulator)
    for key, m in one.flux_monitors.items():
        for name in ('e1_dft', 'e2_dft', 'h1_dft', 'h2_dft'):
            values[f'flux/{key}/{name}'] = measure(getattr(many.flux_monitors[key], name), getattr(m, name))
    # Kahan pairs represent total minus compensation; compare that recorded quantity.
    for i, name in enumerate(one.ntff_data._fields[:6]):
        values['ntff/' + name] = measure(many.ntff_data[i] - many.ntff_data[i+6],
                                        one.ntff_data[i] - one.ntff_data[i+6])
    snapshots = {}
    for i, c in enumerate(components):
        expected = np.stack([frame[i] for frame in frames])
        actual = np.asarray(many.snapshots[c])
        delta = np.max(np.abs(actual.astype(np.float64) - expected), axis=(1, 2, 3))
        bound = 1.2e-7 * np.cumsum(np.max(np.abs(expected), axis=(1, 2, 3)), dtype=np.float64)
        snapshots[c] = dict(delta=delta.tolist(), law_c_bound=bound.tolist())
    record('poles', dict(devices=n_devices, boundary='cpml', medium='Debye'), dict(dfts=values, snapshots=snapshots))
    assert all(v['max_abs'] <= 1e-4*v['peak'] for v in values.values()), ('pole DFT bar', values)
    assert all(np.all(np.asarray(v['delta']) <= v['law_c_bound']) for v in snapshots.values()), ('pole snapshot law C', snapshots)


@pytest.mark.parametrize('mutation', ['double_dft', 'snapshot_late'])
def test_pole_mutations(mutation, monkeypatch):
    from rfx.runners._distributed_records import PlaneRecords, SnapshotRecords
    if mutation == 'double_dft':
        original = PlaneRecords.assemble
        def wrong(self, buffer):
            result = original(self, buffer)
            return tuple(p._replace(accumulator=2*p.accumulator) for p in result) if not self.flux else result
        monkeypatch.setattr(PlaneRecords, 'assemble', wrong)
    else:
        original = SnapshotRecords.update
        monkeypatch.setattr(SnapshotRecords, 'update', lambda self, buffer, state, dt, step, **kw:
            original(self, buffer, state, dt, step - 1, **kw))
    with pytest.raises(AssertionError, match='pole (DFT bar|snapshot law C)') as failure:
        test_debye_cpml(2, monkeypatch)
    mutation_result('poles', mutation, failure.value)
