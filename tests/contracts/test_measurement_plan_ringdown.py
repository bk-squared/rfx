"""Ring-down probe transport and complete-result identity."""
OLD_RingdownRun = 'def run(self, call):\n    """Attach the probes, call the lane\'s runner, detach, complete."""\n    from rfx.api._spec import _ProbeEntry\n\n    _check_run_identification_memory(self)\n    probes = self.sim._probes\n    n_user = len(probes)\n    # rfx attaches port channels; identification.py refuses pad probes earlier, and the port entry judges its position with the user\'s flag.\n    probes.extend(_ProbeEntry(position=_node_position(self.grid, idx),\n                              component=comp, allow_in_absorber=True)\n                  for comp, idx in self.probe_keys)\n    try:\n        kwargs = {"keep_wire_port_sparams": True} if self.lane == "uniform" else {}\n        result = call(**kwargs)\n    finally:\n        del probes[n_user:]\n    return self._finish(result, n_user)'

OLD_RingdownStop = 'def run(self, call):\n    """Attach the probes, call the lane\'s runner with the stop, detach, complete."""\n    from rfx.api._spec import _ProbeEntry\n\n    probes = self.sim._probes\n    n_user = len(probes)\n    self.n_user = n_user\n    probes.extend(_ProbeEntry(position=_node_position(self.grid, idx),\n                              component=comp, allow_in_absorber=True)\n                  for comp, idx in self.base.probe_keys)\n    try:\n        kwargs = {"keep_wire_port_sparams": True} if self.lane == "uniform" else {}\n        result = call(stop_fn=self.after_chunk, stop_interval=self.chunk,\n                      **kwargs)\n    finally:\n        del probes[n_user:]\n    n_rec = int(np.shape(result.time_series)[0])\n    fin = (self.base if n_rec == self.n_max else\n           RingdownRun(self.sim, self.spec, lane=self.lane, n_steps=n_rec,\n                       grid=self.grid, warn_identification=False))\n    if fin.probe_keys != self.base.probe_keys:\n        raise RuntimeError(\n            "run(until_identified=True): the port-channel probes planned for "\n            f"{n_rec} steps differ from those the run carried.")\n    out = fin._finish(result, n_user)\n    return out._replace(ringdown=out.ringdown._replace(stop=self._report(n_rec)))'

OLD_RingdownForward = 'def run(self, call):\n    """Attach the probes, call the lane, detach, complete (traced)."""\n    from rfx.api._spec import _ProbeEntry\n\n    _check_run_identification_memory(self)\n    probes = self.sim._probes\n    n_declared = len(probes)\n    if self.lane == "uniform" and n_declared == 0:\n        # forward()\'s uniform lane records every port\'s own cell when the\n        # model declares no probe; those columns stay first, as they were.\n        probes.extend(_ProbeEntry(position=pe.position, component=pe.component,\n                                  allow_in_absorber=pe.allow_in_absorber)\n                      for pe in self.sim._ports)\n    n_user = len(probes)\n    probes.extend(_ProbeEntry(position=_node_position(self.grid, idx),\n                              component=comp, allow_in_absorber=True)\n                  for comp, idx in self.probe_keys)\n    try:\n        result = call()\n    finally:\n        del probes[n_declared:]\n    self.sim = None          # the plan outlives the call inside the result\n    return self._finish_forward(result, n_user)'

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, ringdown as rd
from rfx._internal._grid_metric import field_index
from rfx.measurement import prepared
from rfx.measurement.identification import read_ringdown_probes, ringdown_probes
from rfx.measurement.projector import plan_tables
from rfx.ringdown._pole_output import RingdownForwardResult
from tests.unit.sparams.test_ringdown_run import _box, DXP, DYP, DZP, FREQS, MM, PULSE

I = (((6.0e-3, 5.0e-3, 2.0e-3), 'ez'), ((6.2e-3, 5.3e-3, 2.3e-3), 'hx'))
ENTRIES = ('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8')


@contextmanager
def old_shape():
    """Use the three run bodies copied verbatim from main f1f6d1c44."""
    from contextlib import ExitStack
    with ExitStack() as stack:
        for cls, source in ((rd.RingdownRun, OLD_RingdownRun),
                            (rd.RingdownStop, OLD_RingdownStop),
                            (rd.RingdownForward, OLD_RingdownForward)):
            namespace = dict(vars(rd))
            exec(source, namespace)
            stack.enter_context(patch.object(cls, 'run', namespace['run']))
        yield


def exact(a, b, path='result'):
    """Compare all declared fields, including floating point representations."""
    if isinstance(a, (np.ndarray, jax.Array, np.generic, float, complex)):
        x, y = np.asarray(a), np.asarray(b)
        assert x.dtype == y.dtype, (path, x.dtype, y.dtype)
        assert x.shape == y.shape, (path, x.shape, y.shape)
        if x.tobytes() != y.tobytes():
            differing = np.flatnonzero(x.reshape(-1).view(np.uint8) != y.reshape(-1).view(np.uint8))
            byte = int(differing[0])
            index = np.unravel_index(byte // x.dtype.itemsize, x.shape) if x.shape else ()
            pytest.fail(f'{path}: {type(a).__name__}, first differing index {index}: '
                        f'{x[index]!r} != {y[index]!r} (byte {byte})')
    elif isinstance(a, RingdownForwardResult):
        exact(a.report, b.report, path + '.report')
        for name in a.__slots__:
            if name not in ('_ctx', '_report'):
                exact(getattr(a, name), getattr(b, name), path + '.' + name)
    elif is_dataclass(a) or hasattr(a, '_fields'):
        names = [f.name for f in fields(a)] if is_dataclass(a) else a._fields
        assert type(a) is type(b), path
        for name in names:
            if type(a).__name__ == 'RingdownStopCheck' and name == 'seconds':
                continue
            exact(getattr(a, name), getattr(b, name), path + '.' + name)
    elif isinstance(a, Mapping):
        assert a.keys() == b.keys(), path
        for key in a:
            exact(a[key], b[key], f'{path}[{key!r}]')
    elif isinstance(a, Sequence) and not isinstance(a, (str, bytes)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f'{path}[{i}]')
    elif callable(a):
        assert (a.__module__, a.__qualname__) == (b.__module__, b.__qualname__), path
    elif type(a).__eq__ is object.__eq__ and (hasattr(a, '__dict__') or hasattr(a, '__slots__')):
        assert type(a) is type(b), path
        if hasattr(a, '__dict__'):
            exact(vars(a), vars(b), path)
        else:
            for name in a.__slots__:
                exact(getattr(a, name), getattr(b, name), path + '.' + name)
    else:
        assert a == b, (path, type(a), a, b)


def box_without_probe(lane):
    """The declared box with its add_probe line omitted."""
    kw = ({'dx_profile': DXP, 'dy_profile': DYP, 'dz_profile': DZP}
          if lane == 'graded' else {})
    sim = Simulation(freq_max=20e9, domain=(12*MM, 11*MM, 5*MM), dx=MM,
                     boundary='pec', precision='float32', stencil_order=2, **kw)
    eps_r, q_d, f_ref = 2.2, 300., 12.5e9
    sim.add_material('fill', eps_r=eps_r,
                     sigma=2*np.pi*f_ref*8.854187817e-12*eps_r/q_d)
    sim.add(Box((0, 0, 0), (12*MM, 11*MM, 5*MM)), material='fill')
    if lane == 'graded':
        pos, ext = (3.3*MM, 3.8*MM, .8*MM), .6*MM
        pos2, ext2 = (9.2*MM, 8.2*MM, .8*MM), .6*MM
    else:
        pos, ext = (2*MM, 2*MM, 0.), MM
    sim.add_port(position=pos, component='ez', impedance=50., extent=ext, waveform=PULSE)
    if lane == 'graded':
        sim.add_port(position=pos2, component='ez', impedance=50., extent=ext2,
                     waveform=PULSE, excite=False)
    return sim


def scene(entry):
    lane = 'graded' if entry in ('E2', 'E4', 'E7', 'E8') else 'uniform'
    return (box_without_probe(lane) if entry in ('E6', 'E8') else
            _box(lane, second_port=lane == 'graded')), lane


def invoke(sim, entry, *, ring=True):
    lane = 'graded' if entry in ('E2', 'E4', 'E7', 'E8') else 'uniform'
    kw = dict(n_steps=1500 if entry in ('E3', 'E4') else 600, skip_preflight=True)
    if ring:
        kw['ringdown'] = rd.RingdownSpec(identification_probes=I)
    if entry in ('E5', 'E6', 'E7', 'E8'):
        return sim.forward(**kw, port_s11_freqs=(FREQS if lane == 'uniform' else
                           np.linspace(sim._freq_max/10, sim._freq_max, 50)))
    if entry in ('E3', 'E4'):
        kw.update(until_identified=True, s_param_freqs=FREQS)
    elif lane == 'uniform':
        kw['s_param_freqs'] = FREQS
    return sim.run(**kw, compute_s_params=True)


def planned(sim, lane):
    grid = sim._build_nonuniform_grid() if lane == 'graded' else sim._build_grid()
    base = rd.RingdownRun(sim, rd.RingdownSpec(), lane=lane, n_steps=600, grid=grid)
    keys = tuple((comp, tuple(field_index(grid, position, comp))) for position, comp in I)
    assert len(set(keys)) == 2 and all(key not in base.probe_keys for key in keys)
    wrapper = rd.RingdownRun(sim, rd.RingdownSpec(identification_probes=I),
                             lane=lane, n_steps=600, grid=grid)
    assert wrapper.identification_keys == keys
    return grid, wrapper.probe_keys


def check_plan(plan, sim, grid, keys, entry):
    owners = plan.owners
    ids = [o for o in owners if o.kind == 'identification']
    assert [o.id for o in ids] == [f'ringdown:{i}' for i in range(len(keys))]
    assert min(owners.index(o) for o in ids) > max(
        (i for i, o in enumerate(owners) if o.kind == 'probe'), default=-1)
    actual = [(n.component, (n.i, n.j, n.k)) for o in ids for c in o.channels for n in c.nodes]
    assert actual == list(keys)
    leading = [(p.component, tuple(field_index(grid, p.position, p.component))) for p in sim._probes]
    if entry == 'E6':
        leading += [(p.component, tuple(field_index(grid, p.position, p.component))) for p in sim._ports]
    expected = leading + list(keys)
    table = plan_tables(plan, n_probes=len(expected), port_specs=(), dtype=np.float32)['probe_table']
    columns = [None] * table.n_channels
    for term in table.components:
        for col, idx, weight in zip(term.channel_ids, zip(*term.indices), term.weights):
            assert weight == 1.
            columns[int(col)] = (term.component, tuple(idx))
    assert columns == expected
    missing = replace(plan, owners=tuple(o for o in owners if o is not ids[-1]))
    with pytest.raises(ValueError, match='probe owners; the runner records'):
        plan_tables(missing, n_probes=len(expected), port_specs=(), dtype=np.float32)


@pytest.mark.parametrize('entry', ENTRIES)
def test_i2_r2_r3(entry):
    old_sim, _ = scene(entry)
    with old_shape():
        expected = invoke(old_sim, entry)
    sim, lane = scene(entry)
    grid, keys = planned(sim, lane)
    probes, declared = sim._probes, tuple(sim._probes)
    attached = []
    original = prepared.attach_measurement

    def spy(subject, conductors, **kw):
        assert subject._probes is probes
        assert tuple(probes) == declared
        out = original(subject, conductors, **kw)
        if subject is sim:
            attached.append(conductors._prepared_solve)
        return out

    with patch.object(prepared, 'attach_measurement', spy):
        actual = invoke(sim, entry)
    assert sim._probes is probes and tuple(probes) == declared
    assert attached
    check_plan(attached[0].measurement, sim, grid, keys, entry)
    exact(actual, expected)
    assert read_ringdown_probes(probes, 'test') == ()


def gradient_call():
    sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001, boundary='pec')
    sim.add_material('fill', eps_r=2.2, sigma=2*np.pi*12.5e9*8.854187817e-12*2.2/5000)
    sim.add(Box((0, 0, 0), (.012, .011, .005)), material='fill')
    sim.add_port(position=(.001, .001, 0), component='ez', impedance=50., extent=.00025,
                 waveform=GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5, amplitude=1.))
    eps = jnp.full(sim._build_grid().shape, 2.2, jnp.float32)

    def f(p):
        r = sim.forward(n_steps=1200, skip_preflight=True, port_s11_freqs=np.array([12.425e9]),
                        eps_override=eps*p,
                        ringdown=rd.RingdownSpec(identification_probes=(I[0],))).ringdown
        return jnp.stack([r.s_params, r.s_params_long]), r

    return jax.jit(lambda p: jax.jvp(f, (p,), (jnp.ones_like(p),), has_aux=True))(jnp.float32(1.))


def test_i2_G():
    with old_shape():
        expected = gradient_call()
    actual = gradient_call()
    exact(actual, expected)


def test_r1b():
    sim, _ = scene('E6')
    actual = invoke(sim, 'E6')
    plain, _ = scene('E6')
    expected = invoke(plain, 'E6', ring=False)
    exact(actual.realized_geometry.probes, expected.realized_geometry.probes)


@pytest.mark.parametrize('entry', ('E1', 'E2', 'E6'))
def test_r5(entry):
    sim, _ = scene(entry)
    fresh, _ = scene(entry)
    expected_plain = invoke(fresh, entry, ring=False)
    first = invoke(sim, entry, ring=False)
    actual = invoke(sim, entry)
    if entry in ('E1', 'E2'):
        exact(sim.realized_geometry(), fresh.realized_geometry())
    last = invoke(sim, entry, ring=False)
    fresh_ring, _ = scene(entry)
    expected_ring = invoke(fresh_ring, entry)
    exact(first, expected_plain)
    exact(last, expected_plain)
    exact(actual, expected_ring)
    if entry in ('E1', 'E2'):
        old_sim, _ = scene(entry)
        with old_shape():
            invoke(old_sim, entry)
        print(f'{entry} Old post-run geometry: {old_sim.realized_geometry()!r}')


@pytest.mark.parametrize('entry', ('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'))
def test_r2_after_raise(entry):
    sim, _ = scene(entry)
    probes, declared = sim._probes, tuple(sim._probes)
    original = prepared.attach_measurement

    def fail(subject, conductors, **kw):
        original(subject, conductors, **kw)
        assert subject._probes is probes and tuple(probes) == declared
        raise LookupError('after attach')

    with patch.object(prepared, 'attach_measurement', fail):
        with pytest.raises(LookupError, match='after attach'):
            invoke(sim, entry)
    assert sim._probes is probes and tuple(probes) == declared
    assert read_ringdown_probes(probes, 'test') == ()


def test_r7_a():
    import rfx.runners.nonuniform as lane
    original = lane.run_nonuniform_path

    def copy_list(sim, **kw):
        if kw.get('_drive_scan'):
            sim._probes = list(sim._probes)
        return original(sim, **kw)

    sim, _ = scene('E2')
    with patch.object(lane, 'run_nonuniform_path', copy_list):
        with pytest.raises(RuntimeError, match='RingdownRun.run.*attach_measurement'):
            invoke(sim, 'E2')


@pytest.mark.parametrize('reader', ('attach_measurement', 'runners/uniform.py',
                                    'runners/nonuniform.py', 'runners/uniform_forward.py'))
@pytest.mark.parametrize('entry_point', ('RingdownRun.run', 'RingdownStop.run', 'RingdownForward.run'))
def test_r7_b(reader, entry_point):
    from rfx.api._spec import _ProbeEntry
    from rfx.ringdown import _node_position
    sim, lane = scene('E1')
    grid, keys = planned(sim, lane)
    probes = sim._probes
    entries = tuple(_ProbeEntry(position=_node_position(grid, idx), component=comp,
                                allow_in_absorber=True) for comp, idx in keys)
    with ringdown_probes(probes, entries, entry_point):
        with pytest.raises(RuntimeError, match=entry_point + '.*' + reader):
            read_ringdown_probes(list(probes), reader)


def test_r8():
    from rfx.stepping import prepared as step
    original_attach, original_advance = prepared.attach_measurement, step.advance

    def record(sim):
        attached, sequences = [], []

        def attach(subject, conductors, **kw):
            attached.append(subject._probes)
            return original_attach(subject, conductors, **kw)

        def advance(product, stepper, *args, **kw):
            sequences.append(tuple(tuple(p) for p in kw['probes']))
            return original_advance(product, stepper, *args, **kw)

        with patch.object(prepared, 'attach_measurement', attach), patch.object(step, 'advance', advance):
            invoke(sim, 'E2')
        return attached, sequences

    old_sim, lane = scene('E2')
    grid, keys = planned(old_sim, lane)
    expected_sequence = tuple((*field_index(grid, p.position, p.component), p.component)
                              for p in old_sim._probes) + tuple((*idx, comp) for comp, idx in keys)
    with old_shape():
        old_calls, expected = record(old_sim)
    assert len(old_calls) == len(expected) == 3, 'R8 start value wrong: lane count'
    assert all(expected), 'R8 start value wrong: empty sequence'
    assert all(seq == expected_sequence for seq in expected)
    sim, _ = scene('E2')
    new_calls, actual = record(sim)
    assert all(p is sim._probes for p in new_calls)
    assert len(new_calls) == len(old_calls)
    exact(actual, expected)
    print(f'R8 Old calls={len(old_calls)} lengths={[len(s) for s in expected]}; '
          f'New calls={len(new_calls)} lengths={[len(s) for s in actual]}')


@pytest.mark.parametrize('lane', ('uniform', 'graded'))
def test_identification_indices(lane):
    sim = _box(lane, second_port=lane == 'graded')
    grid, _ = planned(sim, lane)
    print(lane, [(comp, field_index(grid, pos, comp)) for pos, comp in I])
