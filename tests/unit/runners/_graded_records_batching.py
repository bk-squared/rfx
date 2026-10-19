"""Public graded record solves used by batching and forward-AD witnesses."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding

from rfx import Simulation
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import measure, record, with_bars


def observe_placement(monkeypatch):
    """Check physical shard axes as well as numbers: a divisible wrong axis runs."""
    from rfx.stepping import slab
    original = slab._placement_map

    def check(value, spec):
        if isinstance(value, jax.core.Tracer):
            if hasattr(value, 'primal'):
                check(value.primal, spec)
                check(value.tangent, spec)
            elif hasattr(value, 'batch_dim'):
                axes = list(spec)
                if value.batch_dim is not None:
                    axes += [None] * (value.ndim - len(axes))
                    axes.insert(value.batch_dim, None)
                check(value.val, axes)
        elif isinstance(value, jax.Array) and isinstance(value.sharding, NamedSharding):
            expected = [i for i, axis in enumerate(spec) if axis == 'x']
            actual = [i for i, axis in enumerate(value.sharding.spec) if axis == 'x']
            assert actual == expected, ('batching placement axis', value.shape, actual, expected)

    def checked_map(sharding):
        apply = original(sharding)

        def checked(value):
            result = apply(value)
            check(result, sharding.spec)
            return result
        return checked
    monkeypatch.setattr(slab, '_placement_map', checked_map)


def problem(kind, boundary, monkeypatch):
    observe_placement(monkeypatch)
    # PEC witnesses compare the record kernels; the public radiation guards
    # still refuse surfaces/volumes that exclude the reflecting domain walls.
    if boundary == 'pec' and kind == 'ntff':
        monkeypatch.setattr('rfx.boundaries.features.refuse_ntff_reflecting_faces', lambda *a, **kw: None)
    if boundary == 'pec' and kind == 'moments':
        monkeypatch.setattr('rfx.current_moments.refuse_current_the_monitor_cannot_see', lambda *a, **kw: None)

    def build(amplitude):
        axis = 'z' if kind == 'moments' else 'x'
        count = 6 if axis == 'z' else 13
        sim = Simulation(freq_max=15e9, domain=(15e-3, 8e-3, 8e-3), dx=1e-3,
            boundary=boundary, cpml_layers=2,
            **{f'd{axis}_profile': np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, count), 1e-3]})
        sim.add_source((5.1e-3, 3.2e-3, 4.3e-3), 'ez', amplitude_kind='field',
            waveform=lambda t: amplitude*jnp.exp(-((t-1.2e-11)/5e-12)**2))
        if kind == 'flux':
            # Excite both tangential magnetic components; an Ez-only dipole
            # leaves Hz at cancellation noise even off its source plane.
            sim.add_source((8.1e-3, 2.4e-3, 5.1e-3), 'ex', amplitude_kind='field',
                waveform=lambda t: jnp.exp(-((t-1.2e-11)/5e-12)**2))
        if kind == 'planes':
            sim.add_dft_plane_probe(axis='y', coordinate=4.6e-3, component='ez', freqs=[3e9], name='record')
        elif kind == 'flux':
            sim.add_flux_monitor(axis='y', coordinate=4.6e-3, freqs=[3e9], name='record')
        elif kind == 'ntff':
            sim.add_ntff_box((1.7e-3, 1.3e-3, 1.2e-3), (12.1e-3, 6.1e-3, 7.1e-3), freqs=[3e9])
        else:
            sim.add_current_moment_monitor((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 7e-3),
                                           block_size=2e-3, freqs=[3e9])
        return sim

    sim = build(1.)
    eps = jnp.full(sim._build_nonuniform_grid().shape, 1.3, jnp.float32)

    def solve(alpha, distributed):
        # Moments refuse traced whole-grid eps on both lanes. Source amplitude
        # is a supported differentiable knob and exercises their whole adapter.
        model = build(alpha) if kind == 'moments' else sim
        options = {} if kind == 'moments' else dict(eps_override=eps*alpha)
        result = model.forward(n_steps=12, checkpoint=False, skip_preflight=True, **options,
            **(dict(distributed=True, devices=host_devices(2)) if distributed else {}))
        if kind == 'planes':
            return dict(plane=result.dft_planes['record'].accumulator)
        if kind == 'flux':
            return {name: getattr(result.flux_monitors['record'], name)
                    for name in ('e1_dft', 'e2_dft', 'h1_dft', 'h2_dft')}
        if kind == 'ntff':
            return {name: getattr(result.ntff_data, name) for name in result.ntff_data._fields[:6]}
        return dict(moments=result.current_moment_data[0])
    return solve


def compare(actual, expected, *, kind, boundary, transform):
    values = {name: measure(actual[name], expected[name]) for name in expected}
    # through jax.vmap/jvp the compiled program differs (cross-trace rule, rfx CLAUDE.md)
    with_bars(values, 'relative_peak')
    record('batching', dict(record=kind, boundary=boundary, transform=transform, devices=2), values)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves((actual, expected))), 'batching finite'
    assert max(v['peak'] for v in values.values()) > 0, 'batching nonzero'
    assert all(v['passed'] for v in values.values()), ('batching bar', values)


def mutate(monkeypatch, mutation):
    if mutation == 'raw_placement':
        from rfx.stepping import slab
        # Undo transformation-aware placement without bypassing the public solve.
        monkeypatch.setattr(slab, '_placement_map', lambda sharding: lambda a: jax.device_put(a, sharding))
    else:
        from rfx.runners._distributed_records import SlabRecords
        original = SlabRecords.assemble

        def doubled(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            for kind, value in result.items():
                if kind in ('dft_planes', 'flux_monitors', 'ntff_data', 'current_moment_data') and value is not None:
                    result[kind] = jax.tree.map(lambda a: 2*a if hasattr(a, 'dtype') else a, value)
            return result
        monkeypatch.setattr(SlabRecords, 'assemble', doubled)
