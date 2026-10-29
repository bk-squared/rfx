"""Public calls own diagnostic contexts only until return or refusal (#1432).

These are lifetime checks, not RF accuracy checks: short records are deliberate.
The conductor object on a returned record belongs to its caller (Part B).
"""
import copy
from dataclasses import fields, is_dataclass
import gc
import importlib
import inspect
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.preflight.call_scope import PUBLIC_CALLS, call_scope


MODULE_CALLS = (
    'topology.topology_optimize', 'vmap_sweep.vmap_material_sweep',
    'optimize.optimize', 'optimize.gradient_check', 'profiling.profile_forward',
    'ntff_sweep.ntff_sweep', 'mesh_planner.mesh_intelligence_report',
    'mesh_planner.plan_simulation_mesh', 'artifacts.build_runtime_report',
    'artifacts.export_artifact_bundle', 'fidelity.fidelity_report',
)


@pytest.fixture(autouse=True)
def cpu():
    with jax.default_device(jax.devices('cpu')[0]):
        yield


@pytest.fixture
def contexts(monkeypatch):
    """Track audit constructions AND assembly-backed views, retaining neither."""
    from rfx.preflight import realization
    refs = {'audit': [], 'view': []}
    init = realization._CampaignStaticsContext.__init__
    from_conductors = realization.context_from_conductors

    def register_init(self, *args, **kwargs):
        refs['audit'].append(weakref.ref(self))
        init(self, *args, **kwargs)

    def register_view(*args, **kwargs):
        ctx = from_conductors(*args, **kwargs)
        refs['view'].append(weakref.ref(ctx))
        return ctx

    monkeypatch.setattr(realization._CampaignStaticsContext, '__init__', register_init)
    monkeypatch.setattr(realization, 'context_from_conductors', register_view)
    return refs


def _released(sim, refs):
    assert sim._pf_campaign_ctx is None
    assert sim._realized_geometry_record is None
    assert sim._coax_geometry is None
    gc.collect()
    assert all(ref() is None for group in refs.values() for ref in group)


def _model(*, graded=False, source=True, boundary='pec'):
    kw = {'dz_profile': np.full(8, .001)} if graded else {}
    sim = Simulation(10e9, (.008,) * 3, dx=.001, boundary=boundary, **kw)
    sim.add_material('design', eps_r=2.)
    sim.add(Box((.002, .002, .002), (.006, .006, .006)), material='design')
    if source:
        sim.add_source((.003, .003, .003), 'ez', amplitude_kind='field')
        sim.add_probe((.004, .003, .003), 'ez')
    return sim


def _msl(*, mixed=False):
    # The thru contract's aligned strip, with one drive and an absorbing end.
    sim = Simulation(20e9, (.012, .008, .0032), dx=.0002,
                     boundary='cpml', cpml_layers=8, snap='declared')
    sim.add_material('sub', eps_r=2.2)
    sim.add(Box((0, 0, 0), (.012, .008, .0008)), material='sub')
    sim.add(Box((0, 0, 0), (.012, .008, 0)), material='pec')
    sim.add(Box((.002, .0034, .0008), (.010 if mixed else .012, .0046, .0008)), material='pec')
    if mixed:
        sim.add_port((.002, .004, 0), component='ez', impedance=50.,
                     extent=.0008, terminates=2)
        position, direction = (.010, .004, 0), '-x'
    else:
        position, direction = (.002, .004, 0), '+x'
    sim.add_msl_port(position, width=.0012, height=.0008, direction=direction,
                     impedance=50., eps_r_sub=2.2, n_probe_offset=10,
                     n_probe_spacing=4, n_probes=3)
    return sim


def _coax():
    sim = Simulation(40e9, (.008, .008, .040), dx=.0004,
                     boundary='cpml', cpml_layers=4)
    sim.add_coaxial_port((.004, .004, .020), face='top', pin_length=.005,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    return sim


def _objective(result):
    return jnp.sum(result.time_series ** 2)


def _success(name, tmp_path):
    """Return the model and a real public invocation; no stepping substitutes."""
    if name in ('compute_msl_s_matrix', 'compute_s_matrix', 'compute_mixed_s_matrix'):
        sim = _msl(mixed=name == 'compute_mixed_s_matrix')
        kw = dict(n_steps=24, freqs=np.array([4e9, 6e9, 8e9]))
        if name == 'compute_mixed_s_matrix':
            kw.update(skip_preflight=True, enforce_passivity=False, magnitude_channel='wave')
        return sim, lambda: getattr(sim, name)(**kw)
    if name == 'compute_waveguide_s_matrix':
        from tests._pec_short_advisory_fixture import build
        sim = build(np.array([4e9, 5e9, 6e9]), dx=.002, cpml=2,
                    boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'})
        return sim, lambda: sim.compute_waveguide_s_matrix(n_steps=24, normalize=False)
    if name in ('compute_coaxial_line_reflection', 'compute_coaxial_two_port'):
        sim = _coax()
        # The pulse reaches the probe planes after a few hundred steps; fewer leaves them at zero.
        return sim, lambda: getattr(sim, name)(n_steps=600, freqs=np.array([6e9, 8e9, 10e9]),
            probe_count=3, probe_start_cells=4, probe_spacing_cells=2)
    if name == 'compute_coax_msl_transition':
        from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs
        sim = build_instrument_junction()
        return sim, lambda: sim.compute_coax_msl_transition(**instrument_kwargs(200))

    sim = _model()
    if name in ('run', 'forward'):
        return sim, lambda: getattr(sim, name)(n_steps=2)
    if name == 'preflight':
        return sim, lambda: sim.preflight()
    if name == 'preflight_sparameters':
        return sim, lambda: sim.preflight_sparameters(calculator='run')
    if name == 'realized_geometry':
        return sim, lambda: sim.realized_geometry()
    if name in ('estimate_ad_memory', 'explain_ad_memory'):
        return sim, lambda: getattr(sim, name)(n_steps=2)
    if name in ('plan_ad_memory', 'ad_memory_preflight'):
        return sim, lambda: getattr(sim, name)(n_steps=2, available_memory_gb=1.)
    if name in ('mesh_intelligence_report', 'plan_mesh', 'artifact_report'):
        return sim, lambda: getattr(sim, name)()
    if name == 'export_artifact_bundle':
        return sim, lambda: sim.export_artifact_bundle(tmp_path / name)
    if name == 'fidelity_report':
        return sim, lambda: sim.fidelity_report(print_report=False)

    module, fn_name = name.split('.')
    fn = getattr(importlib.import_module('rfx.' + module), fn_name)
    if module == 'mesh_planner' or fn_name == 'build_runtime_report':
        return sim, lambda: fn(sim=sim)
    if fn_name == 'export_artifact_bundle':
        return sim, lambda: fn(tmp_path / name, sim)
    if fn_name == 'fidelity_report':
        return sim, lambda: fn(sim=sim, print_report=False)
    if fn_name == 'profile_forward':
        return sim, lambda: fn(sim, n_steps=2, warmup_trace=False)
    if fn_name == 'gradient_check':
        return sim, lambda: fn(sim, jnp.array(0.), _objective, n_steps=2)
    if fn_name == 'optimize':
        from rfx.optimize import DesignRegion
        region = DesignRegion((.002, .002, .002), (.006, .006, .006))
        return sim, lambda: fn(sim, region, _objective, n_iters=1, n_steps=2, verbose=False, jit=False)
    if fn_name == 'topology_optimize':
        pytest.importorskip('optax', reason='topology success requires the optional optimization extra')
        from rfx.topology import TopologyDesignRegion
        region = TopologyDesignRegion((.002, .002, .002), (.006, .006, .006), material_fg='design')
        # Initialization is a successful public call with preflight; no FDTD loop.
        return sim, lambda: fn(sim, region, _objective, n_iterations=0, verbose=False, jit=False)
    if fn_name == 'vmap_material_sweep':
        sim = _model(graded=True)  # Exercise the copy/run fallback that reaches the memo.
        return sim, lambda: fn(sim, 'design.eps_r', [2., 3.], n_steps=2)
    if fn_name == 'ntff_sweep':
        sim = _model(boundary='cpml')  # NTFF refuses a PEC-walled box.
        sim.add_ntff_box((.001, .001, .001), (.007, .007, .007), freqs=[4e9])
        return sim, lambda: fn(sim, [4e9, 5e9], batch_size=1,
                              run_kwargs={'n_steps': 2, 'compute_s_params': False})
    raise AssertionError(f'Missing success row: {name}')


# These diagnostic APIs return findings for invalid geometry; they have no
# strict/refusal mode. Do not invent an argument-binding error as a physics row.
_NO_REFUSAL = {
    'realized_geometry', 'fidelity_report', 'fidelity.fidelity_report',
    'artifact_report', 'artifacts.build_runtime_report',
}


def _refusal(name, tmp_path):
    if name in _NO_REFUSAL:
        pytest.skip('diagnostic API returns findings; no concrete pre-stepping refusal mode')
    sim = _model(source=False)
    if name in ('run', 'forward', 'preflight', 'optimize.gradient_check', 'profiling.profile_forward'):
        # A sub-cell PEC volume is refused by assembly before any step.
        sim.add(Box((.002, .002, .002), (.0021, .004, .004)), material='pec')
    if name in ('run', 'forward'):
        return sim, lambda: getattr(sim, name)(n_steps=2), ValueError
    if name == 'preflight':
        return sim, lambda: sim.preflight(strict=True), ValueError
    if name == 'preflight_sparameters':
        return sim, lambda: sim.preflight_sparameters(calculator='invalid'), ValueError
    if name.startswith('compute_'):
        kw = {'junction_x': .001} if name == 'compute_coax_msl_transition' else {}
        return sim, lambda: getattr(sim, name)(**kw), (ValueError, NotImplementedError)
    if name == 'estimate_ad_memory':
        from rfx.api._ad_memory_types import ADMemoryUnavailable
        sim = _model(graded=True)
        return sim, lambda: sim.estimate_ad_memory(n_steps=2, compiled=True), ADMemoryUnavailable
    if name == 'explain_ad_memory':
        return sim, lambda: sim.explain_ad_memory(n_steps=-1), ValueError
    if name in ('plan_ad_memory', 'ad_memory_preflight'):
        return sim, lambda: getattr(sim, name)(n_steps=2, available_memory_gb=-1), ValueError
    if name in ('mesh_intelligence_report', 'plan_mesh'):
        return sim, lambda: getattr(sim, name)(n_steps=-1), ValueError
    if name in ('export_artifact_bundle', 'artifacts.export_artifact_bundle'):
        path = tmp_path / 'file'
        path.write_text('not a directory')
        if '.' not in name:
            return sim, lambda: sim.export_artifact_bundle(path), FileExistsError
        from rfx.artifacts import export_artifact_bundle
        return sim, lambda: export_artifact_bundle(path, sim=sim), FileExistsError
    module, fn_name = name.split('.')
    fn = getattr(importlib.import_module('rfx.' + module), fn_name)
    if module == 'mesh_planner':
        return sim, lambda: fn(sim, n_steps=-1), ValueError
    if fn_name == 'topology_optimize':
        sim = _model(graded=True)
        return sim, lambda: fn(sim, None, None), NotImplementedError
    if fn_name == 'optimize':
        return sim, lambda: fn(sim, None, None, n_starts=0), ValueError
    if fn_name == 'gradient_check':
        return sim, lambda: fn(sim, jnp.array(0.), _objective, n_steps=2), ValueError
    if fn_name == 'profile_forward':
        return sim, lambda: fn(sim, n_steps=2), ValueError
    if fn_name == 'vmap_material_sweep':
        return sim, lambda: fn(sim, 'eps_r', [1.], n_steps=2, return_fields=True), ValueError
    if fn_name == 'ntff_sweep':
        return sim, lambda: fn(sim, [4e9]), ValueError
    raise AssertionError(f'Missing refusal row: {name}')


@pytest.mark.parametrize('name', (*PUBLIC_CALLS, *MODULE_CALLS))
@pytest.mark.parametrize('refused', [False, True], ids=['return', 'refuse'])
def test_call_releases_all_contexts(name, refused, tmp_path, contexts):
    if refused:
        sim, call, error = _refusal(name, tmp_path)
        # Do not retain ExceptionInfo/traceback after the call has closed.
        with pytest.raises(error):
            call()
    else:
        sim, call = _success(name, tmp_path)
        result = call()
        del result
    del call
    _released(sim, contexts)


def test_compiled_estimate_success_requires_gpu():
    pytest.skip('compiled=True success is GPU-only; compiled=False CPU success is in the matrix')


def _assert_host(value):
    from rfx.boundaries.pec import SheetSpec
    if isinstance(value, SheetSpec):
        return  # Its device footprint is the conductor store, Part B's (#1432); Part A does not change SheetSpec.
    assert not isinstance(value, jax.Array)
    if is_dataclass(value):
        for field in fields(value):
            if field.name != 'conductors':
                _assert_host(getattr(value, field.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_host(key)
            _assert_host(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_host(item)


@pytest.mark.parametrize('graded', [False, True])
def test_returned_record_is_host_metadata(graded, contexts):
    sim = _model(graded=graded)
    sim.add(Box((.002, .002, .005), (.006, .006, .005)), material='pec')
    record = sim.realized_geometry()
    _assert_host(record)  # _conductors is outside the dataclass schema (Part B).
    assert record.conductors is not None
    del record
    _released(sim, contexts)


def test_one_port_msl_builds_one_campaign_context(tmp_path, contexts):
    sim, call = _success('compute_msl_s_matrix', tmp_path)
    result = call()
    # Views borrow an existing assembly; only __init__ builds a campaign.
    assert len(contexts['audit']) == 1
    assert contexts['view'], 'also exercise the object.__new__ constructor'
    del result, call
    _released(sim, contexts)


def test_wrappers_preserve_api_and_subclass_hook(contexts):
    for name in PUBLIC_CALLS:
        fn = getattr(Simulation, name)
        assert not isinstance(inspect.getattr_static(Simulation, name), (staticmethod, classmethod))
        assert fn._rfx_public_call
        assert fn.__qualname__ == f'Simulation.{name}'
        assert inspect.signature(fn) == inspect.signature(fn.__wrapped__)
        assert fn.__doc__ == fn.__wrapped__.__doc__
    for path in MODULE_CALLS:
        module, name = path.split('.')
        fn = getattr(importlib.import_module('rfx.' + module), name)
        assert fn._rfx_public_call
        assert inspect.signature(fn) == inspect.signature(fn.__wrapped__)
    chained = []

    class NextHook:
        def __init_subclass__(cls, **kwargs):
            chained.append(cls.__name__)
            super().__init_subclass__(**kwargs)

    class Derived(Simulation, NextHook):
        def realized_geometry(self):
            return super().realized_geometry()

    assert chained == ['Derived']
    assert Derived.run is Simulation.run
    assert Derived.realized_geometry._rfx_public_call
    assert Derived.realized_geometry.__qualname__ == f'{Derived.__qualname__}.realized_geometry'
    derived = Derived(10e9, (.004,) * 3, dx=.001, boundary='pec')
    record = derived.realized_geometry()
    del record
    _released(derived, contexts)
    sim = _model()
    with pytest.raises(TypeError, match=r'Simulation\.run'):
        sim.run(n_stepss=2)
    assert sim._pf_campaign_ctx is sim._realized_geometry_record is sim._coax_geometry is None


@pytest.mark.parametrize('raises', [False, True])
def test_nested_scope_releases_only_on_outer_exit(raises, contexts):
    sim = _model()

    def outer():
        with call_scope(sim):
            first = sim.realized_geometry()
            assert sim.realized_geometry() is first
            assert sim._pf_campaign_ctx is not None
            assert sim._realized_geometry_record[1] is first
            # Release must clear all three, even if this lane makes no stamp.
            sim._coax_geometry = (sim._pf_campaign_ctx[1], first.conductors)
            if raises:
                raise ValueError('after nested calls')

    if raises:
        with pytest.raises(ValueError, match='after nested calls'):
            outer()
    else:
        outer()
    _released(sim, contexts)


@pytest.mark.parametrize('copier', [copy.copy, copy.deepcopy])
def test_copied_model_starts_closed(copier, contexts):
    sim = _model()
    with call_scope(sim):
        record = sim.realized_geometry()
        other = copier(sim)
        inherited = other._pf_campaign_ctx
        first = other._campaign_ctx()
        second = other._campaign_ctx()
        assert first is not second
        assert first is not inherited[1]
        assert other._pf_campaign_ctx is inherited
        assert '_call_scope_depth' not in vars(other)
        del first, second, other, inherited, record
    _released(sim, contexts)


def test_outside_scope_ignores_stale_memos_and_never_stores(contexts):
    from rfx.preflight.realization import _campaign_ctx
    from rfx.realized_geometry import realized_geometry
    sim = _model()
    first = _campaign_ctx(sim)
    second = _campaign_ctx(sim)
    assert first is not second
    assert getattr(sim, '_pf_campaign_ctx', None) is None
    del first, second
    # Poison all reads; an out-of-scope read must not index these sentinels.
    stale = object()
    sim._pf_campaign_ctx = sim._realized_geometry_record = sim._coax_geometry = stale
    first = realized_geometry(sim)
    second = realized_geometry(sim)
    assert first is not second
    assert sim._pf_campaign_ctx is stale
    assert sim._realized_geometry_record is stale
    assert sim._coax_geometry is stale
    import warnings
    sim._validate_cfg_unresolved_pulse(warnings, .001)
    del first, second
    gc.collect()
    assert all(ref() is None for group in contexts.values() for ref in group)


def test_coax_stamp_is_not_stored_outside_a_call(contexts):
    from rfx.model.conductors import coax_kernel_conductors
    from rfx.sources.coaxial_port import stamp_coaxial_line
    from rfx.sparams.coax import _coax_pec_edge_masks
    sim = _coax()
    grid = sim._build_grid()
    entities = []
    materials, _, cells = stamp_coaxial_line(grid, sim._build_materials(grid)[0],
        center_xy=(.004, .004), z_lo_index=grid.pad_z_lo + 2,
        z_hi_index=grid.pad_z_lo + 12, realized_entities=entities)
    root = coax_kernel_conductors(sim, grid, materials, cells,
                                 _coax_pec_edge_masks(cells), entities)
    assert sim._coax_geometry is None
    assert sim._realized_geometry_record is None
    assert getattr(sim, '_pf_campaign_ctx', None) is None
    assert contexts['audit']
    del root, materials, cells, entities, grid
    gc.collect()
    assert all(ref() is None for group in contexts.values() for ref in group)
