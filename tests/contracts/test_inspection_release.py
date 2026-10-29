"""Inspection products are shared until a solve or explicit release (#1432).

Lifetime checks use tiny CPU solves, not RF accuracy assertions. Every
constructor used by audit contexts and assembly-backed views is observed
without retaining its products. Run the two-device rows in a fresh process
with XLA_FLAGS=--xla_force_host_platform_device_count=2.
"""
import gc
import weakref

import jax
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation


COAX = ('compute_coaxial_line_reflection', 'compute_coaxial_two_port',
        'compute_coax_msl_transition')
CALCULATORS = ('compute_msl_s_matrix', 'compute_waveguide_s_matrix', *COAX,
               'compute_mixed_s_matrix')
# Public compute_s_matrix() refuses lumped ports; run()/forward() below
# exercise the supported lumped S path instead (Amendment 8, #1432).
MEMOS = ('_pf_campaign_ctx', '_realized_geometry_record', '_coax_geometry')


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
    gc.collect()
    for name in MEMOS:
        assert getattr(sim, name) is None, name
    assert refs['audit'], 'the row must exercise an inspection context'
    assert all(ref() is None for group in refs.values() for ref in group)


def _prime(sim):
    # A solve must release a record kept by a prior inspection as well as
    # contexts it creates itself. Do not retain the returned record here.
    sim.realized_geometry()
    assert sim._pf_campaign_ctx is not None
    assert sim._realized_geometry_record is not None


def _model(*, graded=False, source=True, boundary='pec'):
    kw = {'dz_profile': np.full(8, .001)} if graded else {}
    sim = Simulation(10e9, (.008,) * 3, dx=.001, boundary=boundary,
                     cpml_layers=2, **kw)
    sim.add_material('design', eps_r=2.)
    sim.add(Box((.002, .002, .002), (.006, .006, .006)), material='design')
    if source:
        sim.add_source((.003, .003, .003), 'ez', amplitude_kind='field')
        sim.add_probe((.004, .003, .003), 'ez')
    return sim


def _msl(*, mixed=False):
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


def _calculator(name):
    if name in ('compute_msl_s_matrix', 'compute_mixed_s_matrix'):
        sim = _msl(mixed=name == 'compute_mixed_s_matrix')
        kw = dict(n_steps=24, freqs=np.array([4e9, 6e9, 8e9]))
        if name == 'compute_mixed_s_matrix':
            kw.update(skip_preflight=True, enforce_passivity=False, magnitude_channel='wave')
        return sim, kw
    if name == 'compute_waveguide_s_matrix':
        from tests._pec_short_advisory_fixture import build
        sim = build(np.array([4e9, 5e9, 6e9]), dx=.002, cpml=2,
                    boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'})
        return sim, dict(n_steps=24, normalize=False)
    if name in COAX[:2]:
        # Fewer steps leave the probe planes at zero before pulse arrival.
        return _coax(), dict(n_steps=600, freqs=np.array([6e9, 8e9, 10e9]),
                            probe_count=3, probe_start_cells=4, probe_spacing_cells=2)
    if name == 'compute_coax_msl_transition':
        from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs
        return build_instrument_junction(), instrument_kwargs(200)
    raise AssertionError(name)


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('lane', ['uniform', 'graded', 'uniform_two', 'graded_two', 'subgridded'])
def test_solve_releases_inspection(entry, lane, contexts):
    kwargs = dict(n_steps=2)
    if lane.endswith('_two'):
        devices = jax.devices('cpu')
        if len(devices) < 2:
            pytest.skip('requires two CPU devices; set XLA_FLAGS=--xla_force_host_platform_device_count=2 before importing JAX')
        kwargs['devices'] = devices[:2]
        if entry == 'forward':
            kwargs['distributed'] = True
    if lane == 'subgridded':
        if entry == 'forward':
            pytest.skip('forward() refuses refinement; there is no subgridded forward lane on CPU or GPU')
        # Same CPU refinement as test_subgrid_requires_experimental.
        sim = Simulation(10e9, (.008, .008, .016), dx=.002, boundary='pec')
        sim.add_refinement((0., .012), ratio=2, validation='research')
        sim.add_source((.004, .004, .004), 'ez', amplitude_kind='field')
        sim.add_probe((.004, .004, .006), 'ez')
    else:
        sim = _model(graded=lane.startswith('graded'))
    _prime(sim)
    result = getattr(sim, entry)(**kwargs)
    del result
    _released(sim, contexts)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_lumped_solve_releases_inspection(entry, contexts):
    sim = _model(source=False)
    sim.add_port((.003, .003, .003), 'ez', impedance=50.,
                 waveform=GaussianPulse(f0=6e9, bandwidth=1.2))
    freqs = np.array([4e9, 6e9, 8e9])
    kwargs = (dict(compute_s_params=True, s_param_freqs=freqs) if entry == 'run'
              else dict(port_s11_freqs=freqs))
    _prime(sim)
    result = getattr(sim, entry)(n_steps=24, **kwargs)
    assert result.s_params is not None
    del result
    _released(sim, contexts)


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('graded', [False, True])
def test_ntff_solve_releases_inspection(entry, graded, contexts):
    sim = _model(graded=graded, boundary='cpml')
    # A conductor inside the box exercises the NTFF realized-geometry reader.
    sim.add(Box((.002, .002, .005), (.006, .006, .005)), material='pec')
    sim.add_ntff_box((.001, .001, .001), (.007, .007, .007), freqs=[4e9])
    _prime(sim)
    # The box sits closer than lambda/2 to the sheet, which preflight refuses; this checks lifetime, not the far field.
    result = getattr(sim, entry)(n_steps=2, skip_preflight=True)
    del result
    _released(sim, contexts)


@pytest.mark.parametrize('name', CALCULATORS)
def test_calculator_releases_inspection(name, contexts):
    sim, kwargs = _calculator(name)
    _prime(sim)
    result = getattr(sim, name)(**kwargs)
    del result
    _released(sim, contexts)
    if name in COAX:
        record = sim.realized_geometry()
        assert record.lane != 'coax'
        assert not any(e.provenance == 'coax stamp' for e in record.entities)
        del record
        sim.clear_inspection_cache()
        _released(sim, contexts)


def test_calculator_refusal_keeps_inspection(contexts):
    sim = _model(source=False)
    _prime(sim)
    cached = tuple(getattr(sim, name, None) for name in MEMOS)
    with pytest.raises(ValueError, match='requires.*cpml'):
        sim.compute_coaxial_line_reflection(n_steps=2)
    assert all(getattr(sim, name, None) is value for name, value in zip(MEMOS, cached))
    assert contexts['audit'][0]() is cached[0][1]
    del cached
    sim.clear_inspection_cache()
    _released(sim, contexts)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('entry', ['preflight', 'realized_geometry'])
def test_inspection_reuses_until_explicit_release(entry, graded, contexts, monkeypatch):
    sim = _model(graded=graded)
    assemblies = []
    assemble = sim._assemble_realized

    def count(*args, **kwargs):
        assemblies.append(1)
        return assemble(*args, **kwargs)

    monkeypatch.setattr(sim, '_assemble_realized', count)
    first = getattr(sim, entry)()
    ctx = sim._campaign_ctx()
    second = getattr(sim, entry)()
    assert sim._campaign_ctx() is ctx
    assert len(contexts['audit']) == len(assemblies) == 1
    assert contexts['audit'][0]() is ctx
    if entry == 'realized_geometry':
        assert first is second
    # Populate all three slots, including the coax stamp on this tiny model.
    record = sim.realized_geometry()
    sim._coax_geometry = (ctx, record.conductors)
    record_ref = weakref.ref(record)
    assert len(assemblies) == 1
    del first, second, ctx, record
    assert sim.clear_inspection_cache() is None
    _released(sim, contexts)
    assert record_ref() is None  # m2: leaving the record must fail this row.
    rebuilt = getattr(sim, entry)()
    assert len(contexts['audit']) == len(assemblies) == 2
    del rebuilt
    sim.clear_inspection_cache()
    _released(sim, contexts)


def test_dispatch_refusal_keeps_inspection(contexts):
    sim = _model()
    _prime(sim)
    cached = tuple(getattr(sim, name, None) for name in MEMOS)
    # This refusal is inside dispatch, after automatic preflight has returned.
    with pytest.raises(ValueError, match='checkpoint_segments must be an int'):
        sim.forward(n_steps=2, checkpoint_segments=(1, 1))
    assert all(getattr(sim, name, None) is value for name, value in zip(MEMOS, cached))
    assert contexts['audit'][0]() is cached[0][1]
    del cached
    sim.clear_inspection_cache()
    _released(sim, contexts)
