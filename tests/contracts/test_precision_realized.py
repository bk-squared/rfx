"""Tracker 1567: field storage realizes the request, or admission refuses.

Materials, coefficients and intermediate spectra retain main's precision.
Only the changed final lumped/wire S casts promise complex128 here.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
from rfx.sources.sources import GaussianPulse
from rfx.sweep import parametric_sweep

FREQS = np.array([2e9, 3e9 + 100, 4e9])
HONOURS = ('run_decay', 'run_lumped', 'run_wire', 'run_wire_single', 'forward',
           'driver_lumped', 'driver_wire', 'driver_refplane', 'sweep', 'batch', 'batch_manifest', 'run_dft', 'forward_dft',
           'run_ntff', 'forward_ntff', 'run_flux', 'ntff_sweep',
           'convergence', 'quick_convergence', 'gradient_check', 'adjoint',
           'compute_waveguide_s_matrix', 'compute_s_matrix', 'run_waveguide', 'forward_waveguide')
HONOURS += ('optimize', 'progressive_optimize', 'compute_msl_s_matrix',
            'compute_mixed_s_matrix', 'topology_optimize',
            'run_ringdown', 'forward_ringdown', 'run_msl', 'forward_msl',
            'run_tfsf', 'forward_tfsf', 'run_window', 'forward_window', 'vmap_sweep_fallback')
# These refusals predate #1567; preserve the original exception and message.
MAIN_REFUSALS = {
    'differentiable_material_fit': (NotImplementedError, "precision='float64'.*#1290"),
    **{name: (ValueError, "requires precision='float32'") for name in (
        'compute_coaxial_line_reflection', 'compute_coaxial_two_port', 'compute_coax_msl_transition')},
    **{name: (NotImplementedError, "precision|float32") for name in (
        'vmap_material_sweep', 'run_adi', 'forward_adi', 'run_nonuniform',
        'forward_nonuniform', 'forward_distributed', 'run_distributed', 'driver_distributed')},
    'run_subgridded': (NotImplementedError, 'subgrid|SBP|precision'),
}
REFUSES = tuple(MAIN_REFUSALS)
# Final S conversion sites retained from this change. Other returned spectra
# are measured below, without imposing a new end-to-end precision contract.
FINAL_S = ('run_decay', 'run_lumped', 'run_wire', 'run_wire_single',
           'driver_lumped', 'driver_wire', 'driver_refplane', 'sweep', 'batch', 'batch_manifest')


def model(precision, entry):
    if entry in ('run_msl', 'forward_msl', 'compute_msl_s_matrix'):
        from tests.unit.autodiff.test_design_box_holds_port import _msl_board
        sim = _msl_board()
        sim._precision = precision
        return sim
    if entry == 'compute_mixed_s_matrix':
        from tests.unit.sparams.test_msl_mixed_spatial_collocation import _case
        sim, _ = _case('+x')
        sim._precision = precision
        sim.add_probe((.004, .0015, .00025), 'ez')
        return sim
    if entry.endswith('_ringdown'):
        from tests.unit.sparams.test_ringdown_run import _box
        return _box('uniform', precision=precision)
    if entry.startswith('compute_coax'):
        return Simulation(freq_max=40e9, domain=(.008, .008, .02), boundary='cpml', precision=precision)
    if entry in ('vmap_material_sweep', 'vmap_sweep_fallback'):
        from tests.unit.runners.test_vmap_sweep import _make_dielectric_sim
        sim = _make_dielectric_sim()
        if entry == 'vmap_sweep_fallback':
            sim.add_flux_monitor(axis='x', coordinate=.01, freqs=FREQS, name='flux')
        sim._precision = precision
        return sim
    if entry.endswith('_tfsf'):
        sim = Simulation(freq_max=8e9, domain=(.08,.006,.006), dx=.001,
                         boundary='cpml', cpml_layers=8, precision=precision)
        sim.add_tfsf_source(f0=4e9, bandwidth=.5, margin=3)
        sim.add_probe((.02,.003,.003), 'ez')
        return sim
    if 'waveguide' in entry or entry == 'compute_s_matrix':
        from tests.unit.sparams.test_waveguide_solver_precision import _guide
        sim = _guide(precision=precision)
        sim.add_probe((.045,.02,.01), 'ez')
        from dataclasses import replace
        sim._waveguide_ports = [replace(p, freqs=jnp.asarray([8e9, 9e9+100], dtype=precision))
                                for p in sim._waveguide_ports]
        if entry == 'run_waveguide':
            sim._waveguide_ports.pop()
        return sim
    if entry == 'driver_refplane':
        from tests.locks.test_refplane_port_waves import _build_thru
        sim = _build_thru(reference_plane_cells=3)
        sim._precision = precision
        return sim
    sim = Simulation(freq_max=5e9, domain=(.018, .012, .012), dx=.003,
                     boundary='pec', precision=precision)
    for x in (.006, .012)[:1 if 'single' in entry else 2]:
        sim.add_port((x, .006, .006), 'ez', impedance=50.,
                     waveform=GaussianPulse(f0=3e9, bandwidth=.8),
                     **({'extent': .003} if 'wire' in entry else {}))
    sim.add_probe((.009, .006, .006), 'ez')
    if entry.endswith('_adi'):
        sim._ports.clear()
        sim.add_source((.006,.006,.006), 'ez', amplitude_kind='field')
        sim._solver = 'adi'
    if entry.endswith('_nonuniform') or entry == 'forward_distributed':
        sim._dz_profile = np.full(4,.003)
    if entry in ('run_nonuniform','forward_nonuniform','forward_distributed'):
        sim._ports.clear()
    if entry == 'run_subgridded':
        sim._refinement = {'z_range':(.003,.009),'ratio':2,'xy_margin':0.,
                           'tau':None,'validation':None,'topology':None}
    if entry.endswith('_dft'):
        sim.add_dft_plane_probe(axis='x', coordinate=.009, component='ez',
                                freqs=FREQS, name='plane')
    if entry.endswith('_ntff') or entry == 'ntff_sweep':
        sim.add_ntff_box((.003,.003,.003),(.015,.009,.009),freqs=FREQS)
    if entry == 'run_flux':
        sim.add_flux_monitor(axis='x',coordinate=.009,freqs=FREQS,name='flux')
    if entry.endswith('_window'):
        sim.add_flux_monitor(axis='x', coordinate=.009, freqs=FREQS, dft_window='hann')
    return sim


def invoke(sim, entry):
    if entry == 'run_decay':
        return sim.run(until_decay=.01,decay_min_steps=8,decay_max_steps=16,
                       decay_check_interval=4,s_param_n_steps=16,s_param_freqs=FREQS,skip_preflight=True)
    if entry == 'optimize':
        from rfx.optimize import optimize, DesignRegion
        region = DesignRegion((.006,.003,.003), (.009,.003,.003))
        return optimize(sim, region, lambda r:jnp.sum(r.time_series**2),
                        n_iters=1, n_steps=8, verbose=False, skip_preflight=True)
    if entry == 'progressive_optimize':
        from rfx.optimize import progressive_optimize, ProgressiveStage, DesignRegion
        region = DesignRegion((.006,.003,.003), (.009,.003,.003))
        return progressive_optimize(lambda _:sim, region, lambda r:jnp.sum(r.time_series**2),
                                    [ProgressiveStage(dx=.003,n_iters=1,n_steps=8)],
                                    verbose=False, skip_preflight=True)
    if entry == 'gradient_check':
        from rfx.optimize import gradient_check
        return gradient_check(sim,np.asarray(.1,dtype=sim._precision),lambda r:jnp.sum(r.time_series**2),n_steps=8)
    if entry == 'ntff_sweep':
        from rfx.ntff_sweep import ntff_sweep
        return ntff_sweep(sim,FREQS,batch_size=2,
                          run_kwargs={'n_steps':8,'skip_preflight':True,'compute_s_params':False})
    if entry in ('convergence','quick_convergence'):
        from rfx.convergence import convergence_study,quick_convergence
        kw={'n_steps':8,'run_kwargs':{'s_param_freqs':FREQS,'skip_preflight':True}}
        if entry == 'convergence':
            return convergence_study(lambda _:sim,[.003,.002],lambda r:float(np.max(r.time_series)),**kw)
        return quick_convergence(sim,dx_factors=[1.,.75],metric='peak_field',**kw)
    if entry == 'differentiable_material_fit':
        from rfx.differentiable_material_fit import differentiable_material_fit
        return differentiable_material_fit(lambda *a: sim, np.ones((1,1,3), dtype=complex), FREQS)
    if entry == 'batch_manifest':
        from tempfile import TemporaryDirectory
        from rfx.batch import run_batch_with_manifest, ParameterSweep
        with TemporaryDirectory() as output:
            return run_batch_with_manifest(lambda **kw: sim, ParameterSweep(unused=[1.]), output,
                include_environment=False,
                run_kwargs={'n_steps':128,'s_param_freqs':FREQS,'skip_preflight':True})[0].result
    if entry == 'batch':
        from rfx.batch import run_batch, ParameterSweep
        return run_batch(lambda **kw: sim, ParameterSweep(unused=[1.]),
                         run_kwargs={'n_steps':128,'s_param_freqs':FREQS,'skip_preflight':True})[0][1]
    if entry in ('compute_waveguide_s_matrix', 'compute_s_matrix'):
        return getattr(sim, entry)(n_steps=32, normalize=False)
    if entry == 'compute_coax_msl_transition':
        return sim.compute_coax_msl_transition(junction_x=.004)
    if entry == 'compute_msl_s_matrix':
        return sim.compute_msl_s_matrix(n_steps=64, freqs=FREQS)
    if entry == 'compute_mixed_s_matrix':
        return sim.compute_mixed_s_matrix(n_steps=64, freqs=FREQS, skip_preflight=True, magnitude_channel='wave')
    if entry.startswith('compute_'):
        return getattr(sim, entry)()
    if entry in ('vmap_material_sweep', 'vmap_sweep_fallback'):
        from rfx.vmap_sweep import vmap_material_sweep
        return vmap_material_sweep(sim, 'substrate.eps_r', [1.], n_steps=8)
    if entry == 'topology_optimize':
        from rfx.topology import topology_optimize, TopologyDesignRegion
        region = TopologyDesignRegion((.006,.003,.003), (.009,.003,.003),
                                      material_bg='air', material_fg='fr4')
        return topology_optimize(sim, region, lambda r:jnp.sum(r.time_series**2),
                                 n_iterations=1, verbose=False, skip_preflight=True)
    if entry == 'driver_distributed':
        return compute_lumped_wire_s_matrix_via_scan(sim,FREQS,n_steps=8,devices=[object(),object()])
    if entry.startswith('driver_'):
        return compute_lumped_wire_s_matrix_via_scan(sim, FREQS, n_steps=32 if entry == 'driver_refplane' else 128)
    if entry == 'sweep':
        return parametric_sweep(lambda _: sim, 'unused', [1.], n_steps=128,
                                run_kwargs={'s_param_freqs': FREQS, 'skip_preflight':True})
    kw = {'n_steps':128, 'skip_preflight':True}
    if entry.endswith('_ringdown'):
        from rfx.ringdown import RingdownSpec
        kw['ringdown'] = RingdownSpec()
        kw['n_steps'] = 1500
    if entry == 'run_distributed':
        kw['devices'] = [object(),object()]
    if entry == 'forward_distributed':
        kw['distributed'] = True
    if entry.startswith('forward'):
        return sim.forward(port_s11_freqs=None if entry in ('forward_nonuniform','forward_distributed','forward_waveguide','forward_tfsf','forward_msl','forward_adi') else FREQS, **kw)
    if entry in ('run_tfsf', 'run_msl', 'run_adi'):
        kw['compute_s_params'] = False
    return sim.run(s_param_freqs=None if entry in ('run_nonuniform','run_waveguide','run_tfsf','run_msl','run_adi') else FREQS, **kw)


def _forbid(*args, **kwargs):
    raise AssertionError('precision admission allocated an array or reached setup')


@pytest.mark.parametrize('entry', HONOURS + REFUSES)
def test_off_refuses_before_allocation(monkeypatch, entry):
    with jax.enable_x64(False):
        sim = model('float64', entry)
        # Includes the zero-sized tracing probe, not just FDTD initialization.
        for name in ('zeros','ones','empty','array','asarray','arange','full'):
            monkeypatch.setattr(jnp, name, _forbid)
        monkeypatch.setattr(sim, '_build_grid', _forbid)
        from rfx._precision import FLOAT64_MODE_ERROR
        with pytest.raises(ValueError, match='Build and run the Simulation') as caught:
            if entry == 'adjoint':
                sim.forward(gradient='adjoint')
            else:
                invoke(sim, entry)
        assert str(caught.value) == FLOAT64_MODE_ERROR


@pytest.mark.parametrize('entry', REFUSES)
def test_on_preserves_main_refusal(entry):
    with jax.enable_x64():
        sim = model('float64', entry)
        error, message = MAIN_REFUSALS[entry]
        with pytest.raises(error, match=message):
            invoke(sim, entry)


def numeric_arrays(result):
    if hasattr(result,'ad_grad'):
        yield result.ad_grad
        yield result.fd_grad
    elif isinstance(result,tuple) and hasattr(result[0],'x_lo'):
        for a in jax.tree.leaves(result[0]):
            if hasattr(a,'dtype') and np.issubdtype(a.dtype,np.complexfloating):
                yield a
    elif hasattr(result, 'results'):
        for item in result.results:
            yield from numeric_arrays(item)
    elif hasattr(result, 'S'):
        yield result.S
    elif hasattr(result, 's_params') and not hasattr(result, 'time_series'):
        yield result.s_params
    elif hasattr(result, 'time_series'):
        if hasattr(result, 'state'):
            for name in ('ex','ey','ez','hx','hy','hz'):
                yield getattr(result.state, name)
        yield result.time_series
        for cfg in (getattr(result, 'waveguide_ports', None) or {}).values():
            for name in ('ey_profile','ez_profile','hy_profile','hz_profile','e_inc_table','h_inc_table',
                         'v_probe_t','v_ref_t','i_probe_t','i_ref_t','v_inc_t'):
                yield getattr(cfg, name)
        for spectra in (getattr(result, 'waveguide_sparams', None) or {}).values():
            yield spectra.s11
            yield spectra.s21
        if getattr(result, 's_params', None) is not None:
            yield result.s_params
        for monitor in (getattr(result, 'dft_planes', None) or {}).values():
            yield monitor.accumulator
        for container in (getattr(result,'ntff_data',None), getattr(result,'flux_monitors',None)):
            for a in jax.tree.leaves(container):
                if hasattr(a,'dtype') and np.issubdtype(a.dtype,np.complexfloating):
                    yield a
    elif hasattr(result, 'eps_design'):
        yield result.eps_design
    elif hasattr(result, 'stages'):
        for item in result.stages:
            yield from numeric_arrays(item)
    elif hasattr(result, 'density'):
        yield result.density
    else:
        yield result[0]


@pytest.mark.parametrize('entry', HONOURS)
@pytest.mark.parametrize('precision', ['float32','float64'])
def test_realized_field_storage(monkeypatch, entry, precision):
    if entry == 'topology_optimize':
        pytest.importorskip('optax')  # optional dependency of topology_optimize's optimizer
    import rfx.simulation as core
    fields, accumulators, records = [], [], []
    initialize, make_step = core.init_state, core.make_core_step
    assert FREQS[1] != np.float32(FREQS[1])
    expected_real = np.dtype(precision)
    expected_complex = np.dtype('complex128' if precision == 'float64' else 'complex64')

    def observe_fields(*args, **kwargs):
        state = initialize(*args, **kwargs)
        fields.extend(getattr(state,n).dtype for n in ('ex','ey','ez','hx','hy','hz'))
        return state

    def observe_step(ctx, *positional, **options):
        step = make_step(ctx, *positional, **options)
        def checked(*args, **kwargs):
            carry, probes, extras = step(*args, **kwargs)
            fields.extend(getattr(carry['fdtd'], n).dtype
                          for n in ('ex','ey','ez','hx','hy','hz'))
            for key in ('waveguide_port_accs', 'sparam_time_records', 'dft_time_records'):
                records.extend(a.dtype for a in jax.tree.leaves(carry.get(key))
                               if hasattr(a, 'dtype') and np.issubdtype(a.dtype, np.floating))
            for key in ('lumped_sparam_accs','wire_sparam_accs','dft_planes','flux_monitors','ntff'):
                for a in jax.tree.leaves(carry.get(key)):
                    if hasattr(a, 'dtype') and np.issubdtype(a.dtype, np.complexfloating):
                        accumulators.append(a.dtype)
            records.extend(a.dtype for a in jax.tree.leaves(probes) if hasattr(a,'dtype'))
            return carry, probes, extras
        return checked

    monkeypatch.setattr(core, 'init_state', observe_fields)
    monkeypatch.setattr(core, 'make_core_step', observe_step)
    try:
        with jax.enable_x64() if precision == 'float64' else jax.enable_x64(False):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if entry == 'adjoint':
                    from tests.unit.autodiff.test_reciprocity_adjoint import fixture
                    from tests.unit.autodiff.test_design_box_tape import BOX_LO, BOX_HI
                    sim, eps = fixture(precision)
                    result = sim.forward(design_box=(BOX_LO,BOX_HI), design_eps_override=eps,
                                         gradient='adjoint', n_steps=32, skip_preflight=True)
                else:
                    result = invoke(model(precision, entry), entry)
            output = list(numeric_arrays(result))

            def check_finite_entries(a):
                if entry == 'compute_mixed_s_matrix':
                    # The per-cell drive frame refuses the whole lumped column;
                    # precision still applies to every retained entry.
                    assert np.isnan(a[:, 0]).all(), entry
                    assert np.isfinite(a[:, 1]).all(), entry
                else:
                    assert np.isfinite(a).all(), entry

            assert fields and all(d == expected_real for d in fields), fields
            assert records and all(d == expected_real for d in records), records
            for a in output:
                if np.iscomplexobj(a):
                    check_finite_entries(a)
            if entry in FINAL_S:
                spectra = [a for a in output if np.iscomplexobj(a)]
                assert spectra
                for a in spectra:
                    assert a.dtype == expected_complex, (entry, a.dtype, expected_complex)
                    check_finite_entries(a)
            print(entry, precision, 'fields', sorted(set(map(str,fields))),
                  'records', sorted(set(map(str,records))),
                  'DFT', sorted(set(map(str,accumulators))),
                  'returned', sorted(set(str(a.dtype) for a in output)))
    finally:
        jax.clear_caches()


def test_scoped_mode_does_not_leak():
    assert not jax.config.x64_enabled
    with jax.enable_x64():
        assert jax.config.x64_enabled
        assert model('float64','run_lumped')._resolve_field_dtype() == jnp.float64
    assert not jax.config.x64_enabled


def test_resolver_itself_refuses_without_x64():
    with jax.enable_x64(False), pytest.raises(ValueError, match='jax.enable_x64'):
        model('float64','run_lumped')._resolve_field_dtype()


def test_preflight_precision_is_error():
    with jax.enable_x64(False):
        sim = model('float64','run_lumped')
        report = sim.preflight(strict=False)
        findings = report.by_code('precision_float64_without_x64')
        assert len(findings) == 1 and findings[0].severity == 'error'
        from rfx._precision import FLOAT64_MODE_ERROR
        assert str(findings[0]) == FLOAT64_MODE_ERROR
        with pytest.raises(ValueError, match='jax.enable_x64'):
            sim.preflight(strict=True)


def test_lumped_two_port_really_uses_float64():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with jax.enable_x64(False):
            s32 = invoke(model('float32','run_lumped'),'run_lumped').s_params
        with jax.enable_x64():
            s64 = invoke(model('float64','run_lumped'),'run_lumped').s_params
    delta = float(np.max(np.abs(s64-s32)))
    rounding = float(np.max(np.abs(s64-s64.astype(np.complex64))))
    print(f'max S delta={delta:.17g}; float32 rounding={rounding:.17g}; ratio={delta/rounding:.9g}')
    assert delta > rounding
    assert s32.dtype == np.complex64 and s64.dtype == np.complex128
    jax.clear_caches()
