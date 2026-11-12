"""B2/A4: drive sampling times in synthetic single-device boxes."""
from contextlib import contextmanager
from dataclasses import replace
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contracts.path_equivalence.builders import build, DX


def assert_field_time(name, value):
    assert value.dtype == np.float64, f'B2 {name} time: {value.dtype}; expected float64'




@contextmanager
def observe_times():
    from rfx.measurement.record_end import source_end_step
    from rfx.preflight.sources import _validate_cfg_unresolved_pulse
    from rfx._internal import _realized
    from rfx.sources.tfsf import line_source_value
    from rfx.model.source_coefficients import uniform_source_table, current_source_samples
    from rfx.sources.port_drive import port_drive_waveform
    observed = set()
    previous = sys.getprofile()

    def inspect(frame, event, arg):
        if event != 'return':
            return
        code, local = frame.f_code, frame.f_locals
        slots = {source_end_step.__code__: ('record-end', 'times'),
                 _validate_cfg_unresolved_pulse.__code__: ('preflight', 'times'),
                 _realized.sources.__code__: ('realized source', 'times'),
                 line_source_value.__code__: ('TFSF', 't'),
                 uniform_source_table.__code__: ('uniform source', 'times'),
                 current_source_samples.__code__: ('current source', 'times'),
                 port_drive_waveform.__code__: ('port drive', 'times')}
        if code in slots:
            name, key = slots[code]
            value = local.get(key)
            if value is not None:
                assert_field_time(name, value)
                if name not in observed:
                    print(f'A4 measured {name}: {value.dtype}')
                observed.add(name)
    try:
        with jax.enable_x64(True), warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            sys.setprofile(inspect)
            yield observed
    finally:
        sys.setprofile(previous)
        jax.clear_caches()


@pytest.mark.parametrize('source, lane, entry', [
    (source, lane, entry)
    for entry in ('run', 'forward')
    for lane in ('uniform', 'nonuniform')
    for source in ('point', 'tfsf')
    if not (source == 'tfsf' and lane == 'nonuniform')
])
def test_float64_record_end_times(entry, lane, source):
    from rfx._internal import _realized
    row = ('_tfsf', 'plane_wave') if source == 'tfsf' else ('_materials', 'eps')
    with observe_times() as observed:
        sim = build(row, 'run_' + lane, precision='float64')
        # Capture also checks the diagnostic's independently sampled raw drive.
        with _realized.capture():
            result = (sim.run(n_steps=12, compute_s_params=False, skip_preflight=True)
                      if entry == 'run' else sim.forward(n_steps=12, checkpoint=False, skip_preflight=True))
            jax.block_until_ready(result.time_series)
        assert 'record-end' in observed
        assert ('TFSF' if source == 'tfsf' else 'realized source') in observed


@pytest.mark.parametrize('lane', ('uniform', 'nonuniform'))
@pytest.mark.parametrize('flux', (False, True))
def test_float64_until_decay_times(lane, flux):
    with observe_times() as observed:
        sim = build(('_tfsf', 'plane_wave'), 'run_' + lane, precision='float64')
        result = sim.run(until_decay=1e-3, decay_max_steps=12, decay_min_steps=12,
                         decay_check_interval=4, compute_s_params=False, skip_preflight=True,
                         **({'radiated_flux_box': ((3*DX, 2*DX, 2*DX), (7*DX, 5*DX, 4*DX))} if flux else {}))
        jax.block_until_ready(result.time_series)
        assert {'record-end', 'TFSF'} <= observed


@pytest.mark.parametrize('lane', ('uniform', 'nonuniform'))
def test_float64_ringdown_source_times(lane):
    from rfx.ringdown import RingdownRun, RingdownForward, RingdownStop, RingdownSpec
    with observe_times() as observed:
        sim = build(('_materials', 'eps'), 'run_' + lane, precision='float64')
        # One early pulse keeps source-off refusal independent of precision.
        sim._ports[0] = replace(sim._ports[0], waveform=lambda t: jnp.where(t == 0, 1., 0.))
        grid = sim._build_realized_grid()
        for cls in (RingdownRun, RingdownForward, RingdownStop):
            observed.clear()
            args = {'n_max': 1200} if cls is RingdownStop else {'n_steps': 1200}
            cls(sim, RingdownSpec(), lane='graded' if lane == 'nonuniform' else lane, grid=grid, **args)
            assert 'record-end' in observed, cls.__name__


@pytest.mark.parametrize('lane', ('uniform', 'nonuniform'))
def test_float64_preflight_sampling(lane):
    from rfx.preflight.sources import _validate_cfg_unresolved_pulse
    from rfx.sources.sources import GaussianPulse
    with observe_times() as observed:
        sim = build(('_materials', 'eps'), 'run_' + lane, precision='float64')
        sim._ports[0] = replace(sim._ports[0], waveform=GaussianPulse(f0=1e11, bandwidth=1.))
        _validate_cfg_unresolved_pulse(sim, warnings, DX)
        assert 'preflight' in observed


@pytest.mark.parametrize('kind', ('normal', 'closed', 'methodB'))
def test_float64_incident_replay_times(kind):
    from rfx.measurement.incident import measure_normal_incident_spectrum, measure_incident_spectrum
    from rfx.sources.tfsf import init_tfsf
    from rfx.sources.tfsf_oblique_open import init_tfsf_methodB
    with observe_times() as observed:
        dt = np.float64(DX / (299792458. * np.sqrt(3)) * .99)
        if kind == 'methodB':
            cfg, state = init_tfsf_methodB(24, 24, DX, dt, nz=24,
                                           cpml_layers=2, tfsf_margin=3, f0=5e9,
                                           bandwidth=.8, theta_deg=20.)
            measure_incident_spectrum(cfg, state, 12, [5e9], DX)
        else:
            cfg, state = init_tfsf(nx=24, ny=24, nz=24, dx=DX, dt=dt,
                                  cpml_layers=2, tfsf_margin=3, f0=5e9,
                                  bandwidth=.8, closed_box=kind == 'closed', n_steps=12)
            measure_normal_incident_spectrum(cfg, state, 12, [5e9], float(dt))
        assert 'TFSF' in observed


@pytest.mark.parametrize('lane', ('uniform', 'nonuniform'))
@pytest.mark.parametrize('source', ('lumped_port', 'msl_port'))
def test_float64_port_sampling(lane, source):
    row = ('_msl_ports', source) if source == 'msl_port' else ('_ports', source)
    with observe_times() as observed:
        sim = build(row, 'run_' + lane, precision='float64')
        result = sim.run(n_steps=12, compute_s_params=False, skip_preflight=True)
        jax.block_until_ready(result.time_series)
        assert 'port drive' in observed


def test_uniform_eigenmode_time_grid():
    from types import SimpleNamespace
    from rfx.sources.msl_port import make_msl_port_sources_jm
    previous = sys.getprofile()
    observed = set()

    def inspect(frame, event, arg):
        if event == 'return' and frame.f_code is make_msl_port_sources_jm.__code__:
            for name in ('times_e', 'times_h'):
                value = frame.f_locals[name]
                assert_field_time('MSL ' + name, value)
                print(f'A4 measured MSL {name}: {value.dtype}')
                observed.add(name)
    try:
        with jax.enable_x64(True), warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            sim = build(('_materials', 'eps'), 'run_uniform', precision='float64')
            grid = sim._build_realized_grid()
            # No spatial edges are needed to observe the actual sampler's two
            # time grids; the mode solve does not participate in time sampling.
            em = SimpleNamespace(j_grid_lo=0, k_grid_lo=0, n_y_grid=1, n_z_grid=1, cell_indices=[])
            port = SimpleNamespace(excitation=lambda t: jnp.cos(t * 2e10), direction='+x')
            sys.setprofile(inspect)
            make_msl_port_sources_jm(grid, port, n_steps=12, materials=None, eigenmode_data=em)
            assert observed == {'times_e', 'times_h'}
    finally:
        sys.setprofile(previous)
        jax.clear_caches()
