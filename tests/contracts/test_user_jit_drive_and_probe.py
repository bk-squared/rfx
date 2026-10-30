"""Plain versus user-jitted public drives, probes and port records (#2241).

Subgrid is out: it is a research lane without forward(). Dedicated coax
calculators are out: coax is refused by run()/forward(). Floquet is out:
it inherits the #2252 sampler. Linux x86 CPU binds the declared bars.
Each entry is a separate row; strict xfails follow design amendment 1.
Set RFX_USER_JIT_METRICS to save one JSON per row, including construction errors.
"""

import gc
import json
import os
from pathlib import Path
import re
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation

N_STEPS = 192
BINS = np.array([4.3e9, 5.1e9, 5.8e9], dtype=np.float32)
# Nonuniform physical widths, including non-whole-cell domain lengths.
DX = np.array([1.] * 7 + [.9, 1.1, .95, 1.05, .9, 1.1, 1.05, .95]
              + [1.3] + [1.] * 8) * 1e-3
# Keep the waveguide port, probe and reference span in equal-width cells.
DX_GUIDE = np.r_[np.ones(10), [.9, 1.1, .95, 1.05, .9, 1.1, 1.05, .95, 1.3],
                 np.ones(5)] * 1e-3
DZ_MSL = np.r_[np.full(6, .5), .5 * 1.25 ** np.arange(1, 6)] * 1e-3


XFAILS = dict.fromkeys((1, 3, 5, 15, 29, 30, 31, 32, 40, 43), pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#2241 a user jax.jit call differs from the plain call on this scene (Linux x86 CPU; cause not localized)"))
XFAILS.update(dict.fromkeys((17, 19, 50), pytest.mark.xfail(
    strict=True, raises=jax.errors.ConcretizationTypeError,
    reason="#2241 graded port assembly calls float() on a traced cell spacing under a user jit (rfx/model/graded_ports.py:77)")))
XFAILS[49] = pytest.mark.xfail(
    strict=True, raises=jax.errors.TracerArrayConversionError,
    reason="#2241 run()'s lumped-port S extraction converts a traced array to numpy under a user jit (rfx/measurement/port_s.py:32)")
XFAILS[33] = pytest.mark.xfail(
    strict=True, raises=ValueError,
    reason="#2241 the closed-box TF/SF vacuum check uses an ordered debug effect, which multi-device lowering refuses under a user jit (rfx/boundaries/tfsf.py:399)")


def _row(path, kind, entry, *, eps=False, always=False):
    name = f'{path}-{kind}{"-eps" if eps else ""}-{entry}'
    marks = [] if always else [pytest.mark.slow]
    if len(ROWS) + 1 in XFAILS:
        marks.append(XFAILS[len(ROWS) + 1])
    if path.startswith('multi-'):
        marks.append(pytest.mark.distributed)
    return pytest.param(path, kind, entry, eps, id=name, marks=marks)


ROWS = []
for _path, _kinds in (
    ('uniform', ('point', 'lumped', 'wire', 'waveguide', 'msl', 'msl-eigenmode',
                 'tfsf-open', 'tfsf-closed')),
    ('graded', ('point', 'lumped', 'wire', 'waveguide', 'msl', 'tfsf-open')),
    ('multi-uniform', ('point', 'lumped', 'wire', 'tfsf-closed')),
    ('multi-graded', ('point', 'lumped', 'wire')),
    ('adi', ('field',)),
):
    for _kind in _kinds:
        _entries = ('forward',) if _kind == 'tfsf-closed' else (
            ('run',) if _kind == 'msl-eigenmode' else ('run', 'forward'))
        for _entry in _entries:
            _always = ((_path in ('uniform', 'graded') and _kind == 'lumped') or (_path == 'multi-uniform' and _kind == 'point')
                       or _path == 'adi' or _kind == 'msl-eigenmode')
            ROWS.append(_row(_path, _kind, _entry, always=_always))
for _path in ('uniform', 'graded', 'multi-graded', 'adi'):
    for _kind in (('field',) if _path == 'adi' else ('point', 'lumped')):
        ROWS.append(_row(_path, _kind, 'forward', eps=True,
                         always=_path == 'multi-graded' and _kind == 'lumped'))


def _devices(path):
    if not path.startswith('multi-'):
        return {}
    devices = jax.devices('cpu')[:2]
    assert len(devices) == 2, 'requires two forced host CPU devices (root conftest.py)'
    return {'devices': devices}


def _scene(path, kind, *, refusal=False, debye=False):
    graded, adi = path.endswith('graded'), path == 'adi'
    msl, guide, tfsf = kind.startswith('msl'), kind == 'waveguide', kind.startswith('tfsf')
    domain = (24.3e-3, 13.4e-3, 11.6e-3)
    mesh = {'dx_profile': DX} if graded else {}
    boundary = 'pec' if adi else 'cpml'
    layers, dx = (0 if adi else 3), 1e-3
    if guide and not adi:
        domain = (24.3e-3, 20.4e-3, 10.6e-3)
        boundary = {'x': 'cpml', 'y': 'pec', 'z': 'pec'}
        if graded:
            mesh = {'dx_profile': DX_GUIDE}
    if guide and adi:
        boundary, layers = 'cpml', 3
    if msl and not adi:
        domain = (24.3e-3, 12.4e-3, float(DZ_MSL.sum()) if graded else 8.3e-3)
        mesh = {'dz_profile': DZ_MSL} if graded else {}
    if tfsf:
        domain, dx = (32.3e-3, 30.4e-3, 28.6e-3), 2e-3
        mesh = {'dx_profile': np.r_[np.full(7, 2.), 1.8, 2.2,
                                   2.3, np.full(6, 2.)] * 1e-3} if graded else {}
    sim = Simulation(freq_max=16e9, domain=domain, dx=dx, boundary=boundary,
                     cpml_layers=layers, solver='adi' if adi else 'yee',
                     snap='declared' if adi or guide else 'strict', **mesh)
    pulse = GaussianPulse(f0=5.1e9, bandwidth=.9)
    near, far = (6.3e-3, 4.2e-3, 4.3e-3), (15.2e-3, 7.3e-3, 6.2e-3)
    if kind in ('point', 'field'):
        sim.add_source(near, 'ez', waveform=pulse,
                       amplitude_kind='field' if kind == 'field' else 'current')
    elif kind in ('lumped', 'wire'):
        sim.add_port(near, 'ez', impedance=50., waveform=pulse,
                     extent=2.3e-3 if kind == 'wire' else None)
    elif guide:
        sim.add_waveguide_port(6.3e-3, f0=10e9, bandwidth=.8, n_modes=1,
                               freqs=np.array([9e9, 10e9]), probe_offset=3, ref_offset=2)
    elif msl:
        if not refusal:
            sim.add_material('substrate', eps_r=3.)
            sim.add(Box((2e-3, 2e-3, 1e-3), (23e-3, 10e-3, 3e-3)), material='substrate')
            sim.add(Box((2e-3, 2e-3, 0), (23e-3, 10e-3, 1e-3)), material='pec')
            sim.add(Box((5e-3, 5e-3, 3e-3), (22e-3, 7e-3, 4e-3)), material='pec')
        sim.add_msl_port((5e-3, 6e-3, 1e-3), width=2e-3, height=2e-3,
                         direction='+x', impedance=50., waveform=pulse,
                         mode='eigenmode' if kind == 'msl-eigenmode' else 'laplace',
                         eps_r_sub=3., n_probe_offset=3, n_probe_spacing=2, n_probes=3)
        near, far = (5e-3, 5.4e-3, 2e-3), (15.2e-3, 6.3e-3, 2e-3)
    elif tfsf:
        sim.add_tfsf_source(f0=10e9, bandwidth=.8, polarization='ez', direction='+x',
                            margin=3, closed_box=kind == 'tfsf-closed',
                            angle_deg=15. if kind == 'tfsf-oblique' else 0.)
        near, far = (8.3e-3, 11.2e-3, 12.3e-3), (19.2e-3, 17.3e-3, 14.2e-3)
    elif kind == 'coax':
        sim.add_coaxial_port((6.3e-3, 4.2e-3, 0.), face='bottom', pin_length=3e-3)
    else:
        raise AssertionError(kind)
    if debye:
        sim.add_material('debye', eps_r=2.5, debye_poles=[DebyePole(delta_eps=1., tau=20e-12)])
        sim.add(Box((3.1e-3, 2.1e-3, 2.1e-3), (9.2e-3, 8.2e-3, 9.1e-3)), material='debye')
    sim.add_probe(near, 'ez')
    sim.add_probe(far, 'ez')
    return sim


def _input_eps(sim, path):
    if path == 'adi':
        return jnp.float32(1.5)
    # Public mesh finalization supplies only input shape; never an expected value.
    return jnp.full(sim.freeze_mesh().shape, 2.5, dtype=jnp.float32)


def _call(sim, path, entry, eps=None, *, spectrum=False):
    options = dict(n_steps=N_STEPS, skip_preflight=True, **_devices(path))
    if entry == 'run':
        if spectrum:
            options['s_param_freqs'] = BINS
        else:
            options['compute_s_params'] = False
    else:
        options['checkpoint'] = False
        if path.startswith('multi-'):
            options['distributed'] = True
        if eps is not None:
            options['eps_override'] = eps
    result = getattr(sim, entry)(**options)
    channels = {f'probe/{i}': result.time_series[:, i] for i in range(2)}
    for i, record in enumerate(result.sparam_time_records or ()):
        for channel in record.channels:
            channels[f'port/{i}/{record.kind}/{channel}'] = record[channel]
    return channels, (result.s_params, result.freqs) if spectrum else ()


def _metrics(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    delta = np.abs(actual.astype(np.complex128 if np.iscomplexobj(actual) else np.float64)
                   - expected)
    peak = float(np.max(np.abs(expected)))
    spacing = float(np.spacing(np.float32(peak)))
    first = np.flatnonzero(delta)
    summed, maximum = float(np.sum(delta)), float(np.max(delta))
    stats = dict(peak=peak, max_abs=maximum, max_peak_ulp=maximum / spacing,
                first_differing_step=int(first[0]) if first.size else None,
                summed_over_peak=summed / peak if peak else (0. if summed == 0 else None),
                finite=bool(np.isfinite(expected).all() and np.isfinite(actual).all()),
                passed=bool(np.isfinite(delta).all() and maximum <= 9 * spacing
                            and summed <= 1e-4 * peak))
    return {key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in stats.items()}


def _error_details(error):
    return dict(error=f'{type(error).__name__}: {error}',
                error_first_line=f'{type(error).__name__}: {str(error).partition(chr(10))[0]}',
                rfx_frames=[dict(file='rfx/' + frame.filename.rsplit('/rfx/', 1)[1],
                                 line=frame.lineno, function=frame.name)
                            for frame in traceback.extract_tb(error.__traceback__)
                            if '/rfx/' in frame.filename])


@pytest.fixture
def evidence(request, record_property):
    report = {'row': request.node.callspec.id, 'channels': {}, 'status': 'started'}
    yield report
    record_property('user_jit_drive', json.dumps(report, allow_nan=False))
    print(json.dumps(report, allow_nan=False))
    output = os.environ.get('RFX_USER_JIT_METRICS')
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'{report["row"]}.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n')
    jax.clear_caches()
    gc.collect()


def _compare(path, kind, entry, eps, evidence, *, spectrum=False):
    try:
        sim = _scene(path, kind)
        value = _input_eps(sim, path) if eps else None

        def call(e=None):
            return _call(sim, path, entry, e, spectrum=spectrum)

        plain = jax.tree.map(np.asarray, call(value))
        evidence['channels'] = {
            name: dict(peak=float(np.max(np.abs(channel))), max_peak_ulp=None,
                       first_differing_step=None, summed_over_peak=None,
                       comparison='user-jitted call not completed')
            for name, channel in plain[0].items()}
        actual = jax.tree.map(np.asarray, jax.jit(call)(value))
        expected_channels, expected_s = plain
        actual_channels, actual_s = actual
        assert expected_channels.keys() == actual_channels.keys()
        evidence['channels'] = {name: _metrics(expected, actual_channels[name])
                                for name, expected in expected_channels.items()}
        failures = [name for name, numbers in evidence['channels'].items()
                    if not numbers['passed']]
        # Guard against a silent no-drive solve, without requiring every optional
        # port channel (e.g. an unavailable reference) to be nonzero.
        assert all(evidence['channels'][f'probe/{i}']['peak'] > 0 for i in range(2))
        if spectrum:
            s0, f0 = expected_s
            s1, f1 = actual_s
            assert s0 is not None and f0 is not None
            assert s1.shape == s0.shape and s1.dtype == s0.dtype
            assert f1.shape == f0.shape and f1.dtype == f0.dtype
            assert f1.tobytes() == f0.tobytes(), 'S frequencies must be bit-equal'
            # S has shape (ports, ports, bins): each bin has its own matrix peak.
            for k in range(len(f0)):
                stats = _metrics(s0[..., k].reshape(-1), s1[..., k].reshape(-1))
                stats['passed'] = bool(stats['finite'] and stats['peak'] > 0
                                       and stats['max_abs'] <= 1e-4 * stats['peak'])
                evidence['channels'][f'S/bin/{k}'] = stats
                if not stats['passed']:
                    failures.append(f'S/bin/{k}')
        evidence['status'] = 'fail' if failures else 'pass'
        assert not failures, json.dumps({name: evidence['channels'][name] for name in failures})
    except Exception as error:
        evidence.update(status='fail', **_error_details(error))
        raise


@pytest.mark.parametrize('path,kind,entry,eps', ROWS)
def test_drive_and_probe(path, kind, entry, eps, evidence):
    if kind == 'msl-eigenmode':
        # Stable row 11: plain run also refuses this unavailable solver.
        try:
            for mode in ('plain', 'jit'):
                sim = _scene(path, kind)

                def call():
                    return _call(sim, path, entry)

                with pytest.raises(NotImplementedError, match='not yet wired in') as refusal:
                    (call if mode == 'plain' else jax.jit(call))()
                evidence[mode + '_refusal'] = _error_details(refusal.value)
            evidence['status'] = 'pass'
        except Exception as error:
            evidence.update(status='fail', **_error_details(error))
            raise
    else:
        _compare(path, kind, entry, eps, evidence)


@pytest.mark.slow
@pytest.mark.parametrize('path,kind', [pytest.param('uniform', 'lumped', id='U4-uniform-lumped-S', marks=XFAILS[49]),
                                      pytest.param('graded', 'wire', id='U5-graded-wire-S', marks=XFAILS[50])])
def test_s_drive_and_reference(path, kind, evidence):
    _compare(path, kind, 'run', False, evidence, spectrum=True)


# Literal message fragments are independent of implementation admission helpers.
REFUSALS = []
for _entry in ('run', 'forward'):
    REFUSALS.append(('uniform', 'coax', _entry, False, False, NotImplementedError,
                     f'add_coaxial_port() is not wired into Simulation.{_entry}().'))
REFUSALS.append(('uniform', 'msl-eigenmode', 'forward', False, False, NotImplementedError,
                 "MSL port mode='eigenmode' is supported on the"))
for _path in ('multi-uniform', 'multi-graded'):
    for _kind, _fragment in (('waveguide', 'waveguide'), ('msl', 'microstrip')):
        for _entry in ('run', 'forward'):
            REFUSALS.append((_path, _kind, _entry, False, False, NotImplementedError, _fragment))
REFUSALS.append(('multi-uniform', 'point', 'forward', True, False, NotImplementedError,
                 'eps_override'))
for _entry in ('run', 'forward'):
    REFUSALS.extend([
        ('graded', 'tfsf-oblique', _entry, False, False, ValueError, 'TFSF oblique incidence'),
        ('graded', 'msl-eigenmode', _entry, False, False, NotImplementedError,
         "NU MSL S-params support mode='laplace'/'uniform'"),
        ('multi-graded', 'tfsf-open', _entry, False, False, NotImplementedError,
         'TFSF plane-wave sources (add_tfsf_source) are not supported on the graded multi-device lane'),
    ])
    for _kind, _fragment in (
        ('point', "ADI implements only amplitude_kind='field'"),
        ('lumped', "solver='adi' currently supports only add_source()-style soft sources"),
        ('wire', "solver='adi' currently supports only add_source()-style soft sources"),
        ('waveguide', 'solver="adi" refuses absorbing boundaries'),
        ('msl', 'a microstrip port (add_msl_port)'),
        ('coax', f'add_coaxial_port() is not wired into Simulation.{_entry}().'),
    ):
        _exception = ValueError if _kind in ('lumped', 'wire', 'waveguide') else NotImplementedError
        REFUSALS.append(('adi', _kind, _entry, False, False, _exception, _fragment))
REFUSALS.append(('multi-graded', 'lumped', 'forward', True, True, NotImplementedError,
                 'Graded multi-device port overrides on Debye/Lorentz live edges'))


@pytest.mark.parametrize('path,kind,entry,eps,debye,exception,fragment', [
    pytest.param(*row, id=f'{row[0]}-{row[1]}{"-eps" if row[3] else ""}'
                 f'{"-debye" if row[4] else ""}-{row[2]}-refused',
                 marks=[pytest.mark.distributed] if row[0].startswith('multi-') else [])
    for row in REFUSALS
])
def test_refused_under_user_jit(path, kind, entry, eps, debye, exception, fragment, evidence):
    try:
        if path == 'adi' and kind == 'waveguide':
            # ADI refuses every CPML scene at construction, before port admission.
            with pytest.raises(exception, match=re.escape(fragment)) as refusal:
                jax.jit(lambda: _call(_scene(path, kind), path, entry))()
            evidence.update(status='pass', refusal_stage='CPML scene construction',
                            refusal=_error_details(refusal.value))
            return
        sim = _scene(path, kind, refusal=path == 'adi' or path.startswith('multi-'),
                     debye=debye)
        value = _input_eps(sim, path) if eps else None
        with pytest.raises(exception, match=re.escape(fragment)) as refusal:
            jax.jit(lambda e: _call(sim, path, entry, e))(value)
        evidence.update(status='pass', refusal=_error_details(refusal.value))
    except Exception as error:
        evidence.update(status='fail', **_error_details(error))
        raise
