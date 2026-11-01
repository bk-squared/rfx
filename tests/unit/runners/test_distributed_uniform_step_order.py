"""Uniform slab probes against the same plain one-device scene."""
import jax
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.materials.lorentz import lorentz_pole


def _model(boundary, position, geometry_devices):
    medium = boundary if boundary in ('debye', 'lorentz') else None
    boundary = 'pec' if medium else boundary
    sim = Simulation(freq_max=15e9, domain=(.031, .015, .015), dx=.001,
                     boundary=boundary, cpml_layers=6 if boundary == 'cpml' else 0)
    grid = sim._build_grid()
    width = (grid.nx + geometry_devices - 1) // geometry_devices
    node = {'off': grid.pad_x_lo + (4 if geometry_devices == 3 else 8),
            'first': width, 'last': width - 1}[position]
    if medium:
        poles = (dict(debye_poles=[DebyePole(delta_eps=1., tau=1e-11)])
                 if medium == 'debye' else dict(lorentz_poles=[lorentz_pole(
                     delta_eps=1., omega_0=2*np.pi*3e9, delta=1e9)]))
        sim.add_material('pole', eps_r=4. if medium == 'debye' else 2., **poles)
        sim.add(Box((.003, .003, .003), (.027, .012, .012)), material='pole')
    sim.add_source(((node - grid.pad_x_lo) * .001, .008, .008), 'ez',
                   amplitude_kind='field')
    for x in (.012, .020):
        sim.add_probe((x, .008, .008), 'ez')
    return sim


def _peak_ulp(reference, actual):
    """Use the whole-record reference-family peak, including every step."""
    assert actual.shape == reference.shape
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    peak = np.max(np.abs(reference))
    if peak == 0:
        return np.where(error == 0, 0., np.inf)
    return error / float(np.spacing(np.float32(peak)))


def _pole_errors(reference, actual):
    """Return the accumulated time law and the identical unwindowed rFFT error."""
    assert actual.shape == reference.shape
    want, got = reference.astype(np.float64), actual.astype(np.float64)
    error = np.abs(got - want)
    denominator = np.cumsum(np.max(np.abs(want), axis=1))[:, None]
    time = np.divide(error, denominator, out=np.where(error == 0, 0., np.inf),
                     where=denominator != 0)
    spectrum = np.fft.rfft(want, axis=0)
    delta = np.max(np.abs(np.fft.rfft(got, axis=0) - spectrum))
    peak = np.max(np.abs(spectrum))
    spectral = delta / peak if peak else (0. if delta == 0 else np.inf)
    return time, spectral


@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('geometry_devices', [2, 3], ids=['g2', 'g3'])
@pytest.mark.parametrize('devices', [2, 3], ids=['d2', 'd3'])
@pytest.mark.parametrize('position', ['off', 'first', 'last'])
@pytest.mark.parametrize('steps', [12, 60, 240])
def test_plain_one_device_probe_peak(boundary, geometry_devices, devices, position, steps):
    if jax.device_count() < devices:
        pytest.skip(f'requires {devices} devices')
    kwargs = dict(n_steps=steps, skip_preflight=True, compute_s_params=False)
    reference = _model(boundary, position, geometry_devices).run(**kwargs)
    actual = _model(boundary, position, geometry_devices).run(
        **kwargs, devices=jax.devices()[:devices])
    np.testing.assert_array_equal(actual.dt, reference.dt)
    score = _peak_ulp(np.asarray(reference.time_series), np.asarray(actual.time_series))
    assert np.isfinite(score).all()
    assert np.max(score) <= 9., f'worst peak-ULP: {np.max(score)}'


@pytest.mark.parametrize('medium', ['debye', 'lorentz'])
@pytest.mark.parametrize('geometry_devices', [2, 3], ids=['g2', 'g3'])
@pytest.mark.parametrize('devices', [2, 3], ids=['d2', 'd3'])
@pytest.mark.parametrize('position', ['off', 'first', 'last'])
@pytest.mark.parametrize('steps', [12, 60, 240])
def test_plain_one_device_pole_records(medium, geometry_devices, devices, position, steps):
    if jax.device_count() < devices:
        pytest.skip(f'requires {devices} devices')
    kwargs = dict(n_steps=steps, skip_preflight=True, compute_s_params=False)
    reference = _model(medium, position, geometry_devices).run(**kwargs)
    actual = _model(medium, position, geometry_devices).run(
        **kwargs, devices=jax.devices()[:devices])
    np.testing.assert_array_equal(actual.dt, reference.dt)
    time, spectral = _pole_errors(np.asarray(reference.time_series), np.asarray(actual.time_series))
    assert np.isfinite(time).all() and np.max(time) <= 1.2e-7, np.max(time)
    assert np.isfinite(spectral) and spectral <= 1e-4, spectral


@pytest.mark.parametrize('scene', ['pec', 'cpml', 'debye', 'lorentz'])
@pytest.mark.parametrize('geometry_devices', [2, 3], ids=['g2', 'g3'])
@pytest.mark.parametrize('position', ['off', 'first', 'last'])
@pytest.mark.parametrize('steps', [12, 60, 240])
def test_two_vs_three_fields_and_probes(scene, geometry_devices, position, steps):
    if jax.device_count() < 3:
        pytest.skip('requires three devices')
    results = [_model(scene, position, geometry_devices).run(
        n_steps=steps, skip_preflight=True, compute_s_params=False,
        devices=jax.devices()[:devices]) for devices in (2, 3)]
    for name in ('time_series', 'ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        arrays = [np.asarray(getattr(result if name == 'time_series' else result.state, name))
                  for result in results]
        assert arrays[0].shape == arrays[1].shape and arrays[0].dtype == arrays[1].dtype
        assert arrays[0].tobytes() == arrays[1].tobytes(), name


def _seam_model(boundary, devices):
    """Observer cells come from the runner's realized ownership ranges."""
    from rfx.runners.distributed_v2 import Slab
    from rfx.simulation import make_probe

    sim = _model(boundary, 'off', devices)
    sim._probes.clear()
    grid = sim._build_grid()
    layout = Slab(grid.nx, devices)
    indices = [i for lo, hi in layout.owned for i in (lo, hi - 1)]
    positions = [((i - grid.pad_x_lo + .5) * grid.dx, .008, .008)
                 for i in indices]
    for i, position in zip(indices, positions):
        assert make_probe(grid, position, 'ez').i == i
        sim.add_probe(position, 'ez')
    sim.add_port(positions[1], 'ez', impedance=50.)
    # Keep the existing U4 plane/flux bins and flux window, at the U1 y row.
    sim.add_dft_plane_probe(axis='y', coordinate=.008, component='ez',
                            freqs=[3e9, 7e9], name='seam_y')
    sim.add_flux_monitor(axis='y', coordinate=.008, freqs=[3e9, 7e9],
                         name='seam_y', dft_window='hann')
    return sim, dict(nx=int(grid.nx), owned=layout.owned, probe_indices=indices,
                     probe_positions=positions, source=sim._ports[0].position,
                     port_position=positions[1], boundary=boundary, devices=devices)


def _seam_arrays(result):
    from rfx.probes.probes import flux_spectrum

    arrays = dict(probes=np.asarray(result.time_series), dt=np.asarray(result.dt),
                  plane=np.asarray(result.dft_planes['seam_y'].accumulator))
    flux = result.flux_monitors['seam_y']
    for component in ('e1_dft', 'e2_dft', 'h1_dft', 'h2_dft'):
        arrays['flux.' + component] = np.asarray(getattr(flux, component))
    arrays['flux.power'] = np.asarray(flux_spectrum(flux))
    if result.s_params is not None:
        arrays['port.S11'] = np.asarray(result.s_params)[0, 0]
    for number, (_, spectra) in enumerate(result.lumped_port_sparams or ()):
        for field, value in zip(('V', 'I', 'V_ref'), spectra):
            arrays[f'port.{number}.{field}'] = np.asarray(value)
    return arrays


def _seam_metrics(reference, actual):
    np.testing.assert_array_equal(actual['dt'], reference['dt'])
    assert actual.keys() == reference.keys()
    score = _peak_ulp(reference['probes'], actual['probes'])
    rows = [dict(row=f'probe/{p}', worst=float(np.max(score[:, p])), bar=9.,
                 passed=bool(np.max(score[:, p]) <= 9.))
            for p in range(score.shape[1])]
    for name in reference.keys() - {'dt', 'probes'}:
        want, got = reference[name], actual[name]
        assert want.shape == got.shape
        dtype = np.complex128 if want.dtype.kind == 'c' else np.float64
        error = float(np.max(np.abs(got.astype(dtype) - want.astype(dtype))))
        peak = float(np.max(np.abs(want)))
        value = error / peak if peak else (0. if error == 0 else float('inf'))
        rows.append(dict(row=name, worst=value, bar=1e-4, passed=value <= 1e-4))
    return rows


def _seam_run(boundary, geometry_devices, run_devices, steps, *, request_s=True):
    sim, metadata = _seam_model(boundary, geometry_devices)
    options = dict(n_steps=steps, skip_preflight=True, compute_s_params=request_s)
    if request_s:
        options.update(s_param_freqs=np.array([1, 2.5, 5, 7.5, 10]) * 1e9,
                       s_param_n_steps=steps)
    if run_devices > 1:
        options['devices'] = jax.devices()[:run_devices]
    return _seam_arrays(sim.run(**options)), metadata


@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('devices', [2, 3], ids=['d2', 'd3'])
@pytest.mark.parametrize('steps', [12, 60, 240])
def test_seam_observers(boundary, devices, steps, tmp_path):
    """Plain one-device reference for probes, port, plane and flux at seams."""
    import json
    from rfx.errors import RfxRefusal

    if jax.device_count() < max(3, devices):
        pytest.skip('three devices needed to compare a possible port refusal')
    outcomes, refusals = {}, {}
    for count in (1, 2, 3):
        try:
            outcomes[count] = _seam_run(boundary, devices, count, steps)
        except RfxRefusal as error:
            refusals[count] = str(error)
    if refusals:
        assert set(refusals) == {1, 2, 3} and len(set(refusals.values())) == 1, refusals
        (tmp_path / 'port-refusal.json').write_text(json.dumps(refusals, indent=2))
        outcomes = {count: _seam_run(boundary, devices, count, steps, request_s=False)
                    for count in (1, devices)}
    reference, metadata = outcomes[1]
    actual, _ = outcomes[devices]
    rows = _seam_metrics(reference, actual)
    np.savez(tmp_path / 'reference.npz', **reference)
    np.savez(tmp_path / 'actual.npz', **actual)
    (tmp_path / 'measurements.json').write_text(json.dumps(
        dict(metadata, steps=steps, refusals=refusals, metrics=rows), indent=2))
    score = _peak_ulp(reference['probes'], actual['probes'])
    probe = np.unravel_index(np.argmax(score), score.shape)[1]
    np.savetxt(tmp_path / 'worst-probe.csv', np.column_stack((
        np.arange(steps), reference['probes'][:, probe], actual['probes'][:, probe],
        score[:, probe])), delimiter=',', header='step,reference,actual,peak_ulp', comments='')
    assert all(row['passed'] for row in rows), rows
