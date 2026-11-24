"""#2219: the one-device sheet update survives both slab runners.

Use the root conftest's CPU-device setup; run three-device evidence with
XLA_FLAGS=--xla_force_host_platform_device_count=3 before starting pytest.
The oracle is a public one-device run, compared only with NumPy arithmetic.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._internal import _grid_metric
from tests.contracts.path_equivalence.comparison import component_peaks


CHANNELS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
SAMPLE_STEPS = (4, 8, 16, 32)


def model(graded, boundary, placement, count, *, sheet=True, drive=True):
    options = {'dy_profile': np.array([.001] + [.0008, .0012] * 4 + [.001])} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.024, .010, .009), dx=.001,
                     boundary=boundary, cpml_layers=3 if boundary == 'cpml' else 0,
                     snap='declared', **options)
    grid = sim._build_realized_grid()
    per = (grid.shape[0] + count - 1) // count
    seam = grid.node_of(0, per)
    plane = grid.node_of(0, per + (placement == 'beside'))
    if placement == 'end_rows':
        shape = Box((grid.node_of(0, 5), grid.node_of(1, 5), .004),
                    (grid.node_of(0, grid.shape[0] - 5), grid.node_of(1, 10), .004))
    elif placement == 'span':
        shape = Box((.00035, .00235, .004), (.02365, .00765, .004))
    else:
        shape = Box((plane, .00235, .00135), (plane, .00765, .00765))
    if sheet:
        sim.add_thin_conductor(shape, sigma_bulk=100., surface_impedance_f0=5e9)
    if drive:
        for axis, component in enumerate(CHANNELS[:3]):
            sim.add_source((seam - .002, .004, .003), component,
                           waveform=lambda t, a=axis: (a + 1) * jnp.exp(-((t / grid.dt - 3) / 2) ** 2),
                           amplitude_kind='field')
    for position in ((plane, .003, .004), (seam + .002, .006, .005),
                     (plane, .007, .006)):
        for component in CHANNELS:
            sim.add_probe(position, component)
    return sim


def record(sim, count, *, forward=False):
    """Independent public runs from zero; Result.state already gathers owners."""
    fields = []
    for n in SAMPLE_STEPS:
        options = dict(n_steps=n, skip_preflight=True)
        if count > 1:
            options['devices'] = jax.devices()[:count]
        if forward:
            result = sim.forward(**options, distributed=count > 1)
        else:
            result = sim.run(**options, compute_s_params=False)
        fields.append(np.stack([np.asarray(getattr(result.state, c), dtype=np.float32)
                                for c in CHANNELS]))
    probes = np.asarray(result.time_series, dtype=np.float32).reshape(n, -1, 6)
    return np.stack(fields), probes


def need_devices(count):
    if len(jax.devices()) < count:
        pytest.skip(f'requires {count} CPU devices')


def peak_ulps(reference, actual, axes):
    peaks = np.max(np.abs(reference), axis=axes)
    differences = np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64)), axis=axes)
    return peaks, differences / np.spacing(peaks)


def source_cells(sim):
    """The point-source component cells written by this fixture's model."""
    grid = sim._build_realized_grid()
    cells = []
    for source in sim._ports:
        assert source.impedance == 0 and source.extent is None
        cells.append((source.component, tuple(int(i) for i in
                      _grid_metric.field_index(grid, source.position, source.component))))
    return cells


def peak_fields(fields, cells):
    """Set aside driven component cells for peaks only; retain all differences."""
    masked = {name: fields[:, channel].copy() for channel, name in enumerate(CHANNELS)}
    for name, index in cells:
        masked[name][(slice(None), *index)] = 0
    return masked


def compare(reference, actual, case, cells):
    fields, probes = reference
    other_fields, other_probes = actual
    assert fields.shape == other_fields.shape
    assert np.all(np.isfinite(fields)) and np.all(np.isfinite(other_fields))
    assert cells, f'{case}: no source-driven cells found in the model'
    family = component_peaks(peak_fields(fields, cells), peak_fields(other_fields, cells))
    peaks = np.array([family[c] for c in CHANNELS], dtype=np.float32)
    differences = np.max(np.abs(other_fields.astype(np.float64) - fields.astype(np.float64)),
                         axis=(0, 2, 3, 4))
    worst = differences / np.spacing(peaks)
    assert np.all(peaks > 0), (case, peaks)
    component_peak, component_worst = peak_ulps(fields, other_fields, (0, 2, 3, 4))
    probe_peaks, probe_worst = peak_ulps(probes, other_probes, (0, 1))
    print('F0_PARITY ' + json.dumps(dict(case=case, steps=SAMPLE_STEPS,
          peaks=peaks.tolist(), max_ulp=worst.tolist(), source_cells=cells,
          component_peaks=component_peak.tolist(), component_max_ulp=component_worst.tolist(),
          bitwise=[fields[:, c].tobytes() == other_fields[:, c].tobytes() for c in range(6)],
          probe_peaks=probe_peaks.tolist(), probe_max_ulp=probe_worst.tolist())), flush=True)
    for channel, name in enumerate(CHANNELS):
        assert worst[channel] <= 9, f'{case} {name}: {worst[channel]} source-excluded family peak ULP > 9'


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('placement', ['span', 'seam', 'beside'])
@pytest.mark.parametrize('count', [2, 3])
def test_v1_sheet_parity(graded, boundary, placement, count):
    if len(jax.devices()) < count:
        pytest.skip(f'requires {count} CPU devices')
    sim = model(graded, boundary, placement, count)
    compare(record(sim, 1), record(sim, count),
            ['graded' if graded else 'uniform', boundary, placement, count], source_cells(sim))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_v3_sheet_removed_control(graded, boundary):
    need_devices(2)
    sim = model(graded, boundary, 'span', 2, sheet=False)
    compare(record(sim, 1), record(sim, 2),
            ['graded' if graded else 'uniform', boundary, 'control', 2], source_cells(sim))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_v4_forward_carries_sheet(graded):
    need_devices(2)
    sim = model(graded, 'pec', 'span', 2)
    assert source_cells(sim), 'no source-driven cells found in the model'
    options = dict(n_steps=32, devices=jax.devices()[:2], skip_preflight=True)
    reference = np.asarray(sim.run(**options, compute_s_params=False).time_series)
    actual = np.asarray(sim.forward(**options, distributed=True).time_series)
    assert reference.shape == actual.shape
    assert reference.dtype == actual.dtype
    assert np.all(np.isfinite(reference)) and np.all(np.isfinite(actual))
    reference = reference.reshape(32, -1, len(CHANNELS))
    actual = actual.reshape(reference.shape)
    bitwise = [reference[:, :, c].tobytes() == actual[:, :, c].tobytes()
               for c in range(len(CHANNELS))]
    differences = np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64)),
                         axis=(0, 1))
    case = ['graded' if graded else 'uniform', 'pec', 'span', 2]
    print('F0_FORWARD ' + json.dumps(dict(case=case, bitwise=bitwise,
          max_abs_diff=differences.tolist())), flush=True)
    for channel, name in enumerate(CHANNELS):
        assert bitwise[channel], f'{case} {name}: forward/run probe records differ bitwise'


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_v1_end_rows(graded, monkeypatch):
    from rfx.materials import thin_conductor

    need_devices(2)
    sim = model(graded, 'cpml', 'end_rows', 2)
    contexts = []
    build = thin_conductor.build_sheet_impedance_ctx

    def capture(*args, **kwargs):
        context = build(*args, **kwargs)
        if context is not None:
            contexts.append(context)
        return context

    with monkeypatch.context() as patch:
        patch.setattr(thin_conductor, 'build_sheet_impedance_ctx', capture)
        reference = record(sim, 1)
    assert contexts
    counts = [0 if getattr(contexts[-1], 'end_' + c) is None else
              int(np.count_nonzero(getattr(contexts[-1], 'end_' + c))) for c in CHANNELS[:3]]
    print('F0_END_ROWS ' + json.dumps(dict(graded=graded, counts=counts)), flush=True)
    assert any(counts), 'end-row fixture has no end-row edges on the one-device context'
    compare(reference, record(sim, 2),
            ['graded' if graded else 'uniform', 'cpml', 'end_rows', 2], source_cells(sim))


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('graded', [False, True], ids=['uniform-tfsf', 'graded-pmc'])
def test_legacy_refuses_sheet_before_step(graded, entry, monkeypatch):
    from rfx.boundaries.spec import BoundarySpec
    from rfx.errors import RfxRefusal

    need_devices(2)
    sim = model(graded, 'pec' if graded else 'cpml', 'span', 2, drive=False)
    if graded:
        sim._boundary_spec = BoundarySpec(x='pmc', y='pec', z='pec')
    else:
        sim.add_tfsf_source(f0=5e9, closed_box=True, margin=3)
    monkeypatch.setattr(jax.lax, 'scan', lambda *a, **k: pytest.fail('stepped a refused sheet'))
    lane = 'graded multi-device legacy step' if graded else 'uniform multi-device TF/SF step'
    with pytest.raises(RfxRefusal, match='surface_impedance_f0.*' + lane) as exc:
        options = dict(n_steps=4, devices=jax.devices()[:2], skip_preflight=True)
        if entry == 'forward':
            sim.forward(**options, distributed=True)
        else:
            sim.run(**options, compute_s_params=False)
    expected_frame = ('slab_wrap.py', 'runner_step_kind') if graded else (
        'distributed_v2.py', 'run_distributed')
    assert any((Path(str(frame.path)).name, frame.name) == expected_frame for frame in exc.traceback)
    print('F0_REFUSAL ' + json.dumps(dict(case=[lane, entry], message=str(exc.value))), flush=True)


SPECTRUM_FREQS = np.linspace(0.5e9, 10e9, 20)


def probe_spectra(series, dt):
    """S_p(f) = sum_{n=1..N} x(n) exp(-i 2 pi f n dt) dt, per probe column (NumPy only)."""
    x = np.asarray(series, dtype=np.float64)
    x = x.reshape(x.shape[0], -1)
    n =np.arange(1, x.shape[0] + 1, dtype=np.float64)
    kernel = np.exp(-2j * np.pi * np.outer(SPECTRUM_FREQS, n) * dt) * dt
    return kernel @ x


def spectra_ratios(reference, actual, dt):
    """Per family: max |S_p(n) - S_p(1)| over probes and bins / largest one-device |S_p|."""
    ref = probe_spectra(reference, dt)
    act = probe_spectra(actual, dt)
    assert ref.shape == act.shape and ref.shape[1] % len(CHANNELS) == 0
    component = np.arange(ref.shape[1]) % len(CHANNELS)
    ratios = {}
    for family, members in (('E', (0, 1, 2)), ('H', (3, 4, 5))):
        columns = np.isin(component, members)
        scale = float(np.max(np.abs(ref[:, columns])))
        assert scale > 0, family
        ratios[family] = float(np.max(np.abs(act[:, columns] - ref[:, columns]))) / scale
    return ratios


def seam_cell(sim, position, count):
    grid = sim._build_realized_grid()
    nx = grid.shape[0]
    per = (nx + count - 1) // count
    i = int(_grid_metric.field_index(grid, position, 'ez')[0])
    return i, nx, per


@pytest.mark.parametrize('count', [2, 3])
def test_lumped_port_s_and_spectra_parity(count):
    if len(jax.devices()) < count:
        pytest.skip(f'requires {count} CPU devices')
    from rfx.sources.sources import GaussianPulse

    sim = model(True, 'cpml', 'seam', count, sheet=True, drive=False)
    grid = sim._build_realized_grid()
    per = (grid.shape[0] + count - 1) // count
    position = (grid.node_of(0, per) + .004, .006, .005)
    sim.add_port(position, 'ez', impedance=50.0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    i, nx, per = seam_cell(sim, position, count)
    assert 3 <= i <= nx - 5, (i, nx)
    assert i % per not in (0, per - 1), (i, per)

    options = dict(n_steps=1000, compute_s_params=True, s_param_freqs=SPECTRUM_FREQS,
                   skip_preflight=True)
    reference = sim.run(**options)
    actual = sim.run(**options, devices=jax.devices()[:count])
    s_ref = np.asarray(reference.s_params).astype(np.complex128)
    s_act = np.asarray(actual.s_params).astype(np.complex128)
    assert s_ref.shape == (1, 1, 20) and s_act.shape == (1, 1, 20)
    assert np.all(np.isfinite(s_ref)) and np.all(np.isfinite(s_act))
    delta_s = float(np.max(np.abs(s_act - s_ref)))
    series_ref = np.asarray(reference.time_series, dtype=np.float32)
    series_act = np.asarray(actual.time_series, dtype=np.float32)
    assert series_ref.shape == series_act.shape
    ratios = spectra_ratios(series_ref, series_act, float(grid.dt))
    bitwise = dict(
        fields=all(np.asarray(getattr(reference.state, c)).tobytes()
                   == np.asarray(getattr(actual.state, c)).tobytes() for c in CHANNELS),
        s=s_ref.tobytes() == s_act.tobytes(),
        probes=series_ref.tobytes() == series_act.tobytes())
    print('F0_PORT_PARITY ' + json.dumps(dict(count=count, cell=i, delta_s=delta_s,
          spectra=ratios, min_abs_s=float(np.min(np.abs(s_ref))), bitwise=bitwise)), flush=True)
    assert delta_s <= 1e-4, delta_s
    for family, ratio in ratios.items():
        assert ratio <= 1e-4, f'{count} devices {family}: spectra ratio {ratio} > 1e-4'


@pytest.mark.parametrize('placement', ['seam', 'beside'])
@pytest.mark.parametrize('count', [2, 3])
def test_point_source_spectra_parity(placement, count):
    if len(jax.devices()) < count:
        pytest.skip(f'requires {count} CPU devices')
    sim = model(True, 'cpml', placement, count)
    grid = sim._build_realized_grid()
    options = dict(n_steps=256, compute_s_params=False, skip_preflight=True)
    reference = np.asarray(sim.run(**options).time_series, dtype=np.float32)
    actual = np.asarray(sim.run(**options, devices=jax.devices()[:count]).time_series,
                        dtype=np.float32)
    assert reference.shape == actual.shape
    assert np.all(np.isfinite(reference)) and np.all(np.isfinite(actual))
    ratios = spectra_ratios(reference, actual, float(grid.dt))
    print('F0_SOURCE_SPECTRA ' + json.dumps(dict(placement=placement, count=count,
          spectra=ratios)), flush=True)
    for family, ratio in ratios.items():
        assert ratio <= 1e-4, f'{placement} {count} devices {family}: spectra ratio {ratio} > 1e-4'


def test_msl_with_f0_sheet_fields_and_planes():
    """Compare uniform MSL plus an off-trace f0 sheet with one device at the B1 bars."""
    from tests._multi_device_scene import host_devices
    from tests.unit.runners.test_distributed_msl_ports import observed_scene

    devices = host_devices(2)
    sim, _ = observed_scene('T1', 'uniform', 'laplace', 2)
    sim.add_thin_conductor(
        Box((.004, .0056, .0008), (.008, .0064, .0008)),
        sigma_bulk=100., surface_impedance_f0=5e9)
    options = dict(compute_s_params=False, skip_preflight=True)
    longest = sim.run(n_steps=2100, **options)
    expected = sim.run(n_steps=1050, **options)
    actual = sim.run(n_steps=1050, devices=devices, **options)
    jax.block_until_ready((actual.time_series, actual.dft_planes))
    a, b = np.asarray(actual.time_series), np.asarray(expected.time_series)
    assert a.shape == b.shape
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    peak = np.max(np.abs(np.asarray(longest.time_series)), axis=0)
    delta = np.max(np.abs(a - b), axis=0)
    report = dict(scene='T1', lane='uniform', mode='laplace', n=2, length=1050,
                  probe_peaks=peak.tolist(), probe_deltas=delta.tolist(),
                  probe_relative=(delta / peak).tolist(), planes={})
    assert actual.dft_planes.keys() == expected.dft_planes.keys()
    for name, plane in expected.dft_planes.items():
        reference = np.asarray(plane.accumulator)
        other = np.asarray(actual.dft_planes[name].accumulator)
        assert reference.shape == other.shape
        assert np.all(np.isfinite(reference)) and np.all(np.isfinite(other))
        p = float(np.max(np.abs(reference)))
        d = float(np.max(np.abs(other - reference)))
        report['planes'][name] = dict(peak=p, delta=d, relative=d / p if p else None)
    print('F0_MSL_PARITY ' + json.dumps(report), flush=True)
    assert np.all(peak > 0), report
    assert np.all(delta <= 9e-6 * peak), report
    for name, row in report['planes'].items():
        assert row['peak'] > 0, (name, row)
        assert row['delta'] <= 1e-4 * row['peak'], (name, row)
