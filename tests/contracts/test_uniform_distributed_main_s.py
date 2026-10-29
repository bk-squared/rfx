"""C1/C2/D1: independent main records, one scan, and identical distributed S."""
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.runners import distributed_v2


def scene(kind, boundary):
    sim = Simulation(freq_max=10e9, domain=(.016, .006, .006), dx=.001,
                     boundary=boundary, cpml_layers=2)
    grid = sim._build_grid()
    seam = (grid.shape[0] + 1) // 2
    x = (seam - grid.pad_x_lo) * grid.dx
    # Lumped Ez's H(i-1) is on rank 0; its E midpoint is on rank 1.
    # The Ex wire straddles the seam; rank 0 alone loses its midpoint.
    position = (x - .001, .003, .003) if kind == 'wire' else (x, .003, .003)
    sim.add_port(position, 'ex' if kind == 'wire' else 'ez', impedance=50.,
                 waveform=GaussianPulse(f0=5e9, bandwidth=1.6),
                 **({'extent': .002} if kind == 'wire' else {}))
    sim.add_probe((x + .002, .003, .003), 'ez')
    return sim


def compare_spectrum(label, got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape
    assert np.isfinite(got).all() and np.isfinite(want).all()
    peak = float(np.max(np.abs(want), initial=0))
    error = float(np.max(np.abs(got.astype(np.complex128) - want), initial=0))
    print(f'{label}: error={error:.12g} peak={peak:.12g} bar={1e-4 * peak:.12g}', flush=True)
    assert error <= 1e-4 * peak, (label, error, peak)


@pytest.mark.parametrize('boundary', ['cpml', 'pec'])
@pytest.mark.parametrize('kind', ['lumped', 'wire'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_main_record_and_s(kind, entry, boundary):
    devices = jax.devices('cpu')[:2]
    assert len(devices) == 2, 'configure two host devices before Python starts'
    bins = np.asarray([2., 4., 6., 8.], dtype=np.float32) * 1e9
    options = dict(n_steps=150, skip_preflight=True)

    def call(distributed):
        sim = scene(kind, boundary)
        if entry == 'run':
            return sim.run(**options, s_param_freqs=bins, compute_s_params=True,
                           **({'devices': devices} if distributed else {}))
        return sim.forward(**options, port_s11_freqs=bins,
                           **({'devices': devices, 'distributed': True} if distributed
                              else {'checkpoint': False}))

    # All expected samples, accumulator values and S come from a public
    # one-device solve. No cut, reduction, sampler or DFT helper is the oracle.
    with patch.object(distributed_v2, 'run_distributed', wraps=distributed_v2.run_distributed) as runs:
        actual = call(True)
    print(f'{entry}/{kind}/{boundary}: runs={runs.call_count}', flush=True)
    assert runs.call_count == 1
    if entry == 'run':
        fwd = scene(kind, boundary).forward(
            **options, distributed=True, devices=devices, checkpoint=False,
            port_s11_freqs=actual.freqs)
        equal = np.array_equal(actual.s_params, fwd.s_params)
        print(f'D1/{kind}/{boundary}: run/forward array_equal={equal}', flush=True)
        assert equal, (kind, boundary, actual.s_params, fwd.s_params)
    expected = call(False)
    assert len(actual.sparam_time_records) == len(expected.sparam_time_records) == 1
    for got, want in zip(actual.sparam_time_records, expected.sparam_time_records, strict=True):
        assert (got.id, got.kind, got.channels) == (want.id, want.kind, want.channels)
        a, b = np.asarray(got.samples), np.asarray(want.samples)
        assert a.shape == b.shape
        assert np.isfinite(a).all()
        for index, channel in enumerate(want.channels):
            peak = np.float32(np.max(np.abs(b[:, index])))
            assert peak > 0, channel
            ulp = float(np.spacing(peak))
            error = float(np.max(np.abs(a[:, index].astype(np.float64) - b[:, index])))
            print(f'{entry}/{kind}/{boundary}/{channel}: error={error:.12g} '
                  f'peak_ULP={error / ulp:.12g} bar=9', flush=True)
            assert error <= 9 * ulp, (entry, kind, boundary, channel, error / ulp)
    field = kind + '_port_sparams'
    got, want = getattr(actual, field), getattr(expected, field)
    assert got is not None and want is not None and len(got) == len(want) == 1
    for (meta, accs), (ref_meta, ref_accs) in zip(got, want, strict=True):
        assert meta._fields == ref_meta._fields
        for name in meta._fields:
            np.testing.assert_array_equal(getattr(meta, name), getattr(ref_meta, name))
        for index, (a, b) in enumerate(zip(accs, ref_accs, strict=True)):
            compare_spectrum(f'{entry}/{kind}/{boundary}/acc{index}', a, b)
    compare_spectrum(f'{entry}/{kind}/{boundary}/S', actual.s_params, expected.s_params)
    np.testing.assert_array_equal(actual.freqs, expected.freqs)
