"""Per-bin pole-tail value checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from rfx.sparams._tail_witness import tail_share_witness

DT = .4
BINS = np.array([.5, 1.])
SOURCE_END = 125


def modes(n):
    t = np.arange(n - SOURCE_END) * DT
    y = np.empty(n, dtype=complex)
    drive_t = np.arange(SOURCE_END) * DT
    y[:SOURCE_END] = np.exp(2j*np.pi*.5*drive_t) + np.exp(2j*np.pi*drive_t)
    y[SOURCE_END:] = (np.exp((-np.pi*.5/50 + 2j*np.pi*.5)*t)
                      + 1e-3*np.exp((-np.pi/5000 + 2j*np.pi)*t))
    return y


def witness(y, bins=BINS, source_end=SOURCE_END):
    return tail_share_witness([('modes', y)], DT, source_end, bins, freq_max=1.)


def test_weak_slow_mode_fails_its_bin_while_fast_bin_passes():
    result = witness(modes(1500))
    assert result.status == 'fail', result
    assert result.share_per_bin[0] < 1e-2
    assert result.share_per_bin[1] > 1e-2
    assert result.worst_freq_hz == 1.
    assert result.db > -40


def test_same_modes_pass_with_long_record():
    result = witness(modes(6000))
    assert result.status == 'pass', result
    assert result.db <= -40


def test_static_offset_and_minus_95_db_residual():
    t = np.arange(1000) * DT
    y = np.exp(2j*np.pi*.5*t)
    y[125:] = 1. + 10**(-95/20)*np.exp((-.1 + 2j*np.pi*.5)*t[:875])
    result = witness(y, bins=[.5])
    assert result.status == 'pass', result
    assert .005 < result.share_per_bin[0] < 1e-2


def test_steady_sinusoid_is_undetermined():
    result = witness(np.exp(2j*np.pi*.5*np.arange(500)*DT), bins=[.5])
    assert result.status == 'undetermined'
    assert result.reason


def test_short_one_percent_two_mode_beat_is_not_pass():
    t = np.arange(200)*DT
    y = np.exp(2j*np.pi*.5*t) + np.exp(2j*np.pi*.505*t)
    result = witness(y, bins=[.5, .505], source_end=0)
    assert result.status == 'undetermined', result
    assert result.reason


def test_records_and_channels_are_order_independent():
    y = modes(1500)
    a = tail_share_witness([('b', y*.3), ('a', np.column_stack([y, 2j*y]))],
                           DT, SOURCE_END, BINS, freq_max=1.)
    b = tail_share_witness([('a', np.column_stack([2j*y, y])), ('b', y*.3)],
                           DT, SOURCE_END, BINS[::-1], freq_max=1.)
    assert a.status == b.status == 'fail'
    np.testing.assert_allclose(a.share_per_bin, b.share_per_bin[::-1], rtol=1e-8)
    assert a.worst_freq_hz == b.worst_freq_hz == 1.


@pytest.mark.parametrize('name,bins,fmax', [
    ('ntff', [3e9], 5e9),
    ('waveguide', np.linspace(8.4e9, 11.6e9, 17), 11.6e9),
])
def test_frozen_record(name, bins, fmax):
    path = Path(__file__).parents[1] / 'fixtures' / 'settling_witness' / f'{name}_worst.npz'
    with np.load(path) as data:
        meta = json.loads(str(data['metadata_json']))
        result = tail_share_witness([('selected_record', data['selected_record'])],
                                    meta['dt'], meta['source_end'], bins, freq_max=fmax)
    print(f'{name}: db={result.db}, status={result.status}, bin={result.worst_freq_hz}, reason={result.reason}')
    assert result.status == 'pass', result
    print('pole contributions:', {key: float(np.max(value))
                                 for key, value in result.pole_contributions.items()})
    assert max(np.max(value) for value in result.pole_contributions.values()) <= 1e-2


def test_identification_exception_is_reported(monkeypatch):
    import rfx.ringdown

    def raises(*args, **kwargs):
        raise np.linalg.LinAlgError('test pencil failure')

    monkeypatch.setattr(rfx.ringdown, 'identify', raises)
    result = witness(modes(300))
    assert result.status == 'undetermined'
    assert 'test pencil failure' in result.reason
    assert np.isnan(result.db)


@pytest.mark.parametrize('source_end', [None, 299, 300])
def test_source_off_window_is_required(source_end):
    result = witness(modes(300), source_end=source_end)
    assert result.status == 'undetermined'
    assert result.reason


def test_absent_without_records_or_bins():
    assert tail_share_witness([], DT, 0, BINS, freq_max=1.).status == 'absent'
    assert witness(modes(300), bins=[]).status == 'absent'


def waveguide_cavity(*, mode_removed=False):
    from rfx import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    sim = Simulation(
        freq_max=12e9, domain=(.06096, .02286, .01016), dx=.00254,
        boundary=BoundarySpec(x=Boundary(lo='cpml', hi='cpml' if mode_removed else 'pec'),
                              y=Boundary(lo='pec', hi='pec'), z=Boundary(lo='pec', hi='pec')),
        cpml_layers=6)
    if not mode_removed:
        sim.add(Box((.03048, 0, 0), (.03302, .00762, .01016)), material='pec')
        sim.add(Box((.03048, .01524, 0), (.03302, .02286, .01016)), material='pec')
    bins = np.linspace(7e9, 11e9, 21)
    sim.add_waveguide_port(.0127, direction='+x', freqs=bins, f0=9e9,
                           bandwidth=.8, name='p')
    return sim, bins


@pytest.mark.parametrize("mode_removed", [False, True])
def test_weakly_coupled_pec_waveguide_cavity(mode_removed):
    from rfx.sources.waveguide_port import settling_db_from_port_records

    sim, bins = waveguide_cavity(mode_removed=mode_removed)
    result = sim.run(n_steps=1000, compute_s_params=False, skip_preflight=True)
    cfg = result.waveguide_ports['p']
    tail = np.asarray(cfg.v_probe_t)[500:]
    fft_freq = np.fft.rfftfreq(len(tail), cfg.dt)
    power = np.abs(np.fft.rfft(tail - tail.mean()))
    power[(fft_freq < bins[0]) | (fft_freq > bins[-1])] = 0
    mode_hz = fft_freq[np.argmax(power)]
    k = int(np.argmin(np.abs(bins - (8.2e9 if mode_removed else mode_hz))))
    db, detail = settling_db_from_port_records([cfg], freqs=bins[k:k+1],
                                              freq_max=12e9, return_detail=True)
    magnitude = float(abs(result.waveguide_sparams['p'].s11[k]))
    print(f'WR90 cavity mode_removed={mode_removed}: FFT mode={mode_hz}, read bin={bins[k]}, |S11|={magnitude}, '
          f'db={db}, status={detail["status"]}, reason={detail["reason"]}')
    if mode_removed:
        assert detail['status'] == 'pass', detail
    else:
        assert np.sqrt(np.mean(tail[-100:]**2)) / np.max(np.abs(tail)) > .1
        assert detail['status'] in {'fail', 'undetermined'}
    assert detail['share_per_bin'].shape == (1,)
    if detail['status'] == 'undetermined':
        assert detail['reason']


def issue_1451_scaled_cavity():
    from rfx import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    fine = .000635
    sim = Simulation(
        freq_max=12.4e9, domain=(36*fine, 16*fine, 129*fine), dx=2*fine,
        boundary=BoundarySpec(x='pec', y='pec', z=Boundary(lo='cpml', hi='pec')),
        cpml_layers=20)
    z0, z1 = 96*fine, 98*fine
    for lo, hi in [((-1, -1, z0), (13*fine, 1, z1)),
                   ((23*fine, -1, z0), (1, 1, z1)),
                   ((-1, -1, z0), (1, 3*fine, z1)),
                   ((-1, 13*fine, z0), (1, 1, z1))]:
        sim.add(Box(lo, hi), material='pec')
    sim.add_waveguide_port(
        32*fine, direction='+z', mode=(1, 0), mode_type='TE', f0=10.3e9,
        bandwidth=.41, waveform='modulated_gaussian',
        freqs=np.linspace(8.2e9, 12.4e9, 421), name='P1')
    return sim


def test_issue_1451_scaled_s_only_cavity_warns_at_worst_bin():
    from rfx.sources.waveguide_port import extract_waveguide_port_waves, settling_db_from_port_records

    sim = issue_1451_scaled_cavity()
    assert sim._ntff is None and not sim._dft_planes
    with pytest.warns(UserWarning, match='settling witness.*worst bin') as caught:
        result = sim.run(n_steps=6000, skip_preflight=True)
    cfg = result.waveguide_ports['P1']
    bins = np.asarray(cfg.freqs)
    incident, _ = extract_waveguide_port_waves(cfg)
    qualifying = np.flatnonzero(np.abs(incident) >= .1*np.max(np.abs(incident)))
    magnitude = np.abs(result.waveguide_sparams['P1'].s11)
    peak = qualifying[np.argmax(magnitude[qualifying])]
    db, detail = settling_db_from_port_records(
        [cfg], freqs=bins[peak:peak+1], freq_max=12.4e9, return_detail=True)
    witness = result.settling_witness
    messages = [str(w.message) for w in caught if 'settling witness' in str(w.message)]
    assert len(messages) == 1
    assert f"worst bin {witness['worst_freq_hz']:.9g} Hz" in messages[0]
    assert witness['status'] in {'fail', 'undetermined'}
    assert detail['status'] in {'fail', 'undetermined'}
    assert detail['worst_freq_hz'] == bins[peak]
    print(f"#1451 scaled: |S11|={magnitude[peak]}, peak_bin={bins[peak]}, "
          f"resonance_db={db}, resonance_status={detail['status']}, "
          f"run_db={result.settling_db}, run_status={witness['status']}, "
          f"worst_bin={witness['worst_freq_hz']}")


def test_wire_s_only_resonance_warns_without_internal_double_fire():
    import warnings
    from tests.unit.sparams.test_ringdown_run import _box, FREQS

    sim = _box('uniform')
    assert sim._ntff is None and not sim._dft_planes
    with pytest.warns(UserWarning, match='settling witness.*worst bin') as caught:
        result = sim.run(n_steps=1500, s_param_freqs=FREQS, skip_preflight=True)
    assert result.s_params is not None
    witness = result.settling_witness
    assert witness['route'] == 's_channels'
    assert witness['status'] in {'fail', 'undetermined'}
    assert np.isfinite(witness['worst_freq_hz'])
    messages = [str(w.message) for w in caught if 'settling witness' in str(w.message)]
    assert len(messages) == 1
    assert f"worst bin {witness['worst_freq_hz']:.9g} Hz" in messages[0]
    sim._internal_probe_indices = {0}
    with warnings.catch_warnings(record=True) as internal:
        warnings.simplefilter('always')
        sim._attach_run_settling_witness(result, n_steps=1500)
    assert not [w for w in internal if 'settling witness' in str(w.message)]
    print(f"wire S-only: db={result.settling_db}, status={witness['status']}, "
          f"worst_bin={witness['worst_freq_hz']}")


@pytest.mark.parametrize('after_source', [True, False])
def test_non_decaying_term_exceeds_read_bin_contribution(after_source):
    t = np.arange(1000)*DT
    y = np.exp(2j*np.pi*.5*t)
    if after_source:
        y[:SOURCE_END] *= 2
    result = witness(y, bins=[.5])
    assert result.status == 'undetermined'
    assert 'non-decaying sinusoidal pole at read bin 0.5 Hz' in result.reason
    assert result.pole_contributions['non-decaying sinusoidal'][0] > 1e-2
    assert 'whole-record plain DFT' in result.pole_rule
