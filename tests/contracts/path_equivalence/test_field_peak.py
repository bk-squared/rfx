"""Field-scale mutations without solver runs (2026-10-06 decision)."""
import numpy as np
import pytest

from .comparison import VACUUM_IMPEDANCE, compare, component_peaks, wave_impedance_range
from .execution import _tree


def check(a, b, name, kind, *, paired_impedance=None):
    report = dict(failures=[], measurements=[])
    _tree(a, b, name, kind, report, paired_impedance=paired_impedance)
    return report


@pytest.mark.parametrize('name,keys', [
    ('state', ('ex', 'ey', 'ez')),
    ('state', ('hx', 'hy', 'hz')),
    ('probe.vector', ('ex', 'ey', 'ez')),
    ('dft_planes.vector', ('hx', 'hy', 'hz')),
    ('flux_monitors.surface', ('e1_dft', 'e2_dft')),
    ('flux_monitors.surface', ('h1_dft', 'h2_dft')),
])
@pytest.mark.parametrize('kind', ['step', 'accumulated'])
@pytest.mark.parametrize('reverse', [False, True])
def test_component_field_peak_mutations(name, keys, kind, reverse):
    peak = np.float32(2.**20)
    ulp = np.spacing(peak)
    a = {key: np.array([1e-8], dtype=np.float32) for key in keys}
    a[keys[0]] = np.array([peak], dtype=np.float32)
    b = {key: value.copy() for key, value in a.items()}
    for count in (1, 2):
        b[keys[-1]] = a[keys[-1]] + count * ulp
        left, right = (b, a) if reverse else (a, b)
        report = check(left, right, name, kind)
        assert not report['failures']
        assert {m['peak'] for m in report['measurements']} == {peak}
        with pytest.raises(AssertionError):
            compare(a[keys[-1]], b[keys[-1]], record='old-own-peak',
                    kind=kind, measurements=[])
    delta = 10 * ulp if kind == 'step' else 1.01e-4 * peak
    b[keys[-1]] = a[keys[-1]] + delta
    report = check(b, a, name, kind) if reverse else check(a, b, name, kind)
    assert len(report['failures']) == 1
    assert report['failures'][0].startswith(f'{name}.{keys[-1]}:')


def test_field_peak_includes_both_traces_and_separates_e_h():
    a = dict(ex=np.array([1.]), ey=np.array([1e-8]), hx=np.array([1e-6]))
    b = dict(ex=np.array([2.]), ey=np.array([1e-8]), hx=np.array([1.1e-6]))
    for left, right in ((a, b), (b, a)):
        report = check(left, right, 'state', 'step')
        peaks = {m['record']: m['peak'] for m in report['measurements']}
        assert peaks == {'state.ex': 2., 'state.ey': 2., 'state.hx': 1.1e-6}
        assert len(report['failures']) == 2


@pytest.mark.parametrize('kind', ['step', 'accumulated'])
@pytest.mark.parametrize('field,slot', [('E', 1), ('H', 3)])
def test_ntff_packed_fields_and_faces(kind, field, slot):
    peak = np.float32(2.**20)
    a = dict(x_lo=np.array([[[[peak, 1e-8, peak / 1024, 1e-8]]]], dtype=np.complex64),
             y_hi=np.full((1, 1, 1, 4), 1e-8, dtype=np.complex64),
             c_x_lo=np.full((1, 1, 1, 4), 1e30, dtype=np.complex64))
    field_peak = peak if field == 'E' else peak / 1024
    ulp = np.spacing(np.float32(field_peak))
    b = {key: value.copy() for key, value in a.items()}
    b['c_x_lo'] *= -1
    for count in (1, 2):
        b['y_hi'][..., slot] = a['y_hi'][..., slot] + count * ulp
        report = check(a, b, 'ntff_data', kind)
        assert not report['failures']
        assert len(report['measurements']) == 4
        for m in report['measurements']:
            assert m['peak'] == (peak if m['record'].endswith('.E') else peak / 1024)
        with pytest.raises(AssertionError):
            compare(a['y_hi'][..., slot], b['y_hi'][..., slot], record='old-component',
                    kind=kind, measurements=[])
    delta = 10 * ulp if kind == 'step' else 1.01e-4 * field_peak
    b['y_hi'][..., slot] = a['y_hi'][..., slot] + delta
    report = check(a, b, 'ntff_data', kind)
    assert len(report['failures']) == 1
    assert report['failures'][0].startswith(f'ntff_data.y_hi.{field}:')


def test_grouping_preserves_exact_missing_shape_and_nonfinite_checks():
    a = dict(ex=np.array([1.]), ey=np.array([1e-8]))
    b = dict(ex=a['ex'], ey=np.array([1e-8 + 1e-9]))
    assert check(a, b, 'state', 'exact')['failures']
    for bad in (None, np.array([np.nan]), np.ones(2)):
        assert check(a, dict(b, ey=bad), 'state', 'step')['failures']
    assert check(a, dict(ex=a['ex']), 'state', 'step')['failures']


def test_existing_leaf_witness_is_checked_in_field_units():
    from .reporting import KnownFinding, assert_record
    a = dict(e1_dft=np.array([1.]), e2_dft=np.array([1e-2]))
    b = dict(e1_dft=np.array([1.]), e2_dft=np.array([.0105]))
    name = 'flux_monitors.surface'
    report = check(a, b, name, 'accumulated')
    key = name + '.e2_dft'
    witness = {key: dict(relative=(.0105 - .01) / .0105, relative_bar=1e-4)}
    with pytest.raises(KnownFinding):
        assert_record(report, 'observers', [key + ': numeric'], witness)
    # Drift above 1e-4 of E's peak must not hide under the existing xfail.
    b['e2_dft'] += .001
    report = check(a, b, name, 'accumulated')
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        assert_record(report, 'observers', [key + ': numeric'], witness)


def test_accumulated_ten_ulps_is_below_the_unchanged_bar():
    peak = np.float32(2.**20)
    a = dict(ex=np.array([peak]), ey=np.array([1e-8], dtype=np.float32))
    b = dict(a, ey=a['ey'] + 10 * np.spacing(peak))
    assert not check(a, b, 'state', 'accumulated')['failures']
    assert check(a, b, 'state', 'step')['failures']


def test_large_e_does_not_hide_flux_or_ntff_h_error():
    a = dict(e1_dft=np.array([1e9]), e2_dft=np.array([1.]),
             h1_dft=np.array([1e-8]), h2_dft=np.array([1e-8]))
    b = dict(a, h2_dft=np.array([1.05e-8]))
    report = check(a, b, 'flux_monitors.surface', 'accumulated')
    assert len(report['failures']) == 1
    assert 'h2_dft:' in report['failures'][0]
    face_a = np.array([[[[1e9, 1., 1e-8, 1e-8]]]], dtype=np.complex64)
    face_b = face_a.copy()
    face_b[..., 3] *= 1.05
    report = check(dict(x_lo=face_a), dict(x_lo=face_b), 'ntff_data', 'accumulated')
    assert len(report['failures']) == 1
    assert 'x_lo.H:' in report['failures'][0]
    compare(face_a, face_b, record='old-mixed-face', kind='accumulated', measurements=[])


@pytest.mark.parametrize('keys', [('ex', 'hx'), ('e1_dft', 'h2_dft')])
@pytest.mark.parametrize('reverse', [False, True])
def test_paired_static_e_small_h_rounding(keys, reverse):
    e, h = keys
    h_peak = np.float32(1e-8)
    a = {e: np.array([1.]), h: np.array([h_peak], dtype=np.float32)}
    b = dict(a, **{h: a[h] - 10000 * np.spacing(h_peak)})
    left, right = (b, a) if reverse else (a, b)
    assert len(check(left, right, 'state', 'step')['failures']) == 1
    report = check(left, right, 'state', 'step', paired_impedance=VACUUM_IMPEDANCE)
    assert not report['failures']
    assert component_peaks(left, right, paired_impedance=VACUUM_IMPEDANCE) == {
        e: 1., h: 1. / VACUUM_IMPEDANCE}


@pytest.mark.parametrize('kind', ['step', 'accumulated'])
@pytest.mark.parametrize('keys', [('ex', 'hx'), ('e2_dft', 'h1_dft')])
def test_paired_scale_real_h_error_fails(keys, kind):
    e, h = keys
    a = {e: np.array([1.]), h: np.array([1e-8])}
    paired_peak = 1. / VACUUM_IMPEDANCE
    delta = (10 * np.spacing(np.float32(paired_peak)) if kind == 'step'
             else 1.01e-4 * paired_peak)
    b = dict(a, **{h: a[h] + delta})
    report = check(a, b, 'state', kind, paired_impedance=VACUUM_IMPEDANCE)
    assert len(report['failures']) == 1
    assert report['failures'][0].startswith(f'state.{h}:')


@pytest.mark.parametrize('kind', ['step', 'accumulated'])
@pytest.mark.parametrize('count', [2, 10])
def test_paired_propagating_peaks_and_verdict_unchanged(kind, count):
    h_peak = 1. / VACUUM_IMPEDANCE
    a = dict(ex=np.array([1., 0.]), hx=np.array([h_peak, 0.]))
    delta = (count * np.spacing(np.float32(h_peak)) if kind == 'step'
             else count * 0.2e-4 * h_peak)
    b = dict(a, hx=np.array([h_peak, delta]))
    assert component_peaks(a, b) == component_peaks(a, b, paired_impedance=VACUUM_IMPEDANCE)
    default = check(a, b, 'state', kind)
    paired = check(a, b, 'state', kind, paired_impedance=VACUUM_IMPEDANCE)
    assert default == paired
    assert bool(paired['failures']) == (count == 10)


@pytest.mark.parametrize('keys', [('ex', 'ez'), ('hx', 'hy'),
                                 ('e1_dft', 'e2_dft'), ('h1_dft', 'h2_dft')])
def test_paired_absent_partner_unchanged(keys):
    a = {keys[0]: np.array([1.]), keys[1]: np.array([1e-8])}
    b = dict(a, **{keys[1]: np.array([2e-8])})
    assert component_peaks(a, b) == component_peaks(a, b, paired_impedance=VACUUM_IMPEDANCE)
    assert check(a, b, 'state', 'step') == check(
        a, b, 'state', 'step', paired_impedance=VACUUM_IMPEDANCE)


def test_paired_h_dominated_nested_dft_groups():
    a = dict(e1_dft=np.array([1e-8]), e2_dft=np.array([0.]),
             h1_dft=np.array([1.]), h2_dft=np.array([0.]),
             ex=np.array([2.]), hx=np.array([1e-8]))
    b = dict(a, e2_dft=np.array([2 * np.spacing(np.float32(VACUUM_IMPEDANCE))]))
    peaks = component_peaks(a, b, paired_impedance=VACUUM_IMPEDANCE)
    assert peaks == dict(e1_dft=VACUUM_IMPEDANCE, e2_dft=VACUUM_IMPEDANCE,
                         h1_dft=1., h2_dft=1., ex=2., hx=2. / VACUUM_IMPEDANCE)
    assert check({'surface': [a]}, {'surface': [b]}, 'flux_monitors', 'step')['failures']
    assert not check({'surface': [a]}, {'surface': [b]}, 'flux_monitors', 'step',
                     paired_impedance=VACUUM_IMPEDANCE)['failures']


@pytest.mark.parametrize('impedance', [0., -1., np.inf, np.nan])
def test_paired_impedance_requires_finite_positive_value(impedance):
    with pytest.raises(ValueError, match='finite and positive'):
        component_peaks({}, {}, paired_impedance=impedance)


@pytest.mark.parametrize('count', [2, 10])
def test_paired_dielectric_wave_range_preserves_peaks_and_verdict(count):
    h = 0.008394009044264874
    a = dict(ex=np.array([1., 0.]), hx=np.array([h, 0.]))
    b = dict(a, hx=np.array([h, count * np.spacing(np.float32(h))]))
    bounds = wave_impedance_range(10)
    assert bounds == pytest.approx((119.13258548169428, 376.73031366686166))
    own = component_peaks(a, b)
    paired = component_peaks(a, b, paired_impedance=bounds)
    assert own == paired == {'ex': 1., 'hx': h}
    default = check(a, b, 'state', 'step')
    assert default == check(a, b, 'state', 'step', paired_impedance=bounds)
    assert bool(default['failures']) == (count == 10)
    vacuum = component_peaks(a, b, paired_impedance=VACUUM_IMPEDANCE)
    assert vacuum == pytest.approx({'ex': 3.1622776601683795, 'hx': h})


@pytest.mark.parametrize('reverse', [False, True])
def test_paired_time_domain_h_dominated(reverse):
    a = dict(ex=np.array([1e-8]), hx=np.array([1.]))
    b = dict(a, ex=np.array([1e-8 + 2 * 0.00000762939453125]))
    left, right = (b, a) if reverse else (a, b)
    assert check(left, right, 'state', 'step')['failures']
    report = check(left, right, 'state', 'step', paired_impedance=(100., 400.))
    assert not report['failures']
    assert {m['record']: m['peak'] for m in report['measurements']} == {
        'state.ex': 100., 'state.hx': 1.}
    b = dict(a, ex=np.array([1e-8 + 10 * 0.00000762939453125]))
    report = check(a, b, 'state', 'step', paired_impedance=(100., 400.))
    assert len(report['failures']) == 1
    assert report['failures'][0].startswith('state.ex:')


@pytest.mark.parametrize('kind', ['step', 'accumulated'])
def test_paired_ntff_small_h_rounding(kind):
    a = dict(x_lo=np.array([[[[1., 0., 1e-8, 0.]]]], dtype=np.complex64))
    b = dict(x_lo=a['x_lo'].copy())
    b['x_lo'][..., 2] -= 10000 * np.spacing(np.float32(1e-8))
    assert len(check(a, b, 'ntff_data', kind)['failures']) == 1
    report = check(a, b, 'ntff_data', kind, paired_impedance=(100., 400.))
    assert not report['failures']
    assert {m['record']: m['peak'] for m in report['measurements']} == {
        'ntff_data.x_lo.E': 1., 'ntff_data.x_lo.H': .0025}


def test_paired_nonfinite_partner_keeps_own_failure():
    a = dict(ex=np.array([1.]), hx=np.array([np.inf]))
    b = dict(a, ex=np.array([2.]))
    assert component_peaks(a, b, paired_impedance=(100., 400.)) == {
        'ex': 2., 'hx': np.inf}
    report = check(a, b, 'state', 'step', paired_impedance=(100., 400.))
    assert len(report['failures']) == 2
    assert any(line.startswith('state.ex:') for line in report['failures'])


@pytest.mark.parametrize('impedance', [True, np.bool_(True), (400., 100.),
                                      (0., 400.), (100., np.inf), (True, 400.)])
def test_paired_invalid_impedances(impedance):
    with pytest.raises(ValueError):
        component_peaks({}, {}, paired_impedance=impedance)


@pytest.mark.parametrize('impedance', [np.float32(100.), (np.float32(100.), np.float32(400.))])
def test_paired_numpy_impedance_returns_python_floats(impedance):
    a = dict(ex=np.array([0.]), hx=np.array([1.]))
    peaks = component_peaks(a, a, paired_impedance=impedance)
    assert peaks == {'ex': 100., 'hx': 1.}
    assert all(type(value) is float for value in peaks.values())


@pytest.mark.parametrize('eps', [True, 0., .5, np.inf, np.nan])
def test_wave_impedance_range_rejects_invalid_permittivity(eps):
    with pytest.raises(ValueError):
        wave_impedance_range(eps)
