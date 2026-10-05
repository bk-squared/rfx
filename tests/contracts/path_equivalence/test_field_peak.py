"""Field-scale mutations without solver runs (2026-10-06 decision)."""
import numpy as np
import pytest

from .comparison import compare
from .execution import _tree


def check(a, b, name, kind):
    report = dict(failures=[], measurements=[])
    _tree(a, b, name, kind, report)
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
