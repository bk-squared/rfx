"""Offline checks for catalog formatting and calibration record provenance."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[3] / 'scripts/showcase/build_catalog.py'
    spec = spec_from_file_location('showcase_catalog_builder', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sci_digits(builder):
    for n, value, expected in (
        (2, 0.003, '3.0×10⁻³'),
        (3, 0.00319, '3.19×10⁻³'),
        (2, 0.00999, '1.0×10⁻²'),
        (3, -1234, '-1.23×10³'),
        (1, 0.00002, '2×10⁻⁵'),
        (2, 0, '0.0×10⁰'),
    ):
        assert builder.sci_digits(n)(value) == expected
    with pytest.raises(ValueError, match='positive'):
        builder.sci_digits(0)
    assert builder.sci(0.00002) == '2×10⁻⁵'


def test_calibration_headlines(builder):
    patch = {
        'hidden': {'eps_r': 2.3, 'sigma_s_per_m': 0.003},
        'start': {'eps_r': 2.2},
        'recovered': {'eps_r': 2.3012, 'sigma_s_per_m': 0.003191},
        'iterates': 51, 'bins': 71, 'band_hz': [2.15e9, 2.50e9],
    }
    records = {
        ('patch-calibration', 'result.json'): patch,
        ('grin-lens', 'mesh_trend.json'): {
            'reported': {'boresight_dbi': [[11, 12]]},
            'no_lens': {'boresight_dbi': [[5, 6]]},
        },
        ('waveguide-filter', 'mesh_trend.json'): {
            'mask_each_mesh': [True, True], 'worst_passband_S11_db': [-20],
            'worst_stopband_S21_db': [-30],
        },
    }
    numbers = builder.headlines(SimpleNamespace(json=lambda *key: records[key]))
    assert set(numbers) == {'patch-calibration', 'waveguide-filter', 'grin-lens'}
    assert list(builder.CASES) == ['patch-calibration', 'waveguide-filter', 'grin-lens']
    assert builder.SHARE == 'https://remilab.cnu.ac.kr/share/f1abd0bff18d/'
    expected = [
        ('hidden permittivity', '2.30', 2.3, '1', 'hidden.eps_r'),
        ('hidden loss', '3.0×10⁻³', 0.003, 'S/m', 'hidden.sigma_s_per_m'),
        ('start permittivity (datasheet)', '2.20', 2.2, '1', 'start.eps_r'),
        ('recovered permittivity', '2.301', 2.3012, '1', 'recovered.eps_r'),
        ('recovered loss', '3.19×10⁻³', 0.003191, 'S/m', 'recovered.sigma_s_per_m'),
        ('updates', '51', 51, '', 'iterates'),
        ('frequency bins', '71', 71, '', 'bins'),
        ('band', '2.15–2.50', [2.15, 2.5], 'GHz', 'band_hz / 1e9'),
    ]
    assert numbers['patch-calibration'] == [
        dict(quantity=q, printed=p, value=v, unit=u,
             routes=['/rfx/showcase/patch-calibration/'],
             **{'from': f'patch-calibration/result.json {field}'})
        for q, p, v, u, field in expected
    ]


def test_source_question_fallback_and_refusal(builder, monkeypatch):
    monkeypatch.setattr(builder, 'CASES', {
        'patch-calibration': ('unused.mdx', ('patch-calibration',)),
    })
    monkeypatch.setattr(builder, 'front_matter', lambda _: {
        'title': 'Calibration', 'description': 'Page question',
    })
    monkeypatch.setattr(builder, 'headlines', lambda _: {'patch-calibration': []})
    record = {'question': 'Explicit question', 'example': 'Example name'}
    share = SimpleNamespace(
        json=lambda *args: record,
        manifest_sha256='digest',
        manifest={'created': 'date', 'cite': 'citation', 'cases': {
            'patch-calibration': {'rfx_commit': 'commit', 'files': {}},
        }},
    )

    def source_question():
        case = builder.build(share)['cases'][0]
        assert case['question'] == 'Page question'
        return case['sources'][0]['question']

    assert source_question() == 'Explicit question'
    del record['question']
    assert source_question() == 'Example name'
    del record['example']
    with pytest.raises(SystemExit, match='patch-calibration/result.json: expected question or example'):
        builder.build(share)
