"""run() and forward() are one pipeline: the leader's C1–C5 contract.

The literal exceptions are progress toward zero at 2.0, not tolerances.
No solver helper supplies an expected value: the other fresh entry does.
"""
import json
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

from tests.contracts.run_forward_census import (
    BUILDERS, SEAMS, capture, compare, compare_l2, l2_differences,
    normalize_inputs, observables, run_l2, port_s_pair,
)

KNOWN_ROWS = (
    ('P-lumped', 'uniform', 'L2', 'D9', "#1373 D9 (ports): run()'s driver discards each drive run's port records (raw V/I/V_ref accumulators, per-step port records) after forming S; forward(port_s11_freqs) returns them"),
    ('P-lumped2', 'uniform', 'L2', 'D9', "#1373 D9 (ports): run()'s driver discards each drive run's port records (raw V/I/V_ref accumulators, per-step port records) after forming S; forward(port_s11_freqs) returns them"),
    ('C-smooth', 'graded', 'L1', 'A6', '#1373 A6 (nu): subpixel smoothing is a run() keyword; forward() cannot request it'),
    ('C-smooth', 'graded', 'admission', 'A6', '#1373 A6 (nu): subpixel smoothing is a run() keyword; forward() cannot request it'),
    ('C-smooth', 'uniform', 'L1', 'A6', '#1373 A6 (nu): subpixel smoothing is a run() keyword; forward() cannot request it'),
    ('C-smooth', 'uniform', 'admission', 'A6', '#1373 A6 (nu): subpixel smoothing is a run() keyword; forward() cannot request it'),
    ('P-lumped', 'multi-device-uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-lumped', 'multi-device-uniform', 'L2', 'D8', '#1373 D8 (ports): run computes the port S-matrix by default; forward returns no S without port_s11_freqs'),
    ('P-lumped', 'uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-lumped2', 'uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-waveguide', 'graded', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-waveguide', 'multi-device-graded', 'L1', 'A5', '#1373 A5 (acceleration): run falls back for a waveguide port while distributed forward refuses'),
    ('P-waveguide', 'multi-device-graded', 'admission', 'A5', '#1373 A5 (acceleration): run falls back for a waveguide port while distributed forward refuses'),
    ('P-waveguide', 'multi-device-uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-waveguide', 'uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-wire', 'multi-device-graded', 'admission', 'A2', 'A2 expected: equal S requests refuse the extra source; forward without an S request admits'),
    ('P-wire', 'graded', 'admission', 'A2', 'A2 expected: equal S requests give equal refusals; forward without an S request admits'),
    ('P-wire', 'multi-device-uniform', 'admission', 'A2', 'A2 expected: equal S requests give equal refusals; forward without an S request admits'),
    ('P-wire', 'uniform', 'admission', 'A2', 'A2 expected: equal S requests give equal refusals; forward without an S request admits'),
)

ASSEMBLY_ROWS = (
    ('B-asym', 'graded'),
    ('B-asym', 'multi-device-graded'),
    ('B-asym', 'uniform'),
    ('B-closed', 'graded'),
    ('B-closed', 'multi-device-graded'),
    ('B-closed', 'uniform'),
    ('B-cpml', 'graded'),
    ('B-cpml', 'multi-device-graded'),
    ('B-cpml', 'uniform'),
    ('B-pecsides', 'graded'),
    ('B-pecsides', 'multi-device-graded'),
    ('B-pecsides', 'uniform'),
    ('B-periodic', 'graded'),
    ('B-periodic', 'uniform'),
    ('B-pmcone', 'graded'),
    ('B-pmcone', 'multi-device-graded'),
    ('B-pmcone', 'uniform'),
    ('B-pmcpair', 'graded'),
    ('B-pmcpair', 'multi-device-graded'),
    ('B-pmcpair', 'uniform'),
    ('B-upml', 'uniform'),
    ('C-film', 'graded'),
    ('C-film', 'multi-device-graded'),
    ('C-film', 'uniform'),
    ('C-pecbox', 'graded'),
    ('C-pecbox', 'multi-device-graded'),
    ('C-pecbox', 'uniform'),
    ('C-pinned', 'graded'),
    ('C-pinned', 'multi-device-graded'),
    ('C-pinned', 'uniform'),
    ('C-sheet', 'graded'),
    ('C-sheet', 'multi-device-graded'),
    ('C-sheet', 'uniform'),
    ('C-smooth', 'graded'),
    ('C-smooth', 'uniform'),
    ('C-wire', 'graded'),
    ('C-wire', 'multi-device-graded'),
    ('C-wire', 'uniform'),
    ('M-debye', 'graded'),
    ('M-debye', 'multi-device-graded'),
    ('M-debye', 'uniform'),
    ('M-diel', 'graded'),
    ('M-diel', 'multi-device-graded'),
    ('M-diel', 'uniform'),
    ('M-drude', 'graded'),
    ('M-drude', 'multi-device-graded'),
    ('M-drude', 'uniform'),
    ('M-lorentz', 'graded'),
    ('M-lorentz', 'multi-device-graded'),
    ('M-lorentz', 'uniform'),
    ('M-lossy', 'graded'),
    ('M-lossy', 'multi-device-graded'),
    ('M-lossy', 'uniform'),
    ('M-mu', 'graded'),
    ('M-mu', 'multi-device-graded'),
    ('M-mu', 'uniform'),
    ('O-cm', 'graded'),
    ('O-cm', 'uniform'),
    ('O-dft', 'graded'),
    ('O-dft', 'uniform'),
    ('O-flux', 'graded'),
    ('O-flux', 'uniform'),
    ('O-ntff', 'graded'),
    ('O-ntff', 'uniform'),
    ('O-vector', 'graded'),
    ('O-vector', 'multi-device-graded'),
    ('O-vector', 'uniform'),
    ('P-floquet', 'uniform'),
    ('P-lumped', 'graded'),
    ('P-lumped', 'uniform'),
    ('P-lumped2', 'graded'),
    ('P-lumped2', 'uniform'),
    ('P-msl', 'graded'),
    ('P-msl', 'uniform'),
    ('P-waveguide', 'graded'),
    ('P-waveguide', 'multi-device-graded'),
    ('P-waveguide', 'uniform'),
    ('PW-A', 'graded'),
    ('PW-A', 'uniform'),
    ('PW-B', 'uniform'),
    ('PW-closed', 'uniform'),
    ('PW-oblique', 'uniform'),
    ('R-rlc', 'graded'),
    ('R-rlc', 'uniform'),
    ('S-field', 'graded'),
    ('S-field', 'multi-device-graded'),
    ('S-field', 'uniform'),
    ('S-polarized', 'graded'),
    ('S-polarized', 'multi-device-graded'),
    ('S-polarized', 'uniform'),
)

FIELD_ROWS = (
    ('B-asym', 'graded'),
    ('B-asym', 'multi-device-graded'),
    ('B-asym', 'multi-device-uniform'),
    ('B-asym', 'uniform'),
    ('B-closed', 'graded'),
    ('B-closed', 'multi-device-graded'),
    ('B-closed', 'multi-device-uniform'),
    ('B-closed', 'uniform'),
    ('B-cpml', 'graded'),
    ('B-cpml', 'multi-device-graded'),
    ('B-cpml', 'multi-device-uniform'),
    ('B-cpml', 'uniform'),
    ('B-pecsides', 'graded'),
    ('B-pecsides', 'multi-device-graded'),
    ('B-pecsides', 'multi-device-uniform'),
    ('B-pecsides', 'uniform'),
    ('B-periodic', 'graded'),
    ('B-periodic', 'uniform'),
    ('B-pmcone', 'graded'),
    ('B-pmcone', 'multi-device-graded'),
    ('B-pmcone', 'multi-device-uniform'),
    ('B-pmcone', 'uniform'),
    ('B-pmcpair', 'graded'),
    ('B-pmcpair', 'multi-device-graded'),
    ('B-pmcpair', 'multi-device-uniform'),
    ('B-pmcpair', 'uniform'),
    ('B-upml', 'uniform'),
    ('C-film', 'graded'),
    ('C-film', 'multi-device-graded'),
    ('C-film', 'multi-device-uniform'),
    ('C-film', 'uniform'),
    ('C-pecbox', 'graded'),
    ('C-pecbox', 'multi-device-graded'),
    ('C-pecbox', 'multi-device-uniform'),
    ('C-pecbox', 'uniform'),
    ('C-pinned', 'graded'),
    ('C-pinned', 'multi-device-graded'),
    ('C-pinned', 'multi-device-uniform'),
    ('C-pinned', 'uniform'),
    ('C-sheet', 'graded'),
    ('C-sheet', 'multi-device-graded'),
    ('C-sheet', 'multi-device-uniform'),
    ('C-sheet', 'uniform'),
    ('C-wire', 'graded'),
    ('C-wire', 'multi-device-graded'),
    ('C-wire', 'multi-device-uniform'),
    ('C-wire', 'uniform'),
    ('M-debye', 'graded'),
    ('M-debye', 'multi-device-graded'),
    ('M-debye', 'multi-device-uniform'),
    ('M-debye', 'uniform'),
    ('M-diel', 'graded'),
    ('M-diel', 'multi-device-graded'),
    ('M-diel', 'multi-device-uniform'),
    ('M-diel', 'uniform'),
    ('M-drude', 'graded'),
    ('M-drude', 'multi-device-graded'),
    ('M-drude', 'multi-device-uniform'),
    ('M-drude', 'uniform'),
    ('M-lorentz', 'graded'),
    ('M-lorentz', 'multi-device-graded'),
    ('M-lorentz', 'multi-device-uniform'),
    ('M-lorentz', 'uniform'),
    ('M-lossy', 'graded'),
    ('M-lossy', 'multi-device-graded'),
    ('M-lossy', 'multi-device-uniform'),
    ('M-lossy', 'uniform'),
    ('M-mu', 'graded'),
    ('M-mu', 'multi-device-graded'),
    ('M-mu', 'multi-device-uniform'),
    ('M-mu', 'uniform'),
    ('O-cm', 'graded'),
    ('O-cm', 'uniform'),
    ('O-dft', 'graded'),
    ('O-dft', 'uniform'),
    ('O-flux', 'graded'),
    ('O-flux', 'uniform'),
    ('O-ntff', 'graded'),
    ('O-ntff', 'multi-device-uniform'),
    ('O-ntff', 'uniform'),
    ('O-vector', 'graded'),
    ('O-vector', 'multi-device-graded'),
    ('O-vector', 'multi-device-uniform'),
    ('O-vector', 'uniform'),
    ('P-floquet', 'uniform'),
    ('P-lumped', 'graded'),
    ('P-lumped', 'multi-device-uniform'),
    ('P-lumped', 'uniform'),
    ('P-lumped2', 'graded'),
    ('P-lumped2', 'uniform'),
    ('P-msl', 'graded'),
    ('P-msl', 'uniform'),
    ('P-waveguide', 'graded'),
    ('P-waveguide', 'multi-device-uniform'),
    ('P-waveguide', 'uniform'),
    ('PW-A', 'multi-device-uniform'),
    ('PW-A', 'uniform'),
    ('PW-B', 'multi-device-uniform'),
    ('PW-B', 'uniform'),
    ('PW-closed', 'uniform'),
    ('PW-oblique', 'multi-device-uniform'),
    ('PW-oblique', 'uniform'),
    ('R-rlc', 'graded'),
    ('R-rlc', 'uniform'),
    ('S-field', 'graded'),
    ('S-field', 'multi-device-graded'),
    ('S-field', 'multi-device-uniform'),
    ('S-field', 'uniform'),
    ('S-polarized', 'graded'),
    ('S-polarized', 'multi-device-graded'),
    ('S-polarized', 'multi-device-uniform'),
    ('S-polarized', 'uniform'),
)

ADMISSION_ROWS = (
    ('B-asym', 'multi-device-uniform'),
    ('B-closed', 'multi-device-uniform'),
    ('B-cpml', 'multi-device-uniform'),
    ('B-pecsides', 'multi-device-uniform'),
    ('B-pmcone', 'multi-device-uniform'),
    ('B-pmcpair', 'multi-device-uniform'),
    ('C-film', 'multi-device-uniform'),
    ('C-pecbox', 'multi-device-uniform'),
    ('C-pinned', 'multi-device-uniform'),
    ('C-sheet', 'multi-device-uniform'),
    ('C-smooth', 'graded'),
    ('C-smooth', 'uniform'),
    ('C-wire', 'multi-device-uniform'),
    ('M-debye', 'multi-device-uniform'),
    ('M-diel', 'multi-device-uniform'),
    ('M-drude', 'multi-device-uniform'),
    ('M-lorentz', 'multi-device-uniform'),
    ('M-lossy', 'multi-device-uniform'),
    ('M-mu', 'multi-device-uniform'),
    ('O-ntff', 'multi-device-uniform'),
    ('O-vector', 'multi-device-uniform'),
    ('P-lumped', 'multi-device-uniform'),
    ('P-waveguide', 'multi-device-graded'),
    ('P-waveguide', 'multi-device-uniform'),
    ('P-wire', 'graded'),
    ('P-wire', 'multi-device-uniform'),
    ('P-wire', 'uniform'),
    ('PW-A', 'graded'),
    ('PW-A', 'multi-device-uniform'),
    ('PW-B', 'multi-device-uniform'),
    ('PW-oblique', 'multi-device-uniform'),
    ('S-field', 'multi-device-uniform'),
    ('S-polarized', 'multi-device-uniform'),
    ('P-wire', 'multi-device-graded'),
)


BOTH_REFUSE_ROWS = (
    ('B-periodic', 'multi-device-graded'),
    ('B-periodic', 'multi-device-uniform'),
    ('B-upml', 'graded'),
    ('B-upml', 'multi-device-graded'),
    ('B-upml', 'multi-device-uniform'),
    ('C-smooth', 'multi-device-graded'),
    ('C-smooth', 'multi-device-uniform'),
    ('O-cm', 'multi-device-graded'),
    ('O-cm', 'multi-device-uniform'),
    ('O-dft', 'multi-device-graded'),
    ('O-dft', 'multi-device-uniform'),
    ('O-flux', 'multi-device-graded'),
    ('O-flux', 'multi-device-uniform'),
    ('O-ntff', 'multi-device-graded'),
    ('P-coax', 'graded'),
    ('P-coax', 'multi-device-graded'),
    ('P-coax', 'multi-device-uniform'),
    ('P-coax', 'uniform'),
    ('P-floquet', 'multi-device-uniform'),
    ('P-lumped2', 'multi-device-uniform'),
    ('P-msl', 'multi-device-graded'),
    ('P-msl', 'multi-device-uniform'),
    ('PW-A', 'multi-device-graded'),
    ('PW-B', 'graded'),
    ('PW-B', 'multi-device-graded'),
    ('PW-closed', 'graded'),
    ('PW-closed', 'multi-device-graded'),
    ('PW-closed', 'multi-device-uniform'),
    ('PW-oblique', 'graded'),
    ('PW-oblique', 'multi-device-graded'),
    ('R-rlc', 'multi-device-graded'),
    ('R-rlc', 'multi-device-uniform'),
)

# Item 20: only common time_series differences on this L2 pair
# are consequences. Other measurements, outcomes, and L1 remain exact;
# O-flux has equal fields and is not a consequence pair.
CONSEQUENCE_PAIRS = (
)
ID_PATTERNS = {
    'D4': {'L1': ('/carry_init/flux_monitors/*', '/init/flux_monitors/*',
                  '/ctx/flux_meta/*', '/invariants/ctx/flux_meta/*',
                  '/ctx/use_flux_monitors', '/invariants/ctx/use_flux_monitors'),
           'L2': ('flux_monitors/*',)},
    'D5': {'L1': (), 'L2': ('settling_witness/*',)},
    'D7': {'L1': (), 'L2': ('waveguide_ports/*', 'waveguide_sparams/*')},
    'D8': {'L1': (), 'L2': ('s_params', 'freqs', 'freqs_hz')},
    'D9': {'L1': (), 'L2': ('lumped_port_sparams*', 'sparam_time_records*')},
    'A2': {'run': ('raised', 'NotImplementedError',
            'Lumped/wire S-matrix requests do not support plain sources / 0-ohm ports'),
           'forward': ('admitted',)},
    'A5': {'run': ('admitted',), 'forward': ('raised', 'NotImplementedError',
            'Waveguide ports are not supported on the distributed forward path')},
    'A6': {'run': ('admitted',), 'forward': ('raised', 'TypeError',
            "'subpixel_smoothing' is not a recognised forward() keyword argument")},
}
DISTRIBUTED_L1_SLOW = False


def parameters(rows, level, *, slow=False):
    return [pytest.param(d, p, id=f'{d}-{p}', marks=[pytest.mark.slow] if
                         slow or (level == 'L1' and p == 'multi-device-graded' and DISTRIBUTED_L1_SLOW)
                         else []) for d, p in rows]


def entry_options(declaration):
    return {'subpixel_smoothing': True} if declaration == 'C-smooth' else {}


def known(declaration, path, level):
    return {ident: reason for d, p, lev, ident, reason in KNOWN_ROWS
            if (d, p, lev) == (declaration, path, level)}


def finish(declaration, path, level, ids, matched, unexplained):
    if unexplained:
        pytest.fail(f'unexplained {declaration}/{path}/{level}: ' + json.dumps(unexplained, default=str))
    for ident in ids:
        if ident not in matched:
            pytest.fail(f'{ident} no longer applies to {declaration}/{path}/{level}: remove it from KNOWN_ROWS')
    if ids:
        pytest.xfail('; '.join(ids.values()))


def outcomes(declaration, path, level, pair):
    ids = known(declaration, path, level)
    capabilities = {i: reason for i, reason in ids.items() if i.startswith('A')}
    success = 'captured' if level == 'L1' else 'admitted'
    if capabilities:
        matched, unexplained = set(), []
        for ident in capabilities:
            matches = True
            for entry, pattern in ID_PATTERNS[ident].items():
                record = pair[entry]
                wanted = pattern[0]
                if wanted == 'admitted' and level == 'L1':
                    wanted = 'fallback' if ident == 'A5' and entry == 'run' else 'captured'
                ok = record['outcome'] == wanted
                if pattern[0] == 'raised':
                    ok = ok and record.get('type') == pattern[1] and pattern[2] in record.get('message', '')
                if not ok:
                    matches = False
                    # The original cause disappearing is reported separately.
                    if record['outcome'] != success:
                        unexplained.append({entry: {k: v for k, v in record.items() if k != 'value'}})
            if matches:
                matched.add(ident)
        finish(declaration, path, level, ids, matched, unexplained)
    failures = {e: {k: v for k, v in r.items() if k != 'value'}
                for e, r in pair.items() if r['outcome'] != success}
    if failures:
        pytest.fail(f'unexpected outcomes {declaration}/{path}/{level}: {failures}')
    return ids


def classify(declaration, path, level, differences, ids):
    matched, unexplained = set(), []
    for difference in differences:
        explained = set()
        for ident in ids:
            if ident.startswith('A'):
                continue
            common = 'bytes_equal' in difference
            run_only = (difference.get('run') == 'array'
                        and difference.get('forward') != 'array' and not common)
            consequence = (level == 'L2' and common and difference['path'] == 'time_series'
                           and (declaration, path, ident) in CONSEQUENCE_PAIRS)
            pattern_match = any(fnmatchcase(difference['path'], pattern)
                                for pattern in ID_PATTERNS[ident][level])
            if level == 'L2' and ident in ('D4', 'D7', 'D8'):
                pattern_match = pattern_match and run_only
            if level == 'L2' and ident == 'D5':
                shapes = difference.get('shape')
                shape_change = common and shapes is not None and shapes[0] != shapes[1]
                # Item 21: both spellings represent per-drive witness channels.
                run_record = run_only and difference['path'].startswith(
                    ('settling_witness/runs/', 'settling_witness/per_drive/'))
                pattern_match = pattern_match and (shape_change or run_record)
            if consequence or pattern_match:
                explained.add(ident)
        matched.update(explained)
        if not explained:
            unexplained.append(difference)
    finish(declaration, path, level, ids, matched, unexplained)


@pytest.mark.parametrize('declaration,path', parameters(ASSEMBLY_ROWS, 'L1'))
def test_assembly(declaration, path):
    # Uniform seam spelling resolves before the step. Compare the complete ctx,
    # effective invariants, setup carry, actual scan init and xs. Graded compares
    # its scan builder, not the outer seam's conductor maps.
    differences = []
    for level in (('a', 'b') if path == 'multi-device-graded' else ('b',)):
        pair = {entry: capture(BUILDERS[declaration], entry, path, level,
                               entry_options=entry_options(declaration), seams=SEAMS)
                for entry in ('run', 'forward')}
        ids = outcomes(declaration, path, 'L1', pair)
        required = {'ctx', 'invariants', 'carry_init', 'init', 'xs'} if path == 'uniform' else (
            {'arguments', 'result'} if path == 'graded' else {'arguments'})
        for cap in pair.values():
            assert set(cap['value']) == required, cap['seam_sites']
        differences.extend(compare(normalize_inputs(pair['run']['value'], path),
                                   normalize_inputs(pair['forward']['value'], path)))
    classify(declaration, path, 'L1', differences, ids)


def check_fields(declaration, path):
    pair = port_s_pair(declaration, path) if declaration in ('P-lumped', 'P-lumped2') and path == 'uniform' else None
    if pair is None:
        pair = {entry: run_l2(BUILDERS[declaration], entry, path,
                              entry_options=entry_options(declaration)) for entry in ('run', 'forward')}
    ids = outcomes(declaration, path, 'L2', pair)
    values = {entry: observables(value['result']) for entry, value in pair.items()}
    if declaration in ('P-lumped', 'P-lumped2') and path == 'uniform':
        # S was checked above on its measured column, with the cross-trace bar.
        for value in values.values():
            value['arrays'].pop('s_params', None)
            value['arrays'].pop('freqs_hz', None)  # run alias of the bins checked above
    comparison = compare_l2(values['run']['arrays'], values['forward']['arrays'],
                            left_raises=values['run']['raises'], right_raises=values['forward']['raises'])
    differences = l2_differences(comparison)
    classify(declaration, path, 'L2', differences, ids)


ALWAYS_ON_FIELDS = tuple(row for row in FIELD_ROWS if row[0] == 'B-cpml' or row in (
    ('O-dft', 'uniform'), ('O-ntff', 'uniform'), ('O-cm', 'uniform'), ('P-msl', 'uniform'),
    ('B-closed', 'uniform')) or any(
    (d, p, level) == (*row, 'L2') for d, p, level, ident, reason in KNOWN_ROWS))


@pytest.mark.parametrize('declaration,path', parameters(ALWAYS_ON_FIELDS, 'L2'))
def test_fields(declaration, path):
    check_fields(declaration, path)


@pytest.mark.parametrize('declaration,path', parameters(FIELD_ROWS, 'L2', slow=True))
def test_fields_slow(declaration, path):
    check_fields(declaration, path)


@pytest.mark.parametrize('declaration,path', parameters(ADMISSION_ROWS, 'admission', slow=True))
def test_admission(declaration, path):
    pair = {entry: run_l2(BUILDERS[declaration], entry, path,
                          entry_options=entry_options(declaration)) for entry in ('run', 'forward')}
    if declaration == 'P-wire' and path in ('uniform', 'graded'):
        with pytest.raises(NotImplementedError) as exc:
            BUILDERS[declaration](path).forward(n_steps=1, skip_preflight=True,
                                               port_s11_freqs=[2e9])
        assert str(exc.value) == pair['run']['message']
    ids = outcomes(declaration, path, 'admission', pair)
    finish(declaration, path, 'admission', ids, set(), [])


# Appendix I: these full-suite rows refuse while declaring the closed box.
DECLARATION_REFUSAL_ROWS = (
    ('PW-closed', 'graded'),
    ('PW-closed', 'multi-device-graded'),
)


def both_refused(declaration, path, pair):
    if all(record['outcome'] == 'raised' for record in pair.values()):
        return True
    if (declaration, path) not in DECLARATION_REFUSAL_ROWS:
        return False
    if not all(record['outcome'] == 'not_called' for record in pair.values()):
        return False
    construction = pair['run']['construction']
    if construction != pair['forward']['construction']:
        return False
    source_root = Path(__file__).resolve().parents[2] / 'rfx'
    raising_path = Path(construction['raising_site'].rsplit(':', 1)[0]).resolve()
    return (construction['outcome'] == 'raised'
            and construction['type'] in ('NotImplementedError', 'ValueError')
            and 'closed_box TFSF require a uniform' in construction['message']
            and raising_path.is_relative_to(source_root))


@pytest.mark.parametrize('declaration,path', parameters(BOTH_REFUSE_ROWS, 'refusal'))
def test_both_refuse(declaration, path):
    # Refusals occur before stepping; one step bounds an unexpectedly admitted call.
    pair = {entry: run_l2(BUILDERS[declaration], entry, path, n_steps=1,
                          entry_options=entry_options(declaration))
            for entry in ('run', 'forward')}
    assert both_refused(declaration, path, pair), pair


@pytest.mark.parametrize('error', (TypeError, ValueError, NotImplementedError))
@pytest.mark.parametrize('declaration,path', DECLARATION_REFUSAL_ROWS)
def test_declaration_refusal_rejects_test_builder_defect(error, declaration, path):
    def broken_builder(path):
        raise error('defect raised by the test builder')
    pair = {entry: run_l2(broken_builder, entry, path, n_steps=1)
            for entry in ('run', 'forward')}
    assert not both_refused(declaration, path, pair), pair


def test_declaration_refusal_does_not_extend_to_unlisted_rows():
    pair = {entry: run_l2(BUILDERS['PW-closed'], entry, 'graded', n_steps=1)
            for entry in ('run', 'forward')}
    assert both_refused('PW-closed', 'graded', pair), pair
    assert not both_refused('PW-A', 'graded', pair), pair


@pytest.mark.parametrize('declaration', ('P-lumped', 'P-lumped2'))
def test_multi_device_graded_ports_reach_both_entries(declaration):
    pair = {entry: run_l2(BUILDERS[declaration], entry, 'multi-device-graded', n_steps=16)
            for entry in ('run', 'forward')}
    assert all(item['outcome'] == 'admitted' for item in pair.values()), pair
    import numpy as np
    for a, b in zip(pair['run']['result'].sparam_time_records,
                    pair['forward']['result'].sparam_time_records, strict=True):
        np.testing.assert_array_equal(a.samples, b.samples)


@pytest.mark.parametrize('declaration', ['P-lumped', 'P-lumped2'])
@pytest.mark.parametrize('path', ['uniform', 'graded', 'multi-device-graded'])
def test_port_s_equal_requests(declaration, path):
    pair = port_s_pair(declaration, path)
    if path in ('graded', 'multi-device-graded'):
        assert pair is None  # public explicit lumped run S request is refused
    else:
        assert pair is not None
