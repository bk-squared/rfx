"""run() and forward() are one pipeline: the leader's C1–C5 contract.

The literal exceptions are progress toward zero at 2.0, not tolerances.
No solver helper supplies an expected value: the other fresh entry does.
"""
import json
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

import numpy as np

from tests.contracts.run_forward_census import (
    BUILDERS, SEAMS, FrozenArray, capture, compare, compare_l2, equal_request_pair, freeze,
    l2_differences, normalize_inputs, observables, run_l2, port_s_pair, port_drive_witness_leaves,
)

KNOWN_ROWS = (
)

# Declarations with ports compare equal S requests in both directions (#1373):
# (a) run's default against forward(port_s11_freqs=run's bins);
# (b) run(compute_s_params=False) against forward's default.
PORTED = tuple(d for d in BUILDERS if d.startswith('P-') or d == 'C-full-tensor')
KNOWN_DIRECTIONS = {}
# Under (a) both entries refuse the S request with the same error: a plain
# source fires in every port drive.
EQUAL_S_REFUSAL_ROWS = tuple(('P-wire', path, 'a') for path in (
    'uniform', 'graded', 'multi-device-uniform', 'multi-device-graded'))

ASSEMBLY_ROWS = (
    ('C-full-tensor', 'graded'),
    ('O-cm', 'multi-device-graded'),
    ('O-dft', 'multi-device-graded'),
    ('O-flux', 'multi-device-graded'),
    ('O-ntff', 'multi-device-graded'),

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
    ('P-waveguide', 'uniform'),
    ('PW-A', 'graded'),
    ('PW-A', 'uniform'),
    ('PW-B', 'uniform'),
    ('PW-closed', 'multi-device-uniform'),
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
    ('C-full-tensor', 'graded'),
    ('O-cm', 'multi-device-graded'),
    ('O-dft', 'multi-device-graded'),
    ('O-flux', 'multi-device-graded'),
    ('O-ntff', 'multi-device-graded'),

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
    ('O-cm', 'multi-device-uniform'),
    ('O-cm', 'uniform'),
    ('O-dft', 'graded'),
    ('O-dft', 'multi-device-uniform'),
    ('O-dft', 'uniform'),
    ('O-flux', 'graded'),
    ('O-flux', 'multi-device-uniform'),
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
    ('P-waveguide', 'uniform'),
    ('PW-A', 'uniform'),
    ('PW-B', 'uniform'),
    ('PW-closed', 'multi-device-uniform'),
    ('PW-closed', 'uniform'),
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
    ('O-cm', 'multi-device-graded'),
    ('O-dft', 'multi-device-graded'),
    ('O-flux', 'multi-device-graded'),
    ('O-ntff', 'multi-device-graded'),

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
    ('O-cm', 'multi-device-uniform'),
    ('O-dft', 'multi-device-uniform'),
    ('O-flux', 'multi-device-uniform'),
    ('O-ntff', 'multi-device-uniform'),
    ('O-vector', 'multi-device-uniform'),
    ('P-lumped', 'multi-device-uniform'),
    ('P-wire', 'graded'),
    ('P-wire', 'multi-device-uniform'),
    ('P-wire', 'uniform'),
    ('PW-A', 'graded'),
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
    ('P-coax', 'graded'),
    ('P-coax', 'multi-device-graded'),
    ('P-coax', 'multi-device-uniform'),
    ('P-coax', 'uniform'),
    ('P-floquet', 'multi-device-uniform'),
    ('P-lumped2', 'multi-device-uniform'),
    ('P-msl', 'multi-device-graded'),
    ('P-msl', 'multi-device-uniform'),
    ('P-waveguide', 'multi-device-graded'),
    ('P-waveguide', 'multi-device-uniform'),
    ('P-waveguide-2mode', 'graded'),
    ('P-waveguide-2mode', 'multi-device-graded'),
    ('P-waveguide-2mode', 'multi-device-uniform'),
    ('P-waveguide-2mode', 'uniform'),
    ('PW-A', 'multi-device-graded'),
    ('PW-A', 'multi-device-uniform'),
    ('PW-B', 'graded'),
    ('PW-B', 'multi-device-graded'),
    ('PW-B', 'multi-device-uniform'),
    ('PW-closed', 'graded'),
    ('PW-closed', 'multi-device-graded'),
    ('PW-oblique', 'graded'),
    ('PW-oblique', 'multi-device-graded'),
    ('PW-oblique', 'multi-device-uniform'),
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
    'D7': {'L1': (), 'L2': ('waveguide_ports/*', 'waveguide_sparams/*')},
}
DISTRIBUTED_L1_SLOW = False


def directions(declaration):
    return ('a', 'b') if declaration in PORTED else (None,)


# Both entries refuse, but from two guards, not one check both call
# (rfx-core-desk 2026-10-27). The graded multi-device MSL and RLC rows left
# with #2076 and the coaxial port's 8 rows with #2075: both entries now refuse
# each of them from one shared check.
TWO_GUARD_REFUSALS = {
}


def parameters(rows, level, *, slow=False):
    def marks(d, p):
        found = [pytest.mark.slow] if (
            slow or (level == 'L1' and p == 'multi-device-graded' and DISTRIBUTED_L1_SLOW)) else []
        if level == 'refusal' and (d, p) in TWO_GUARD_REFUSALS:
            found.append(pytest.mark.xfail(strict=True, reason=TWO_GUARD_REFUSALS[d, p]))
        return found
    return [pytest.param(d, p, direction, id=f'{d}-{p}' + (f'-{direction}' if direction else ''),
                         marks=marks(d, p)) for d, p in rows for direction in directions(d)]


def entry_options(declaration):
    if declaration == 'C-full-tensor':
        return {'subpixel_smoothing': 'full_tensor'}
    return {'subpixel_smoothing': True} if declaration == 'C-smooth' else {}


def known(declaration, path, level, direction=None):
    return {ident: reason for d, p, lev, ident, reason in KNOWN_ROWS
            if (d, p, lev) == (declaration, path, level)
            and direction in KNOWN_DIRECTIONS.get((d, p, lev, ident), (direction,))}


def finish(declaration, path, level, ids, matched, unexplained):
    if unexplained:
        pytest.fail(f'unexplained {declaration}/{path}/{level}: ' + json.dumps(unexplained, default=str))
    for ident in ids:
        if ident not in matched:
            pytest.fail(f'{ident} no longer applies to {declaration}/{path}/{level}: remove it from KNOWN_ROWS')
    if ids:
        pytest.xfail('; '.join(ids.values()))


def outcomes(declaration, path, level, pair, direction=None):
    ids = known(declaration, path, level, direction)
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
                    wanted = 'captured'
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
            if level == 'L2' and ident in ('D4', 'D7'):
                pattern_match = pattern_match and run_only
            if consequence or pattern_match:
                explained.add(ident)
        matched.update(explained)
        if not explained:
            unexplained.append(difference)
    finish(declaration, path, level, ids, matched, unexplained)


@pytest.mark.parametrize('declaration,path,direction', parameters(ASSEMBLY_ROWS, 'L1'))
def test_assembly(declaration, path, direction):
    # Uniform seam spelling resolves before the step. Compare the complete ctx,
    # effective invariants, setup carry, actual scan init and xs. Graded compares
    # its scan builder, not the outer seam's conductor maps.
    differences = []
    for level in (('a', 'b') if path == 'multi-device-graded' else ('b',)):
        if direction:
            pair = equal_request_pair(declaration, path, direction, level=level,
                                      entry_options=entry_options(declaration))
        else:
            pair = {entry: capture(BUILDERS[declaration], entry, path, level,
                                   entry_options=entry_options(declaration), seams=SEAMS)
                    for entry in ('run', 'forward')}
        ids = outcomes(declaration, path, 'L1', pair, direction)
        capture_paths = {cap['capture_path'] for cap in pair.values()}
        assert len(capture_paths) == 1, capture_paths
        capture_path = capture_paths.pop()
        required = {'ctx', 'invariants', 'carry_init', 'init', 'xs'} if capture_path == 'uniform' else (
            {'arguments', 'result'} if capture_path == 'graded' and level == 'b' else {'arguments'})
        for cap in pair.values():
            assert set(cap['value']) == required, cap['seam_sites']
        if path == 'graded' and declaration in ('P-lumped', 'P-lumped2', 'C-full-tensor'):
            # run's default bins arrive as the float64 host grid, forward's
            # request as float32; the runner solves on float32 bins either way
            # (result/sp_freqs is compared exactly below).
            for cap in pair.values():
                arguments = cap['value']['arguments']
                freqs = arguments.get('s_param_freqs')
                if isinstance(freqs, FrozenArray):
                    arguments['s_param_freqs'] = freeze(freqs.array().astype(np.float32))
            # Stated fact (#1373 decision C): a graded run() that asks for S (direction
            # a) on two ports takes it from its drive scan, so its main record's port
            # specs say record_s=False; forward's one record forms S and carries no
            # such key. One excited lumped port reads its main record, as on uniform.
            for section in ('arguments', 'result'):
                ran = pair['run']['value'].get(section, {}).get('wire_ports') or ()
                fwd = pair['forward']['value'].get(section, {}).get('wire_ports') or ()
                assert len(ran) == len(fwd)
                assert [port.pop('record_s', None) for port in ran] == (
                    [False] * len(ran) if direction == 'a' and declaration == 'P-lumped2'
                    else [None] * len(ran))
                assert all('record_s' not in port for port in fwd)
        if level == 'a' and capture_path == 'graded':
            # Judge a full-domain fallback at the same scan-builder seam as
            # a direct graded call, after driver defaults are resolved.
            continue
        comparison_path = path if level == 'a' else capture_path
        differences.extend(compare(normalize_inputs(pair['run']['value'], comparison_path),
                                   normalize_inputs(pair['forward']['value'], comparison_path)))
    classify(declaration, path, 'L1', differences, ids)


def check_fields(declaration, path, direction=None):
    port_s = direction == 'a' and declaration in ('P-lumped', 'P-lumped2') and path in ('uniform', 'graded')
    if port_s:
        pair = port_s_pair(declaration, path)
    elif direction:
        pair = equal_request_pair(declaration, path, direction,
                                  entry_options=entry_options(declaration))
    else:
        pair = {entry: run_l2(BUILDERS[declaration], entry, path,
                              entry_options=entry_options(declaration)) for entry in ('run', 'forward')}
    ids = outcomes(declaration, path, 'L2', pair, direction)
    values = {entry: observables(value['result']) for entry, value in pair.items()}
    if port_s:
        # S was checked above on its measured column, with the cross-trace bar.
        for value in values.values():
            value['arrays'].pop('s_params', None)
        if declaration == 'P-lumped2':
            leaves = port_drive_witness_leaves(BUILDERS[declaration](path),
                                               pair['run']['result'], pair['forward']['result'])
            assert len(leaves) == 8
            for leaf in leaves:
                assert leaf in values['run']['arrays'], leaf
                assert leaf not in values['forward']['arrays'], leaf
                values['run']['arrays'].pop(leaf)
    comparison = compare_l2(values['run']['arrays'], values['forward']['arrays'],
                            left_raises=values['run']['raises'], right_raises=values['forward']['raises'])
    differences = l2_differences(comparison)
    classify(declaration, path, 'L2', differences, ids)


ALWAYS_ON_FIELDS = tuple(row for row in FIELD_ROWS if row[0] == 'B-cpml' or row in (
    ('PW-closed', 'multi-device-uniform'),
    ('C-full-tensor', 'graded'),
    ('O-dft', 'uniform'), ('O-ntff', 'uniform'), ('O-cm', 'uniform'), ('P-msl', 'uniform'),
    ('B-closed', 'uniform'), ('P-lumped', 'multi-device-uniform'), ('P-lumped2', 'uniform')) or any(
    (d, p, level) == (*row, 'L2') for d, p, level, ident, reason in KNOWN_ROWS))


@pytest.mark.parametrize('declaration,path,direction', parameters(ALWAYS_ON_FIELDS, 'L2'))
def test_fields(declaration, path, direction):
    check_fields(declaration, path, direction)


@pytest.mark.parametrize('declaration,path,direction', parameters(FIELD_ROWS, 'L2', slow=True))
def test_fields_slow(declaration, path, direction):
    check_fields(declaration, path, direction)


@pytest.mark.parametrize('declaration,path,direction', parameters(ADMISSION_ROWS, 'admission', slow=True))
def test_admission(declaration, path, direction):
    if direction:
        pair = equal_request_pair(declaration, path, direction,
                                  entry_options=entry_options(declaration))
    else:
        pair = {entry: run_l2(BUILDERS[declaration], entry, path,
                              entry_options=entry_options(declaration)) for entry in ('run', 'forward')}
    if (declaration, path, direction) in EQUAL_S_REFUSAL_ROWS:
        run, forward = pair['run'], pair['forward']
        assert run['outcome'] == forward['outcome'] == 'raised', pair
        assert (run['type'], run['message']) == (forward['type'], forward['message']), pair
        assert run['type'] == 'NotImplementedError', pair
        assert 'Lumped/wire S-matrix requests do not support plain sources' in run['message'], pair
        return
    ids = outcomes(declaration, path, 'admission', pair, direction)
    finish(declaration, path, 'admission', ids, set(), [])


# Appendix I: these full-suite rows refuse while declaring the closed box.
DECLARATION_REFUSAL_ROWS = (
    ('PW-closed', 'graded'),
    ('PW-closed', 'multi-device-graded'),
)


#: The exception classes rfx raises to refuse a request it does not solve
#: (admission rows and lane refusals: NotImplementedError; configuration
#: guards, PreflightConfigError included: ValueError).
REFUSAL_TYPES = ('NotImplementedError', 'ValueError')


def raised_in_rfx(record):
    source_root = Path(__file__).resolve().parents[2] / 'rfx'
    raising_path = Path(record['raising_site'].rsplit(':', 1)[0]).resolve()
    return raising_path.is_relative_to(source_root)


def same_raising_line(run, forward):
    # One refusal both entries call raises from one line; two guards with
    # the same text do not (rfx-core-desk 2026-10-27).
    return (Path(run['raising_site'].rsplit(':', 1)[0]).resolve(),
            run['raising_site'].rsplit(':', 1)[1]) == (
        Path(forward['raising_site'].rsplit(':', 1)[0]).resolve(),
        forward['raising_site'].rsplit(':', 1)[1])


def both_refused(declaration, path, pair):
    # A crash on both entries is not a refusal: both must raise the same
    # refusal type from the same line inside rfx. Messages are not compared.
    run, forward = pair['run'], pair['forward']
    if run['outcome'] == forward['outcome'] == 'raised':
        return (run['type'] == forward['type'] and run['type'] in REFUSAL_TYPES
                and raised_in_rfx(run) and raised_in_rfx(forward)
                and same_raising_line(run, forward))
    if (declaration, path) not in DECLARATION_REFUSAL_ROWS:
        return False
    if not all(record['outcome'] == 'not_called' for record in pair.values()):
        return False
    construction = pair['run']['construction']
    if construction != pair['forward']['construction']:
        return False
    return (construction['outcome'] == 'raised'
            and construction['type'] in REFUSAL_TYPES
            and 'closed_box TFSF require a uniform' in construction['message']
            and raised_in_rfx(construction))


@pytest.mark.parametrize('declaration,path,direction', parameters(BOTH_REFUSE_ROWS, 'refusal'))
def test_both_refuse(declaration, path, direction):
    # Refusals occur before stepping; one step bounds an unexpectedly admitted call.
    if direction:
        pair = equal_request_pair(declaration, path, direction, n_steps=1,
                                  entry_options=entry_options(declaration))
    else:
        pair = {entry: run_l2(BUILDERS[declaration], entry, path, n_steps=1,
                              entry_options=entry_options(declaration))
                for entry in ('run', 'forward')}
    assert both_refused(declaration, path, pair), pair
    if path.startswith('multi-device') and declaration in ('P-waveguide', 'PW-A', 'PW-B', 'PW-oblique'):
        expected = ('waveguide ports (add_waveguide_port) are not supported on'
                    if declaration == 'P-waveguide' else
                    'TFSF plane-wave sources (add_tfsf_source) are not supported on the graded multi-device lane'
                    if path == 'multi-device-graded' else {
                        'PW-A': 'open-slab normal-incidence TFSF (periodic y/z is not on slabs)',
                        'PW-B': 'TFSF Method B oblique (is_tfsf_methodB)',
                        'PW-oblique': 'TFSF 2-D Bloch line (is_tfsf_2d)',
                    }[declaration])
        for record in pair.values():
            assert record['type'] == 'NotImplementedError', record
            assert expected in record['message'], record
        print(json.dumps(dict(declaration=declaration, path=path, direction=direction,
                              dispositions=pair)), file=sys.stderr)
    if declaration == 'P-waveguide-2mode':
        # The per-mode list once reached the single-run assembly and raised
        # AttributeError on both entries (#2070); pin the refusal's text too.
        expected = ('waveguide ports (add_waveguide_port) are not supported on'
                    if path.startswith('multi-device') else 'compute_waveguide_s_matrix()')
        for record in pair.values():
            assert record['type'] == 'NotImplementedError', record
            assert expected in record['message'], record


def _raised(kind, message, site='rfx/runners/_admission.py:1'):
    root = Path(__file__).resolve().parents[2]
    return {'outcome': 'raised', 'type': kind, 'message': message,
            'raising_site': f'{root / site}'}


@pytest.mark.parametrize('case,run,forward,refused', [
    ('one refusal', _raised('NotImplementedError', 'm'), _raised('NotImplementedError', 'm'), True),
    ('crash on both', _raised('AttributeError', "'list' object has no attribute 'v_probe_t'"),
     _raised('AttributeError', "'list' object has no attribute 'v_probe_t'"), False),
    ('different types', _raised('NotImplementedError', 'm'), _raised('ValueError', 'm'), False),
    ('one entry crashes', _raised('AttributeError', 'm'), _raised('NotImplementedError', 'm'), False),
    ('messages differ by entry name', _raised('ValueError', 'm run()'), _raised('ValueError', 'm forward()'), True),
    ('two guards with one text', _raised('ValueError', 'm', 'rfx/api/_dispatch.py:252'),
     _raised('ValueError', 'm', 'rfx/api/_dispatch.py:190'), False),
    ('two guards in two files', _raised('ValueError', 'm'), _raised('ValueError', 'm', 'rfx/runners/distributed_v2.py:1'), False),
    ('raised outside rfx', _raised('ValueError', 'm', 'tests/x.py:1'),
     _raised('ValueError', 'm', 'tests/x.py:1'), False),
])
def test_both_refused_requires_one_refusal_from_rfx(case, run, forward, refused):
    """#2070: a crash on both entries read as 'both refuse'."""
    assert both_refused('P-waveguide', 'uniform', {'run': run, 'forward': forward}) is refused, case


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


@pytest.mark.parametrize('direction', ('a', 'b'))
@pytest.mark.parametrize('declaration', ('P-lumped', 'P-lumped2'))
def test_multi_device_graded_ports_reach_both_entries(declaration, direction):
    pair = equal_request_pair(declaration, 'multi-device-graded', direction, n_steps=16)
    assert all(item['outcome'] == 'admitted' for item in pair.values()), pair
    import numpy as np
    for a, b in zip(pair['run']['result'].sparam_time_records,
                    pair['forward']['result'].sparam_time_records, strict=True):
        np.testing.assert_array_equal(a.samples, b.samples)


@pytest.mark.parametrize('declaration', ['P-lumped', 'P-lumped2'])
@pytest.mark.parametrize('path', ['uniform', 'graded', 'multi-device-graded'])
def test_port_s_equal_requests(declaration, path):
    assert port_s_pair(declaration, path) is not None
