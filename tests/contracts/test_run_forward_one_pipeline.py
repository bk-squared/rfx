"""run() and forward() are one pipeline: the leader's C1–C5 contract.

The literal exceptions are progress toward zero at 2.0, not tolerances.
No solver helper supplies an expected value: the other fresh entry does.
"""
import json

import pytest

from tests.contracts.run_forward_census import (
    BUILDERS, SEAMS, capture, compare, compare_l2, l2_differences,
    normalize_inputs, observables, run_l2,
)

KNOWN_ROWS = (
    ('B-asym', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('B-closed', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('B-closed', 'uniform', 'L1', 'D2', '#1373 D2 (ports): closed-box point-source drive table differs'),
    ('B-closed', 'uniform', 'L2', 'D2', '#1373 D2 (ports): closed-box point-source drive table differs'),
    ('B-cpml', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('B-pecsides', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('C-film', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('C-pecbox', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('C-pinned', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('C-sheet', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('C-smooth', 'graded', 'L1', 'A6', '#1373 A6 (nu): subpixel smoothing is a run() keyword; forward() cannot request it'),
    ('C-smooth', 'graded', 'admission', 'A6', '#1373 A6 (nu): subpixel smoothing is a run() keyword; forward() cannot request it'),
    ('C-smooth', 'uniform', 'L1', 'A6', '#1373 A6 (nu): subpixel smoothing is a run() keyword; forward() cannot request it'),
    ('C-smooth', 'uniform', 'admission', 'A6', '#1373 A6 (nu): subpixel smoothing is a run() keyword; forward() cannot request it'),
    ('C-wire', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('M-debye', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('M-diel', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('M-drude', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('M-lorentz', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('M-lossy', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('M-mu', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('O-flux', 'graded', 'L2', 'D4', '#1373 D4 (ports): forward omits the declared flux measurement'),
    ('O-flux', 'graded', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('O-flux', 'uniform', 'L1', 'D4', '#1373 D4 (ports): forward omits the declared flux measurement'),
    ('O-flux', 'uniform', 'L2', 'D4', '#1373 D4 (ports): forward omits the declared flux measurement'),
    ('O-flux', 'uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('O-ntff', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('O-vector', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('P-lumped', 'multi-device-graded', 'admission', 'A4', '#1373 A4 (ports): run refusal names the distributed=True forward path'),
    ('P-lumped', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('P-lumped', 'uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-lumped', 'uniform', 'L2', 'D8', '#1373 D8 (ports): run computes the port S-matrix by default; forward returns no S without port_s11_freqs'),
    ('P-lumped2', 'uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-lumped2', 'uniform', 'L2', 'D8', '#1373 D8 (ports): run computes the port S-matrix by default; forward returns no S without port_s11_freqs'),
    ('P-waveguide', 'graded', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-waveguide', 'graded', 'L2', 'D7', '#1373 D7 (ports): forward() result carries no waveguide port records and S'),
    ('P-waveguide', 'multi-device-graded', 'L1', 'A5', '#1373 A5 (acceleration): run falls back for a waveguide port while distributed forward refuses'),
    ('P-waveguide', 'multi-device-graded', 'admission', 'A5', '#1373 A5 (acceleration): run falls back for a waveguide port while distributed forward refuses'),
    ('P-waveguide', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('P-waveguide', 'uniform', 'L2', 'D5', '#1373 D5 (ports): settling witness arrays differ between entries'),
    ('P-waveguide', 'uniform', 'L2', 'D7', '#1373 D7 (ports): forward() result carries no waveguide port records and S'),
    ('P-wire', 'graded', 'admission', 'A2', '#1373 A2 (ports): run refuses the wire port that forward admits'),
    ('P-wire', 'uniform', 'admission', 'A2', '#1373 A2 (ports): run refuses the wire port that forward admits'),
    ('PW-A', 'graded', 'L1', 'A3', '#1373 A3 (absorber): forward refuses the graded normal plane wave'),
    ('PW-A', 'graded', 'admission', 'A3', '#1373 A3 (absorber): forward refuses the graded normal plane wave'),
    ('PW-A', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('PW-B', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('PW-oblique', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('PW-oblique', 'uniform', 'L2', 'D3', '#1373 D3 (absorber): oblique plane-wave fields differ'),
    ('R-rlc', 'uniform', 'L1', 'D6', '#1373 D6 (ports): lumped RLC step inputs and fields differ'),
    ('R-rlc', 'uniform', 'L2', 'D6', '#1373 D6 (ports): lumped RLC step inputs and fields differ'),
    ('S-field', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
    ('S-polarized', 'multi-device-uniform', 'admission', 'A1', '#1373 A1 (acceleration): forward refuses the multi-device uniform path'),
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
    ('B-periodic', 'uniform'),
    ('B-pmcone', 'graded'),
    ('B-pmcone', 'uniform'),
    ('B-pmcpair', 'graded'),
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
    ('B-periodic', 'uniform'),
    ('B-pmcone', 'graded'),
    ('B-pmcone', 'uniform'),
    ('B-pmcpair', 'graded'),
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

ADMISSION_ROWS = (
    ('B-asym', 'multi-device-uniform'),
    ('B-closed', 'multi-device-uniform'),
    ('B-cpml', 'multi-device-uniform'),
    ('B-pecsides', 'multi-device-uniform'),
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
    ('P-lumped', 'multi-device-graded'),
    ('P-lumped', 'multi-device-uniform'),
    ('P-waveguide', 'multi-device-graded'),
    ('P-waveguide', 'multi-device-uniform'),
    ('P-wire', 'graded'),
    ('P-wire', 'uniform'),
    ('PW-A', 'graded'),
    ('PW-A', 'multi-device-uniform'),
    ('PW-B', 'multi-device-uniform'),
    ('PW-oblique', 'multi-device-uniform'),
    ('S-field', 'multi-device-uniform'),
    ('S-polarized', 'multi-device-uniform'),
)


# The budget rule may move only this L1 path to the weekly set after measurement.
DISTRIBUTED_L1_SLOW = False


def parameters(rows, level, *, slow=False):
    result = []
    for declaration, path in rows:
        marks = []
        reasons = [reason for d, p, lev, ident, reason in KNOWN_ROWS
                   if (d, p, lev) == (declaration, path, level)]
        if reasons:
            marks.append(pytest.mark.xfail(strict=True, raises=AssertionError,
                                           reason='; '.join(reasons)))
        if slow or (level == 'L1' and path == 'multi-device-graded' and DISTRIBUTED_L1_SLOW):
            marks.append(pytest.mark.slow)
        result.append(pytest.param(declaration, path, marks=marks,
                                   id=f'{declaration}-{path}'))
    return result


def entry_options(declaration):
    return {'subpixel_smoothing': True} if declaration == 'C-smooth' else {}


def explain(value):
    return json.dumps(value, indent=2, default=str)


@pytest.mark.parametrize('declaration,path', parameters(ASSEMBLY_ROWS, 'L1'))
def test_assembly(declaration, path):
    # Uniform seam arguments (periodic/cpml_axes/list spelling) resolve before
    # the step. Compare ctx, invariants, carry_init and xs, not that spelling.
    # Graded compares the scan builder, not the outer seam's conductor maps.
    levels = ('a', 'b') if path == 'multi-device-graded' else ('b',)
    for level in levels:
        pair = {entry: capture(BUILDERS[declaration], entry, path, level,
                               entry_options=entry_options(declaration), seams=SEAMS)
                for entry in ('run', 'forward')}
        assert all(cap['outcome'] == 'captured' for cap in pair.values()), explain(pair)
        required = {'ctx', 'invariants', 'carry_init', 'xs'} if path == 'uniform' else (
            {'arguments', 'result'} if path == 'graded' else {'arguments'})
        for cap in pair.values():
            assert set(cap['value']) == required, explain(cap['seam_sites'])
        differences = compare(normalize_inputs(pair['run']['value'], path),
                              normalize_inputs(pair['forward']['value'], path))
        assert not differences, explain(differences)


def check_fields(declaration, path):
    pair = {entry: run_l2(BUILDERS[declaration], entry, path,
                          entry_options=entry_options(declaration))
            for entry in ('run', 'forward')}
    assert all(value['outcome'] == 'admitted' for value in pair.values()), explain(pair)
    values = {entry: observables(value['result']) for entry, value in pair.items()}
    comparison = compare_l2(values['run']['arrays'], values['forward']['arrays'],
                            left_raises=values['run']['raises'],
                            right_raises=values['forward']['raises'])
    differences = l2_differences(comparison)
    assert not differences, explain(differences)


# C3/addendum 9: three base paths plus every measured D row before merge.
ALWAYS_ON_FIELDS = tuple(row for row in FIELD_ROWS if row[0] == 'B-cpml' or any(
    (d, p, level) == (*row, 'L2') for d, p, level, ident, reason in KNOWN_ROWS))


@pytest.mark.parametrize('declaration,path', parameters(ALWAYS_ON_FIELDS, 'L2'))
def test_fields(declaration, path):
    check_fields(declaration, path)


# The complete 150-step matrix, including the cheap subset, runs weekly.
@pytest.mark.parametrize('declaration,path', parameters(FIELD_ROWS, 'L2', slow=True))
def test_fields_slow(declaration, path):
    check_fields(declaration, path)


# Admission calls execute 150 steps when accepted. They belong to the complete
# weekly matrix; the required L1 capability rows remain in the assembly set.
@pytest.mark.parametrize('declaration,path', parameters(ADMISSION_ROWS, 'admission', slow=True))
def test_admission(declaration, path):
    pair = {entry: run_l2(BUILDERS[declaration], entry, path,
                          entry_options=entry_options(declaration))
            for entry in ('run', 'forward')}
    if (declaration, path) == ('P-lumped', 'multi-device-graded'):
        # A4 is specifically the run-side wording of a both-refuse row.
        assert pair['run']['outcome'] == pair['forward']['outcome'] == 'raised', explain(pair)
        assert 'distributed=True forward path' not in pair['run']['message'], explain(pair)
    else:
        assert all(value['outcome'] == 'admitted' for value in pair.values()), explain(pair)
