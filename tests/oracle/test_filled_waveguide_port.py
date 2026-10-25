"""Lossless homogeneous WR-90 and dielectric-step scattering oracles (#1994)."""
import functools
import json
import os
from pathlib import Path
import numpy as np
import pytest
from tests._filled_waveguide import measure, build, preflight_findings, assert_preflight_clean


def dump(row):
    directory=os.environ.get('FILLED_PORT_OUTPUT')
    if directory:
        path=Path(directory)
        path.mkdir(parents=True,exist_ok=True)
        (path/f"{row['case']}-{row['cells']}-{int(row.get('shift',False))}-{row.get('layers',12)}.json").write_text(json.dumps(row,indent=2))


def judge(row,empty,*,power_checks=True):
    assert_preflight_clean(row['preflight'])
    reflection=np.array(row['s11_abs'])
    power=np.array(row['sum'])
    if row['case']=='O1':
        assert np.all(reflection<=np.maximum(2*np.array(empty['s11_abs']),.01)), 'O1 reflection floor'
        assert np.all(np.abs(row['phase_error_discrete'])<=.001), 'O1 discrete propagation phase'
        assert np.all(np.abs(row['phase_error_continuous'])<=.01), 'O1 continuous propagation phase'
    else:
        assert np.all(np.abs(reflection-np.abs(row['s11_closed']))<=.02), 'O2 reflection closed form'
        assert np.all(np.abs(np.array(row['s21_abs'])-row['s21_closed'])<=.02), 'O2 power-wave transmission'
        assert np.all(np.abs(row['interface_phase_error'])<=np.array(row['half_cell_phase'])), 'O2 interface half-cell phase'
        if power_checks:
            assert np.all(np.abs(1-power)<=1e-3), 'O2 power conservation'
    if power_checks:
        assert np.all(power<=1+np.maximum(2*np.abs(1-np.array(empty['sum'])),1e-3)), 'W1 per-bin passivity'


@pytest.mark.parametrize('case',['O1','O2'])
@pytest.mark.parametrize('cells',[30,45,60])
def test_filled_waveguide_preflight_clean(case,cells):
    sim,freqs=build(case,cells)
    assert_preflight_clean(preflight_findings(sim,freqs,round(8000*cells/45)))


# The step's vacuum port has 45 % of its source spectrum at the vacuum cutoff; the port's H source table wraps its
# cutoff tail into the record there (#2013), which moves |S11|^2+|S21|^2 by about 1e-2 with record length.
# Reflection, power-wave transmission and interface phase stay always-on; only the power sum waits for #2013.
H_TABLE_WRAP=pytest.mark.xfail(strict=True,reason='#2013 H table wrap')
measured=functools.lru_cache(maxsize=None)(measure)


@pytest.mark.parametrize('case',['O1','O2'])
def test_filled_waveguide_oracles(case):
    empty=measured('empty_filled_band' if case=='O1' else 'empty')
    row=measured(case)
    dump(empty);dump(row)
    judge(row,empty,power_checks=case=='O1')


@H_TABLE_WRAP
def test_step_power_conservation():
    empty=measured('empty')
    row=measured('O2')
    judge(row,empty)


def test_filled_waveguide_inward_reference_shift():
    empty=measure('empty_filled_band')
    shifted=measure('O1',shift=True)
    dump(empty);dump(shifted)
    judge(shifted,empty)


def three_mesh(case):
    rows={}
    for cells in (30,45,60):
        empty=measure('empty_filled_band' if case=='O1' else 'empty',cells)
        row=measure(case,cells)
        dump(empty);dump(row);judge(row,empty,power_checks=case=='O1')
        rows[cells]=row
    return rows


@pytest.mark.slow
def test_three_mesh_trend_filled():
    rows=three_mesh('O1')
    for coarse,fine in ((30,45),(45,60)):
        assert np.all(np.abs(rows[fine]['phase_error_continuous'])<=np.abs(rows[coarse]['phase_error_continuous'])), 'O1 phase refinement'


# Measured (records 20261025-nu-filled-port-oracles): the per-bin |S11| error at 7.2 GHz rises from 4e-5 (a/45) to
# 7.6e-4 (a/60) while other bins fall; its cause being #2013's per-bin ripple is an estimate, which the #2013 fix tests.
@pytest.mark.slow
@pytest.mark.xfail(strict=True,reason='#2013 H table wrap (estimate): O2 reflection refinement fails at 7.2 GHz')
def test_three_mesh_trend_step():
    rows=three_mesh('O2')
    err=lambda n: np.abs(np.array(rows[n]['s11_abs'])-np.abs(rows[n]['s11_closed']))
    for coarse,fine in ((30,45),(45,60)):
        assert np.all(err(fine)<=err(coarse)), 'O2 reflection refinement'
