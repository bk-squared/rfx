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
            # rfx-lead L:921b26, chosen after the settled-record result (rfx-archive rfx/records/20261025-nu-settled-wr90):
            # Amendment 1's W1 control form; the fill may add at most 1e-3 to the empty guide's own |1-sum|.
            assert np.all(np.abs(1-power)<=np.abs(1-np.array(empty['sum']))+1e-3), 'O2 power conservation against the empty guide'
    if power_checks:
        assert np.all(power<=1+np.maximum(2*np.abs(1-np.array(empty['sum'])),1e-3)), 'W1 per-bin passivity'


@pytest.mark.parametrize('case',['O1','O2'])
@pytest.mark.parametrize('cells',[30,45,60])
def test_filled_waveguide_preflight_clean(case,cells):
    sim,freqs=build(case,cells)
    assert_preflight_clean(preflight_findings(sim,freqs,round(8000*cells/45)))


measured=functools.lru_cache(maxsize=None)(measure)
# Settled record at a/45 near cutoff: 8 times the default; the power sum stops moving from about 32k steps on (record
# 20261025-nu-settled-wr90).
SETTLED=64000


@pytest.mark.parametrize('case',['O1','O2'])
def test_filled_waveguide_oracles(case):
    empty=measured('empty_filled_band' if case=='O1' else 'empty')
    row=measured(case)
    dump(empty);dump(row)
    judge(row,empty,power_checks=case=='O1')


# Measured on this code (record 20261025-nu-settled-wr90/pr2016-code): the port's H source table wraps its cutoff
# tail into the record (#2013), so the empty guide reads |1-sum| up to 2.3e-2 at 64k steps and the step exceeds it by
# 2.8e-3. The #2013 fix removes this mark.
@pytest.mark.slow
@pytest.mark.xfail(strict=True,reason='#2013 H table wrap: empty guide off 2e-2 on settled records')
def test_step_power_conservation():
    R={}
    for n in (SETTLED,SETTLED*3//2):
        empty=measured('empty',n_steps=n)
        row=measured('O2',n_steps=n)
        dump(row);judge(row,empty)
        R[n]=np.max(np.abs(1-np.array(row['sum'])))
    assert abs(R[SETTLED*3//2]-R[SETTLED])<=.3*max(R.values()), 'record settled: R(T) against R(1.5T)'


def test_filled_waveguide_inward_reference_shift():
    empty=measure('empty_filled_band')
    shifted=measure('O1',shift=True)
    dump(empty);dump(shifted)
    judge(shifted,empty)


@functools.lru_cache(maxsize=None)
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


@pytest.mark.slow
def test_three_mesh_trend_step():
    # Closed-form checks at a/30, a/45 and a/60 stay visible while the refinement order waits for #2013.
    three_mesh('O2')


# Measured: at the default record the |S11| error at 7.2 GHz rises from 4e-5 (a/45) to 7.6e-4 (a/60) (record
# 20261025-nu-filled-port-oracles); on settled records with this code the 7.0 GHz error rises 2.3e-3, 3.6e-3, 4.2e-3
# (pr2016-code). Its cause being #2013 is an estimate, which the #2013 fix tests.
@pytest.mark.slow
@pytest.mark.xfail(strict=True,reason='#2013 H table wrap (estimate): O2 reflection refinement fails')
def test_three_mesh_refinement_step():
    rows=three_mesh('O2')
    err=lambda n: np.abs(np.array(rows[n]['s11_abs'])-np.abs(rows[n]['s11_closed']))
    for coarse,fine in ((30,45),(45,60)):
        assert np.all(err(fine)<=err(coarse)), 'O2 reflection refinement'
