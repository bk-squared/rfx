"""Lossless homogeneous WR-90 and dielectric-step scattering oracles (#1994)."""
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


def judge(row,empty):
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
    assert np.all(power<=1+np.maximum(2*np.abs(1-np.array(empty['sum'])),1e-3)), 'W1 per-bin passivity'


@pytest.mark.parametrize('case',['O1','O2'])
@pytest.mark.parametrize('cells',[30,45,60])
def test_filled_waveguide_preflight_clean(case,cells):
    sim,freqs=build(case,cells)
    assert_preflight_clean(preflight_findings(sim,freqs,round(8000*cells/45)))


@pytest.mark.parametrize('case',['O1','O2'])
def test_filled_waveguide_oracles(case):
    empty=measure('empty_filled_band' if case=='O1' else 'empty')
    row=measure(case)
    dump(empty);dump(row)
    judge(row,empty)


def test_filled_waveguide_inward_reference_shift():
    empty=measure('empty_filled_band')
    shifted=measure('O1',shift=True)
    dump(empty);dump(shifted)
    judge(shifted,empty)


@pytest.mark.slow
def test_three_mesh_trend():
    rows={}
    for cells in (30,45,60):
        for case in ('O1','O2'):
            empty=measure('empty_filled_band' if case=='O1' else 'empty',cells)
            row=measure(case,cells)
            dump(empty);dump(row);judge(row,empty)
            rows[cells,case]=row
    for coarse,fine in ((30,45),(45,60)):
        assert np.all(np.abs(rows[fine,'O1']['phase_error_continuous'])<=np.abs(rows[coarse,'O1']['phase_error_continuous'])), 'O1 phase refinement'
        err=lambda n: np.abs(np.array(rows[n,'O2']['s11_abs'])-np.abs(rows[n,'O2']['s11_closed']))
        assert np.all(err(fine)<=err(coarse)), 'O2 reflection refinement'
