"""Lossless homogeneous WR-90 and dielectric-step scattering oracles (#1994), judged by bars B1-B5 and the group delay
(rfx-lead L:fd8d39)."""
import functools
import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest
from tests._electrical_length import ELECTRICAL_LENGTH_FRAC, electrical_length_ratio, transmitting_bins
from tests._filled_waveguide import C0, EPS, measure, build, beta, preflight_findings, assert_preflight_clean, record_steps


def dump(row):
    directory=os.environ.get('FILLED_PORT_OUTPUT')
    if directory:
        path=Path(directory)
        path.mkdir(parents=True,exist_ok=True)
        (path/f"{row['case']}-{row['cells']}-{int(row.get('shift',False))}-{row.get('layers',12)}.json").write_text(json.dumps(row,indent=2))


@pytest.mark.parametrize('case',['O1','O2'])
@pytest.mark.parametrize('cells',[30,45,60])
def test_filled_waveguide_preflight_clean(case,cells):
    sim,freqs=build(case,cells)
    assert_preflight_clean(preflight_findings(sim,freqs,record_steps(cells)))


measured=functools.lru_cache(maxsize=None)(measure)
COMPARATOR=dict(O1='empty_filled_band',O2='empty')
# Case A fills the whole guide; case B is the step; A-shift is case A with its reference planes moved inward.
CASES={'A':('O1',False),'B':('O2',False),'A-shift':('O1',True)}


def passivity_excess(row,*,record_end=True):
    """B1 (rfx-lead L:1c3609, form (i)): sum <= 1 + delta + 2s + s^2 per bin, s the record check's per-bin share (pole
    tail plus the known drive beyond the record): a relative error s per entry moves |S11|^2+|S21|^2 by at most 2s + s^2
    (derivation: rfx-archive rfx/records/20261026-nu-record-end-drive-tail/DESIGN.md). Checked before adoption at a/45 on
    16000, 32000 and 64000 steps against the 64000-step sum, every bin inside (record-end-truncation.json there).
    record_end=False is the settled form (ii): rounding only."""
    # derived estimate: float32 rounding of the DFT ratio over N steps; not measured
    delta=np.sqrt(row['n_steps'])*2**-24*10
    share=np.zeros(len(row['sum']))
    if record_end:
        for w in row['settling_witness']:
            share=np.maximum(share,np.asarray(w['share_per_bin'],dtype=float))
    return np.array(row['sum'])-1-delta-2*share-share**2


def db(x):
    return 20*np.log10(np.asarray(x,dtype=float))


def cutoff_errors(row):
    """B3: each realized port cutoff against c0/(2 a_r sqrt(eps_r mu_r)), a_r the realized aperture."""
    assert {p['name'] for p in row['ports']}=={'p1','p2'}, f"B3 read no port config the run built: {row['ports']}"
    for p in row['ports']:
        assert np.isclose(p['aperture_cells']*row['dx'],row['realized_a']), f"B3 aperture {p['aperture_cells']} cells against {row['realized_a']} m"
    return [p['f_cutoff']/(C0/(2*row['realized_a']*np.sqrt(p['expected_eps_r']*1.)))-1 for p in row['ports']]


def group_delay(row):
    """v2 item 5: the electrical length between the reference planes against exp(-j beta L) in the fill."""
    f=np.array(row['freqs'])
    s21=np.array([complex(*z) for z in row['s21_complex']])
    length=abs(np.diff(row['planes'])[0])
    return electrical_length_ratio(f,s21,np.exp(-1j*beta(f,EPS,None,discrete=False)*length),transmitting_bins(s21))


def shares(row):
    return [(w['status'],float(np.nanmax(np.asarray(w.get('share_per_bin') or [np.nan],dtype=float))))
            for w in row['settling_witness']]


def log(name,row,empty):
    out=lambda *a: print(f"{name} a/{row['cells']} N={row['n_steps']}",*a,file=sys.stderr)
    out('freqs',row['freqs'])
    out('1-sum case',(1-np.array(row['sum'])).tolist())
    out('1-sum comparator',(1-np.array(empty['sum'])).tolist())
    s11,s21=np.array(row['s11_abs']),np.array(row['s21_abs'])
    if row['case']=='O1':
        out('B2 |S21| dB',db(s21).tolist())
        out('B2 |S11|/(2|S11|_empty)',(s11/(2*np.array(empty['s11_abs']))).tolist())
        out('|S11| case',s11.tolist())
        out('group delay ratio',group_delay(row))
    else:
        out('B2 |S11| dB error',db(s11/np.abs(row['s11_closed'])).tolist())
        out('B2 |S21| dB error',db(s21/np.array(row['s21_closed'])).tolist())
        out('||S11|-|Gamma||',np.abs(s11-np.abs(row['s11_closed'])).tolist())
    out('B3 cutoff error case',cutoff_errors(row),'comparator',cutoff_errors(empty))
    out('B4 witness case',shares(row),'comparator',shares(empty))


@functools.lru_cache(maxsize=None)
def rows(name,cells=45):
    case,shift=CASES[name]
    empty,row=measured(COMPARATOR[case],cells,shift=False),measured(case,cells,shift=shift)
    dump(empty);dump(row)
    assert_preflight_clean(row['preflight'])
    # The planes the group delay and B2 read are the planes asked for (build()), within half a cell.
    asked=(.050,.070) if shift else ((row['interface']-.040,row['interface']+.040) if case=='O2' else (.020,.100))
    assert np.allclose(row['planes'],asked,atol=row['dx']/2), f"reference planes {row['planes']} against {asked}"
    log(name,row,empty)
    return row,empty


ROWS=[('A','case'),('A','comparator'),('B','case'),('B','comparator'),('A-shift','case')]
@pytest.mark.parametrize('name,role',ROWS)
def test_passivity(name,role):
    row=dict(zip(('case','comparator'),rows(name)))[role]
    excess=passivity_excess(row)
    assert np.all(excess<=0), f'B1 passivity {name} {role}: sum-1-delta per bin {excess.tolist()}'


@pytest.mark.parametrize('name',['A','A-shift'])
def test_filled_magnitude(name):
    row,empty=rows(name)
    s11,s21=np.array(row['s11_abs']),np.array(row['s21_abs'])
    assert np.all(np.abs(db(s21))<=2), f'B2 |S21| within 2 dB of 1: {db(s21).tolist()}'
    # falsifier separation: a port on vacuum constants reads its own mismatch |Gamma| = 0.53 at 7.0 GHz and 0.38 at
    # 7.9 GHz (closed form, declared aperture)
    assert np.all(s11<=2*np.array(empty['s11_abs'])), f"B2 |S11| at most twice the empty guide's: {s11.tolist()}"


def test_step_magnitude():
    row,_=rows('B')
    reflection=db(np.array(row['s11_abs'])/np.abs(row['s11_closed']))
    transmission=db(np.array(row['s21_abs'])/np.array(row['s21_closed']))
    assert np.all(np.abs(reflection)<=2), f'B2 |S11| within 2 dB of |Gamma|: {reflection.tolist()}'
    assert np.all(np.abs(transmission)<=2), f'B2 |S21| within 2 dB of T: {transmission.tolist()}'


@pytest.mark.parametrize('name',CASES)
def test_cutoff(name):
    for role,row in zip(('case','comparator'),rows(name)):
        errors=cutoff_errors(row)
        assert np.all(np.abs(errors)<=.01), f'B3 realized cutoff within 1 % ({name} {role}): {errors}'


@pytest.mark.parametrize('name,role',ROWS)
def test_record(name,role):
    row=dict(zip(('case','comparator'),rows(name)))[role]
    for w in row['settling_witness']:
        if w['status']=='undetermined':
            print(f"B4 {name} {role}: status {w['status']}, reason {w.get('reason')}",file=sys.stderr)
        assert w['status']!='undetermined', f"B4 cannot judge ({name} {role}): {w.get('reason')}"
        share=np.asarray(w['share_per_bin'],dtype=float)
        assert w['status']=='pass' and np.all(share<=1e-2), f"B4 record ({name} {role}): {w['status']}, share {share.tolist()}"


@pytest.mark.parametrize('name',['A','A-shift'])
def test_group_delay(name):
    ratio=group_delay(rows(name)[0])
    assert abs(ratio)<=ELECTRICAL_LENGTH_FRAC, f'group delay between the reference planes: ratio {ratio}'


def three_mesh(name):
    return {cells: rows(name,cells) for cells in (30,45,60)}


@pytest.mark.slow
def test_three_mesh_trend_filled():
    s11={cells: np.array(row['s11_abs']) for cells,(row,_) in three_mesh('A').items()}
    assert np.all(s11[45]<s11[30]) and np.all(s11[60]<s11[45]), f'B5 case A |S11| falls a/30, a/45, a/60: {[s11[c].tolist() for c in (30,45,60)]}'


@pytest.mark.slow
def test_three_mesh_trend_step():
    err={cells: np.abs(np.array(row['s11_abs'])-np.abs(row['s11_closed'])) for cells,(row,_) in three_mesh('B').items()}
    assert np.all(err[45]<err[30]) and np.all(err[60]<err[45]), f'B5 case B ||S11|-|Gamma|| falls a/30, a/45, a/60: {[err[c].tolist() for c in (30,45,60)]}'


@pytest.mark.slow
@pytest.mark.parametrize('name',['A','B'])
def test_settled_passivity(name):
    """B1 form (ii), the weekly run: settled records, 64000*cells/45 steps, rounding margin only."""
    case,_=CASES[name]
    for cells in (30,45,60):
        n=round(64000*cells/45)
        for role,row in (('case',measured(case,cells,n_steps=n)),('comparator',measured(COMPARATOR[case],cells,n_steps=n))):
            excess=passivity_excess(row,record_end=False)
            print(f'B1 settled {name} {role} a/{cells} N={n} sum-1-delta',excess.tolist(),file=sys.stderr)
            assert np.all(excess<=0), f'B1 settled passivity {name} {role} a/{cells}: {excess.tolist()}'
