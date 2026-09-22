from __future__ import annotations
import contextlib
import gc
import json
import subprocess
import sys
import time
import traceback
from measure_common import *

def settings(row, sim, mod, captured):
    if captured is not None:
        return dict(method='compute_msl_s_matrix',kwargs=serial(captured))
    path=row['path']
    if path.endswith(('test_msl_internal_probe_advisories.py','test_msl_passivity_enforcement.py')):
        return dict(method='compute_msl_s_matrix',kwargs=dict(freqs=serial(mod.FREQS),num_periods=2.0))
    if path.endswith('test_coax_two_port_smatrix.py'):
        return dict(method='compute_coaxial_two_port',kwargs=dict(n_steps=6000,freqs=serial(mod.BAND)))
    if path.endswith('test_coax_msl_transition.py'):
        return dict(method='compute_coax_msl_transition',kwargs=dict(junction_x=mod.JUNCTION_X,eps_r_sub=mod.EPS_SUB,n_steps=mod.N_STEPS,n_freqs=3,probe_count=6,probe_start_cells=4,probe_spacing_cells=2,skip_preflight=True,strict_passivity=True))
    if path.endswith('06b_msl_notch_filter_uniform.py'):
        return dict(method='compute_msl_s_matrix',kwargs=dict(n_freqs=100,num_periods=20.0))
    if path.endswith('build_msl_thru_phase_dx50um_reference.py'):
        return dict(method='compute_msl_s_matrix',kwargs=dict(n_freqs=mod.N_FREQS,num_periods=mod.NUM_PERIODS))
    return None

def one(index):
    row=catalog()[index]
    start=time.perf_counter()
    record=dict(**row)
    try:
        sim,mod,captured=build(row)
        record['lane'],record['port_counts']=lane(sim)
        record['settings']=settings(row,sim,mod,captured)
        grid,nu,refusal=grid_for(sim)
        record.update(nonuniform=int(nu),uniform_grid_refusal=refusal)
        record.update(measurement(sim,grid,nu))
        record['x64']=int(jax.config.jax_enable_x64)
        if record['settings']:
            kw=record['settings']['kwargs']
            steps=kw.get('n_steps')
            if steps is None:
                f0=sim._freq_max/2
                steps=int(kw.get('num_periods',20)/(f0*float(grid.dt)))
                record['steps_for_ranking_formula']='int(num_periods / ((freq_max/2) * dt)); exact driver steps recorded before solve'
            record['steps_for_ranking']=steps
            record['cells_times_steps']=int(record['cells']*steps)
        record['completed']=1
    except BaseException as exc:
        record.update(completed=0,exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
        print(record['traceback'],flush=True)
    record['wall_s']=time.perf_counter()-start
    write_json(OUT/'census_records'/f'{index:03d}.json',record)
    print(json.dumps({k:record.get(k) for k in ('index','id','completed','lane','affected','cells','wall_s','exception')}),flush=True)

def main():
    if len(sys.argv)>1:
        return one(int(sys.argv[1]))
    rows=catalog()
    write_json(OUT/'catalog.json',rows)
    for row in rows:
        index=row['index']
        print(f"{index+1}/{len(rows)} {row['id']}",flush=True)
        cmd=[sys.executable,'-B',str(OUT/'census_driver.py'),str(index)]
        start=time.perf_counter()
        with (OUT/'census_records'/f'{index:03d}.log').open('x') as log:
            try:
                proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                rc=proc.returncode
            except subprocess.TimeoutExpired as exc:
                rc=124
                print(str(exc),file=log)
        path=OUT/'census_records'/f'{index:03d}.json'
        if not path.exists():
            write_json(path,dict(**row,completed=0,returncode=rc,wall_s=time.perf_counter()-start,exception=(OUT/'census_records'/f'{index:03d}.log').read_text()))
        rec=json.loads(path.read_text())
        print(json.dumps({k:rec.get(k) for k in ('index','completed','lane','affected','cells','wall_s','exception')}),flush=True)
    write_json(OUT/'census.json',dict(variants=[json.loads(p.read_text()) for p in sorted((OUT/'census_records').glob('*.json'))]))

if __name__=='__main__':
    main()
