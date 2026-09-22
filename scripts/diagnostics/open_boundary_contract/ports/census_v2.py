from __future__ import annotations
import contextlib
import io
import subprocess
import time
import traceback
from measure_common import *
import census_driver as first

def catalog_v2():
    rows=catalog()
    unit='tests/unit/sparams/'
    extras=[
        dict(path=unit+'test_mixed_port_sparam.py',builder='test_mixed_probe_fed_msl_plumbing_smoke',label='default',capture_mixed=True),
        dict(path=unit+'test_msl_plane_primitives_parity.py',builder='_build_thru',label='aligned',kwargs={'dx':254e-6/3},manufactured=1),
        dict(path=unit+'test_msl_plane_primitives_parity.py',builder='_build_thru',label='bisecting',kwargs={'dx':80e-6},manufactured=1),
        dict(path=unit+'test_msl_power_normalization.py',builder='_case',label='unequal',kwargs={'equal':False},prefix=True,manufactured=1),
        dict(path=unit+'test_msl_power_normalization.py',builder='_case',label='equal',kwargs={'equal':True},prefix=True,manufactured=1),
    ]
    for x in extras:
        x.update(kind='additional',index=len(rows))
        x.setdefault('kwargs',{})
        x['id']=f"{x['path']}::{x['builder']}::{x['label']}"
        rows.append(x)
    return rows

def build_v2(row):
    if row.get('capture_mixed'):
        from unittest.mock import patch
        mod=load_module(row['path'])
        captured={}
        def intercept(sim,*a,**kw):
            captured.update(sim=sim,kwargs=kw)
            raise Captured()
        try:
            with patch.object(Simulation,'compute_mixed_s_matrix',intercept):
                getattr(mod,row['builder'])()
        except Captured:
            return captured['sim'],mod,captured['kwargs']
        raise RuntimeError('no mixed call captured')
    if row.get('prefix'):
        mod=load_module(row['path'])
        tree=ast.parse(inspect.getsource(getattr(mod,row['builder'])))
        f=tree.body[0]
        cut=next(i for i,s in enumerate(f.body) if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='grid' for t in s.targets))
        f.body=f.body[:cut]+[ast.Return(value=ast.Name(id='sim',ctx=ast.Load()))]
        f.decorator_list=[]
        ast.fix_missing_locations(tree)
        ns=dict(vars(mod))
        exec(compile(tree,str(OUT/'census_v2.py'),'exec'),ns)
        return ns[f.name](None,**row['kwargs']),mod,None
    return build(row)

def grid_v2(sim):
    try:
        return sim._build_grid(),False,None
    except (ValueError,NotImplementedError) as exc:
        if any(getattr(sim,a,None) is not None for a in ('_dx_profile','_dy_profile','_dz_profile')):
            return sim._build_nonuniform_grid(),True,str(exc)
        raise

def lane_v2(sim):
    counts={key:len(getattr(sim,key,[])) for key in ('_msl_ports','_coaxial_ports','_waveguide_ports')}
    counts['lumped_ports']=sum(p.impedance>0 and p.extent is None for p in sim._ports)
    counts['wire_ports']=sum(p.impedance>0 and p.extent is not None for p in sim._ports)
    counts['sources']=sum(p.impedance==0 for p in sim._ports)
    lw=counts['lumped_ports']+counts['wire_ports']
    if counts['_coaxial_ports'] and counts['_msl_ports']:
        name='coax_msl_transition'
    elif counts['_msl_ports'] and lw:
        name='mixed'
    elif counts['_coaxial_ports']:
        name='coax_two_port'
    elif counts['_msl_ports']:
        name='msl_two_port'
    elif counts['_waveguide_ports']:
        name='waveguide'
    elif lw:
        name='lumped_wire'
    else:
        name='none'
    return name,counts

def settings_v2(row,sim,mod,captured):
    if row.get('manufactured'):
        return None
    if row.get('capture_mixed'):
        return dict(method='compute_mixed_s_matrix',kwargs=serial(captured))
    setting=first.settings(row,sim,mod,captured)
    if setting:
        return setting
    if row['path'].endswith('test_msl_sheet_threading.py'):
        return dict(method='compute_msl_s_matrix',kwargs=dict(freqs=serial(mod.FREQS),num_periods=12.0))
    if row['path'].endswith('test_msl_sparse_dft.py'):
        return dict(method='compute_msl_s_matrix',kwargs=dict(n_freqs=3,num_periods=3.0))
    calls=[]
    for node in ast.walk(ast.parse((SRC/row['path']).read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('compute_msl_s_matrix','compute_mixed_s_matrix','compute_waveguide_s_matrix'):
            kwargs={}
            for kw in node.keywords:
                try:
                    kwargs[kw.arg]=serial(eval(compile(ast.Expression(kw.value),'settings','eval'),vars(mod)))
                except Exception:
                    break
            else:
                calls.append(dict(method=node.func.attr,kwargs=kwargs,source_line=node.lineno))
    if calls:
        return min(calls,key=lambda c:c['kwargs'].get('n_steps',float(c['kwargs'].get('num_periods',40))*1e6))
    return None

def one(index):
    row=catalog_v2()[index]
    start=time.perf_counter()
    record=dict(**row)
    previous_path=OUT/'census_records'/f'{index:03d}.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
    try:
        sim,mod,captured=build_v2(row)
        record['lane'],record['port_counts']=lane_v2(sim)
        record['settings']=settings_v2(row,sim,mod,captured)
        grid,nu,refusal=grid_v2(sim)
        record.update(nonuniform=int(nu),uniform_grid_refusal=refusal)
        if previous and previous.get('completed'):
            for key in ('domain_m','grid_shape','cells','dt_s','face_pads','boundary','entities','affected','assembly_tuple_length','total_cells','total_edges','total_eps_not_1'):
                record[key]=previous[key]
            record['measurement_file']=str(previous_path.relative_to(OUT))
            record['measurement_wall_s']=previous['wall_s']
        elif previous and previous.get('exception_type') not in ('NotImplementedError',):
            raise RuntimeError('Previous item stopped: '+previous.get('exception',''))
        else:
            record.update(measurement(sim,grid,nu))
        record['x64']=int(jax.config.jax_enable_x64)
        if record['settings']:
            kw=record['settings']['kwargs']
            steps=kw.get('n_steps')
            if steps is None:
                steps=int(np.ceil(kw.get('num_periods',40)/(sim._freq_max*float(grid.dt))))
            record['steps_for_ranking']=steps
            record['cells_times_steps']=int(record['cells']*steps)
        record['completed']=1
    except BaseException as exc:
        record.update(completed=0,exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
        print(record['traceback'],flush=True)
    record['wall_s']=time.perf_counter()-start
    write_json(OUT/'census_records_v2'/f'{index:03d}.json',record)
    print(json.dumps({k:record.get(k) for k in ('index','completed','lane','affected','cells','wall_s','exception')}),flush=True)

def main():
    if len(sys.argv)>1:
        return one(int(sys.argv[1]))
    rows=catalog_v2()
    write_json(OUT/'catalog_v2.json',rows)
    for row in rows:
        i=row['index']
        print(f"{i+1}/{len(rows)} {row['id']}",flush=True)
        cmd=[sys.executable,'-B',str(OUT/'census_v2.py'),str(i)]
        with (OUT/'census_records_v2'/f'{i:03d}.log').open('x') as log:
            try:
                proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                rc=proc.returncode
            except subprocess.TimeoutExpired as exc:
                rc=124
                print(str(exc),file=log)
        path=OUT/'census_records_v2'/f'{i:03d}.json'
        if not path.exists():
            write_json(path,dict(**row,completed=0,returncode=rc,exception=(OUT/'census_records_v2'/f'{i:03d}.log').read_text()))
        rec=json.loads(path.read_text())
        print(json.dumps({k:rec.get(k) for k in ('index','completed','lane','affected','cells','wall_s','exception')}),flush=True)
    write_json(OUT/'census.json',dict(variants=[json.loads(p.read_text()) for p in sorted((OUT/'census_records_v2').glob('*.json'))]))

if __name__=='__main__':
    main()
