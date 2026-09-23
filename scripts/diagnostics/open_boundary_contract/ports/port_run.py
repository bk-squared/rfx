from __future__ import annotations
import argparse
import contextlib
import textwrap
import time
import traceback
import warnings
from unittest.mock import patch
from census_v2 import *
from rfx import Box
from rfx import simulation as lowlevel

class Tee:
    def __init__(self,*streams):
        self.streams=streams
    def write(self,text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()

def copied_builder(row,baseline,grid,entities):
    mod=load_module(row['path'])
    changes=[]
    lookup={}
    for e in entities:
        if not e['affected_faces']:
            continue
        if e['shape_type']!='Box':
            raise NotImplementedError('affected conductor is not a Box: '+e['shape_type'])
        lo,hi=[list(x) for x in e['bounds_m']]
        newlo,newhi=lo.copy(),hi.copy()
        for face in e['affected_faces']:
            a='xyz'.index(face[0])
            delta=(pads(grid)[face]+1)*float(grid.dx)
            if face.endswith('lo'):
                newlo[a]=-delta
            else:
                newhi[a]=float(baseline._domain[a])+delta
        key=(tuple(lo),tuple(hi))
        lookup[key]=(tuple(newlo),tuple(newhi))
        changes.append(dict(collection=e['collection'],index=e['index'],old_bounds_m=[lo,hi],new_bounds_m=[newlo,newhi],faces=e['affected_faces']))
    calls=[]
    def extended_box(*args,**kwargs):
        old=Box(*args,**kwargs)
        key=tuple(tuple(map(float,x)) for x in old.bounding_box())
        if key in lookup:
            calls.append(key)
            return Box(*lookup[key])
        return old
    class ReplaceBox(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='Box':
                node.func=ast.Name(id='_ports_extended_box',ctx=ast.Load())
            return node
    ns=dict(vars(mod))
    ns['_ports_extended_box']=extended_box
    defs=[]
    for node in ast.parse((SRC/row['path']).read_text()).body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            node.decorator_list=[]
            defs.append(ReplaceBox().visit(node))
    tree=ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[]))
    exec(compile(tree,str(OUT/'port_run.py')+'::builder_copy','exec'),ns)
    fn=ns[row['builder']]
    if row['kind']=='audited':
        b=next(x for x in lib.CLASSIFICATION[row['path']].builders if x.fn==row['builder'])
        v=next(x for x in b.variants if x.label==row['label'])
        with lib.build_only():
            result=fn(**v.kwargs(mod))
        sim=result if b.result_index is None else result[b.result_index]
    elif row.get('capture') or row.get('capture_mixed'):
        capture={}
        def intercept(sim,*args,**kwargs):
            capture['sim']=sim
            raise Captured()
        method='compute_mixed_s_matrix' if row.get('capture_mixed') else 'compute_msl_s_matrix'
        try:
            with patch.object(Simulation,method,intercept):
                fn(**row.get('kwargs',{}))
        except Captured:
            sim=capture['sim']
        else:
            raise RuntimeError('copied builder reached no public S-parameter call')
    else:
        with lib.build_only():
            sim=fn(**row.get('kwargs',{}))
    assert len(calls)==len(changes),(len(calls),len(changes))
    allowed={(c['collection'],c['index']):c for c in changes}
    for collection in ('_geometry','_thin_conductors'):
        before=getattr(baseline,collection)
        after=getattr(sim,collection)
        assert len(before)==len(after)
        for i,(b,a) in enumerate(zip(before,after)):
            if (collection,i) in allowed:
                assert serial(a.shape.bounding_box())==allowed[(collection,i)]['new_bounds_m']
                assert dataclasses.asdict(dataclasses.replace(a,shape=b.shape))==dataclasses.asdict(b)
            else:
                assert serial(a)==serial(b),(collection,i)
    for name in ('_ports','_msl_ports','_coaxial_ports','_waveguide_ports','_probes','_boundary_spec','_materials'):
        if hasattr(baseline,name):
            assert serial(getattr(baseline,name))==serial(getattr(sim,name)),name
    return sim,changes

def arrays_for(grid,materials,kwargs):
    arrays={k:np.asarray(getattr(materials,k)) for k in ('eps_r','mu_r','sigma')}
    mask=kwargs.get('pec_mask')
    arrays['pec_mask']=np.zeros(grid.shape,bool) if mask is None else np.asarray(mask)
    edges=kwargs.get('pec_edge_masks')
    for i,axis in enumerate('xyz'):
        arrays['pec_edge_'+axis]=np.zeros(grid.shape,bool) if edges is None else np.asarray(edges[i])
    return arrays

def result_dict(result):
    if dataclasses.is_dataclass(result):
        return {f.name:getattr(result,f.name) for f in dataclasses.fields(result)}
    if hasattr(result,'_asdict'):
        return result._asdict()
    if isinstance(result,dict):
        return result
    return vars(result)

def execute(record,variant,out,dry_run=False):
    start=time.perf_counter()
    status=dict(index=record['index'],lane=record['lane'],variant=variant,completed=0,solve_calls=0,dry_run=int(dry_run))
    try:
        row=catalog_v2()[record['index']]
        baseline,mod,unused=build_v2(row)
        grid,nu,refusal=grid_v2(baseline)
        before=assemble(baseline,grid,nu)
        changes=[]
        sim=baseline
        if variant=='continued':
            if nu:
                raise NotImplementedError('continued builder uses uniform boundary cell size; nonuniform selected')
            sim,changes=copied_builder(row,baseline,grid,record['entities'])
        expected=assemble(sim,grid,nu)
        material_equal={k:int(np.array_equal(np.asarray(getattr(before[0][0],k)),np.asarray(getattr(expected[0][0],k)))) for k in ('eps_r','mu_r','sigma')}
        assert all(material_equal.values()),material_equal
        mm=measurement(sim,grid,nu)
        write_json(out/'assembly_before_solve.json',dict(**mm,changes=changes,materials_equal_to_baseline=material_equal))
        save_npz(out/'assembly_before_solve.npz',pec_mask=expected[1],pec_edge_x=expected[2][0],pec_edge_y=expected[2][1],pec_edge_z=expected[2][2],eps_r=np.asarray(expected[0][0].eps_r))
        print(json.dumps(dict(variant=variant,grid=mm['grid_shape'],cells=mm['total_cells'],edges=mm['total_edges'],changes=changes)),flush=True)
        if variant=='continued':
            for change in changes:
                entity=next(e for e in mm['entities'] if e['collection']==change['collection'] and e['index']==change['index'])
                for face in change['faces']:
                    count=entity['volume_cells'][face]['absorber']+sum(entity['edges'][a][face]['absorber'] for a in 'xyz')
                    if count==0:
                        raise RuntimeError('continuation_absorber_count=0: '+str(change)+' '+face)
        with (out/'preflight.txt').open('x') as preflight:
            with contextlib.redirect_stdout(preflight),contextlib.redirect_stderr(preflight):
                try:
                    issues=sim.preflight(strict=False)
                    for issue in issues:
                        print(serial(issue))
                except Exception:
                    traceback.print_exc()
        orig=lowlevel.run
        def observe(run_grid,materials,n_steps,*args,**kwargs):
            n=status['solve_calls']
            arrays=arrays_for(run_grid,materials,kwargs)
            counts={k:face_counts(v,run_grid) for k,v in arrays.items() if k.startswith('pec_')}
            entry=dict(call=n,n_steps=int(n_steps),grid_shape=list(run_grid.shape),face_pads=pads(run_grid),counts=counts,sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()},pec_mask_present=int(kwargs.get('pec_mask') is not None),pec_edge_masks_present=int(kwargs.get('pec_edge_masks') is not None))
            edge_equal={axis:int(np.array_equal(arrays['pec_edge_'+axis],expected[2][i])) for i,axis in enumerate('xyz')}
            entry['edges_equal_to_manual_assembly']=edge_equal
            status['solve_calls']+=1
            save_npz(out/f'assembly_received_{n:02d}.npz',**arrays)
            write_json(out/f'assembly_received_{n:02d}.json',entry)
            print(json.dumps(dict(readback=entry)),flush=True)
            if not all(edge_equal.values()):
                raise RuntimeError('solver_edge_masks_differ_from_manual_assembly: '+str(edge_equal))
            if dry_run:
                raise Captured()
            result=orig(run_grid,materials,n_steps,*args,**kwargs)
            rd=result if isinstance(result,dict) else result_dict(result)
            arrays={k:np.asarray(v) for k,v in rd.items() if v is not None and k in ('time_series','energy_history','energy','max_field_history')}
            if arrays:
                save_npz(out/f'witness_series_{n:02d}.npz',**arrays)
            return result
        kwargs=dict(record['settings']['kwargs'])
        for key in ('freqs','s_param_freqs'):
            if key in kwargs:
                kwargs[key]=np.asarray(kwargs[key])
        write_json(out/'settings.json',record['settings'])
        lowlevel.run=observe
        try:
            result=getattr(sim,record['settings']['method'])(**kwargs)
        except Captured:
            if not dry_run:
                raise
            status['completed']=1
            status['dry_readback_completed']=1
            return status
        finally:
            lowlevel.run=orig
        rd=result_dict(result)
        S=rd.get('S',rd.get('s_params'))
        freqs=rd.get('freqs',kwargs.get('freqs',kwargs.get('s_param_freqs')))
        if S is None or freqs is None:
            raise RuntimeError('public result has no S matrix or frequency vector')
        save_npz(out/'s.npz',S=np.asarray(S),freqs=np.asarray(freqs))
        numeric={k:np.asarray(v) for k,v in rd.items() if isinstance(v,(np.ndarray,jax.Array,np.generic,float,int,bool,complex))}
        save_npz(out/'diagnostics.npz',**numeric)
        write_json(out/'diagnostics.json',rd)
        status['completed']=1
        status['frequencies']=int(len(freqs))
        status['max_column_power']=float(np.max(np.sum(np.abs(np.asarray(S))**2,axis=0)))
        return status
    except BaseException as exc:
        status.update(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
        print(status['traceback'],flush=True)
        return status
    finally:
        status['wall_s']=time.perf_counter()-start
        write_json(out/'status.json',status)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('index',type=int)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--require-gpu',action='store_true')
    args=ap.parse_args()
    if args.require_gpu:
        assert all(d.platform=='gpu' for d in jax.devices()),jax.devices()
    record=json.loads((OUT/'census_records_v2'/f'{args.index:03d}.json').read_text())
    dest=OUT/('dry_readback' if args.dry_run else record['lane'])/f'fixture_{args.index:03d}'
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    write_json(dest/'provenance.json',dict(devices=[str(d) for d in jax.devices()],x64=int(jax.config.jax_enable_x64),source_tree=str(SRC),source_record=(BASE/'PROVENANCE.txt').read_text(),sha256={str(p.relative_to(BASE)):sha(p) for p in (Path(__file__),OUT/'measure_common.py',OUT/'census_driver.py',OUT/'census_v2.py',SRC/'rfx/boundaries/cpml.py',SRC/record['path'])}))
    for variant in ('baseline','continued'):
        out=dest/variant
        out.mkdir()
        with (out/'run.log').open('x') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                status=execute(record,variant,out,args.dry_run)
        print(json.dumps(status),flush=True)
        if not status['completed']:
            break

if __name__=='__main__':
    main()
