from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/root/workspace/bk-workspace/.801-measure/ports')

def fmt(x):
    if x is None:
        return 'NA'
    if isinstance(x,(bool,np.bool_)):
        return str(int(x))
    if isinstance(x,(int,np.integer)):
        return str(x)
    if isinstance(x,(float,np.floating)):
        return format(float(x),'.12g')
    if isinstance(x,(list,tuple)):
        return ', '.join(fmt(y) for y in x)
    return str(x).replace('|','\\|').replace('\n',' ')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])+'\n'

def dump(path,obj):
    with path.open('x') as f:
        json.dump(obj,f,indent=2)
        f.write('\n')

def flatten(value,path=''):
    if isinstance(value,dict):
        for k,v in value.items():
            yield from flatten(v,path+'.'+str(k) if path else str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):
            yield from flatten(v,path+'['+str(i)+']')
    elif isinstance(value,(float,int,bool)):
        yield path,value

def census_text(records):
    face_order=['x_lo','x_hi','y_lo','y_hi','z_lo','z_hi']
    main=[]
    entries=[]
    faces=[]
    for r in records:
        main.append([r['index'],r['id'],r['lane'],r.get('grid_shape'),[r.get('face_pads',{}).get(f) for f in face_order],[r.get('boundary',{}).get(f) for f in face_order],r.get('nonuniform'),r.get('completed'),r.get('affected')])
        for i,e in enumerate(r.get('entities',[])):
            label=e.get('collection','stamp')+'['+str(e.get('index',i))+']'
            entries.append([r['index'],label,e.get('material',e.get('name')),e['conductor'],e.get('shape_type','Cylinder difference' if e.get('name')=='shell' else 'Cylinder'),e['bounds_m'][0],e['bounds_m'][1],e.get('sheet_count'),e.get('wire_count')])
            for face,reach in e['reaches'].items():
                v=e.get('volume_cells',{}).get(face,{})
                edges=e.get('edges',{})
                er=e.get('eps_not_1') or {}
                ec=er.get(face,{})
                sig=(e.get('sigma_pec_cells') or {}).get(face,{})
                faces.append([r['index'],label,face,int(reach),v.get('last_interior'),v.get('absorber'),*[edges.get(a,{}).get(face,{}).get(k) for k in ('last_interior','absorber') for a in 'xyz'],ec.get('last_interior'),ec.get('absorber'),sig.get('last_interior'),sig.get('absorber'),int(face in e['affected_faces'])])
    counts=collections.Counter(r['lane'] for r in records)
    affected=collections.defaultdict(list)
    for r in records:
        if r.get('affected'):
            affected[r['lane']].append(r['index'])
    text='Face order: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi. Bounds: m. Boundary column: declared BoundarySpec.\n\n'
    text+='Entry counts: isolated entry assembly. Combined counts: census.json total_cells, total_edges, total_eps_not_1.\n\n'
    text+='Last interior index: pad_lo; shape[axis] - pad_hi - 1. Absorber indices: [0,pad_lo); [shape-pad_hi,shape).\n\n'
    text+='Declared reach tolerance: domain_length × 1e-9. Affected: declared reach = 1 and absorber conductor cells + edges = 0.\n\n'
    text+=table(['lane','records','affected records','affected indices'],[[l,n,len(affected[l]),affected[l]] for l,n in sorted(counts.items())])+'\n'
    text+=table(['index','source :: builder :: variant','lane / port family','shape','face pads','declared boundaries','nonuniform','completed','affected'],main)+'\n'
    text+=table(['index','entry','material / name','conductor','shape','lo m','hi m','sheets','wires'],entries)+'\n'
    text+=table(['index','entry','face','declared reach','interior PEC cells','absorber PEC cells','interior Ex','interior Ey','interior Ez','absorber Ex','absorber Ey','absorber Ez','interior eps≠1','absorber eps≠1','interior sigma≥1e6','absorber sigma≥1e6','affected'],faces)
    return text

def main():
    records=json.loads((ROOT/'census.json').read_text())['variants']
    assert len({r['id'] for r in records})==len(records)
    assert sum(r['kind']=='audited' for r in records)==62
    assert all(r['completed']==1 for r in records)
    rows=[]
    metrics={}
    checks=[]
    def add(lane,index,q,b=None,c=None,d=None,unit='1'):
        rows.append([lane,index,q,b,c,d,unit])
    for lane,index in [('msl_two_port',62),('mixed',81)]:
        root=ROOT/lane/f'fixture_{index:03d}'
        arrays=[]
        diags=[]
        statuses=[]
        for variant in ('baseline','continued'):
            p=root/variant
            with np.load(p/'s.npz') as f:
                S=np.asarray(f['S'],dtype=np.complex128)
                freq=np.asarray(f['freqs'],dtype=np.float64)
            assert np.isfinite(S).all() and np.isfinite(freq).all()
            assert S.shape[-1]==len(freq)
            arrays.append((S,freq))
            diags.append(json.loads((p/'diagnostics.json').read_text()))
            statuses.append(json.loads((p/'status.json').read_text()))
            assert statuses[-1]['completed']==1 and statuses[-1]['solve_calls']==2
            with np.load(p/'assembly_before_solve.npz') as manual:
                for rbpath in sorted(p.glob('assembly_received_*.npz')):
                    with np.load(rbpath) as rb:
                        for component in 'xyz':
                            key='pec_edge_'+component
                            equal=int(np.array_equal(manual[key],rb[key]))
                            checks.append(dict(lane=lane,index=index,variant=variant,readback=rbpath.name,component=component,equal=equal))
                            assert equal
        B,F=arrays[0]
        C,G=arrays[1]
        assert np.array_equal(F,G)
        result={}
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                q=f'S{i+1}{j+1}'
                b=np.abs(B[i,j]); c=np.abs(C[i,j])
                with np.errstate(divide='ignore',invalid='ignore'):
                    dbdiff=np.abs(20*np.log10(c)-20*np.log10(b))
                    linear=np.abs(c-b)
                    phase=np.abs(np.angle(C[i,j]*np.conj(B[i,j]),deg=True))
                    vals=dict(max_abs_magnitude_db_difference=float(dbdiff.max()),mean_abs_magnitude_db_difference=float(dbdiff.mean()),max_abs_phase_difference_deg=float(phase.max()),mean_abs_phase_difference_deg=float(phase.mean()),max_abs_linear_magnitude_difference=float(linear.max()),mean_abs_linear_magnitude_difference=float(linear.mean()),db_of_max_abs_linear_magnitude_difference=float(20*np.log10(linear.max())),db_of_mean_abs_linear_magnitude_difference=float(20*np.log10(linear.mean())))
                result[q]=vals
                for k,v in vals.items():
                    unit='deg' if '_deg' in k else ('dB' if 'db' in k else '1')
                    add(lane,index,q+'.'+k,d=v,unit=unit)
        pb=np.sum(np.abs(B)**2,axis=0)
        pc=np.sum(np.abs(C)**2,axis=0)
        add(lane,index,'max_column_power',float(pb.max()),float(pc.max()))
        for j in range(B.shape[1]):
            add(lane,index,f'max_column_power.port{j+1}',float(pb[j].max()),float(pc[j].max()))
        add(lane,index,'max_abs_S12_minus_S21',float(np.max(np.abs(B[0,1]-B[1,0]))),float(np.max(np.abs(C[0,1]-C[1,0]))))
        b11=np.abs(B[0,0]);c11=np.abs(C[0,0])
        ib=int(np.argmin(b11));ic=int(np.argmin(c11))
        add(lane,index,'S11.global_minimum_frequency',float(F[ib]),float(G[ic]),float(100*(G[ic]-F[ib])/F[ib]),'Hz; delta %')
        local_b=np.flatnonzero((b11[1:-1]<b11[:-2])&(b11[1:-1]<=b11[2:]))+1
        local_c=np.flatnonzero((c11[1:-1]<c11[:-2])&(c11[1:-1]<=c11[2:]))+1
        add(lane,index,'S11.interior_local_minimum_count',len(local_b),len(local_c))
        for n in range(max(len(local_b),len(local_c))):
            fb=float(F[local_b[n]]) if n<len(local_b) else None
            fc=float(G[local_c[n]]) if n<len(local_c) else None
            add(lane,index,f'S11.interior_local_minimum_frequency[{n}]',fb,fc,100*(fc-fb)/fb if fb is not None and fc is not None else None,'Hz; delta %')
        add(lane,index,'returned_resonance_frequency_fields',sum('resonan' in k.lower() for k in diags[0]),sum('resonan' in k.lower() for k in diags[1]))
        add(lane,index,'wall_time',statuses[0]['wall_s'],statuses[1]['wall_s'],unit='s')
        add(lane,index,'frequency_bins',len(F),len(G))
        add(lane,index,'frequency_low',float(F[0]),float(G[0]),unit='Hz')
        add(lane,index,'frequency_high',float(F[-1]),float(G[-1]),unit='Hz')
        for key in ('S_raw','S_wave'):
            for k,d in enumerate(diags):
                if d.get(key) is not None:
                    ss=np.array([[[complex(v['real'],v['imag']) for v in line] for line in port] for port in d[key]])
                    result.setdefault(key,{})[('baseline','continued')[k]]=float(np.max(np.sum(np.abs(ss)**2,axis=0)))
            if key in result:
                add(lane,index,key+'.max_column_power',result[key].get('baseline'),result[key].get('continued'))
        skip={'S','s_params','S_raw','S_wave','freqs'}
        bd=dict(flatten({k:v for k,v in diags[0].items() if k not in skip}))
        cd=dict(flatten({k:v for k,v in diags[1].items() if k not in skip}))
        for key in sorted(set(bd)|set(cd)):
            unit='ohm' if key.startswith(('Z0','reference_impedances','z0_ref')) else 'rad/m' if key.startswith('beta[') else 'dB' if key.startswith('settling_db') else 'm' if '_m' in key else '1'
            add(lane,index,'diagnostic.'+key,bd.get(key),cd.get(key),unit=unit)
        for key in sorted(set(diags[0])|set(diags[1])):
            add(lane,index,'returned_field_present.'+key,int(diags[0].get(key) is not None),int(diags[1].get(key) is not None))
        result['global_minimum_hz']={'baseline':float(F[ib]),'continued':float(G[ic])}
        result['local_minima_hz']={'baseline':F[local_b].tolist(),'continued':G[local_c].tolist()}
        result['max_column_power']={'baseline':float(pb.max()),'continued':float(pc.max())}
        metrics[lane]=result
    for variant in ('baseline','continued'):
        status=json.loads((ROOT/'dry_coax'/'fixture_077'/variant/'status.json').read_text())
        add('coax_two_port',77,'dry_readback_calls.'+variant,status['solve_calls'],unit='count')
        add('coax_two_port',77,'FDTD_steps.'+variant,0,unit='count')
    content='ΔdB = abs(20 log10(abs(S_cont)) − 20 log10(abs(S_base))). Δphase = abs(arg(S_cont × conj(S_base))), principal angle.\n\n'
    content+='Band: every returned frequency bin. Minimum frequencies: sampled bins; interpolation count 0. Interior local minima: strict left, non-strict right; paired by ascending-frequency index.\n\n'
    content+='Reduction floating-point bits: 64. NA: no value.\n\n'
    content+=table(['lane','fixture','quantity','baseline','continued','difference / comparison','unit'],rows)
    content+='\nCoax continuation, verbatim:\n\n```text\n'+json.loads((ROOT/'dry_coax/fixture_077/continued/status.json').read_text())['traceback']+'```\n'
    counts=dict(records=len(records),audited=sum(r['kind']=='audited' for r in records),completed=sum(r['completed'] for r in records),affected=sum(r.get('affected',0) for r in records),nonuniform=sum(r.get('nonuniform',0) for r in records),audited_nonuniform=sum(r.get('nonuniform',0) for r in records if r['kind']=='audited'),readback_component_checks=len(checks),readback_component_equal=sum(r['equal'] for r in checks))
    ct=census_text(records)
    for path in (ROOT/'CENSUS.md',ROOT/'TABLE.md',ROOT/'comparison.json',ROOT/'verification.json'):
        assert not path.exists(),path
    with (ROOT/'CENSUS.md').open('x') as f:
        f.write(ct)
    with (ROOT/'TABLE.md').open('x') as f:
        f.write(content)
    dump(ROOT/'comparison.json',metrics)
    dump(ROOT/'verification.json',dict(counts=counts,checks=checks))
    print(json.dumps(counts))
    print(json.dumps({lane:{k:v for k,v in m.items() if k.startswith('S1') or k=='max_column_power'} for lane,m in metrics.items()}))

if __name__=='__main__':
    main()
