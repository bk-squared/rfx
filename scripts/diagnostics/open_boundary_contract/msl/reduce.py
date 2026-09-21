import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from render_edges import compact_record, fmt, render_record, table

sys.dont_write_bytecode = True
OUT = Path('/root/workspace/bk-workspace/.801-measure/msl')
SRC = OUT.parent / 'src-main'
FIXTURES = ('cv06b','cv20')
VARIANTS = ('baseline','continued')

def read(p):
    return json.loads(Path(p).read_text())

def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

def module(rel):
    spec=importlib.util.spec_from_file_location('_msl_reduce_'+Path(rel).stem,SRC/rel)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def stats(a):
    x=np.asarray(a,dtype=float)
    return dict(count=int(x.size), finite_count=int(np.isfinite(x).sum()),
                min_signed=float(np.min(x)), max_signed=float(np.max(x)),
                mean_signed=float(np.mean(x)), max_abs=float(np.max(abs(x))),
                mean_abs=float(np.mean(abs(x))))

def reduce_fixture(fixture):
    result=dict(fixture=fixture,variants={},edges={},checks={})
    data={}
    for variant in VARIANTS:
        p=OUT/fixture/variant
        st=read(p/'status.json')
        row=dict(status=st)
        result['variants'][variant]=row
        edge_paths=sorted(p.glob('assembly_received_*.json'))
        records=[compact_record(ep) for ep in edge_paths]
        result['edges'][variant]=records
        if len(records)==2:
            row['edge_arrays_equal_between_drives']={k:int(records[0]['sha256'][k]==records[1]['sha256'][k])
                for k in ('pec_mask','pec_edge_x','pec_edge_y','pec_edge_z','eps_r','mu_r')}
        if not st['completed']:
            continue
        with np.load(p/'s.npz') as z:
            s=np.asarray(z['S'],dtype=np.complex128)
            freqs=np.asarray(z['freqs'],dtype=float)
        with np.load(p/'diagnostics.npz') as z:
            diag={k:np.array(z[k]) for k in z.files}
        djson=read(p/'diagnostics.json')
        data[variant]=(freqs,s,diag)
        raw=diag.get('S_raw',s)
        power=np.sum(abs(s)**2,axis=0)
        raw_power=np.sum(abs(raw)**2,axis=0)
        reciprocity=abs(s[0,1]-s[1,0])
        row.update(freqs_hz=freqs.tolist(), s11_mag=abs(s[0,0]).tolist(),s21_mag=abs(s[1,0]).tolist(),
                   s11_db=(20*np.log10(abs(s[0,0]))).tolist(),s21_db=(20*np.log10(abs(s[1,0]))).tolist(),
                   s21_phase_deg=np.angle(s[1,0],deg=True).tolist(),
                   max_column_power=float(np.max(power)),column_power=power.tolist(),
                   max_column_power_raw=float(np.max(raw_power)),column_power_raw=raw_power.tolist(),
                   reciprocity=reciprocity.tolist(),max_reciprocity=float(np.max(reciprocity)),
                   mean_reciprocity=float(np.mean(reciprocity)),
                   max_reciprocity_raw=float(np.max(abs(raw[0,1]-raw[1,0]))),
                   settling_db=diag['settling_db'].tolist(),
                   reliable_true_per_port=np.sum(diag['reliable'],axis=-1).astype(int).tolist(),
                   beta_railed_true_per_port=np.sum(diag['beta_railed'],axis=-1).astype(int).tolist(),
                   max_passivity_correction=float(np.max(diag.get('passivity_correction',np.zeros(len(freqs))))),
                   result_fields=list(djson),probe_clearance=djson['probe_clearance'],
                   reference_impedances_ohm=diag['reference_impedances'].tolist(),
                   fitted_Z0_real_ohm=diag['Z0'].real.tolist(),fitted_Z0_imag_ohm=diag['Z0'].imag.tolist(),
                   fitted_beta_real_rad_per_m=diag['beta'].real.tolist(),fitted_beta_imag_rad_per_m=diag['beta'].imag.tolist())
        witness=[]
        for i in range(st['solve_calls']):
            with np.load(p/f'witness_series_{i:02d}.npz') as z:
                ts=np.asarray(z['time_series'],dtype=float)
            e=ts**2
            tail=max(1,len(ts)//10)
            ratio=(np.mean(e[-tail:],axis=0)+np.finfo(float).tiny)/(np.max(e,axis=0)+np.finfo(float).tiny)
            db=10*np.log10(ratio)
            witness.append(dict(drive=i,steps=len(ts),probes=ts.shape[1],tail_steps=tail,
                                per_probe_db=db.tolist(),worst_db=float(np.max(db)),
                                result_minus_recomputed_db=float(diag['settling_db'][i]-np.max(db))))
        row['ringdown_recomputed']=witness
        if fixture=='cv06b':
            sf=module('validation/crossval/comparators/spectral_features.py')
            notch=sf.refined_extremum(freqs,abs(s[1,0]),transform='log')
            i=notch['index']
            notch['frequency_method']='refined_extremum(transform="log"): 3-point parabola in log(|S21|), local bin spacing'
            notch['sampled_depth_db']=notch.pop('depth_db')
            if 0<i<len(freqs)-1:
                ys=20*np.log10(abs(s[1,0,i-1:i+2]))
                d=notch['sub_bin_shift']
                a=(ys[0]-2*ys[1]+ys[2])/2
                b=(ys[2]-ys[0])/2
                notch['parabolic_depth_db']=float(ys[1]+b*d+a*d*d)
                notch['stencil_bin_indices']=[i-1,i,i+1]
                notch['stencil_freqs_hz']=freqs[i-1:i+2].tolist()
                notch['stencil_s21_db']=ys.tolist()
            row['notch']=notch
        else:
            cv=module('validation/crossval/20_msl_phase_referee.py')
            er=records[0]
            dx=er['dx_m']
            lower=min(er['realized_trace_z_indices'])
            p0=next(p0 for p0 in er['planes'] if p0['z_index']==lower)
            low_y,high_y=p0['trace_y_index_slice']
            width=(high_y-low_y-1)*dx
            with np.load(p/'assembly_received_00.npz') as z:
                eps=z['eps_r']
                i=er['grid_shape'][0]//2
                j=p0['y_index']
                subrows=np.flatnonzero(eps[i,j,:]>1)
                hsub=(int(subrows[-1])+1-er['face_pads']['z_lo'])*dx
            meta=dict(h_sub_realized_m=hsub, trace_realization_kind='volume',
                      trace_wall_planes_realized=er['realized_trace_z_indices'],dx_m=dx,
                      trace_wall_planes_realized_z_m=[(k-er['face_pads']['z_lo'])*dx for k in er['realized_trace_z_indices']])
            height=cv._h_dielectric_under_strip(meta)
            eps_eff=cv._hammerstad_jensen_eps_eff(width,height,cv.B_EPS_R)
            analytic_exception=None
            try:
                witness=cv._analytic_beta_witness(freqs,diag['beta'],eps_eff=eps_eff,
                    tol_frac=cv.B_BETA_ANALYTIC_TOL_FRAC,label='stage_b_analytic_beta',solver='rfx')
            except RuntimeError as exc:
                analytic_exception=dict(type=type(exc).__name__,text=str(exc),traceback=traceback.format_exc())
                witness=ast.literal_eval(str(exc).split('Full result: ',1)[1])
            mask=cv._gate_band_mask(freqs)
            row['analytic_beta']=dict(function='20_msl_phase_referee.py::_analytic_beta_witness',
                  eps_eff_function='20_msl_phase_referee.py::_hammerstad_jensen_eps_eff',
                  height_function='20_msl_phase_referee.py::_h_dielectric_under_strip',
                  eps_eff_key='eps_eff_hammerstad_jensen', eps_eff=eps_eff, width_m=width,
                  height_m=height,eps_r=cv.B_EPS_R,c0_m_per_s=cv._C0,
                  gated_bin_indices=np.flatnonzero(mask).tolist(),freqs_hz=freqs[mask].tolist(),
                  beta_hj_rad_per_m=witness['beta_analytic_rad_per_m'],
                  beta_fitted_rad_per_m=witness['beta_measured_rad_per_m'],
                  signed_deviation_pct=((np.array(witness['beta_ratio_measured_over_analytic'])-1)*100).tolist(),
                  phase_s21_deg=np.angle(s[1,0,mask],deg=True).tolist(),
                  reliable=diag['reliable'][:,mask].astype(int).tolist(),
                  beta_railed=diag['beta_railed'][:,mask].astype(int).tolist(),exception=analytic_exception)
    if len(data)==2:
        f,b,db=data['baseline']
        fc,c,dc=data['continued']
        assert np.array_equal(f,fc)
        result['checks']['frequency_vectors_equal']=1
        result['deltas']={}
        for i in range(2):
            for j in range(2):
                lin=abs(c[i,j])-abs(b[i,j])
                decibels=20*np.log10(abs(c[i,j]))-20*np.log10(abs(b[i,j]))
                phase=np.angle(c[i,j]*np.conj(b[i,j]),deg=True)
                phase[(abs(b[i,j])==0)|(abs(c[i,j])==0)]=np.nan
                result['deltas'][f'S{i+1}{j+1}']=dict(linear=stats(lin),dB=stats(decibels),phase_deg=stats(phase))
        for field in ('eps_r','mu_r','sigma','pec_mask','pec_edge_x','pec_edge_y','pec_edge_z'):
            with np.load(OUT/fixture/'baseline/assembly_received_00.npz') as z:
                ba=z[field]
            with np.load(OUT/fixture/'continued/assembly_received_00.npz') as z:
                ca=z[field]
            result['checks'][field+'_changed_entries']=int(np.count_nonzero(ba!=ca))
        if fixture=='cv06b':
            bn=result['variants']['baseline']['notch']
            cn=result['variants']['continued']['notch']
            result['notch_change']=dict(bin_frequency_shift_pct=100*(cn['bin_f']/bn['bin_f']-1),
                refined_frequency_shift_pct=100*(cn['refined_f']/bn['refined_f']-1),
                sampled_depth_change_db=cn['sampled_depth_db']-bn['sampled_depth_db'],
                parabolic_depth_change_db=cn['parabolic_depth_db']-bn['parabolic_depth_db'])
    write(OUT/(fixture+'_reduced.json'),result)

def render_table():
    lines=['Delta = continued - baseline; phase delta = arg(S_continued * conj(S_baseline)) in degrees.',
           'S = public result.S; S_raw = public result.S_raw when present, otherwise S.',
           'Ring-down = result.settling_db; 10 log10(last-10%-mean(Ez²) / peak(Ez²)), maximum over the 10 point probes, per drive.', '']
    for fixture in FIXTURES:
        r=read(OUT/(fixture+'_reduced.json'))
        lines += [f'**{fixture}**','']
        rows=[]
        for v in VARIANTS:
            d=r['variants'][v]
            st=d['status']
            rows.append([v,st['completed'],st['solve_calls'],st['timestepping_calls'],st.get('frequencies'),st.get('public_call_wall_s'),st['wall_s']])
        lines += [table(['run','completed (0/1)','solve calls','time-stepping calls','bins','S-call wall (s)','total wall (s)'],rows)]
        if 'deltas' in r:
            rows=[]
            for entry,metrics in r['deltas'].items():
                for unit,stat in metrics.items():
                    rows.append([entry,unit,stat['max_abs'],stat['mean_abs'],stat['min_signed'],stat['max_signed'],stat['mean_signed']])
            lines += [table(['S entry','delta unit','max abs delta','mean abs delta','min signed delta','max signed delta','mean signed delta'],rows)]
        rows=[]
        for v,d in r['variants'].items():
            if 'max_column_power' in d:
                rows.append([v,d['max_column_power'],d['max_column_power_raw'],d['max_reciprocity'],d['mean_reciprocity'],d['max_reciprocity_raw'],d['max_passivity_correction']])
        lines += [table(['run','max column power S','max column power S_raw','max |S12-S21|','mean |S12-S21|','max raw |S12-S21|','max passivity correction'],rows)]
        rows=[]
        for v,d in r['variants'].items():
            for w in d.get('ringdown_recomputed',[]):
                i=w['drive']
                rows.append([v,i,d['settling_db'][i],w['worst_db'],w['result_minus_recomputed_db'],
                             d['reliable_true_per_port'][i],d['beta_railed_true_per_port'][i]])
        lines += [table(['run','drive','settling (dB)','recomputed (dB)','difference (dB)','reliable true bins','beta_railed true bins'],rows)]
        lines += ['Array comparison, baseline versus continued:','',table(['array/check','count/value'],list(r['checks'].items()))]
        if fixture=='cv06b' and 'notch_change' in r:
            lines += ['Notch frequency: bin argmin and `spectral_features.refined_extremum(transform="log")`, 3-point log-magnitude parabola. Depth: sampled minimum and the same parabola at its vertex.','']
            rows=[]
            for v,d in r['variants'].items():
                n=d['notch']
                rows.append([v,n['index'],n['bin_f']/1e9,n['refined_f']/1e9,n['sampled_depth_db'],n['parabolic_depth_db'],n['sub_bin_shift']])
            lines += [table(['run','bin index','bin f (GHz)','3-point f (GHz)','sampled depth (dB)','3-point depth (dB)','offset (bins)'],rows),
                      table(['delta','value'],list(r['notch_change'].items()))]
        if fixture=='cv20':
            lines += ['Analytic beta: `20_msl_phase_referee.py::_analytic_beta_witness`; eps_eff: `_hammerstad_jensen_eps_eff`, key `eps_eff_hammerstad_jensen`; dielectric height: `_h_dielectric_under_strip`. Signed deviation = 100 * (Re(beta) / beta_HJ - 1).','']
            rows=[]
            for v,d in r['variants'].items():
                if 'analytic_beta' not in d:
                    continue
                a=d['analytic_beta']
                rows.append([v,a['width_m']*1e6,a['height_m']*1e6,a['eps_r'],a['eps_eff'],a['c0_m_per_s']])
            lines += [table(['run','w (µm)','h (µm)','eps_r','eps_eff','c0 (m/s)'],rows)]
            for v,d in r['variants'].items():
                if 'analytic_beta' not in d:
                    continue
                a=d['analytic_beta']
                lines += [f'{v}, gated bins 3.0–4.5 GHz:','',
                    table(['f (GHz)','fitted beta (rad/m)','HJ beta (rad/m)','signed deviation (%)','angle(S21) (deg)','reliable p0/p1','beta_railed p0/p1'],
                          [[f/1e9,b,h,dev,phase,f"{a['reliable'][0][i]}/{a['reliable'][1][i]}",f"{a['beta_railed'][0][i]}/{a['beta_railed'][1][i]}"]
                           for i,(f,b,h,dev,phase) in enumerate(zip(a['freqs_hz'],a['beta_fitted_rad_per_m'],a['beta_hj_rad_per_m'],a['signed_deviation_pct'],a['phase_s21_deg']))])]
                if a['exception']:
                    lines += ['```text',a['exception']['text'],'```','']
                lines += [f'{v}, angle(S21), all bins:','',table(['f (GHz)','angle(S21) (deg)'],
                      [[f/1e9,p] for f,p in zip(d['freqs_hz'],d['s21_phase_deg'])])]
        for v,d in r['variants'].items():
            if 'freqs_hz' not in d:
                lines += ['```text',d['status'].get('traceback',''),'```','']
                continue
            for s in ('s11','s21'):
                lines += [f'{fixture}/{v}, |{s.upper()}| (linear magnitude):','',
                          table(['f (GHz)',f'|{s.upper()}|'],[[f/1e9,y] for f,y in zip(d['freqs_hz'],d[s+'_mag'])])]
        for variant,records in r['edges'].items():
            if records:
                lines += [render_record(f'{fixture}/{variant}: GPU input arrays, drive 0',records[0])]
                d=r['variants'][variant]
                if len(records)>1:
                    lines += ['Drive 0 / drive 1 array equality (0/1):','',table(['array','equal'],list(d['edge_arrays_equal_between_drives'].items())),
                              f"Drive 1 source x indices received = {records[1]['source_x_indices_received']}; readback = `{records[1]['record_file']}`.",'']
    with (OUT/'TABLE.md').open('x') as f:
        f.write('\n'.join(lines))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('item',choices=(*FIXTURES,'table'))
    args=ap.parse_args()
    if args.item=='table':
        render_table()
    else:
        reduce_fixture(args.item)

if __name__=='__main__':
    main()
