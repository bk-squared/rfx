import ast
import json
from pathlib import Path
import traceback
import numpy as np
import measure_base as m
from readback_report import FACTORS, table

OUT = m.OUT

def read(path):
    return json.loads(path.read_text())

def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(count=int(x.size),finite_count=int(np.isfinite(x).sum()),
                min_signed=float(x.min()),max_signed=float(x.max()),mean_signed=float(x.mean()),
                max_abs=float(abs(x).max()),mean_abs=float(abs(x).mean()))

def analytic(folder, freqs, beta):
    cv=m.load_module('validation/crossval/20_msl_phase_referee.py')
    er=read(folder/'assembly_received_00.json')
    dx=er['dx_m']
    lower=min(er['realized_trace_z_indices'])
    plane=next(p for p in er['planes'] if p['z_index']==lower)
    yl,yh=plane['trace_y_index_slice']
    width=(yh-yl-1)*dx
    with np.load(folder/'assembly_received_00.npz') as z:
        eps=z['eps_r']
        subrows=np.flatnonzero(eps[er['grid_shape'][0]//2,plane['y_index'],:]>1)
        hsub=(int(subrows[-1])+1-er['face_pads']['z_lo'])*dx
    meta=dict(h_sub_realized_m=hsub,trace_realization_kind='volume',dx_m=dx,
              trace_wall_planes_realized=er['realized_trace_z_indices'],
              trace_wall_planes_realized_z_m=[(k-er['face_pads']['z_lo'])*dx for k in er['realized_trace_z_indices']])
    height=cv._h_dielectric_under_strip(meta)
    eff=cv._hammerstad_jensen_eps_eff(width,height,cv.B_EPS_R)
    error=None
    try:
        witness=cv._analytic_beta_witness(freqs,beta,eps_eff=eff,tol_frac=cv.B_BETA_ANALYTIC_TOL_FRAC,
                                         label='stage_b_analytic_beta',solver='rfx')
    except RuntimeError as exc:
        error=dict(type=type(exc).__name__,text=str(exc),traceback=traceback.format_exc())
        witness=ast.literal_eval(str(exc).split('Full result: ',1)[1])
    mask=cv._gate_band_mask(freqs)
    assert int(mask.sum())==9
    deviation=100*(np.asarray(witness['beta_ratio_measured_over_analytic'])-1)
    return dict(width_m=width,height_m=height,eps_r=cv.B_EPS_R,eps_eff=eff,c0_m_s=cv._C0,
                indices=np.flatnonzero(mask).tolist(),freqs_hz=freqs[mask].tolist(),
                beta_fitted_rad_m=witness['beta_measured_rad_per_m'],
                beta_analytic_rad_m=witness['beta_analytic_rad_per_m'],
                signed_deviation_pct=deviation.tolist(),statistics_pct=stats(deviation),exception=error)

def tail_metrics(ts, dt, probe):
    y=np.asarray(ts[:,probe],dtype=float)
    assert np.isfinite(y).all()
    start=int(.4*len(y))
    inds=np.arange(start,len(y))
    blocks=np.array_split(inds,12)
    maxima=np.array([np.max(abs(y[b])) for b in blocks])
    centres=np.array([np.mean(b)*dt*1e9 for b in blocks])
    assert np.all(maxima>0)
    rate,intercept=np.polyfit(centres,np.log(maxima),1)
    prediction=rate*centres+intercept
    sse=np.sum((np.log(maxima)-prediction)**2)
    sst=np.sum((np.log(maxima)-np.log(maxima).mean())**2)
    segment=y[start:]
    spectrum=abs(np.fft.rfft((segment-segment.mean())*np.hanning(len(segment))))
    freqs=np.fft.rfftfreq(len(segment),dt)
    ranked=np.argsort(-spectrum[1:],kind='stable')[:3]+1
    local=np.flatnonzero((spectrum[1:-1]>spectrum[:-2]) & (spectrum[1:-1]>=spectrum[2:]))+1
    peaks=local[np.argsort(-spectrum[local],kind='stable')[:3]]
    return dict(probe=probe,steps=len(y),segment_start_index=start,segment_steps=len(segment),
                segment_duration_ns=len(segment)*dt*1e9,fft_spacing_GHz=(freqs[1]-freqs[0])/1e9,
                blocks=[dict(start_index=int(b[0]),stop_exclusive=int(b[-1]+1),steps=len(b),
                             centre_ns=float(c),max_abs=float(v)) for b,c,v in zip(blocks,centres,maxima)],
                rate_per_ns=float(rate),reciprocal_ns=float(1/rate),decay_time_ns=float(-1/rate),
                intercept=float(intercept),r_squared=float(1-sse/sst),
                strongest_bins=[dict(bin=int(i),frequency_GHz=float(freqs[i]/1e9),magnitude=float(spectrum[i])) for i in ranked],
                strongest_local_maxima=[dict(bin=int(i),frequency_GHz=float(freqs[i]/1e9),magnitude=float(spectrum[i])) for i in peaks])

def main():
    result=dict(cv20={},patch={},exceptions=[])
    data={}
    for label,factor in FACTORS:
        folder=OUT/f'cv20_{label}'
        status=read(folder/'status.json')
        row=dict(factor=factor,status=status)
        result['cv20'][label]=row
        if not status['completed']:
            result['exceptions'].append(dict(item=folder.name,record=status))
            continue
        with np.load(folder/'s.npz') as z:
            freqs=np.asarray(z['freqs'],dtype=float)
            s=np.asarray(z['S'],dtype=np.complex128)
        with np.load(folder/'diagnostics.npz') as z:
            diag={k:np.array(z[k]) for k in z.files}
        raw=diag['S_raw']
        data[label]=(freqs,s)
        row.update(max_reciprocity=float(abs(s[0,1]-s[1,0]).max()),
                   max_column_power_raw=float(np.sum(abs(raw)**2,axis=0).max()),
                   max_column_power_corrected=float(np.sum(abs(s)**2,axis=0).max()),
                   max_passivity_correction=float(diag['passivity_correction'].max()),
                   ringdown_db=diag['settling_db'].tolist(), ringdown_recomputed=[])
        dt=read(folder/'assembly_received_00.json')['dt_s']
        for drive in (0,1):
            with np.load(folder/f'witness_series_{drive:02d}.npz') as z:
                ts=np.asarray(z['time_series'],dtype=float)
            n_tail=max(1,len(ts)//10)
            energy=ts**2
            db=10*np.log10((energy[-n_tail:].mean(axis=0)+np.finfo(float).tiny)/(energy.max(axis=0)+np.finfo(float).tiny))
            row['ringdown_recomputed'].append(dict(drive=drive,steps=len(ts),probes=ts.shape[1],tail_steps=n_tail,
                per_probe_db=db.tolist(),worst_db=float(db.max()),difference_db=float(diag['settling_db'][drive]-db.max())))
            if drive==0:
                row['tail']=[tail_metrics(ts,dt,p) for p in (0,9)]
        row['analytic_beta']=analytic(folder,freqs,diag['beta'])
        if row['analytic_beta']['exception']:
            result['exceptions'].append(dict(item=folder.name+'/analytic_beta',record=row['analytic_beta']['exception']))
    if 'f1' in data:
        bf,bs=data['f1']
        for label,(freqs,s) in data.items():
            assert np.array_equal(freqs,bf)
            deltas={}
            for i in range(2):
                for j in range(2):
                    db=20*np.log10(abs(s[i,j]))-20*np.log10(abs(bs[i,j]))
                    phase=np.angle(s[i,j]*np.conj(bs[i,j]),deg=True)
                    phase[(abs(s[i,j])==0)|(abs(bs[i,j])==0)]=np.nan
                    deltas[f'S{i+1}{j+1}']=dict(dB=stats(db),phase_deg=stats(phase))
            result['cv20'][label]['deltas_against_1']=deltas
    for label,factor in FACTORS:
        result['patch'][label]=dict(factor=factor,**read(OUT/f'patch_{label}'/'result.json'))
    m.write_json(OUT/'reduced.json',result)
    lines=['S = public result.S; raw = public result.S_raw.',
           'Ring-down (dB) = 10 log10(last-10%-mean(Ez²) / peak(Ez²)); maximum over probes, per drive.',
           'Delta dB = 20 log10(abs(S_factor)) - 20 log10(abs(S_1)); phase delta = arg(S_factor * conj(S_1)), degrees.',
           'Beta deviation (%) = 100 * (Re(beta) / beta_HJ - 1); 9 bins, 3.0–4.5 GHz.',
           'Tail = indices floor(0.4*N):N; 12 blocks via numpy.array_split; log(max(abs(Ez))) fitted against block-centre time in ns.',
           'Signed reciprocal (ns) = 1/rate; decay time (ns) = -1/rate.',
           'Spectrum = abs(rfft((segment - mean(segment))*numpy.hanning(N_segment))); positive frequencies; no zero padding.',
           'FFT-bin ranks = descending magnitude; local-maximum ranks = descending magnitude among adjacent-bin maxima.', '']
    good=[r for r in result['cv20'].values() if 'max_reciprocity' in r]
    lines += ['cv20', '',table(['factor','drive 0 ring-down (dB)','drive 1 ring-down (dB)','max abs(S12-S21)',
                'raw max column power','corrected max column power','max passivity correction'],
                [[r['factor'],*r['ringdown_db'],r['max_reciprocity'],r['max_column_power_raw'],r['max_column_power_corrected'],r['max_passivity_correction']] for r in good]),'']
    lines += [table(['factor','beta min deviation (%)','beta max deviation (%)','beta mean deviation (%)'],
                    [[r['factor']]+[r['analytic_beta']['statistics_pct'][k] for k in ('min_signed','max_signed','mean_signed')] for r in good]),'']
    delta_rows=[]
    for r in good:
        for entry,values in r.get('deltas_against_1',{}).items():
            for unit,v in values.items():
                delta_rows.append([r['factor'],entry,unit,v['max_abs'],v['mean_abs'],v['min_signed'],v['max_signed'],v['mean_signed']])
    lines += [table(['factor','S entry','unit','max abs change','mean abs change','min signed change','max signed change','mean signed change'],delta_rows),'']
    lines += [table(['factor','probe','rate (1/ns)','1/rate (ns)','-1/rate (ns)','R²','start index','segment steps','FFT spacing (GHz)'],
                    [[r['factor'],t['probe'],t['rate_per_ns'],t['reciprocal_ns'],t['decay_time_ns'],t['r_squared'],t['segment_start_index'],t['segment_steps'],t['fft_spacing_GHz']] for r in good for t in r['tail']]),'']
    for key in ('strongest_bins','strongest_local_maxima'):
        lines += [key, '',table(['factor','probe','rank','bin','f (GHz)','FFT magnitude'],
                   [[r['factor'],t['probe'],i+1,s['bin'],s['frequency_GHz'],s['magnitude']]
                    for r in good for t in r['tail'] for i,s in enumerate(t[key])]),'']
    lines += ['Patch: n = 2; pad_h = 10; CPML layers = 4; periods = 150.', '',
              table(['factor','settling (dB)','probe 0 rate (1/step)','probe 1 rate (1/step)','probe 2 rate (1/step)','probe 3 rate (1/step)','worst rate (1/step)'],
                    [[r['factor'],r['settling_db'],*r['rates_per_step'],r['worst_rate_per_step']] for r in result['patch'].values() if r['completed']]),'']
    for r in good:
        a=r['analytic_beta']
        lines += [f"cv20 factor = {r['factor']:.17g}; beta: w = {a['width_m']*1e6:.17g} µm; h = {a['height_m']*1e6:.17g} µm; eps_r = {a['eps_r']:.17g}; eps_eff = {a['eps_eff']:.17g}.", '',
                  table(['bin','f (GHz)','beta fitted (rad/m)','beta HJ (rad/m)','signed deviation (%)'],
                        [[i,f/1e9,b,h,d] for i,f,b,h,d in zip(a['indices'],a['freqs_hz'],a['beta_fitted_rad_m'],a['beta_analytic_rad_m'],a['signed_deviation_pct'])]),'']
    with (OUT/'TABLE.md').open('x') as f:
        f.write('\n'.join(lines))
    print(json.dumps(dict(cv20_factors=len(good),patch_factors=sum(r['completed'] for r in result['patch'].values()),exceptions=len(result['exceptions']))))

if __name__=='__main__':
    main()
