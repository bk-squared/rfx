"""Read source-work witnesses; preserve distinction from first-plane MSL S."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from source_work_observable import waves,assemble,dft


def summary(s,f):
    gain=np.linalg.svd(s.transpose(2,0,1),compute_uv=False)[:,0]**2
    k=int(np.argmax(gain))
    return dict(max_coherent_gain=float(gain[k]),frequency_hz=float(f[k]),
                max_column_power=float(np.max(np.sum(abs(s)**2,axis=0))),
                max_entry_magnitude=float(np.max(abs(s))),
                reciprocity_max_abs=float(np.max(abs(s-s.transpose(1,0,2)))))


def audit(root,out):
    plan=json.loads((root/'plan.json').read_text());dt=plan['dt_s'];ports=plan['ports']
    profiles=[p['profile_e'] for p in ports];volumes=[p['volumes_m3'] for p in ports]
    sigmas=[p['added_sigma_s_per_m'] for p in ports]
    records=[];quality=[]
    for drive in range(2):
        with np.load(root/f'drive-{drive}-source-work.npz',allow_pickle=False) as data:
            post=[data[f'p{p}_ez_post'] for p in range(2)];u=data['source_u'];f=data['actual_freqs_hz'].astype(float)
            record=waves(post,u,profiles,volumes,sigmas,dt,f)
            checks=[]
            for p,port in enumerate(ports):
                g=np.asarray(profiles[p]);m=np.asarray(volumes[p]);norm=record['profile_norm'][p]
                projected_post=post[p].astype(float)@(m*g)/norm
                projected_mid=(projected_post+np.concatenate([[0.],projected_post[:-1]]))*.5
                tail=max(1,len(projected_mid)//10)
                ratio=np.mean(projected_mid[-tail:]**2)/max(np.max(projected_mid**2),1e-300)
                stream=np.asarray(data[f'p{p}_ez_dft'],dtype=complex)
                ylo,_,zlo,_=port['dft_region'];cells=np.asarray(port['cells'])
                local=stream[:,cells[:,1]-ylo,cells[:,2]-zlo]
                projected_stream=local@(m*g)/norm
                # Separate clock check. Streaming E DFT is post-E at n+1
                # with production float32 time phases; this independent
                # transform uses float64 time phases on the same samples.
                post_clock=dft(projected_post[:,None],dt,f)[:,0]*np.exp(-1j*np.pi*f*dt)
                delta=abs(projected_stream-post_clock)
                checks.append(dict(port=p,source_projection_settling_db=float(10*np.log10(max(ratio,1e-300))),
                                   streaming_vs_offline_post_relative_peak=float(np.max(delta)/max(np.max(abs(post_clock)),1e-300))))
            records.append(record)
            quality.append(dict(drive=drive,source_point_checks=checks,
                                active_source_wave_min_relative_to_band_peak=float(np.min(abs(record['a'][drive]))/max(np.max(abs(record['a'][drive])),1e-300))))
    s,cond=assemble(records)
    with np.load(root/'raw-vi.npz',allow_pickle=False) as data:
        msl=np.asarray(data['production_smatrix'],dtype=complex)
    report=dict(scope='power-conjugate source-model network; NOT the first-probe-plane MSL S or its calibration',
                source_reference_ohm=records[0]['reference_impedances'].tolist(),
                source_work_response=summary(s,f),first_plane_msl_response=summary(msl,f),
                source_drive_condition_max=float(np.max(cond)),quality=quality,
                source_projection_settling_definition='10*log10(mean(phi_mid[-floor(N/10):]**2)/max(phi_mid**2)); last10percent RMS over whole-record peak, not tail-peak decay',
                actual_freqs_hz=f.tolist(),finite_window_error_bound=None,
                source_shape_rounding_error='base u and ideal e recorded; source fl(Cb*e*u) equality is only to float32 precision',
                verdict='diagnostic only; no MSL support promotion')
    assert not out.exists() and not out.with_suffix('.npz').exists()
    np.savez_compressed(out.with_suffix('.npz'),source_work_s=s,first_plane_msl_s=msl,freqs_hz=f,
                        source_reference_ohm=records[0]['reference_impedances'],source_cond_a=cond,
                        source_a=np.stack([r['a'] for r in records]),source_b=np.stack([r['b'] for r in records]),
                        source_voltage=np.stack([r['voltage'] for r in records]),source_current=np.stack([r['source_current'] for r in records]))
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();audit(a.root,a.out)
